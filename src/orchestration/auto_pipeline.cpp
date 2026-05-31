#include "neurograph/orchestration/auto_pipeline.hpp"

#include "neurograph/core/network.hpp"
#include "neurograph/dataqc/scanner.hpp"
#include "neurograph/eval/baselines.hpp"
#include "neurograph/eval/classification.hpp"
#include "neurograph/eval/learning_curve.hpp"
#include "neurograph/eval/regression.hpp"
#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/reporting/manifest.hpp"
#include "neurograph/reporting/model_card.hpp"
#include "neurograph/snapshot/snapshot_metadata.hpp"
#include "neurograph/training/data_loader.hpp"
#include "neurograph/training/io_manager.hpp"
#include "neurograph/training/loss.hpp"
#include "neurograph/training/training_engine.hpp"
#include "neurograph/util/uuid.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace neurograph::orchestration {

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Bundle artifact filenames — kept in lockstep with reporting/manifest.cpp's
// kArtifactDataset / kArtifactPipeline / kArtifactModel constants. P1-07
// (predict.exe) and P1-08 (DLL) read these by the same names.
// ---------------------------------------------------------------------------
constexpr const char* kBundlePipeline = "pipeline.ngpipe";
constexpr const char* kBundleModel    = "model.ngm";
constexpr const char* kBundleManifest = "manifest.json";
constexpr const char* kBundleCard     = "card.md";

// Stage progress weights — they sum to 1.0 and decide how each stage's
// internal completion maps onto `Progress::overall_fraction`. The
// training stage gets the lion's share because, for real models, that's
// where the wall time goes; the linear-probe stand-in finishes much
// faster but we want the bar to behave the same when the real trainer
// lands. Order matches the Stage enum (Scanning ... Packaging).
constexpr double kStageWeights[] = {
    0.05,  // Scanning
    0.10,  // Preprocessing
    0.02,  // AutoConfig
    0.45,  // Training
    0.08,  // Evaluating
    0.10,  // Baselining
    0.15,  // LearningCurve
    0.05,  // Packaging
};
constexpr int kNumStages = sizeof(kStageWeights) / sizeof(kStageWeights[0]);

// Sum of weights for stages strictly before `s`. Used to convert
// (stage, stage_fraction) → overall_fraction.
double overall_before(Stage s) {
    const int idx = static_cast<int>(s);
    double sum = 0.0;
    for (int i = 0; i < idx && i < kNumStages; ++i) sum += kStageWeights[i];
    return sum;
}

double overall_at(Stage s, double stage_fraction) {
    const int idx = static_cast<int>(s);
    if (idx >= kNumStages) return 1.0;
    return overall_before(s) + kStageWeights[idx] * std::clamp(stage_fraction, 0.0, 1.0);
}

void emit(const ProgressCb& cb, Stage s, double sf, std::string msg) {
    if (!cb) return;
    Progress p;
    p.stage            = s;
    p.stage_fraction   = std::clamp(sf, 0.0, 1.0);
    p.overall_fraction = overall_at(s, sf);
    p.message          = std::move(msg);
    cb(p);
}

bool is_cancelled(const CancelCb& cb) {
    return cb && cb();
}

// ---------------------------------------------------------------------------
// Stand-in trainer.
//
// The orchestrator's "Training" stage uses the same numerically-tested
// linear-probe core that powers `eval::compare_to_baselines`. This is a
// deliberate placeholder so the orchestrator can ship end-to-end now;
// when the real `TrainingEngine` adapter lands, only this struct + the
// two `train_*` helpers below need to change. The model artifact format
// `model.ngm` is also defined here — `predict.exe` reads it with the
// same field names.
// ---------------------------------------------------------------------------

struct LinearModelCls {
    int                              n_classes = 0;
    int                              n_features = 0;
    std::vector<std::vector<double>> W;       // K × D
    std::vector<double>              b;       // K
    std::vector<std::string>         class_labels;
};

struct LinearModelReg {
    int                  n_features = 0;
    std::vector<double>  w;                   // D
    double               bias = 0.0;
};

void softmax_inplace(std::vector<double>& z) {
    if (z.empty()) return;
    const double m = *std::max_element(z.begin(), z.end());
    double s = 0.0;
    for (double& v : z) { v = std::exp(v - m); s += v; }
    if (s > 0.0) for (double& v : z) v /= s;
}

// argmax of a one-hot / probability row.
int argmax_row(const std::vector<double>& v) {
    if (v.empty()) return -1;
    int k = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[k]) k = i;
    return k;
}

LinearModelCls train_classification(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>&                 y_idx,
    int                                     n_classes,
    int                                     n_iters,
    double                                  lr) {

    LinearModelCls m;
    m.n_classes = n_classes;
    if (X_train.empty() || n_classes < 2) return m;

    const std::size_t N = X_train.size();
    const std::size_t D = X_train.front().size();
    m.n_features = static_cast<int>(D);
    m.W.assign(n_classes, std::vector<double>(D, 0.0));
    m.b.assign(n_classes, 0.0);

    for (int it = 0; it < n_iters; ++it) {
        std::vector<std::vector<double>> probs(N, std::vector<double>(n_classes, 0.0));
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<double> z(n_classes, 0.0);
            for (int k = 0; k < n_classes; ++k) {
                double s = m.b[k];
                for (std::size_t d = 0; d < D; ++d) s += m.W[k][d] * X_train[i][d];
                z[k] = s;
            }
            softmax_inplace(z);
            probs[i] = z;
        }
        std::vector<std::vector<double>> dW(n_classes, std::vector<double>(D, 0.0));
        std::vector<double>              db(n_classes, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            for (int k = 0; k < n_classes; ++k) {
                const double err = probs[i][k] - ((y_idx[i] == k) ? 1.0 : 0.0);
                db[k] += err;
                for (std::size_t d = 0; d < D; ++d) dW[k][d] += err * X_train[i][d];
            }
        }
        const double inv_n = 1.0 / static_cast<double>(N);
        for (int k = 0; k < n_classes; ++k) {
            m.b[k] -= lr * inv_n * db[k];
            for (std::size_t d = 0; d < D; ++d) m.W[k][d] -= lr * inv_n * dW[k][d];
        }
    }
    return m;
}

std::vector<std::vector<double>> predict_classification(
    const LinearModelCls& m,
    const std::vector<std::vector<double>>& X) {

    std::vector<std::vector<double>> out(X.size(),
                                         std::vector<double>(m.n_classes, 0.0));
    if (m.n_classes < 2) return out;
    for (std::size_t i = 0; i < X.size(); ++i) {
        std::vector<double> z(m.n_classes, 0.0);
        for (int k = 0; k < m.n_classes; ++k) {
            double s = m.b[k];
            for (std::size_t d = 0; d < X[i].size() && d < static_cast<std::size_t>(m.n_features); ++d) {
                s += m.W[k][d] * X[i][d];
            }
            z[k] = s;
        }
        softmax_inplace(z);
        out[i] = std::move(z);
    }
    return out;
}

LinearModelReg train_regression(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<double>&              y_train,
    int                                     n_iters,
    double                                  lr,
    double                                  l2) {

    LinearModelReg m;
    if (X_train.empty()) return m;
    const std::size_t N = X_train.size();
    const std::size_t D = X_train.front().size();
    m.n_features = static_cast<int>(D);
    m.w.assign(D, 0.0);
    m.bias = 0.0;

    for (int it = 0; it < n_iters; ++it) {
        std::vector<double> errs(N, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            double yhat = m.bias;
            for (std::size_t d = 0; d < D; ++d) yhat += m.w[d] * X_train[i][d];
            errs[i] = yhat - y_train[i];
        }
        std::vector<double> dw(D, 0.0);
        double              db_acc = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            db_acc += errs[i];
            for (std::size_t d = 0; d < D; ++d) dw[d] += errs[i] * X_train[i][d];
        }
        const double inv_n = 1.0 / static_cast<double>(N);
        m.bias -= lr * 2.0 * inv_n * db_acc;
        for (std::size_t d = 0; d < D; ++d) {
            m.w[d] -= lr * (2.0 * inv_n * dw[d] + 2.0 * l2 * m.w[d]);
        }
    }
    return m;
}

std::vector<double> predict_regression(const LinearModelReg& m,
                                       const std::vector<std::vector<double>>& X) {
    std::vector<double> out(X.size(), 0.0);
    for (std::size_t i = 0; i < X.size(); ++i) {
        double yhat = m.bias;
        for (std::size_t d = 0; d < X[i].size() && d < static_cast<std::size_t>(m.n_features); ++d) {
            yhat += m.w[d] * X[i][d];
        }
        out[i] = yhat;
    }
    return out;
}

// ---------------------------------------------------------------------------
// SON-trained model (P1-15) — replaces the placeholder for the main
// Training stage. The network is constructed fresh per run; self-org
// grows hidden neurons during training when interleave_self_org is on.
//
// We hold the trained artifacts (network, IO mapping, normalization
// stats, predictions) in a single struct so the orchestrator can
// pass them to the bundle-write step without re-running training.
// ---------------------------------------------------------------------------

struct SonTrainedModel {
    // Owned: the network grew under self-org so the orchestrator owns
    // it (rather than letting TrainingEngine destruct it on scope exit).
    std::unique_ptr<core::Network>           network;
    // Snapshotted from the engine's IOManager — neuron-id→feature/output
    // mapping needed by the bundle loader.
    std::vector<std::string>                 input_neuron_ids;
    std::vector<std::string>                 output_neuron_ids;
    std::vector<std::string>                 class_labels;
    int                                      n_features = 0;
    int                                      n_outputs  = 0;
    // Final-epoch metric numbers, surfaced into the manifest.
    int                                      epochs_run = 0;
    double                                   final_train_loss = 0.0;
    double                                   final_val_loss   = 0.0;
};

// Convert a TransformedDataset row pair into a Sample list. The
// preprocess pipeline already produced normalized features and one-hot
// targets (cls) or scalar-in-vector targets (reg), which is exactly
// the schema TrainingEngine::load_prepared_dataset expects.
std::vector<training::Sample> to_samples(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& y,
    bool is_cls,
    const std::vector<std::string>&         class_labels) {

    std::vector<training::Sample> out;
    out.reserve(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) {
        training::Sample s;
        s.features = X[i];
        s.target   = y[i];
        if (is_cls) {
            const int cls_idx = argmax_row(y[i]);
            if (cls_idx >= 0 && cls_idx < static_cast<int>(class_labels.size())) {
                s.label = class_labels[cls_idx];
            }
        }
        out.push_back(std::move(s));
    }
    return out;
}

// Compute a sane epoch budget from the wizard's time_budget_seconds.
// Rough mapping (matches how the wizard's Quick/Balanced/Best radio
// translates):
//   - <60s    (Quick)    →  30 epochs
//   - 60-300s (Balanced) →  80 epochs
//   - >300s   (Best)     → 200 epochs
// The optimizer's early-stopping (patience=20) often bails earlier.
int epochs_from_budget(int time_budget_seconds) {
    if (time_budget_seconds <  60) return 30;
    if (time_budget_seconds < 300) return 80;
    return 200;
}

// Train the SON network end-to-end. Returns the trained network +
// inference metadata, or an Error suitable for `fail()` rendering.
util::Result<SonTrainedModel> train_with_engine(
    const preprocess::TransformedDataset& ds,
    bool                                  is_cls,
    int                                   n_classes,
    int                                   time_budget_seconds,
    std::uint64_t                         seed,
    const ProgressCb&                     progress,
    const CancelCb&                       cancel) {

    // Deterministic UUIDs + deterministic self-org for the duration of
    // the training stage so `model.ngm` is byte-reproducible across runs
    // with the same seed.
    //   - util::DeterministicUuidScope (P1-15): routes
    //     core::NeuronId::generate() through a seeded RNG.
    //   - core::DeterministicSelfOrgScope (P1-19): same idea for the
    //     RNG inside Network::self_organize / grow_neurons / etc., so
    //     enabling interleave_self_org below doesn't break determinism.
    // Distinct seed offsets so the two RNG streams aren't aliased
    // (sharing one seed across two independent random sources is a
    // classic source of subtle correlation bugs).
    util::DeterministicUuidScope         uuid_scope(seed);
    core::DeterministicSelfOrgScope      selforg_scope(seed ^ 0x5a5a5a5a5a5a5a5aULL);

    SonTrainedModel out;
    out.network    = std::make_unique<core::Network>();
    out.n_features = static_cast<int>(ds.X_train.front().size());
    out.n_outputs  = is_cls ? n_classes : 1;
    if (is_cls) out.class_labels = ds.class_labels;

    // P1-16 + P1-19 bounded growth: self-org is now safe to enable in
    // the orchestrator. Conservative caps so growth stays subtle on
    // the small datasets typical of the wizard's first-run flow:
    //   - max_neurons = 256 (hard ceiling).
    //   - max_split_fraction_per_cycle = 0.05 (each cycle splits at
    //     most 5% of the network — chosen so 80-epoch training with
    //     self-org every 10 epochs produces ~8 growth events that
    //     compound to ~1.05^8 ≈ 1.5× initial size, not so large
    //     that it disrupts the regression's gradient descent).
    //   - growth_patience = 10 (gate on real loss-stall, not just
    //     epoch count, so well-fitting models don't grow needlessly).
    {
        core::TopologyConfig tcfg;
        tcfg.max_neurons                  = 256;
        tcfg.max_split_fraction_per_cycle = 0.05;
        tcfg.max_hidden_neurons           = 200;
        tcfg.use_stall_counter            = true;
        tcfg.growth_patience              = 10;
        out.network->set_topology_config(tcfg);
    }

    training::TrainingEngine engine(*out.network);

    auto train_samples = to_samples(ds.X_train, ds.y_train, is_cls, ds.class_labels);
    auto val_samples   = to_samples(ds.X_val,   ds.y_val,   is_cls, ds.class_labels);

    // P1-18: explicit hidden-layer sizing. Auto-pick (clamp(2 * features, 8, 64))
    // is fine for the orchestrator's tabular workloads.
    auto load_r = engine.load_prepared_dataset(
        std::move(train_samples), std::move(val_samples),
        out.n_features, out.n_outputs, out.class_labels, seed,
        training::TrainingEngine::kAutoHiddenSize);
    if (!load_r.ok()) {
        return util::Error("Training setup failed: " + load_r.error().message);
    }

    // P1-17: for classification, override the output-neuron activation
    // from the IOManager default (SIGMOID for cls heads) to LINEAR so
    // the values feeding CrossEntropyLoss's internal softmax are raw
    // logits with full gradient magnitude. Regression already gets
    // LINEAR by default per the IOManager heuristic.
    if (is_cls) {
        if (auto* iom = engine.io_manager()) {
            for (const auto& out_id : iom->output_ids()) {
                if (auto* n = out.network->get_neuron(out_id)) {
                    n->set_activation_type(core::ActivationType::LINEAR);
                }
            }
        }
    }

    // Snapshot the IO mapping right after load — these neuron ids are
    // stable for the rest of the network's life because IO neurons are
    // marked protected before self_organize() runs. Convert from
    // strong-typed core::NeuronId to plain std::string for the JSON
    // bundle schema (the bundle loader rebuilds NeuronId from string).
    if (auto* iom = engine.io_manager()) {
        out.input_neuron_ids.reserve(iom->input_ids().size());
        for (const auto& id : iom->input_ids())  out.input_neuron_ids.push_back(id.value());
        out.output_neuron_ids.reserve(iom->output_ids().size());
        for (const auto& id : iom->output_ids()) out.output_neuron_ids.push_back(id.value());
    }

    training::SupervisedConfig cfg;
    cfg.epochs              = epochs_from_budget(time_budget_seconds);
    // Cap batch_size at quarter-of-train so small datasets don't end
    // up with a single batch (which kills the SGD-noise benefit).
    cfg.batch_size          = std::max<std::size_t>(
        1, std::min<std::size_t>(16, ds.X_train.size() / 4));
    // P1-15/17: cls now uses CrossEntropyLoss on LINEAR-output logits
    // (the canonical setup) — the loss applies softmax internally and
    // produces well-conditioned gradients. Reg still uses MSE.
    cfg.loss_type           = is_cls ? training::LossType::CROSS_ENTROPY
                                     : training::LossType::MSE;
    cfg.optimizer_name      = "adam";
    // LR = 0.01 — verified to converge in the P1-18 unit test
    // (`TrainingEngine.HiddenLayerNetworkLearnsSeparableData`) at this
    // value; matches the proven Iris-classification setup in
    // `tests/integration/test_all_phases.cpp`.
    cfg.optimizer_config.learning_rate = 0.01;
    // P1-20: self-org re-enabled in growth-only mode. The full
    // `Network::self_organize` runs 9 phases, three of which (rewire,
    // Hebbian-strengthen, activation-mutation) overwrite weights the
    // supervised optimizer has been tuning and break convergence
    // (iris went 0.93 → 0.33 with full self-org running every 10
    // epochs). `growth_only_self_organize` runs only the prune-synapses
    // + prune-dead-neurons + grow-stressed phases, all of which adapt
    // capacity without disrupting backprop's gradient signal.
    // Combined with P1-16 bounded growth + P1-19 deterministic RNG,
    // the wizard now ships actual self-organizing capacity adaptation.
    // For classification: turn on. For regression: keep off — small
    // datasets + split events still cause modest fit regression even
    // with growth-only (R² 0.85 → 0.7 in EndToEndRegressionProducesBundle),
    // so we hold this off until we can size growth more conservatively.
    cfg.interleave_self_org = is_cls;
    cfg.self_org_interval   = 10;
    cfg.self_org_steps      = 1;
    cfg.self_org_growth_only = true;
    cfg.dropout_rate        = 0.0;      // Small nets; dropout hurts more than helps.
    cfg.batch_norm          = false;    // ditto
    cfg.patience            = 20;
    cfg.min_delta           = 1e-3;
    cfg.warmup_epochs       = std::min(5, cfg.epochs / 4);

    // Forward training progress to the wizard's progress bar.
    // Cancellation is checked every callback (every epoch) so users
    // get sub-second responsiveness.
    training::TrainingCallback cb =
        [&, total = cfg.epochs](const training::EpochResult& er) {
        const double frac = static_cast<double>(er.epoch + 1)
                          / static_cast<double>(total);
        std::ostringstream msg;
        msg << "Epoch " << (er.epoch + 1) << "/" << total
            << " — train_loss=" << std::fixed << std::setprecision(4)
            << er.train_loss
            << " val_loss="   << er.val_loss;
        emit(progress, Stage::Training, frac, msg.str());
        out.epochs_run       = er.epoch + 1;
        out.final_train_loss = er.train_loss;
        out.final_val_loss   = er.val_loss;
        if (is_cancelled(cancel)) {
            engine.request_stop();
        }
    };

    auto train_r = engine.train(cfg, cb);
    if (!train_r.ok()) {
        return util::Error("Training failed: " + train_r.error().message);
    }
    if (is_cancelled(cancel)) {
        return util::Error("Cancelled by user.");
    }

    return out;
}

// Predictions on a held-out set, going through the same forward pass
// the bundle's predict.exe will use. For classification we softmax the
// raw output activations because the network's outputs aren't
// constrained to the simplex by default.
// Build a snapshot::IOMapping from the SonTrainedModel so a fresh
// IOManager can be re-attached to the trained network without
// re-creating neurons. The neuron ids in the model are the strings
// we snapshotted right after `load_prepared_dataset` ran; they're
// stable because IO neurons are protected from self-org.
snapshot::IOMapping mapping_from_model(const SonTrainedModel& model) {
    snapshot::IOMapping m;
    m.input_neuron_ids  = model.input_neuron_ids;
    m.output_neuron_ids = model.output_neuron_ids;
    m.input_features    = model.n_features;
    m.output_classes    = model.n_outputs;
    m.class_labels      = model.class_labels;
    return m;
}

std::vector<std::vector<double>> predict_with_engine_cls(
    core::Network&                          network,
    const SonTrainedModel&                  model,
    const std::vector<std::vector<double>>& X) {

    // Re-attach an IOManager + Backprop to the trained network. Use
    // load_io_mapping (NOT ensure_io_neurons) so we bind to the
    // existing trained IO neurons rather than creating new ones.
    training::IOManager iom(network);
    auto io_r = iom.load_io_mapping(mapping_from_model(model));
    if (!io_r.ok()) return {};
    training::Backprop bp(network, iom);
    bp.set_training_mode(false);
    bp.rebuild_topo_order();

    std::vector<std::vector<double>> out;
    out.reserve(X.size());
    for (const auto& xv : X) {
        bp.forward(xv);
        auto z = iom.read_outputs();
        softmax_inplace(z);  // LINEAR outputs → softmax to probabilities
        out.push_back(std::move(z));
    }
    return out;
}

std::vector<double> predict_with_engine_reg(
    core::Network&                          network,
    const SonTrainedModel&                  model,
    const std::vector<std::vector<double>>& X) {

    training::IOManager iom(network);
    auto io_r = iom.load_io_mapping(mapping_from_model(model));
    if (!io_r.ok()) return {};
    training::Backprop bp(network, iom);
    bp.set_training_mode(false);
    bp.rebuild_topo_order();

    std::vector<double> out;
    out.reserve(X.size());
    for (const auto& xv : X) {
        bp.forward(xv);
        const auto z = iom.read_outputs();
        out.push_back(z.empty() ? 0.0 : z[0]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// model.ngm serialization.
//
// Two schemas live side-by-side now:
//   v1 — kind="classification"|"regression": linear-probe weights.
//        Still used by the learning-curve sub-runs (4 trainings) to
//        keep curve cost bounded; predict.exe still loads these.
//   v2 — kind="neural_cls_v1"|"neural_reg_v1": full SON-trained
//        network serialized via core::Network::to_json + the IO mapping
//        the bundle loader uses to wire IOManager back up.
// predict.exe / runtime/bundle.cpp dispatches on `kind`.
// ---------------------------------------------------------------------------
constexpr int kModelSchemaVersion = 2;

void save_model_classification(const LinearModelCls& m, const fs::path& path) {
    nlohmann::json j;
    j["schema_version"]    = 1;  // linear probe is the v1 schema
    j["kind"]              = "classification";
    j["n_classes"]         = m.n_classes;
    j["n_features"]        = m.n_features;
    j["class_labels"]      = m.class_labels;
    j["W"]                 = m.W;
    j["b"]                 = m.b;
    std::ofstream f(path);
    if (!f) throw std::runtime_error("auto_pipeline: cannot open " + path.string());
    f << j.dump(2);
}

void save_model_regression(const LinearModelReg& m, const fs::path& path) {
    nlohmann::json j;
    j["schema_version"] = 1;  // linear probe is the v1 schema
    j["kind"]           = "regression";
    j["n_features"]     = m.n_features;
    j["w"]              = m.w;
    j["bias"]           = m.bias;
    std::ofstream f(path);
    if (!f) throw std::runtime_error("auto_pipeline: cannot open " + path.string());
    f << j.dump(2);
}

// SON model serialization. The full Network grew under self-org, so we
// dump it via Network::to_json and pair it with the IO mapping (input
// neuron ids in feature order, output neuron ids in class order). The
// bundle loader reconstructs Network::from_json + IOManager + Backprop
// in inference mode.
void save_model_neural(const SonTrainedModel& sm,
                       bool                   is_cls,
                       const fs::path&        path) {
    nlohmann::json j;
    j["schema_version"]    = kModelSchemaVersion;  // 2
    j["kind"]              = is_cls ? "neural_cls_v1" : "neural_reg_v1";
    j["n_features"]        = sm.n_features;
    j["n_outputs"]         = sm.n_outputs;
    // bundle.cpp's neural_cls_v1 loader reads `n_classes` (mirrors the
    // linear probe's schema) — emit it as a duplicate of n_outputs for
    // cls bundles so the loader doesn't have to special-case missing
    // fields.
    if (is_cls) j["n_classes"] = sm.n_outputs;
    j["class_labels"]      = sm.class_labels;       // empty for reg
    j["network"]           = sm.network->to_json();
    j["io_mapping"]        = {
        {"input_neuron_ids",  sm.input_neuron_ids},
        {"output_neuron_ids", sm.output_neuron_ids},
        {"input_features",    sm.n_features},
        {"output_classes",    sm.n_outputs},
        {"class_labels",      sm.class_labels},
    };
    j["training_summary"]  = {
        {"epochs_run",        sm.epochs_run},
        {"final_train_loss",  sm.final_train_loss},
        {"final_val_loss",    sm.final_val_loss},
    };
    std::ofstream f(path);
    if (!f) throw std::runtime_error("auto_pipeline: cannot open " + path.string());
    f << j.dump(2);
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

// Convert one-hot training rows to integer class indices.
std::vector<int> argmax_rows(const std::vector<std::vector<double>>& rows) {
    std::vector<int> out;
    out.reserve(rows.size());
    for (const auto& r : rows) out.push_back(argmax_row(r));
    return out;
}

std::vector<double> flatten_scalar(const std::vector<std::vector<double>>& rows) {
    std::vector<double> out;
    out.reserve(rows.size());
    for (const auto& r : rows) out.push_back(r.empty() ? 0.0 : r[0]);
    return out;
}

// Per-stage iteration count derived from time_budget_seconds. Linear
// probe doesn't actually consume the full budget — this keeps the
// API contract honest so the swap to TrainingEngine can scale epochs
// from `time_budget_seconds * kStageWeights[Training]` later.
int iters_for_training(int time_budget_seconds) {
    // 100 iters fixed for the placeholder. We document the weight
    // calc so the future swap is obvious.
    (void)time_budget_seconds;  // placeholder uses fixed iter count
    return 100;
}

// One-liner the wizard's success toast renders. Format depends on
// task kind.
std::string make_summary(bool is_cls,
                         const eval::ClassificationMetrics* cls,
                         const eval::RegressionMetrics*     reg,
                         const eval::BaselineReport&        bl) {
    std::ostringstream os;
    if (is_cls && cls) {
        os << "Trained classifier: accuracy = "
           << std::fixed << std::setprecision(3) << cls->accuracy
           << " (lift +" << std::setprecision(3) << bl.lift_vs_best_baseline
           << " over best baseline). " << bl.verdict;
    } else if (!is_cls && reg) {
        os << "Trained regressor: R\xc2\xb2 = "
           << std::fixed << std::setprecision(3) << reg->r2
           << " (RMSE " << reg->rmse << "). " << bl.verdict;
    } else {
        os << "Training complete. " << bl.verdict;
    }
    return os.str();
}

// Pipeline → JSON for the manifest's embedded config blob.
nlohmann::json pipeline_config_to_json(const preprocess::PipelineConfig& c) {
    nlohmann::json j;
    j["target_column"] = c.target_column;
    j["val_fraction"]  = c.val_fraction;
    j["test_fraction"] = c.test_fraction;
    j["stratify"]      = c.stratify;
    j["split_seed"]    = c.split_seed;
    nlohmann::json cols = nlohmann::json::array();
    for (const auto& col : c.columns) {
        nlohmann::json cj;
        cj["name"]       = col.name;
        cj["is_target"]  = col.is_target;
        cj["is_numeric"] = col.is_numeric;
        cj["drop"]       = col.drop;
        cols.push_back(std::move(cj));
    }
    j["columns"] = std::move(cols);
    return j;
}

// Training config → JSON. P1-15: orchestrator now drives the SON
// `TrainingEngine`, so the manifest reflects the real network's
// hyperparameters and final-epoch metrics.
nlohmann::json training_config_to_json_son(const SonTrainedModel& sm,
                                            std::uint64_t          seed,
                                            int                    epochs,
                                            std::size_t            batch_size,
                                            bool                   is_cls) {
    nlohmann::json j;
    j["trainer"]              = "neurograph_son_v1";
    j["epochs_configured"]    = epochs;
    j["epochs_run"]           = sm.epochs_run;
    j["batch_size"]           = batch_size;
    j["optimizer"]            = "adam";
    j["learning_rate"]        = 0.01;
    j["loss"]                 = is_cls ? "cross_entropy" : "mse";
    j["interleave_self_org"]  = is_cls;   // P1-20: cls-only growth-only mode
    j["self_org_interval"]    = 10;
    j["self_org_growth_only"] = true;
    j["max_neurons_cap"]      = 256;
    j["seed"]                 = seed;
    j["final_train_loss"]     = sm.final_train_loss;
    j["final_val_loss"]       = sm.final_val_loss;
    j["network_neuron_count"] = sm.network ? sm.network->neuron_count() : 0;
    j["network_synapse_count"]= sm.network ? sm.network->synapse_count() : 0;
    return j;
}

// Linear-probe variant retained for the learning-curve sub-runs which
// still use the linear trainer (4 trainings; SON would 4x wall time).
// Not currently wired into the manifest path but kept to avoid
// dead-code warnings if another caller wants it later.
[[maybe_unused]] nlohmann::json training_config_to_json(
    int iters, double lr, std::uint64_t seed) {
    nlohmann::json j;
    j["trainer"]       = "linear_probe_v1";
    j["iters"]         = iters;
    j["learning_rate"] = lr;
    j["seed"]          = seed;
    return j;
}

util::Error fail(std::string msg, int code = 1) {
    return util::Error(std::move(msg), code);
}

} // namespace

// ---------------------------------------------------------------------------
// run — the orchestrator. Each stage emits start/end progress, polls
// cancel, and converts its native error type to util::Error.
// ---------------------------------------------------------------------------

util::Result<AutoResult>
run(const std::filesystem::path& csv,
    const std::string&           target_column,
    const AutoOptions&           opts,
    ProgressCb                   progress,
    CancelCb                     cancel) {

    AutoResult res;

    // --- Stage 1: Scanning -------------------------------------------------
    emit(progress, Stage::Scanning, 0.0, "Scanning dataset");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    res.dq = dataqc::scan_csv(csv);
    emit(progress, Stage::Scanning, 1.0, res.dq.summary_line);

    if (!res.dq.trainable) {
        std::string detail;
        for (const auto& f : res.dq.findings) {
            if (f.severity == dataqc::Severity::Error) {
                detail = f.message;
                break;
            }
        }
        return fail("Cannot train — " + (detail.empty() ? "data quality scan failed."
                                                        : detail));
    }
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- Stage 2: Preprocessing -------------------------------------------
    emit(progress, Stage::Preprocessing, 0.0, "Fitting preprocessing pipeline");

    preprocess::PipelineConfig pcfg;
    pcfg.target_column = target_column;
    pcfg.split_seed    = opts.seed;
    // columns left empty triggers auto-spec inference inside fit_pipeline.

    preprocess::TransformedDataset ds;
    std::unique_ptr<preprocess::FittedPipeline> pipe;
    try {
        pipe = preprocess::fit_pipeline(csv, pcfg, ds);
    } catch (const std::exception& e) {
        return fail(std::string("Pipeline fit failed: ") + e.what());
    }
    if (!pipe) return fail("Pipeline fit failed: returned null pipeline.");
    if (ds.X_train.empty() || ds.X_val.empty()) {
        return fail("Pipeline fit failed: train or val split empty after fit.");
    }
    emit(progress, Stage::Preprocessing, 1.0, "Pipeline fitted");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- Stage 3: AutoConfig ----------------------------------------------
    emit(progress, Stage::AutoConfig, 0.0, "Selecting model configuration");
    const bool is_cls   = ds.is_classification;
    const int  n_classes = is_cls ? static_cast<int>(ds.class_labels.size()) : 0;
    const int  iters     = iters_for_training(opts.time_budget_seconds);
    const double lr      = is_cls ? 0.1 : 0.05;
    emit(progress, Stage::AutoConfig, 1.0,
         is_cls ? "Classification model selected" : "Regression model selected");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- Stage 4: Training ------------------------------------------------
    //
    // P1-15 (final): the orchestrator now drives the real
    // `training::TrainingEngine`. Network is built fresh per run with:
    //   - Auto-allocated hidden layer (P1-18, ReLU + Glorot init).
    //   - LINEAR output activation for cls (P1-17) so CrossEntropyLoss's
    //     internal softmax sees raw logits with full gradient magnitude.
    //   - Bounded self-organization (P1-16: max_neurons=256, per-cycle
    //     fraction cap=10%) so the network can grow into hard examples
    //     without blowing up wall time.
    // The linear-probe `train_classification` / `train_regression`
    // helpers below are retained for the learning-curve sub-runs only,
    // where comparable shape under bounded wall time matters more than
    // absolute accuracy.
    emit(progress, Stage::Training, 0.0, "Training model");

    SonTrainedModel                  son_model;
    std::vector<std::vector<double>> val_probs;   // cls
    std::vector<double>              val_yhat;    // reg
    int    son_epochs_configured = epochs_from_budget(opts.time_budget_seconds);
    std::size_t son_batch_size   = std::max<std::size_t>(
        1, std::min<std::size_t>(16, ds.X_train.size() / 4));

    try {
        auto train_r = train_with_engine(
            ds, is_cls, n_classes,
            opts.time_budget_seconds, opts.seed,
            progress, cancel);
        if (!train_r.ok()) {
            return fail(train_r.error().message);
        }
        son_model = std::move(train_r).value();

        if (is_cls) {
            val_probs = predict_with_engine_cls(*son_model.network, son_model, ds.X_val);
            if (val_probs.empty()) {
                return fail("Inference failed: could not re-attach IOManager "
                            "for val-set prediction (saved IO neuron ids "
                            "may be stale).");
            }
        } else {
            val_yhat  = predict_with_engine_reg(*son_model.network, son_model, ds.X_val);
            if (val_yhat.empty()) {
                return fail("Inference failed: could not re-attach IOManager "
                            "for val-set prediction.");
            }
        }
    } catch (const std::exception& e) {
        return fail(std::string("Training failed: ") + e.what());
    }
    emit(progress, Stage::Training, 1.0,
         "Training complete — " + std::to_string(son_model.epochs_run) + " epochs, "
         + std::to_string(son_model.network ? son_model.network->neuron_count() : 0)
         + " neurons, "
         + std::to_string(son_model.network ? son_model.network->synapse_count() : 0)
         + " synapses");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- Stage 5: Evaluating ---------------------------------------------
    emit(progress, Stage::Evaluating, 0.0, "Evaluating on val set");
    if (is_cls) {
        const auto y_val_idx = argmax_rows(ds.y_val);
        eval::ClassificationMetrics cm;
        try {
            cm = eval::evaluate_classification(y_val_idx, val_probs, ds.class_labels);
        } catch (const std::exception& e) {
            return fail(std::string("Evaluation failed: ") + e.what());
        }
        res.metrics = cm;
    } else {
        const auto y_val_scalar = flatten_scalar(ds.y_val);
        res.metrics = eval::evaluate_regression(y_val_scalar, val_yhat);
    }
    emit(progress, Stage::Evaluating, 1.0, "Evaluation complete");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- Stage 6: Baselining ---------------------------------------------
    emit(progress, Stage::Baselining, 0.0, "Comparing against baselines");
    {
        // baselines module wants prob rows for cls and 1-element rows for reg.
        std::vector<std::vector<double>> model_val_preds;
        if (is_cls) {
            model_val_preds = val_probs;
        } else {
            model_val_preds.reserve(val_yhat.size());
            for (double y : val_yhat) model_val_preds.push_back({y});
        }
        res.baselines = eval::compare_to_baselines(ds, model_val_preds, is_cls, opts.seed);
    }
    emit(progress, Stage::Baselining, 1.0, res.baselines.verdict);
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- Stage 7: Learning curve -----------------------------------------
    if (opts.do_learning_curve) {
        emit(progress, Stage::LearningCurve, 0.0, "Computing learning curve");

        // Build a TrainerFn that wraps our linear probe. Same trainer
        // for every point so the curve is comparing apples to apples.
        const int  curve_iters = std::max(20, iters / 4);
        eval::TrainerFn trainer =
            [is_cls, n_classes, lr, curve_iters](
                const std::vector<std::vector<double>>& X_sub,
                const std::vector<std::vector<double>>& y_sub,
                const std::vector<std::vector<double>>& X_val,
                const std::vector<std::vector<double>>& y_val,
                int /*epochs*/, std::uint64_t /*seed*/) -> std::pair<double, double> {

                if (is_cls) {
                    auto y_sub_idx = argmax_rows(y_sub);
                    auto y_val_idx = argmax_rows(y_val);
                    auto m = train_classification(X_sub, y_sub_idx, n_classes,
                                                  curve_iters, lr);
                    auto train_probs = predict_classification(m, X_sub);
                    auto val_probs2  = predict_classification(m, X_val);

                    auto labels = std::vector<std::string>();
                    labels.reserve(n_classes);
                    for (int i = 0; i < n_classes; ++i)
                        labels.push_back(std::to_string(i));

                    const auto train_m = eval::evaluate_classification(
                        y_sub_idx, train_probs, labels);
                    const auto val_m   = eval::evaluate_classification(
                        y_val_idx, val_probs2, labels);
                    return { train_m.accuracy, val_m.accuracy };
                }
                auto y_sub_s = flatten_scalar(y_sub);
                auto y_val_s = flatten_scalar(y_val);
                auto m = train_regression(X_sub, y_sub_s, curve_iters, lr, 1e-4);
                auto train_yhat = predict_regression(m, X_sub);
                auto val_yhat2  = predict_regression(m, X_val);
                const auto train_m = eval::evaluate_regression(y_sub_s, train_yhat);
                const auto val_m   = eval::evaluate_regression(y_val_s, val_yhat2);
                return { train_m.r2, val_m.r2 };
            };

        eval::LearningCurveConfig lcc;
        lcc.seed             = opts.seed;
        lcc.epochs_per_point = curve_iters;
        res.curve = eval::compute_learning_curve(ds, is_cls, lcc, trainer);

        emit(progress, Stage::LearningCurve, 1.0, res.curve.message);
        if (is_cancelled(cancel)) return fail("Cancelled by user.");
    } else {
        emit(progress, Stage::LearningCurve, 1.0, "Learning curve skipped");
    }

    // --- Compose the one-line summary now that all metrics exist. ---------
    {
        const eval::ClassificationMetrics* cm =
            std::get_if<eval::ClassificationMetrics>(&res.metrics);
        const eval::RegressionMetrics*     rm =
            std::get_if<eval::RegressionMetrics>(&res.metrics);
        res.summary_one_liner = make_summary(is_cls, cm, rm, res.baselines);
    }

    // --- Stage 8: Packaging ----------------------------------------------
    emit(progress, Stage::Packaging, 0.0, "Writing bundle");
    if (opts.do_export_bundle) {
        if (opts.bundle_out_dir.empty()) {
            return fail("Bundle write failed: bundle_out_dir is empty.");
        }
        std::error_code ec;
        fs::create_directories(opts.bundle_out_dir, ec);
        if (ec) {
            return fail("Bundle write failed: " + opts.bundle_out_dir.string()
                        + ": " + ec.message());
        }

        const auto pipe_path     = opts.bundle_out_dir / kBundlePipeline;
        const auto model_path    = opts.bundle_out_dir / kBundleModel;
        const auto manifest_path = opts.bundle_out_dir / kBundleManifest;
        const auto card_path     = opts.bundle_out_dir / kBundleCard;

        try {
            preprocess::save_pipeline(*pipe, pipe_path);
            // P1-15: write schema_v2 SON bundle. The runtime/bundle.cpp
            // loader handles it via the neural_cls_v1 / neural_reg_v1
            // kind dispatcher — predict.exe / DLL / wizard all share
            // the same path now.
            save_model_neural(son_model, is_cls, model_path);
        } catch (const std::exception& e) {
            return fail(std::string("Bundle write failed: ") + e.what());
        }
        emit(progress, Stage::Packaging, 0.5, "Pipeline + model written");

        // Manifest — hashes pipeline + model. Dataset is referenced by
        // path on the input struct (see model card) but we don't copy
        // the raw CSV into the bundle (privacy). Empty hash for dataset
        // means verify_against will skip it.
        reporting::Manifest mf;
        mf.neurograph_version = "0.1.0";
        mf.build_git_sha      = "";          // populated by CI when known
        mf.dataset_sha256     = "";          // not tracked
        try {
            mf.pipeline_sha256 = reporting::sha256_of_file(pipe_path);
            mf.model_sha256    = reporting::sha256_of_file(model_path);
        } catch (const std::exception& e) {
            return fail(std::string("Bundle write failed: ") + e.what());
        }
        mf.rng_seed         = opts.seed;
        mf.timestamp_iso8601 = reporting::current_iso8601_utc();
        reporting::populate_host_info(mf);
        mf.pipeline_config = pipeline_config_to_json(pcfg);
        mf.training_config = training_config_to_json_son(
            son_model, opts.seed, son_epochs_configured, son_batch_size, is_cls);
        try {
            reporting::save_manifest(mf, manifest_path);
        } catch (const std::exception& e) {
            return fail(std::string("Bundle write failed: ") + e.what());
        }

        // Model card. The reporting module wants a SupervisedConfig
        // and pointers into the metrics variant — feed it placeholder
        // values that match what the linear probe actually did.
        reporting::ModelCardInputs mc;
        mc.model_name             = "Auto-trained model";
        mc.task_kind              = is_cls ? "classification" : "regression";
        mc.dq_report              = res.dq;
        mc.pipe_config            = pcfg;
        mc.train_config                       = training::SupervisedConfig{};
        mc.train_config.epochs                = son_epochs_configured;
        mc.train_config.batch_size            = son_batch_size;
        mc.train_config.optimizer_name        = "adam";
        mc.train_config.optimizer_config.learning_rate = 0.01;
        mc.train_config.loss_type             = is_cls
            ? training::LossType::CROSS_ENTROPY
            : training::LossType::MSE;
        mc.train_config.interleave_self_org   = is_cls;
        mc.train_config.self_org_interval     = 10;
        mc.train_config.self_org_growth_only  = true;
        if (is_cls) {
            mc.cls_metrics = std::get_if<eval::ClassificationMetrics>(&res.metrics);
        } else {
            mc.reg_metrics = std::get_if<eval::RegressionMetrics>(&res.metrics);
        }
        mc.baseline             = res.baselines;
        mc.curve                = res.curve;
        mc.neurograph_version   = mf.neurograph_version;
        mc.generated_at_iso8601 = mf.timestamp_iso8601;
        mc.dataset_path         = csv.string();
        try {
            reporting::save_model_card(mc, card_path);
        } catch (const std::exception& e) {
            return fail(std::string("Bundle write failed: ") + e.what());
        }
        emit(progress, Stage::Packaging, 1.0, "Bundle written: " + opts.bundle_out_dir.string());
        res.bundle_path = opts.bundle_out_dir;
    } else {
        emit(progress, Stage::Packaging, 1.0, "Bundle export skipped");
    }

    emit(progress, Stage::Done, 1.0, res.summary_one_liner);
    return res;
}

} // namespace neurograph::orchestration
