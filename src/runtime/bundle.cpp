#include "neurograph/runtime/bundle.hpp"

#include "neurograph/core/network.hpp"
#include "neurograph/models/model.hpp"
#include "neurograph/snapshot/snapshot_metadata.hpp"
#include "neurograph/training/backprop.hpp"
#include "neurograph/training/io_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurograph::runtime {

// ---------------------------------------------------------------------------
// PIMPL definition for NeuralNet{Classification,Regression}Model.
//
// The bundle.hpp header forward-declares NeuralNetInferenceState and
// declares out-of-line special members so the header doesn't have to
// include training/network internals. P1-15 wires the orchestrator to
// produce schema_v2 bundles holding a SON-trained network; the loader
// here reconstructs Network + IOManager + Backprop in inference mode
// so predict_row can drive forward passes.
//
// Defined at namespace-scope (not inside the anonymous namespace below)
// because the bundle.hpp PIMPL contracts reference the type by name —
// putting it inside the anonymous namespace would create a different
// type per TU and break the unique_ptr destructor link-out.
struct NeuralNetInferenceState {
    std::unique_ptr<core::Network>           network;
    std::unique_ptr<training::IOManager>     io;
    std::unique_ptr<training::Backprop>      bp;     // inference mode

    // Serializes concurrent forward() calls. Backprop holds internal
    // state (activation map, scratch buffers); two threads racing on
    // the same instance can double-free std::vector internals (caught
    // by ASan in `CapiTest.ConcurrentPredictIsSafeAndDeterministic`).
    // The C ABI promises *thread-safe* predict calls on a shared
    // model — not parallel throughput. Lock-and-serialize satisfies
    // that contract; future ticket: per-thread Backprop pool for
    // actual parallelism.
    mutable std::mutex                       forward_mutex;
};

namespace {

namespace fs = std::filesystem;

// Same constants the orchestrator uses to write the bundle. Keep these
// in sync with src/orchestration/auto_pipeline.cpp's `kBundle*` block.
constexpr const char* kBundlePipeline = "pipeline.ngpipe";
constexpr const char* kBundleModel    = "model.ngm";
constexpr const char* kBundleManifest = "manifest.json";

// Bundle loader accepts both schema versions:
//   v1 — linear probe (older bundles, kind="classification"|"regression")
//   v2 — SON-trained neural net (kind="neural_cls_v1"|"neural_reg_v1")
// The orchestrator currently writes v2 for the main Training stage;
// v1 still loads for backwards compat with anything packaged earlier.
constexpr int kModelSchemaVersionMin = 1;
//   v3 — native model chosen by the automatic trainer (kind names the family)
constexpr int kModelSchemaVersionMax = 3;

// Read a file into a string. On failure returns nullopt; the caller
// shapes the error message so it can name the missing file.
std::string slurp(const fs::path& p, std::string& err) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        err = "cannot open " + p.string();
        return {};
    }
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

util::Result<LinearClassificationModel>
parse_classification_model(const nlohmann::json& j) {
    LinearClassificationModel m;
    try {
        m.n_classes  = j.at("n_classes").get<int>();
        m.n_features = j.at("n_features").get<int>();
        if (m.n_classes < 2) {
            return util::Error("model.ngm: n_classes must be >= 2");
        }
        if (m.n_features < 1) {
            return util::Error("model.ngm: n_features must be >= 1");
        }
        m.W = j.at("W").get<std::vector<std::vector<double>>>();
        m.b = j.at("b").get<std::vector<double>>();
        m.class_labels = j.value("class_labels", std::vector<std::string>{});
        if (static_cast<int>(m.W.size()) != m.n_classes) {
            return util::Error("model.ngm: W has " + std::to_string(m.W.size()) +
                               " rows, expected " + std::to_string(m.n_classes));
        }
        for (const auto& row : m.W) {
            if (static_cast<int>(row.size()) != m.n_features) {
                return util::Error("model.ngm: W row width " +
                                   std::to_string(row.size()) +
                                   " ≠ n_features " + std::to_string(m.n_features));
            }
        }
        if (static_cast<int>(m.b.size()) != m.n_classes) {
            return util::Error("model.ngm: b has " + std::to_string(m.b.size()) +
                               " entries, expected " + std::to_string(m.n_classes));
        }
    } catch (const std::exception& e) {
        return util::Error(std::string("model.ngm: classification parse: ") + e.what());
    }
    return m;
}

util::Result<LinearRegressionModel>
parse_regression_model(const nlohmann::json& j) {
    LinearRegressionModel m;
    try {
        m.n_features = j.at("n_features").get<int>();
        if (m.n_features < 1) {
            return util::Error("model.ngm: n_features must be >= 1");
        }
        m.w    = j.at("w").get<std::vector<double>>();
        m.bias = j.at("bias").get<double>();
        if (static_cast<int>(m.w.size()) != m.n_features) {
            return util::Error("model.ngm: w has " + std::to_string(m.w.size()) +
                               " entries, expected " + std::to_string(m.n_features));
        }
    } catch (const std::exception& e) {
        return util::Error(std::string("model.ngm: regression parse: ") + e.what());
    }
    return m;
}

// Build the inference state for a SON neural-net bundle. Network is
// reconstructed from JSON, then IOManager is restored from the saved
// io_mapping (so feature ordering matches what the orchestrator wrote).
// Backprop is held in inference mode: forward-only, dropout/batchnorm
// off, topo order pre-built. The returned PIMPL holds all three so
// predict_row can drive forward passes without re-touching the raw
// JSON. Definition appears further down in this TU.
util::Result<std::unique_ptr<NeuralNetInferenceState>>
build_neural_state(const nlohmann::json& j,
                   int                   n_features,
                   int                   n_outputs,
                   const std::vector<std::string>& class_labels);

void softmax_inplace(std::vector<double>& z) {
    if (z.empty()) return;
    const double m = *std::max_element(z.begin(), z.end());
    double s = 0.0;
    for (double& v : z) { v = std::exp(v - m); s += v; }
    if (s > 0.0) for (double& v : z) v /= s;
}

int argmax_(const std::vector<double>& v) {
    if (v.empty()) return -1;
    int k = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[k]) k = i;
    return k;
}

} // namespace

// ---------------------------------------------------------------------------
// ModelBundle::class_label
// ---------------------------------------------------------------------------
std::string ModelBundle::class_label(int i) const {
    if (const auto* m = std::get_if<LinearClassificationModel>(&model)) {
        if (i < 0 || i >= static_cast<int>(m->class_labels.size())) return {};
        return m->class_labels[i];
    }
    if (const auto* m = std::get_if<NeuralNetClassificationModel>(&model)) {
        if (i < 0 || i >= static_cast<int>(m->class_labels.size())) return {};
        return m->class_labels[i];
    }
    if (const auto* m = std::get_if<NativeModel>(&model)) {
        if (i < 0 || i >= static_cast<int>(m->class_labels.size())) return {};
        return m->class_labels[i];
    }
    return {};
}

int ModelBundle::n_features() const noexcept {
    if (const auto* m = std::get_if<LinearClassificationModel>(&model))    return m->n_features;
    if (const auto* m = std::get_if<LinearRegressionModel>(&model))        return m->n_features;
    if (const auto* m = std::get_if<NeuralNetClassificationModel>(&model)) return m->n_features;
    if (const auto* m = std::get_if<NeuralNetRegressionModel>(&model))     return m->n_features;
    if (const auto* m = std::get_if<NativeModel>(&model))                  return m->n_features;
    return 0;
}

std::string ModelBundle::description() const {
    if (const auto* m = std::get_if<NativeModel>(&model)) {
        if (m->impl) return m->impl->describe();
    }
    return {};
}

std::vector<std::string>
missing_input_columns(const ModelBundle& bundle,
                      const std::vector<std::string>& header) {
    std::vector<std::string> missing;
    if (!bundle.pipeline) return missing;
    for (const auto& need : bundle.pipeline->input_columns()) {
        if (std::find(header.begin(), header.end(), need) == header.end()) {
            missing.push_back(need);
        }
    }
    return missing;
}

// ---------------------------------------------------------------------------
// load_bundle — strict, never throws across the API boundary
// ---------------------------------------------------------------------------
util::Result<ModelBundle> load_bundle(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return util::Error("bundle: not a directory: " + dir.string());
    }

    const auto pipe_path     = dir / kBundlePipeline;
    const auto model_path    = dir / kBundleModel;
    const auto manifest_path = dir / kBundleManifest;

    // Pipeline.
    auto pipe = preprocess::load_pipeline(pipe_path);
    if (!pipe) {
        return util::Error("bundle: cannot load " + pipe_path.string() +
                           " (missing or schema mismatch)");
    }

    // Model.
    std::string err;
    const auto model_text = slurp(model_path, err);
    if (!err.empty()) return util::Error("bundle: " + err);

    nlohmann::json mj;
    try {
        mj = nlohmann::json::parse(model_text);
    } catch (const std::exception& e) {
        return util::Error("bundle: " + model_path.string() +
                           " is not valid JSON: " + e.what());
    }

    int sv = 0;
    try { sv = mj.at("schema_version").get<int>(); }
    catch (...) { return util::Error("bundle: model.ngm missing schema_version"); }
    if (sv < kModelSchemaVersionMin || sv > kModelSchemaVersionMax) {
        return util::Error("bundle: model.ngm schema_version " +
                           std::to_string(sv) + " not supported (expected " +
                           std::to_string(kModelSchemaVersionMin) + "-" +
                           std::to_string(kModelSchemaVersionMax) + ")");
    }

    std::string kind;
    try { kind = mj.at("kind").get<std::string>(); }
    catch (...) { return util::Error("bundle: model.ngm missing 'kind' field"); }

    ModelBundle bundle;
    bundle.pipeline   = std::move(pipe);
    bundle.source_dir = dir;

    if (sv >= 3) {
        NativeModel nm;
        try {
            nm.impl = std::shared_ptr<const models::Model>(
                models::model_from_json(mj.at("model")));
            nm.class_labels = mj.value("class_labels", std::vector<std::string>{});
        } catch (const std::exception& e) {
            return util::Error(std::string("bundle: model.ngm: ") + e.what());
        }
        // The top-level kind must name the model actually stored, so a
        // file edited by hand (or from a newer version) is refused rather
        // than half-understood.
        if (kind != nm.impl->kind()) {
            return util::Error("bundle: model.ngm has unknown kind '" + kind +
                               "' (its model section is '" +
                               std::string(nm.impl->kind()) + "')");
        }
        nm.n_features = nm.impl->n_features();
        nm.n_classes  = (nm.impl->task() == models::Task::Classification)
                      ? nm.impl->n_outputs() : 0;
        if (nm.n_classes > 0
            && static_cast<int>(nm.class_labels.size()) != nm.n_classes) {
            return util::Error("bundle: model.ngm lists " +
                               std::to_string(nm.class_labels.size()) +
                               " class labels but the model has " +
                               std::to_string(nm.n_classes) + " classes");
        }
        const std::size_t pipe_width = bundle.pipeline->feature_names_out().size();
        if (static_cast<std::size_t>(nm.n_features) != pipe_width) {
            return util::Error("bundle: the model takes " +
                               std::to_string(nm.n_features) +
                               " inputs but pipeline.ngpipe produces " +
                               std::to_string(pipe_width) +
                               " (the two files are from different training runs)");
        }
        bundle.model = std::move(nm);
    } else if (kind == "classification") {
        // schema_v1 linear probe — kept for backwards compat with older bundles.
        auto r = parse_classification_model(mj);
        if (!r.ok()) return util::Error(r.error().message);
        bundle.model = std::move(r).value();
    } else if (kind == "regression") {
        auto r = parse_regression_model(mj);
        if (!r.ok()) return util::Error(r.error().message);
        bundle.model = std::move(r).value();
    } else if (kind == "neural_cls_v1") {
        // schema_v2 SON-trained classification network.
        NeuralNetClassificationModel m;
        try {
            m.n_classes    = mj.at("n_classes").get<int>();
            m.n_features   = mj.at("n_features").get<int>();
            m.class_labels = mj.value("class_labels", std::vector<std::string>{});
        } catch (const std::exception& e) {
            return util::Error(std::string("model.ngm: neural_cls_v1 parse: ")
                               + e.what());
        }
        auto state_r = build_neural_state(mj, m.n_features, m.n_classes,
                                          m.class_labels);
        if (!state_r.ok()) return util::Error(state_r.error().message);
        m.state = std::move(state_r).value();
        bundle.model = std::move(m);
    } else if (kind == "neural_reg_v1") {
        NeuralNetRegressionModel m;
        try {
            m.n_features = mj.at("n_features").get<int>();
            m.n_outputs  = mj.value("n_outputs", 1);
        } catch (const std::exception& e) {
            return util::Error(std::string("model.ngm: neural_reg_v1 parse: ")
                               + e.what());
        }
        auto state_r = build_neural_state(mj, m.n_features, m.n_outputs, {});
        if (!state_r.ok()) return util::Error(state_r.error().message);
        m.state = std::move(state_r).value();
        bundle.model = std::move(m);
    } else {
        return util::Error("bundle: model.ngm unknown kind '" + kind + "'");
    }

    // Manifest is optional. If present and unparseable, we surface it
    // — silently swallowing would hide tampering. If the file isn't
    // there at all, that's fine (older bundles).
    if (fs::exists(manifest_path, ec)) {
        const auto mtxt = slurp(manifest_path, err);
        if (err.empty()) {
            try {
                bundle.manifest = nlohmann::json::parse(mtxt);
            } catch (const std::exception& e) {
                return util::Error("bundle: " + manifest_path.string() +
                                   " is not valid JSON: " + e.what());
            }
        }
    }
    return bundle;
}

// ---------------------------------------------------------------------------
// predict_row
// ---------------------------------------------------------------------------
Prediction predict_row(const ModelBundle& bundle, const nlohmann::json& row) {
    Prediction out;
    if (!bundle.pipeline) return out;

    // apply_pipeline never throws per its contract — missing keys are
    // imputed, unknowns routed to the reserved bucket.
    std::vector<double> x;
    try {
        x = preprocess::apply_pipeline(*bundle.pipeline, row);
    } catch (...) {
        return out;
    }

    return predict_vec(bundle, x);
}

// ---------------------------------------------------------------------------
// predict_vec — same dispatch as predict_row but takes a pre-transformed
// feature vector (skips apply_pipeline). Used by the C ABI's
// ng_predict_vec fast path; lives here so NeuralNetInferenceState stays
// opaque outside this TU.
// ---------------------------------------------------------------------------
Prediction predict_vec(const ModelBundle& bundle,
                       const std::vector<double>& x) {
    Prediction out;

    if (const auto* nm = std::get_if<NativeModel>(&bundle.model)) {
        if (!nm->impl) return out;
        // Pad / truncate so a caller-supplied vector of the wrong width
        // cannot read out of bounds.
        std::vector<double> xin(static_cast<std::size_t>(nm->n_features), 0.0);
        std::copy_n(x.begin(), std::min(x.size(), xin.size()), xin.begin());
        if (nm->n_classes >= 2) {
            std::vector<double> p(static_cast<std::size_t>(nm->n_classes), 0.0);
            nm->impl->predict(xin.data(), p.data());
            out.predicted_class = argmax_(p);
            out.probabilities   = std::move(p);
        } else {
            double y = 0.0;
            nm->impl->predict(xin.data(), &y);
            out.value = y;
        }
        return out;
    }

    if (const auto* m = std::get_if<LinearClassificationModel>(&bundle.model)) {
        if (m->n_classes < 2) return out;
        std::vector<double> z(m->n_classes, 0.0);
        const std::size_t D = std::min(x.size(),
                                       static_cast<std::size_t>(m->n_features));
        for (int k = 0; k < m->n_classes; ++k) {
            double s = m->b[k];
            for (std::size_t d = 0; d < D; ++d) s += m->W[k][d] * x[d];
            z[k] = s;
        }
        softmax_inplace(z);
        out.predicted_class = argmax_(z);
        out.probabilities   = std::move(z);
    } else if (const auto* mr = std::get_if<LinearRegressionModel>(&bundle.model)) {
        double yhat = mr->bias;
        const std::size_t D = std::min(x.size(),
                                       static_cast<std::size_t>(mr->n_features));
        for (std::size_t d = 0; d < D; ++d) yhat += mr->w[d] * x[d];
        out.value = yhat;
    } else if (const auto* nc = std::get_if<NeuralNetClassificationModel>(&bundle.model)) {
        if (!nc->state || !nc->state->bp || !nc->state->io) return out;
        std::vector<double> x_padded(nc->n_features, 0.0);
        const std::size_t D = std::min(x.size(),
                                       static_cast<std::size_t>(nc->n_features));
        for (std::size_t d = 0; d < D; ++d) x_padded[d] = x[d];
        // Lock per predict_row's contract — see that function for rationale.
        std::vector<double> z;
        {
            std::lock_guard<std::mutex> lk(nc->state->forward_mutex);
            nc->state->bp->forward(x_padded);
            z = nc->state->io->read_outputs();
        }
        if (z.empty()) return out;
        softmax_inplace(z);
        out.predicted_class = argmax_(z);
        out.probabilities   = std::move(z);
    } else if (const auto* nr = std::get_if<NeuralNetRegressionModel>(&bundle.model)) {
        if (!nr->state || !nr->state->bp || !nr->state->io) return out;
        std::vector<double> x_padded(nr->n_features, 0.0);
        const std::size_t D = std::min(x.size(),
                                       static_cast<std::size_t>(nr->n_features));
        for (std::size_t d = 0; d < D; ++d) x_padded[d] = x[d];
        std::vector<double> z;
        {
            std::lock_guard<std::mutex> lk(nr->state->forward_mutex);
            nr->state->bp->forward(x_padded);
            z = nr->state->io->read_outputs();
        }
        out.value = z.empty() ? 0.0 : z[0];
    }
    return out;
}

// ---------------------------------------------------------------------------
// CSV helpers — RFC 4180-ish. Quotes around fields with comma/quote/CRLF;
// "" inside a quoted field is an escaped quote.
// ---------------------------------------------------------------------------

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(c);
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                out.push_back(std::move(field));
                field.clear();
            } else if (c == '\r') {
                // strip trailing CR
            } else {
                field.push_back(c);
            }
        }
    }
    if (!field.empty() || !line.empty()) out.push_back(std::move(field));
    return out;
}

std::string quote_csv_field(const std::string& s) {
    const bool needs_quote =
        s.find_first_of(",\"\r\n") != std::string::npos;
    if (!needs_quote) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// (NeuralNetInferenceState struct is defined at namespace-scope near
// the top of this TU; only the factory body lives here. Re-opening the
// anonymous namespace so the definition has the same internal linkage
// as the forward declaration up top.)
namespace {
util::Result<std::unique_ptr<NeuralNetInferenceState>>
build_neural_state(const nlohmann::json& j,
                   int                   n_features,
                   int                   n_outputs,
                   const std::vector<std::string>& class_labels) {
    if (!j.contains("network")) {
        return util::Error("model.ngm: neural model missing 'network' field");
    }
    if (!j.contains("io_mapping")) {
        return util::Error("model.ngm: neural model missing 'io_mapping' field");
    }

    auto state = std::make_unique<NeuralNetInferenceState>();
    try {
        // Network::from_json is a static factory that returns by value;
        // wrap it in unique_ptr so the model's PIMPL semantics work.
        state->network = std::make_unique<core::Network>(
            core::Network::from_json(j.at("network")));
    } catch (const std::exception& e) {
        return util::Error(std::string("model.ngm: network deserialize failed: ")
                           + e.what());
    }

    // Reconstruct IOManager from the snapshot::IOMapping we serialized
    // in the orchestrator. This re-binds neuron-id strings → typed
    // input/output IDs so apply_pipeline output rows map to the right
    // input neurons.
    state->io = std::make_unique<training::IOManager>(*state->network);
    snapshot::IOMapping mapping = snapshot::IOMapping::from_json(j.at("io_mapping"));
    if (mapping.input_features  == 0) mapping.input_features  = n_features;
    if (mapping.output_classes  == 0) mapping.output_classes  = n_outputs;
    if (mapping.class_labels.empty()) mapping.class_labels    = class_labels;
    auto io_r = state->io->load_io_mapping(mapping);
    if (!io_r.ok()) {
        return util::Error("model.ngm: io_mapping restore failed: "
                           + io_r.error().message);
    }

    // Backprop in inference mode — disables dropout, prebuilds topo
    // order so predict_row's forward() stays O(neurons + synapses).
    state->bp = std::make_unique<training::Backprop>(*state->network, *state->io);
    state->bp->set_training_mode(false);
    state->bp->rebuild_topo_order();
    if (state->bp->topo_order().empty()) {
        return util::Error("model.ngm: no path from input to output neurons "
                           "in deserialized network");
    }

    return state;
}
} // namespace

NeuralNetClassificationModel::NeuralNetClassificationModel() = default;
NeuralNetClassificationModel::~NeuralNetClassificationModel() = default;
NeuralNetClassificationModel::NeuralNetClassificationModel(
    NeuralNetClassificationModel&&) noexcept = default;
NeuralNetClassificationModel&
NeuralNetClassificationModel::operator=(
    NeuralNetClassificationModel&&) noexcept = default;

NeuralNetRegressionModel::NeuralNetRegressionModel() = default;
NeuralNetRegressionModel::~NeuralNetRegressionModel() = default;
NeuralNetRegressionModel::NeuralNetRegressionModel(
    NeuralNetRegressionModel&&) noexcept = default;
NeuralNetRegressionModel&
NeuralNetRegressionModel::operator=(
    NeuralNetRegressionModel&&) noexcept = default;

} // namespace neurograph::runtime
