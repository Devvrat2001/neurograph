// Dense ReLU network (and, with no hidden layer, a linear model).
//
// Training is ordinary mini-batch Adam on contiguous arrays, with biases,
// decoupled weight decay, and early stopping that keeps the best weights
// seen on the validation split.
//
// Self-organizing width. With exactly one hidden layer and `grow` set,
// the layer starts small. Each time validation loss stalls, the units
// that carried the largest error signal over the last epoch ("stressed"
// units) are split in two: the clone copies the parent's incoming weights
// plus a little noise, and the pair share the parent's outgoing weights
// half each. The network computes almost the same function right after a
// split, so training carries on from where it was rather than restarting.
// Growth stops as soon as a split fails to improve the best validation
// loss.

#include "neurograph/models/model.hpp"
#include "detail.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace neurograph::models {

namespace {

class MlpModel final : public Model {
public:
    Task                             task_ = Task::Classification;
    std::vector<int>                 sizes_;       // [n_in, hidden..., n_out]
    std::vector<std::vector<double>> W_;           // W_[l]: sizes_[l+1] x sizes_[l], row-major
    std::vector<std::vector<double>> b_;           // b_[l]: sizes_[l+1]
    double                           y_mean_  = 0.0;   // regression target standardisation
    double                           y_scale_ = 1.0;
    int                              grown_from_ = 0;  // starting hidden width when the layer grew, else 0

    const char* kind() const noexcept override {
        return sizes_.size() <= 2 ? "linear_v1" : "mlp_v1";
    }
    Task task() const noexcept override { return task_; }
    int  n_features() const noexcept override { return sizes_.empty() ? 0 : sizes_.front(); }
    int  n_outputs()  const noexcept override {
        return task_ == Task::Classification ? sizes_.back() : 1;
    }

    void predict(const double* x, double* out) const override {
        const int widest = *std::max_element(sizes_.begin(), sizes_.end());
        std::vector<double> a(static_cast<std::size_t>(widest));
        std::vector<double> z(static_cast<std::size_t>(widest));
        std::copy(x, x + sizes_[0], a.begin());
        const int L = static_cast<int>(W_.size());
        for (int l = 0; l < L; ++l) {
            const int n_in  = sizes_[l];
            const int n_out = sizes_[l + 1];
            const double* W = W_[l].data();
            const bool hidden = (l + 1 < L);
            for (int j = 0; j < n_out; ++j) {
                const double* w = W + static_cast<std::size_t>(j) * n_in;
                double s = b_[l][j];
                for (int i = 0; i < n_in; ++i) s += w[i] * a[i];
                z[j] = (hidden && s < 0.0) ? 0.0 : s;
            }
            a.swap(z);
        }
        if (task_ == Task::Classification) {
            const int k = sizes_.back();
            detail::softmax(a.data(), k);
            std::copy(a.begin(), a.begin() + k, out);
        } else {
            out[0] = a[0] * y_scale_ + y_mean_;
        }
    }

    nlohmann::json to_json() const override {
        nlohmann::json j;
        j["kind"]       = kind();
        j["task"]       = (task_ == Task::Classification) ? "classification" : "regression";
        j["sizes"]      = sizes_;
        j["W"]          = W_;
        j["b"]          = b_;
        j["y_mean"]     = y_mean_;
        j["y_scale"]    = y_scale_;
        j["grown_from"] = grown_from_;
        return j;
    }

    std::string describe() const override {
        std::ostringstream os;
        if (sizes_.size() <= 2) {
            os << "Linear model (" << parameter_count() << " weights)";
            return os.str();
        }
        os << "Neural network (";
        for (std::size_t l = 1; l + 1 < sizes_.size(); ++l) {
            if (l > 1) os << " + ";
            os << sizes_[l];
        }
        os << " hidden units";
        if (grown_from_ > 0 && sizes_.size() == 3 && grown_from_ < sizes_[1]) {
            os << ", grown from " << grown_from_;
        }
        os << ")";
        return os.str();
    }

    std::size_t parameter_count() const noexcept override {
        std::size_t n = 0;
        for (const auto& w : W_) n += w.size();
        for (const auto& b : b_) n += b.size();
        return n;
    }
};

// Everything the optimiser needs for one layer.
struct LayerState {
    std::vector<double> gW, gb;     // gradients for the current batch
    std::vector<double> mW, vW;     // Adam moments
    std::vector<double> mb, vb;
};

int pick_batch_size(std::size_t n, int requested) {
    if (requested > 0) return requested;
    if (n < 500)   return 16;
    if (n < 5000)  return 32;
    if (n < 50000) return 64;
    return 128;
}

} // namespace

// ---------------------------------------------------------------------------
// fit_mlp
// ---------------------------------------------------------------------------

FitReport fit_mlp(const TrainSet& d, const MlpOptions& opt, const FitControl& ctl) {
    const detail::Deadline deadline(ctl.time_limit_seconds);
    FitReport rep;
    rep.name = opt.hidden.empty() ? "Linear model" : "Neural network";

    const bool        cls   = (d.task == Task::Classification);
    const std::size_t n     = d.X_train.rows;
    const std::size_t nv    = d.X_val.rows;
    const int         n_in  = static_cast<int>(d.X_train.cols);
    const int         n_out = cls ? d.n_classes : 1;
    if (n == 0 || n_in == 0 || n_out <= 0) return rep;

    // ---- Targets -----------------------------------------------------------
    // Regression targets are standardised for training so the learning
    // rate means the same thing whatever units the target is in.
    double y_mean = 0.0, y_scale = 1.0;
    std::vector<double> yt, yv;
    if (!cls) {
        for (double y : d.y_train_reg) y_mean += y;
        y_mean /= static_cast<double>(n);
        double var = 0.0;
        for (double y : d.y_train_reg) var += (y - y_mean) * (y - y_mean);
        var /= static_cast<double>(n);
        y_scale = (var > 1e-24) ? std::sqrt(var) : 1.0;
        yt.resize(n);
        for (std::size_t i = 0; i < n; ++i) yt[i] = (d.y_train_reg[i] - y_mean) / y_scale;
        yv.resize(nv);
        for (std::size_t i = 0; i < nv; ++i) yv[i] = (d.y_val_reg[i] - y_mean) / y_scale;
    }

    // ---- Model -------------------------------------------------------------
    std::mt19937_64 rng(ctl.seed);
    auto model = std::make_unique<MlpModel>();
    model->task_    = d.task;
    model->y_mean_  = y_mean;
    model->y_scale_ = y_scale;
    model->sizes_.push_back(n_in);
    for (int h : opt.hidden) model->sizes_.push_back(std::max(1, h));
    model->sizes_.push_back(n_out);

    const bool can_grow_at_all = opt.grow && opt.hidden.size() == 1;
    // No wider than half the training rows: on a small file a network with
    // more hidden units than examples only memorises them.
    const int max_hidden = opt.hidden.size() == 1
        ? std::max(opt.hidden[0], std::min(opt.max_hidden, static_cast<int>(n / 2)))
        : opt.max_hidden;
    const int  start_width     = opt.hidden.empty() ? 0 : std::max(1, opt.hidden[0]);

    auto init_layer = [&](int l) {
        const int fan_in  = model->sizes_[l];
        const int fan_out = model->sizes_[l + 1];
        const bool hidden = (l + 2 < static_cast<int>(model->sizes_.size()));
        // He initialisation for ReLU layers, a smaller Xavier-style draw
        // for the output layer so initial predictions stay near the prior.
        const double stddev = hidden ? std::sqrt(2.0 / fan_in)
                                     : std::sqrt(1.0 / fan_in) * 0.5;
        std::normal_distribution<double> nd(0.0, stddev);
        auto& W = model->W_[l];
        W.resize(static_cast<std::size_t>(fan_in) * fan_out);
        for (double& w : W) w = nd(rng);
        model->b_[l].assign(static_cast<std::size_t>(fan_out), 0.0);
    };
    const int L = static_cast<int>(model->sizes_.size()) - 1;
    model->W_.resize(L);
    model->b_.resize(L);
    for (int l = 0; l < L; ++l) init_layer(l);

    if (cls) {
        // Start the output biases at the log class priors.
        std::vector<double> prior(static_cast<std::size_t>(n_out), 0.0);
        for (int y : d.y_train_cls) if (y >= 0 && y < n_out) prior[y] += 1.0;
        for (int k = 0; k < n_out; ++k) {
            model->b_[L - 1][k] = std::log((prior[k] + 1.0) / (static_cast<double>(n) + n_out));
        }
    }

    std::vector<LayerState> st(L);
    auto reset_optimizer = [&]() {
        for (int l = 0; l < L; ++l) {
            const std::size_t nw = model->W_[l].size();
            const std::size_t nb = model->b_[l].size();
            st[l].gW.assign(nw, 0.0); st[l].mW.assign(nw, 0.0); st[l].vW.assign(nw, 0.0);
            st[l].gb.assign(nb, 0.0); st[l].mb.assign(nb, 0.0); st[l].vb.assign(nb, 0.0);
        }
    };
    reset_optimizer();
    long long adam_t = 0;

    const int B = static_cast<int>(std::min<std::size_t>(n, pick_batch_size(n, opt.batch_size)));

    // Activations A[l] (B x sizes[l]) and error signals D[l] (B x sizes[l]).
    std::vector<std::vector<double>> A(L + 1), D(L + 1);
    auto size_buffers = [&]() {
        for (int l = 0; l <= L; ++l) {
            A[l].assign(static_cast<std::size_t>(B) * model->sizes_[l], 0.0);
            D[l].assign(static_cast<std::size_t>(B) * model->sizes_[l], 0.0);
        }
    };
    size_buffers();

    // Forward pass over `bn` rows already copied into A[0].
    auto forward = [&](int bn) {
        for (int l = 0; l < L; ++l) {
            const int ni = model->sizes_[l];
            const int no = model->sizes_[l + 1];
            const double* W    = model->W_[l].data();
            const double* bias = model->b_[l].data();
            const bool hidden  = (l + 1 < L);
            for (int r = 0; r < bn; ++r) {
                const double* a = A[l].data() + static_cast<std::size_t>(r) * ni;
                double*       z = A[l + 1].data() + static_cast<std::size_t>(r) * no;
                for (int j = 0; j < no; ++j) {
                    const double* w = W + static_cast<std::size_t>(j) * ni;
                    double s = bias[j];
                    for (int i = 0; i < ni; ++i) s += w[i] * a[i];
                    z[j] = (hidden && s < 0.0) ? 0.0 : s;
                }
            }
        }
    };

    // Mean loss on a dataset, in training units (log-loss, or MSE on the
    // standardised target).
    auto dataset_loss = [&](const Matrix& X, const std::vector<int>& yc,
                            const std::vector<double>& yr) {
        if (X.rows == 0) return std::numeric_limits<double>::infinity();
        double total = 0.0;
        for (std::size_t start = 0; start < X.rows; start += static_cast<std::size_t>(B)) {
            const int bn = static_cast<int>(std::min<std::size_t>(B, X.rows - start));
            for (int r = 0; r < bn; ++r) {
                std::copy(X.row(start + r), X.row(start + r) + n_in,
                          A[0].data() + static_cast<std::size_t>(r) * n_in);
            }
            forward(bn);
            for (int r = 0; r < bn; ++r) {
                double* z = A[L].data() + static_cast<std::size_t>(r) * n_out;
                if (cls) {
                    detail::softmax(z, n_out);
                    const int y = yc[start + r];
                    total += -std::log(std::max(z[y], detail::kProbFloor));
                } else {
                    const double diff = z[0] - yr[start + r];
                    total += diff * diff;
                }
            }
        }
        return total / static_cast<double>(X.rows);
    };

    // ---- Training loop -----------------------------------------------------
    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double lr = opt.learning_rate;

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});

    // Best-so-far snapshot.
    std::vector<int>                 best_sizes = model->sizes_;
    std::vector<std::vector<double>> best_W = model->W_, best_b = model->b_;
    double best_val   = std::numeric_limits<double>::infinity();
    int    best_epoch = -1;

    bool   can_grow          = can_grow_at_all;
    int    growth_events     = 0;
    double best_at_last_grow = std::numeric_limits<double>::infinity();
    int    since_best        = 0;     // epochs since the best validation loss
    int    since_event       = 0;     // epochs since best / last growth / last LR cut
    const int plateau        = std::max(3, opt.patience / 4);
    int    grace             = 0;     // extra epochs a freshly widened layer gets before it is judged
    int    lr_cuts           = 0;

    std::vector<double> stress;       // per hidden unit, accumulated over an epoch
    if (can_grow_at_all) stress.assign(static_cast<std::size_t>(model->sizes_[1]), 0.0);

    // Widen the single hidden layer by splitting `k` stressed units.
    auto grow = [&](int k) {
        const int h  = model->sizes_[1];
        const int nh = h + k;
        std::vector<int> idx(static_cast<std::size_t>(h));
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return stress[a] > stress[b]; });

        std::vector<double> W0(static_cast<std::size_t>(nh) * n_in);
        std::vector<double> b0(static_cast<std::size_t>(nh));
        std::copy(model->W_[0].begin(), model->W_[0].end(), W0.begin());
        std::copy(model->b_[0].begin(), model->b_[0].end(), b0.begin());

        std::vector<double> W1(static_cast<std::size_t>(n_out) * nh);
        for (int o = 0; o < n_out; ++o) {
            std::copy(model->W_[1].begin() + static_cast<std::ptrdiff_t>(o) * h,
                      model->W_[1].begin() + static_cast<std::ptrdiff_t>(o + 1) * h,
                      W1.begin() + static_cast<std::ptrdiff_t>(o) * nh);
        }

        // Half of the new units are splits of the most stressed units; the
        // rest are fresh units that start with no say in the output. Splits
        // add capacity where the error signal is largest, fresh units add
        // directions the layer does not have yet. Either way the network
        // computes (almost) the same function straight after growing.
        const int n_split = std::min(h, (k + 1) / 2);
        for (int c = 0; c < k; ++c) {
            const int child = h + c;
            double*   dst   = W0.data() + static_cast<std::size_t>(child) * n_in;
            if (c < n_split) {
                const int parent = idx[static_cast<std::size_t>(c)];
                const double* src = W0.data() + static_cast<std::size_t>(parent) * n_in;
                double rms = 0.0;
                for (int i = 0; i < n_in; ++i) rms += src[i] * src[i];
                rms = std::sqrt(rms / n_in);
                // Small noise so parent and clone receive different gradients
                // and can specialise; without it they would stay identical.
                std::normal_distribution<double> nd(0.0, 0.1 * rms + 1e-3);
                for (int i = 0; i < n_in; ++i) dst[i] = src[i] + nd(rng);
                b0[child] = b0[parent];
                for (int o = 0; o < n_out; ++o) {
                    double& wp = W1[static_cast<std::size_t>(o) * nh + parent];
                    wp *= 0.5;
                    W1[static_cast<std::size_t>(o) * nh + child] = wp;
                }
            } else {
                std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / n_in));
                for (int i = 0; i < n_in; ++i) dst[i] = nd(rng);
                b0[child] = 0.0;
                for (int o = 0; o < n_out; ++o) {
                    W1[static_cast<std::size_t>(o) * nh + child] = 0.0;
                }
            }
        }

        model->sizes_[1] = nh;
        model->W_[0] = std::move(W0);
        model->b_[0] = std::move(b0);
        model->W_[1] = std::move(W1);
        reset_optimizer();
        adam_t = 0;
        size_buffers();
        stress.assign(static_cast<std::size_t>(nh), 0.0);
    };

    bool stopped_by_time = false;
    bool cancelled       = false;
    // An epoch counts as progress only if it beats the best so far by 0.01%
    // of the starting loss (see the same rule in gbt.cpp).
    const double start_val = (nv > 0) ? dataset_loss(d.X_val, d.y_val_cls, yv)
                                      : dataset_loss(d.X_train, d.y_train_cls, yt);
    const double min_gain  = std::isfinite(start_val) ? std::max(1e-12, 1e-4 * start_val)
                                                      : 1e-7;

    int  epoch           = 0;

    for (; epoch < opt.max_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        if (!stress.empty()) std::fill(stress.begin(), stress.end(), 0.0);

        for (std::size_t start = 0; start < n; start += static_cast<std::size_t>(B)) {
            const int bn = static_cast<int>(std::min<std::size_t>(B, n - start));

            for (int r = 0; r < bn; ++r) {
                const double* src = d.X_train.row(order[start + r]);
                std::copy(src, src + n_in, A[0].data() + static_cast<std::size_t>(r) * n_in);
            }
            forward(bn);

            // Output error signal.
            for (int r = 0; r < bn; ++r) {
                double* z   = A[L].data() + static_cast<std::size_t>(r) * n_out;
                double* dlt = D[L].data() + static_cast<std::size_t>(r) * n_out;
                if (cls) {
                    detail::softmax(z, n_out);
                    const int y = d.y_train_cls[order[start + r]];
                    for (int k = 0; k < n_out; ++k) dlt[k] = z[k];
                    dlt[y] -= 1.0;
                } else {
                    dlt[0] = z[0] - yt[order[start + r]];
                }
            }

            // Backward pass.
            for (int l = L - 1; l >= 0; --l) {
                const int ni = model->sizes_[l];
                const int no = model->sizes_[l + 1];
                auto& gW = st[l].gW;
                auto& gb = st[l].gb;
                std::fill(gW.begin(), gW.end(), 0.0);
                std::fill(gb.begin(), gb.end(), 0.0);
                const double* W = model->W_[l].data();

                for (int r = 0; r < bn; ++r) {
                    const double* a   = A[l].data() + static_cast<std::size_t>(r) * ni;
                    const double* dlt = D[l + 1].data() + static_cast<std::size_t>(r) * no;
                    double* dprev = (l > 0) ? D[l].data() + static_cast<std::size_t>(r) * ni : nullptr;
                    if (dprev) std::fill(dprev, dprev + ni, 0.0);
                    for (int j = 0; j < no; ++j) {
                        const double dj = dlt[j];
                        if (dj == 0.0) continue;
                        gb[j] += dj;
                        double* g = gW.data() + static_cast<std::size_t>(j) * ni;
                        for (int i = 0; i < ni; ++i) g[i] += dj * a[i];
                        if (dprev) {
                            const double* w = W + static_cast<std::size_t>(j) * ni;
                            for (int i = 0; i < ni; ++i) dprev[i] += dj * w[i];
                        }
                    }
                    if (dprev) {
                        // ReLU derivative: no signal through units that were off.
                        for (int i = 0; i < ni; ++i) if (a[i] <= 0.0) dprev[i] = 0.0;
                        if (l == 1 && !stress.empty()) {
                            for (int i = 0; i < ni; ++i) stress[i] += std::fabs(dprev[i]);
                        }
                    }
                }
            }

            // Adam step with decoupled weight decay.
            ++adam_t;
            const double inv_b = 1.0 / bn;
            const double c1 = 1.0 - std::pow(beta1, static_cast<double>(adam_t));
            const double c2 = 1.0 - std::pow(beta2, static_cast<double>(adam_t));
            const double step = lr * std::sqrt(c2) / c1;
            for (int l = 0; l < L; ++l) {
                auto& W = model->W_[l];
                auto& s = st[l];
                for (std::size_t i = 0; i < W.size(); ++i) {
                    const double g = s.gW[i] * inv_b;
                    s.mW[i] = beta1 * s.mW[i] + (1.0 - beta1) * g;
                    s.vW[i] = beta2 * s.vW[i] + (1.0 - beta2) * g * g;
                    W[i] -= step * s.mW[i] / (std::sqrt(s.vW[i]) + eps)
                          + lr * opt.weight_decay * W[i];
                }
                auto& bias = model->b_[l];
                for (std::size_t i = 0; i < bias.size(); ++i) {
                    const double g = s.gb[i] * inv_b;
                    s.mb[i] = beta1 * s.mb[i] + (1.0 - beta1) * g;
                    s.vb[i] = beta2 * s.vb[i] + (1.0 - beta2) * g * g;
                    bias[i] -= step * s.mb[i] / (std::sqrt(s.vb[i]) + eps);
                }
            }

            if (deadline.expired()) { stopped_by_time = true; break; }
            if (detail::is_cancelled(ctl)) { cancelled = true; break; }
        }
        if (cancelled) break;

        // With no validation rows, fall back to the training loss so the
        // snapshot logic still has something to compare.
        const double val = (nv > 0) ? dataset_loss(d.X_val, d.y_val_cls, yv)
                                    : dataset_loss(d.X_train, d.y_train_cls, yt);
        if (std::isfinite(val) && val < best_val - min_gain) {
            best_val   = val;
            best_epoch = epoch;
            best_sizes = model->sizes_;
            best_W     = model->W_;
            best_b     = model->b_;
            since_best  = 0;
            since_event = 0;
        } else {
            ++since_best;
            ++since_event;
        }

        {
            std::ostringstream msg;
            msg << rep.name << ": epoch " << (epoch + 1);
            if (model->sizes_.size() == 3) msg << ", " << model->sizes_[1] << " hidden units";
            detail::report(ctl, static_cast<double>(epoch + 1) / opt.max_epochs, msg.str());
        }

        if (stopped_by_time) break;
        if (!std::isfinite(val)) break;                 // diverged; keep the best snapshot
        if (since_best >= opt.patience) break;

        if (since_event >= plateau + grace) {
            since_event = 0;
            grace       = 0;
            // A split that did not improve the best loss means the layer is
            // wide enough; stop growing and fall through to learning-rate cuts.
            if (can_grow && growth_events > 0
                && !(best_val < best_at_last_grow * (1.0 - 1e-3))) {
                can_grow = false;
            }
            if (can_grow && model->sizes_[1] < max_hidden && growth_events < 8) {
                // Continue from the best weights, widened.
                model->sizes_ = best_sizes;
                model->W_     = best_W;
                model->b_     = best_b;
                if (static_cast<int>(stress.size()) != model->sizes_[1]) {
                    stress.assign(static_cast<std::size_t>(model->sizes_[1]), 0.0);
                }
                const int h = model->sizes_[1];
                const int k = std::min(max_hidden - h, std::max(4, h / 2));
                best_at_last_grow = best_val;
                grow(k);
                ++growth_events;
                since_best = 0;
                grace      = plateau;
            } else if (lr_cuts < 4) {
                lr *= 0.5;
                ++lr_cuts;
            }
        }
    }

    // Restore the best weights.
    model->sizes_ = best_sizes;
    model->W_     = best_W;
    model->b_     = best_b;
    if (can_grow_at_all && model->sizes_.size() == 3 && model->sizes_[1] > start_width) {
        model->grown_from_ = start_width;
    }

    rep.stopped_by_time = stopped_by_time;
    rep.iterations      = best_epoch + 1;
    rep.seconds         = deadline.elapsed();
    if (best_epoch < 0) {
        // Not a single completed epoch (cancelled or out of time immediately).
        if (cancelled) return rep;
        best_val = (nv > 0) ? dataset_loss(d.X_val, d.y_val_cls, yv)
                            : dataset_loss(d.X_train, d.y_train_cls, yt);
        rep.iterations = 0;
    }
    // Report the loss in the target's own units so it is comparable with
    // the other model families.
    rep.val_loss = cls ? best_val : best_val * y_scale * y_scale;

    {
        std::ostringstream os;
        if (model->sizes_.size() == 3) {
            os << model->sizes_[1] << " hidden units";
            if (model->grown_from_ > 0) os << " (grew from " << model->grown_from_ << ")";
        } else if (model->sizes_.size() > 3) {
            os << (model->sizes_.size() - 2) << " hidden layers";
        }
        rep.detail = os.str();
    }
    rep.model = std::move(model);
    return rep;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

std::unique_ptr<Model> detail::mlp_from_json(const nlohmann::json& j) {
    auto m = std::make_unique<MlpModel>();
    const std::string task = j.at("task").get<std::string>();
    if (task == "classification")  m->task_ = Task::Classification;
    else if (task == "regression") m->task_ = Task::Regression;
    else throw std::runtime_error("mlp: unknown task '" + task + "'");

    m->sizes_      = j.at("sizes").get<std::vector<int>>();
    m->W_          = j.at("W").get<std::vector<std::vector<double>>>();
    m->b_          = j.at("b").get<std::vector<std::vector<double>>>();
    m->y_mean_     = j.value("y_mean", 0.0);
    m->y_scale_    = j.value("y_scale", 1.0);
    m->grown_from_ = j.value("grown_from", 0);

    if (m->sizes_.size() < 2) throw std::runtime_error("mlp: needs at least an input and output size");
    const std::size_t L = m->sizes_.size() - 1;
    if (m->W_.size() != L || m->b_.size() != L) {
        throw std::runtime_error("mlp: layer count does not match 'sizes'");
    }
    for (std::size_t l = 0; l < L; ++l) {
        if (m->sizes_[l] <= 0 || m->sizes_[l + 1] <= 0) {
            throw std::runtime_error("mlp: layer sizes must be positive");
        }
        const std::size_t want = static_cast<std::size_t>(m->sizes_[l]) * m->sizes_[l + 1];
        if (m->W_[l].size() != want
            || m->b_[l].size() != static_cast<std::size_t>(m->sizes_[l + 1])) {
            throw std::runtime_error("mlp: weight shape does not match 'sizes'");
        }
    }
    return m;
}

} // namespace neurograph::models
