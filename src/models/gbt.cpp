// Histogram gradient-boosted decision trees.
//
// The standard recipe for tabular data: bucket every feature into at most
// 255 bins once, then grow each tree leaf-wise by scanning per-bin
// gradient sums instead of sorted values. Each boosting round fits one
// tree per output score to the gradient of the loss (squared error,
// logistic, or softmax). Rounds stop when the validation loss has not
// improved for `patience` rounds, and the model is cut back to its best
// round.
//
// Deterministic: no row or feature subsampling, no threads.

#include "neurograph/models/model.hpp"
#include "detail.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace neurograph::models {

namespace {

struct Tree {
    std::vector<int>    feature;     // -1 marks a leaf
    std::vector<double> threshold;   // go left when x[feature] <= threshold
    std::vector<int>    left, right;
    std::vector<double> value;       // leaf output, learning rate already applied

    int add_leaf() {
        feature.push_back(-1);
        threshold.push_back(0.0);
        left.push_back(-1);
        right.push_back(-1);
        value.push_back(0.0);
        return static_cast<int>(feature.size()) - 1;
    }

    double predict(const double* x) const {
        int n = 0;
        while (feature[n] >= 0) {
            n = (x[feature[n]] <= threshold[n]) ? left[n] : right[n];
        }
        return value[n];
    }

    int leaf_count() const {
        int c = 0;
        for (int f : feature) if (f < 0) ++c;
        return c;
    }
};

class GbtModel final : public Model {
public:
    Task                task_       = Task::Regression;
    int                 n_features_ = 0;
    int                 n_classes_  = 0;   // 0 for regression
    int                 n_scores_   = 1;   // 1 for regression and binary, K for multiclass
    std::vector<double> base_;             // starting score per output
    std::vector<Tree>   trees_;            // round-major: trees_[round * n_scores_ + k]
    int                 max_leaves_ = 31;

    const char* kind() const noexcept override { return "gbt_v1"; }
    Task task() const noexcept override { return task_; }
    int  n_features() const noexcept override { return n_features_; }
    int  n_outputs()  const noexcept override {
        return task_ == Task::Classification ? n_classes_ : 1;
    }

    void predict(const double* x, double* out) const override {
        double  small[16];
        std::vector<double> big;
        double* s = small;
        if (n_scores_ > 16) { big.resize(static_cast<std::size_t>(n_scores_)); s = big.data(); }
        for (int k = 0; k < n_scores_; ++k) s[k] = base_[k];
        const std::size_t nt = trees_.size();
        for (std::size_t t = 0; t < nt; ++t) {
            s[t % static_cast<std::size_t>(n_scores_)] += trees_[t].predict(x);
        }
        if (task_ == Task::Regression) {
            out[0] = s[0];
        } else if (n_scores_ == 1) {
            const double p = detail::sigmoid(s[0]);
            out[0] = 1.0 - p;
            out[1] = p;
        } else {
            detail::softmax(s, n_scores_);
            for (int k = 0; k < n_scores_; ++k) out[k] = s[k];
        }
    }

    nlohmann::json to_json() const override {
        nlohmann::json j;
        j["kind"]       = kind();
        j["task"]       = (task_ == Task::Classification) ? "classification" : "regression";
        j["n_features"] = n_features_;
        j["n_classes"]  = n_classes_;
        j["n_scores"]   = n_scores_;
        j["max_leaves"] = max_leaves_;
        j["base"]       = base_;
        j["trees"]      = nlohmann::json::array();
        for (const auto& t : trees_) {
            nlohmann::json tj;
            tj["f"] = t.feature;
            tj["t"] = t.threshold;
            tj["l"] = t.left;
            tj["r"] = t.right;
            tj["v"] = t.value;
            j["trees"].push_back(std::move(tj));
        }
        return j;
    }

    std::string describe() const override {
        std::ostringstream os;
        os << "Gradient-boosted trees (" << trees_.size() << " trees, up to "
           << max_leaves_ << " leaves each)";
        return os.str();
    }

    std::size_t parameter_count() const noexcept override {
        std::size_t n = 0;
        for (const auto& t : trees_) n += t.feature.size();
        return n;
    }
};

// ---------------------------------------------------------------------------
// Binning
// ---------------------------------------------------------------------------

struct Binned {
    std::vector<std::vector<double>>       cuts;     // per feature, ascending
    std::vector<std::vector<std::uint8_t>> bins;     // per feature, one entry per row
    std::vector<int>                       n_bins;   // cuts.size() + 1
    std::vector<int>                       offset;   // start of each feature in a histogram
    int                                    total_bins = 0;
};

// bin(x) = number of cuts strictly below x, so "bin <= b" means "x <= cuts[b]".
inline std::uint8_t bin_of(const std::vector<double>& cuts, double x) {
    return static_cast<std::uint8_t>(
        std::lower_bound(cuts.begin(), cuts.end(), x) - cuts.begin());
}

Binned make_bins(const Matrix& X, int max_bins) {
    Binned b;
    const std::size_t n = X.rows;
    const int F = static_cast<int>(X.cols);
    max_bins = std::clamp(max_bins, 2, 256);
    b.cuts.resize(F);
    b.bins.resize(F);
    b.n_bins.resize(F);
    b.offset.resize(F);

    std::vector<double> col(n);
    for (int f = 0; f < F; ++f) {
        for (std::size_t i = 0; i < n; ++i) col[i] = X.row(i)[f];
        std::vector<double> sorted = col;
        std::sort(sorted.begin(), sorted.end());
        std::vector<double> uniq = sorted;
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

        auto& cuts = b.cuts[f];
        if (static_cast<int>(uniq.size()) <= max_bins) {
            // One bin per distinct value; cut half-way between neighbours.
            for (std::size_t i = 0; i + 1 < uniq.size(); ++i) {
                cuts.push_back(uniq[i] + (uniq[i + 1] - uniq[i]) * 0.5);
            }
        } else {
            // Equal-frequency cuts.
            for (int q = 1; q < max_bins; ++q) {
                const std::size_t pos = static_cast<std::size_t>(
                    static_cast<double>(q) * static_cast<double>(n) / max_bins);
                const double c = sorted[std::min(pos, n - 1)];
                if (cuts.empty() || c > cuts.back()) cuts.push_back(c);
            }
            // A cut at the maximum would leave an empty right side.
            while (!cuts.empty() && cuts.back() >= sorted.back()) cuts.pop_back();
        }

        b.n_bins[f] = static_cast<int>(cuts.size()) + 1;
        b.offset[f] = b.total_bins;
        b.total_bins += b.n_bins[f];

        auto& bins = b.bins[f];
        bins.resize(n);
        for (std::size_t i = 0; i < n; ++i) bins[i] = bin_of(cuts, col[i]);
    }
    return b;
}

// ---------------------------------------------------------------------------
// Tree growing
// ---------------------------------------------------------------------------

struct HistBin {
    double g = 0.0;
    double h = 0.0;
    int    c = 0;
};

struct OpenLeaf {
    int    begin = 0, end = 0;     // range in the row-index array
    double G = 0.0, H = 0.0;
    int    depth = 0;
    int    node  = 0;              // index in the tree arrays
    std::vector<HistBin> hist;

    // Best split found for this leaf (gain <= 0 means none).
    double gain = 0.0;
    int    feat = -1;
    int    bin  = -1;
    double GL = 0.0, HL = 0.0;
    int    cL = 0;
};

struct GrowParams {
    int    max_leaves;
    int    max_depth;
    int    min_samples_leaf;
    double l2;
    double learning_rate;
};

void build_hist(OpenLeaf& leaf, const Binned& b, const std::vector<int>& idx,
                const std::vector<double>& g, const std::vector<double>& h) {
    leaf.hist.assign(static_cast<std::size_t>(b.total_bins), HistBin{});
    const int F = static_cast<int>(b.bins.size());
    for (int f = 0; f < F; ++f) {
        if (b.n_bins[f] < 2) continue;                 // constant feature
        const std::uint8_t* col = b.bins[f].data();
        HistBin* hb = leaf.hist.data() + b.offset[f];
        for (int i = leaf.begin; i < leaf.end; ++i) {
            const int r = idx[static_cast<std::size_t>(i)];
            HistBin& e = hb[col[r]];
            e.g += g[static_cast<std::size_t>(r)];
            e.h += h[static_cast<std::size_t>(r)];
            ++e.c;
        }
    }
}

void find_split(OpenLeaf& leaf, const Binned& b, const GrowParams& p) {
    leaf.gain = 0.0;
    leaf.feat = -1;
    const int count = leaf.end - leaf.begin;
    if (leaf.depth >= p.max_depth || count < 2 * p.min_samples_leaf) return;

    const double parent_score = leaf.G * leaf.G / (leaf.H + p.l2);
    const int F = static_cast<int>(b.bins.size());
    for (int f = 0; f < F; ++f) {
        const int nb = b.n_bins[f];
        if (nb < 2) continue;
        const HistBin* hb = leaf.hist.data() + b.offset[f];
        double GL = 0.0, HL = 0.0;
        int    cL = 0;
        for (int bin = 0; bin + 1 < nb; ++bin) {
            GL += hb[bin].g;
            HL += hb[bin].h;
            cL += hb[bin].c;
            const int cR = count - cL;
            if (cL < p.min_samples_leaf) continue;
            if (cR < p.min_samples_leaf) break;
            const double GR = leaf.G - GL;
            const double HR = leaf.H - HL;
            if (HL < 1e-12 || HR < 1e-12) continue;
            const double gain = GL * GL / (HL + p.l2) + GR * GR / (HR + p.l2) - parent_score;
            if (gain > leaf.gain + 1e-12) {
                leaf.gain = gain;
                leaf.feat = f;
                leaf.bin  = bin;
                leaf.GL   = GL;
                leaf.HL   = HL;
                leaf.cL   = cL;
            }
        }
    }
}

// Grows one tree on gradients (g, h) and adds its leaf outputs to
// `score[row * stride + k]` for the training rows.
Tree grow_tree(const Binned& b, std::vector<int>& idx,
               const std::vector<double>& g, const std::vector<double>& h,
               const GrowParams& p,
               std::vector<double>& score, int stride, int k) {
    Tree tree;
    std::vector<OpenLeaf> leaves;
    leaves.reserve(static_cast<std::size_t>(p.max_leaves) * 2);

    {
        OpenLeaf root;
        root.begin = 0;
        root.end   = static_cast<int>(idx.size());
        for (int r : idx) { root.G += g[static_cast<std::size_t>(r)]; root.H += h[static_cast<std::size_t>(r)]; }
        root.node = tree.add_leaf();
        build_hist(root, b, idx, g, h);
        find_split(root, b, p);
        leaves.push_back(std::move(root));
    }

    int n_leaves = 1;
    while (n_leaves < p.max_leaves) {
        int best = -1;
        for (int i = 0; i < static_cast<int>(leaves.size()); ++i) {
            if (leaves[i].feat >= 0 && (best < 0 || leaves[i].gain > leaves[best].gain)) best = i;
        }
        if (best < 0) break;

        // Take the leaf out of the open list by value; its slot is reused.
        OpenLeaf parent = std::move(leaves[static_cast<std::size_t>(best)]);
        const std::uint8_t* col = b.bins[static_cast<std::size_t>(parent.feat)].data();
        const int split_bin = parent.bin;
        auto mid_it = std::partition(
            idx.begin() + parent.begin, idx.begin() + parent.end,
            [&](int r) { return col[r] <= split_bin; });
        const int mid = static_cast<int>(mid_it - idx.begin());

        OpenLeaf lo, hi;
        lo.begin = parent.begin; lo.end = mid;
        lo.G = parent.GL;        lo.H = parent.HL;
        hi.begin = mid;          hi.end = parent.end;
        hi.G = parent.G - parent.GL;
        hi.H = parent.H - parent.HL;
        lo.depth = hi.depth = parent.depth + 1;
        lo.node = tree.add_leaf();
        hi.node = tree.add_leaf();

        tree.feature[static_cast<std::size_t>(parent.node)]   = parent.feat;
        tree.threshold[static_cast<std::size_t>(parent.node)] =
            b.cuts[static_cast<std::size_t>(parent.feat)][static_cast<std::size_t>(parent.bin)];
        tree.left[static_cast<std::size_t>(parent.node)]  = lo.node;
        tree.right[static_cast<std::size_t>(parent.node)] = hi.node;

        // Scan the smaller child; get the other by subtracting from the parent.
        OpenLeaf& small_child = (lo.end - lo.begin <= hi.end - hi.begin) ? lo : hi;
        OpenLeaf& large_child = (&small_child == &lo) ? hi : lo;
        build_hist(small_child, b, idx, g, h);
        large_child.hist = std::move(parent.hist);
        for (std::size_t i = 0; i < large_child.hist.size(); ++i) {
            large_child.hist[i].g -= small_child.hist[i].g;
            large_child.hist[i].h -= small_child.hist[i].h;
            large_child.hist[i].c -= small_child.hist[i].c;
        }
        find_split(lo, b, p);
        find_split(hi, b, p);

        leaves[static_cast<std::size_t>(best)] = std::move(lo);
        leaves.push_back(std::move(hi));
        ++n_leaves;
    }

    for (const auto& leaf : leaves) {
        const double v = -leaf.G / (leaf.H + p.l2) * p.learning_rate;
        tree.value[static_cast<std::size_t>(leaf.node)] = v;
        for (int i = leaf.begin; i < leaf.end; ++i) {
            score[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]) * stride + k] += v;
        }
    }
    return tree;
}

int pick_min_samples_leaf(std::size_t n, int requested) {
    if (requested > 0) return requested;
    if (n < 100)  return 2;
    if (n < 500)  return 5;
    if (n < 2000) return 10;
    return 20;
}

} // namespace

// ---------------------------------------------------------------------------
// fit_gbt
// ---------------------------------------------------------------------------

FitReport fit_gbt(const TrainSet& d, const GbtOptions& opt, const FitControl& ctl) {
    const detail::Deadline deadline(ctl.time_limit_seconds);
    FitReport rep;
    rep.name = "Gradient-boosted trees";

    const bool        cls = (d.task == Task::Classification);
    const std::size_t n   = d.X_train.rows;
    const std::size_t nv  = d.X_val.rows;
    const int         F   = static_cast<int>(d.X_train.cols);
    if (n == 0 || F == 0) return rep;
    if (cls && d.n_classes < 2) return rep;

    auto model = std::make_unique<GbtModel>();
    model->task_       = d.task;
    model->n_features_ = F;
    model->n_classes_  = cls ? d.n_classes : 0;
    model->n_scores_   = (cls && d.n_classes > 2) ? d.n_classes : 1;
    model->max_leaves_ = opt.max_leaves;
    const int S = model->n_scores_;

    // ---- Starting scores ---------------------------------------------------
    model->base_.assign(static_cast<std::size_t>(S), 0.0);
    if (!cls) {
        double mean = 0.0;
        for (double y : d.y_train_reg) mean += y;
        model->base_[0] = mean / static_cast<double>(n);
    } else if (S == 1) {
        double pos = 0.0;
        for (int y : d.y_train_cls) pos += (y == 1) ? 1.0 : 0.0;
        const double pr = std::clamp(pos / static_cast<double>(n), 1e-6, 1.0 - 1e-6);
        model->base_[0] = std::log(pr / (1.0 - pr));
    } else {
        std::vector<double> prior(static_cast<std::size_t>(S), 0.0);
        for (int y : d.y_train_cls) prior[static_cast<std::size_t>(y)] += 1.0;
        for (int k = 0; k < S; ++k) {
            model->base_[static_cast<std::size_t>(k)] =
                std::log((prior[static_cast<std::size_t>(k)] + 1.0) / (static_cast<double>(n) + S));
        }
    }

    const Binned binned = make_bins(d.X_train, opt.max_bins);

    GrowParams gp;
    gp.max_leaves       = std::max(2, opt.max_leaves);
    gp.max_depth        = std::max(1, opt.max_depth);
    gp.min_samples_leaf = pick_min_samples_leaf(n, opt.min_samples_leaf);
    gp.l2               = opt.l2;
    gp.learning_rate    = opt.learning_rate;

    std::vector<double> score(n * static_cast<std::size_t>(S));
    std::vector<double> vscore(nv * static_cast<std::size_t>(S));
    for (std::size_t i = 0; i < n; ++i)
        for (int k = 0; k < S; ++k) score[i * S + k] = model->base_[static_cast<std::size_t>(k)];
    for (std::size_t i = 0; i < nv; ++i)
        for (int k = 0; k < S; ++k) vscore[i * S + k] = model->base_[static_cast<std::size_t>(k)];

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> g(n), h(n);
    std::vector<double> prob;                    // multiclass only: n x S
    if (S > 1) prob.resize(n * static_cast<std::size_t>(S));

    // Mean loss from raw scores.
    auto loss_of = [&](const std::vector<double>& sc, std::size_t rows,
                       const std::vector<int>& yc, const std::vector<double>& yr) {
        if (rows == 0) return std::numeric_limits<double>::infinity();
        double total = 0.0;
        std::vector<double> tmp(static_cast<std::size_t>(S));
        for (std::size_t i = 0; i < rows; ++i) {
            if (!cls) {
                const double diff = sc[i] - yr[i];
                total += diff * diff;
            } else if (S == 1) {
                const double p  = detail::sigmoid(sc[i]);
                const double py = (yc[i] == 1) ? p : 1.0 - p;
                total += -std::log(std::max(py, detail::kProbFloor));
            } else {
                for (int k = 0; k < S; ++k) tmp[static_cast<std::size_t>(k)] = sc[i * S + k];
                detail::softmax(tmp.data(), S);
                total += -std::log(std::max(tmp[static_cast<std::size_t>(yc[i])], detail::kProbFloor));
            }
        }
        return total / static_cast<double>(rows);
    };

    // A round counts as progress only if it beats the best so far by 0.01%
    // of the starting loss (the loss of predicting the class frequencies or
    // the mean). On rows the trees can separate perfectly the loss keeps
    // shrinking by ever smaller amounts, and without this boosting would
    // run to max_rounds and build a needlessly large model. Gains are
    // measured against the best so far, so steady small steps still count.
    const double start_loss = (nv > 0) ? loss_of(vscore, nv, d.y_val_cls, d.y_val_reg)
                                       : loss_of(score,  n,  d.y_train_cls, d.y_train_reg);
    const double min_gain = std::isfinite(start_loss) ? std::max(1e-12, 1e-4 * start_loss)
                                                      : 1e-9;

    double best_loss  = std::numeric_limits<double>::infinity();
    int    best_round = -1;
    int    since_best = 0;
    bool   stopped_by_time = false;
    bool   cancelled       = false;
    // Without validation rows there is nothing to stop on; cap the rounds.
    const int max_rounds = (nv > 0) ? opt.max_rounds : std::min(opt.max_rounds, 100);

    for (int round = 0; round < max_rounds; ++round) {
        if (S > 1) {
            for (std::size_t i = 0; i < n; ++i) {
                double* pr = prob.data() + i * S;
                for (int k = 0; k < S; ++k) pr[k] = score[i * S + k];
                detail::softmax(pr, S);
            }
        }
        for (int k = 0; k < S; ++k) {
            if (!cls) {
                for (std::size_t i = 0; i < n; ++i) { g[i] = score[i] - d.y_train_reg[i]; h[i] = 1.0; }
            } else if (S == 1) {
                for (std::size_t i = 0; i < n; ++i) {
                    const double p = detail::sigmoid(score[i]);
                    g[i] = p - ((d.y_train_cls[i] == 1) ? 1.0 : 0.0);
                    h[i] = std::max(p * (1.0 - p), 1e-12);
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    const double p = prob[i * S + k];
                    g[i] = p - ((d.y_train_cls[i] == k) ? 1.0 : 0.0);
                    h[i] = std::max(p * (1.0 - p), 1e-12);
                }
            }
            Tree tree = grow_tree(binned, idx, g, h, gp, score, S, k);
            for (std::size_t i = 0; i < nv; ++i) {
                vscore[i * S + k] += tree.predict(d.X_val.row(i));
            }
            model->trees_.push_back(std::move(tree));
        }

        const double loss = (nv > 0) ? loss_of(vscore, nv, d.y_val_cls, d.y_val_reg)
                                     : loss_of(score,  n,  d.y_train_cls, d.y_train_reg);
        if (std::isfinite(loss) && loss < best_loss - min_gain) {
            best_loss  = loss;
            best_round = round;
            since_best = 0;
        } else {
            ++since_best;
        }

        if ((round & 7) == 7 || round + 1 == max_rounds) {
            std::ostringstream msg;
            msg << rep.name << ": " << (round + 1) << " rounds";
            detail::report(ctl, static_cast<double>(round + 1) / max_rounds, msg.str());
        }

        if (nv > 0 && since_best >= opt.patience) break;
        if (!std::isfinite(loss)) break;
        if (deadline.expired()) { stopped_by_time = true; break; }
        if (detail::is_cancelled(ctl)) { cancelled = true; break; }
    }

    rep.seconds         = deadline.elapsed();
    rep.stopped_by_time = stopped_by_time;
    if (best_round < 0) {
        if (cancelled) return rep;
        // No round completed with a finite loss; return the prior-only model.
        model->trees_.clear();
        rep.val_loss = (nv > 0) ? loss_of(vscore, nv, d.y_val_cls, d.y_val_reg)
                                : std::numeric_limits<double>::infinity();
        rep.model = std::move(model);
        return rep;
    }

    // Cut back to the best round.
    model->trees_.resize(static_cast<std::size_t>(best_round + 1) * S);
    rep.iterations = best_round + 1;
    rep.val_loss   = best_loss;
    {
        std::ostringstream os;
        os << model->trees_.size() << " trees";
        rep.detail = os.str();
    }
    rep.model = std::move(model);
    return rep;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

std::unique_ptr<Model> detail::gbt_from_json(const nlohmann::json& j) {
    auto m = std::make_unique<GbtModel>();
    const std::string task = j.at("task").get<std::string>();
    if (task == "classification")  m->task_ = Task::Classification;
    else if (task == "regression") m->task_ = Task::Regression;
    else throw std::runtime_error("gbt: unknown task '" + task + "'");

    m->n_features_ = j.at("n_features").get<int>();
    m->n_classes_  = j.value("n_classes", 0);
    m->n_scores_   = j.at("n_scores").get<int>();
    m->max_leaves_ = j.value("max_leaves", 31);
    m->base_       = j.at("base").get<std::vector<double>>();

    if (m->n_features_ <= 0) throw std::runtime_error("gbt: n_features must be positive");
    if (m->n_scores_ <= 0 || static_cast<int>(m->base_.size()) != m->n_scores_) {
        throw std::runtime_error("gbt: 'base' does not match n_scores");
    }
    if (m->task_ == Task::Classification) {
        const int want = (m->n_classes_ > 2) ? m->n_classes_ : 1;
        if (m->n_classes_ < 2 || m->n_scores_ != want) {
            throw std::runtime_error("gbt: n_classes and n_scores disagree");
        }
    } else if (m->n_scores_ != 1) {
        throw std::runtime_error("gbt: regression needs exactly one score");
    }

    for (const auto& tj : j.at("trees")) {
        Tree t;
        t.feature   = tj.at("f").get<std::vector<int>>();
        t.threshold = tj.at("t").get<std::vector<double>>();
        t.left      = tj.at("l").get<std::vector<int>>();
        t.right     = tj.at("r").get<std::vector<int>>();
        t.value     = tj.at("v").get<std::vector<double>>();
        const std::size_t nn = t.feature.size();
        if (nn == 0 || t.threshold.size() != nn || t.left.size() != nn
            || t.right.size() != nn || t.value.size() != nn) {
            throw std::runtime_error("gbt: malformed tree arrays");
        }
        // Reject anything that could send predict() out of bounds or into a loop:
        // children must exist and come after their parent.
        for (std::size_t i = 0; i < nn; ++i) {
            if (t.feature[i] < 0) continue;
            if (t.feature[i] >= m->n_features_) throw std::runtime_error("gbt: feature index out of range");
            const int l = t.left[i], r = t.right[i];
            if (l <= static_cast<int>(i) || r <= static_cast<int>(i)
                || l >= static_cast<int>(nn) || r >= static_cast<int>(nn)) {
                throw std::runtime_error("gbt: child index out of range");
            }
        }
        m->trees_.push_back(std::move(t));
    }
    if (m->trees_.size() % static_cast<std::size_t>(m->n_scores_) != 0) {
        throw std::runtime_error("gbt: tree count is not a multiple of n_scores");
    }
    return m;
}

} // namespace neurograph::models
