#include "neurograph/models/automl.hpp"
#include "detail.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <sstream>
#include <utility>

namespace neurograph::models {

double evaluate_score(const Model& model,
                      const Matrix& X,
                      const std::vector<int>&    y_cls,
                      const std::vector<double>& y_reg) {
    if (X.rows == 0) return 0.0;
    std::vector<double> out(static_cast<std::size_t>(std::max(1, model.n_outputs())));
    if (model.task() == Task::Classification) {
        std::size_t correct = 0;
        for (std::size_t i = 0; i < X.rows; ++i) {
            model.predict(X.row(i), out.data());
            const int pred = static_cast<int>(
                std::max_element(out.begin(), out.end()) - out.begin());
            if (pred == y_cls[i]) ++correct;
        }
        return static_cast<double>(correct) / static_cast<double>(X.rows);
    }
    double mean = 0.0;
    for (std::size_t i = 0; i < X.rows; ++i) mean += y_reg[i];
    mean /= static_cast<double>(X.rows);
    double sse = 0.0, sst = 0.0;
    for (std::size_t i = 0; i < X.rows; ++i) {
        model.predict(X.row(i), out.data());
        sse += (out[0] - y_reg[i]) * (out[0] - y_reg[i]);
        sst += (y_reg[i] - mean) * (y_reg[i] - mean);
    }
    return (sst > 0.0) ? 1.0 - sse / sst : 0.0;
}

namespace {

// A way of training one model from a TrainSet. Recipes capture nothing
// by reference, so they stay valid after auto_fit returns (the learning
// curve calls the chosen one again on smaller samples).
using Recipe = std::function<FitReport(const TrainSet&, const FitControl&)>;

struct Plan {
    double weight;    // share of the time budget
    Recipe run;
};

// Validation predictions of one model, row-major (rows x outputs).
std::vector<double> predict_all(const Model& m, const Matrix& X) {
    const int k = std::max(1, m.n_outputs());
    std::vector<double> out(X.rows * static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < X.rows; ++i) m.predict(X.row(i), out.data() + i * k);
    return out;
}

double loss_of_predictions(const std::vector<double>& preds, int k, bool cls,
                           const std::vector<int>& y_cls,
                           const std::vector<double>& y_reg) {
    const std::size_t rows = preds.size() / static_cast<std::size_t>(k);
    if (rows == 0) return std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        if (cls) {
            total += -std::log(std::max(preds[i * k + y_cls[i]], detail::kProbFloor));
        } else {
            const double d = preds[i * k] - y_reg[i];
            total += d * d;
        }
    }
    return total / static_cast<double>(rows);
}

// Loss of `preds` (row-major, k outputs) on each check row.
std::vector<double> per_row_losses(const std::vector<double>& preds, int k, bool cls,
                                   const std::vector<int>& y_cls,
                                   const std::vector<double>& y_reg) {
    const std::size_t rows = preds.size() / static_cast<std::size_t>(k);
    std::vector<double> out(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        if (cls) {
            out[i] = -std::log(std::max(preds[i * k + y_cls[i]], detail::kProbFloor));
        } else {
            const double d = preds[i * k] - y_reg[i];
            out[i] = d * d;
        }
    }
    return out;
}

// True when the model with row losses `a` beats the one with `b` by more
// than one standard error of their row-by-row difference. With a few dozen
// check rows, smaller gaps are mostly noise.
bool clearly_better(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    if (n == 0) return false;
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += b[i] - a[i];
    mean /= static_cast<double>(n);
    if (n < 2) return mean > 0.0;
    double var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = (b[i] - a[i]) - mean;
        var += d * d;
    }
    var /= static_cast<double>(n - 1);
    return mean > std::sqrt(var / static_cast<double>(n));
}

// Trains two recipes and averages them, `w` on the first. The time limit
// is shared between the two in proportion to their plan weights.
Recipe blend_recipe(Recipe ra, double wa, Recipe rb, double wb, double w) {
    return [ra = std::move(ra), rb = std::move(rb), wa, wb, w](
               const TrainSet& d, const FitControl& c) {
        const double share_a = wa / (wa + wb);
        FitControl ca = c, cb = c;
        if (c.time_limit_seconds > 0.0) {
            ca.time_limit_seconds = c.time_limit_seconds * share_a;
            cb.time_limit_seconds = c.time_limit_seconds * (1.0 - share_a);
        }
        cb.seed = c.seed + 7919ull;
        if (c.progress) {
            ca.progress = [&c, share_a](double f, const std::string& m) {
                c.progress(f * share_a, m);
            };
            cb.progress = [&c, share_a](double f, const std::string& m) {
                c.progress(share_a + f * (1.0 - share_a), m);
            };
        }
        FitReport a = ra(d, ca);
        if (!a.model) return a;
        FitReport b = rb(d, cb);
        if (!b.model) return a;

        FitReport out;
        out.name            = "Blend of the two best";
        out.seconds         = a.seconds + b.seconds;
        out.iterations      = a.iterations + b.iterations;
        out.stopped_by_time = a.stopped_by_time || b.stopped_by_time;
        out.model = make_blend({std::shared_ptr<const Model>(std::move(a.model)),
                                std::shared_ptr<const Model>(std::move(b.model))},
                               {w, 1.0 - w});
        if (d.X_val.rows > 0) {
            out.val_loss = evaluate_loss(*out.model, d.X_val, d.y_val_cls, d.y_val_reg);
        }
        return out;
    };
}

} // namespace

AutoFitResult auto_fit(const TrainSet& data,
                       const AutoFitOptions& opts,
                       std::function<bool()> cancelled,
                       std::function<void(double, const std::string&)> progress) {
    AutoFitResult result;
    const detail::Deadline total(opts.time_budget_seconds);
    const bool cls  = (data.task == Task::Classification);
    const int  n_in = static_cast<int>(data.X_train.cols);
    if (data.X_train.rows == 0 || n_in == 0) return result;

    // ---- What to train -----------------------------------------------------
    std::vector<Plan> plans;
    if (opts.use_linear) {
        plans.push_back({0.10, [](const TrainSet& d, const FitControl& c) {
            MlpOptions o;
            o.hidden     = {};
            o.max_epochs = 200;
            o.patience   = 15;
            return fit_mlp(d, o, c);
        }});
    }
    if (opts.use_gbt) {
        plans.push_back({0.40, [](const TrainSet& d, const FitControl& c) {
            return fit_gbt(d, GbtOptions{}, c);
        }});
    }
    if (opts.use_mlp) {
        plans.push_back({0.50, [](const TrainSet& d, const FitControl& c) {
            MlpOptions o;
            o.hidden = {std::clamp(static_cast<int>(d.X_train.cols), 8, 32)};
            o.grow   = true;
            return fit_mlp(d, o, c);
        }});
    }
    if (opts.thorough) {
        if (opts.use_gbt) {
            plans.push_back({0.50, [](const TrainSet& d, const FitControl& c) {
                GbtOptions o;
                o.learning_rate = 0.05;
                o.max_leaves    = 63;
                o.max_rounds    = 2000;
                o.patience      = 60;
                FitReport r = fit_gbt(d, o, c);
                r.name = "Gradient-boosted trees (large)";
                return r;
            }});
        }
        if (opts.use_mlp) {
            plans.push_back({0.50, [](const TrainSet& d, const FitControl& c) {
                MlpOptions o;
                o.hidden   = {64, 32};
                o.grow     = false;
                o.patience = 30;
                FitReport r = fit_mlp(d, o, c);
                r.name = "Neural network (two layers)";
                return r;
            }});
        }
    }
    if (plans.empty()) return result;

    // ---- Train each candidate ----------------------------------------------
    std::vector<std::shared_ptr<const Model>> models;      // parallel to leaderboard
    std::vector<std::size_t>                  plan_of;     // parallel to leaderboard
    const double n_plans = static_cast<double>(plans.size());
    for (std::size_t i = 0; i < plans.size(); ++i) {
        if (cancelled && cancelled()) { result.cancelled = true; break; }

        FitControl ctl;
        ctl.seed      = opts.seed + 7919ull * i;
        ctl.cancelled = cancelled;
        ctl.progress  = [&, i](double f, const std::string& msg) {
            if (progress) progress((static_cast<double>(i) + f) / n_plans, msg);
        };
        if (opts.time_budget_seconds > 0.0) {
            const double remaining = opts.time_budget_seconds - total.elapsed();
            // Out of time: skip the rest once at least one model exists.
            if (remaining <= 0.0 && !models.empty()) break;
            double weight_left = 0.0;
            for (std::size_t j = i; j < plans.size(); ++j) weight_left += plans[j].weight;
            ctl.time_limit_seconds =
                std::max(0.25, remaining * plans[i].weight / weight_left);
        }

        FitReport rep = plans[i].run(data, ctl);
        if (!rep.model) {
            if (cancelled && cancelled()) { result.cancelled = true; break; }
            continue;                                       // this family could not fit; try the next
        }

        Candidate c;
        c.name            = rep.name;
        c.kind            = rep.model->kind();
        c.detail          = rep.detail;
        c.val_loss        = rep.val_loss;
        c.seconds         = rep.seconds;
        c.iterations      = rep.iterations;
        c.stopped_by_time = rep.stopped_by_time;
        std::shared_ptr<const Model> shared(std::move(rep.model));
        if (data.X_val.rows > 0) {
            c.val_score = evaluate_score(*shared, data.X_val, data.y_val_cls, data.y_val_reg);
        }
        result.leaderboard.push_back(std::move(c));
        models.push_back(std::move(shared));
        plan_of.push_back(i);
    }
    if (models.empty()) return result;

    // ---- Pick the best -----------------------------------------------------
    std::vector<std::size_t> order(models.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return result.leaderboard[a].val_loss < result.leaderboard[b].val_loss;
    });
    std::size_t best = order[0];

    // One-standard-error rule (as in CART): keep the straight-line model
    // unless the winner beats it by more than the noise in the check rows.
    // Small files often produce near-ties, and the simpler model then
    // generalises at least as well.
    bool kept_simplest = false;
    if (data.X_val.rows > 0 && result.leaderboard[best].kind != "linear_v1") {
        for (std::size_t i = 0; i < models.size(); ++i) {
            if (result.leaderboard[i].kind != "linear_v1"
                || !std::isfinite(result.leaderboard[i].val_loss)) continue;
            const int k = std::max(1, models[best]->n_outputs());
            const auto l_best = per_row_losses(predict_all(*models[best], data.X_val), k, cls,
                                               data.y_val_cls, data.y_val_reg);
            const auto l_lin  = per_row_losses(predict_all(*models[i], data.X_val), k, cls,
                                               data.y_val_cls, data.y_val_reg);
            if (!clearly_better(l_best, l_lin)) {
                best          = i;
                kept_simplest = true;
            }
            break;
        }
    }

    result.model                 = models[best];
    result.refit                 = plans[plan_of[best]].run;
    result.refit_seconds         = result.leaderboard[best].seconds;
    result.refit_stopped_by_time = result.leaderboard[best].stopped_by_time;
    const double best_loss = result.leaderboard[best].val_loss;

    // ---- Try averaging the two best ----------------------------------------
    bool blended = false;
    if (opts.try_blend && !kept_simplest && models.size() >= 2 && data.X_val.rows >= 20
        && std::isfinite(result.leaderboard[order[1]].val_loss)) {
        const std::size_t a = order[0], b = order[1];
        const int k = std::max(1, models[a]->n_outputs());
        const std::vector<double> pa = predict_all(*models[a], data.X_val);
        const std::vector<double> pb = predict_all(*models[b], data.X_val);
        double best_w = 1.0, best_blend = best_loss;
        std::vector<double> mix(pa.size());
        for (double w : {0.75, 0.5, 0.25}) {
            for (std::size_t i = 0; i < pa.size(); ++i) mix[i] = w * pa[i] + (1.0 - w) * pb[i];
            const double l = loss_of_predictions(mix, k, cls, data.y_val_cls, data.y_val_reg);
            if (l < best_blend) { best_blend = l; best_w = w; }
        }
        // Keep the single model unless the blend has at least 2% lower loss.
        // (No one-standard-error test here: a blend of two models that were
        // each trained without seeing these rows tunes a single weight from
        // three choices, so it barely overfits them, and on digits the
        // stricter test threw away a blend worth 1.5 points.)
        if (best_w < 1.0 && best_blend < best_loss * 0.98) {
            try {
                std::shared_ptr<const Model> blend(
                    make_blend({models[a], models[b]}, {best_w, 1.0 - best_w}));
                Candidate c;
                c.name       = "Blend of the two best";
                c.kind       = blend->kind();
                {
                    std::ostringstream os;
                    os << static_cast<int>(std::lround(best_w * 100.0)) << "% "
                       << result.leaderboard[a].name << ", "
                       << static_cast<int>(std::lround((1.0 - best_w) * 100.0)) << "% "
                       << result.leaderboard[b].name;
                    c.detail = os.str();
                }
                c.val_loss   = best_blend;
                c.val_score  = evaluate_score(*blend, data.X_val, data.y_val_cls, data.y_val_reg);
                c.seconds    = 0.0;
                c.chosen     = true;

                result.refit = blend_recipe(plans[plan_of[a]].run, plans[plan_of[a]].weight,
                                            plans[plan_of[b]].run, plans[plan_of[b]].weight,
                                            best_w);
                result.refit_seconds = result.leaderboard[a].seconds
                                     + result.leaderboard[b].seconds;
                result.refit_stopped_by_time = result.leaderboard[a].stopped_by_time
                                            || result.leaderboard[b].stopped_by_time;
                result.leaderboard.push_back(std::move(c));
                result.model = std::move(blend);
                blended = true;
            } catch (const std::exception&) {
                // Shapes disagreed; fall back to the single best model.
            }
        }
    }
    if (!blended) result.leaderboard[best].chosen = true;
    if (blended) {
        result.choice_reason = "Averaging the two best models was clearly better on the "
                               "check rows than either one alone.";
    } else if (kept_simplest) {
        result.choice_reason = "The more complex models were not clearly better on the "
                               "check rows, so the simplest was kept: it is likely to "
                               "hold up at least as well on new data.";
    } else {
        result.choice_reason = "It made the smallest errors on the check rows.";
    }
    return result;
}

} // namespace neurograph::models
