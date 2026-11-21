// Implementation of the P1-08 C ABI. All public symbols live in
// `extern "C"` blocks below; the file is C++ internally so we can
// reuse the existing runtime::ModelBundle loader and predict helpers.
//
// Contract notes restated from the header:
//   - ng_model is a tagged wrapper around runtime::ModelBundle. The
//     caller sees only the opaque pointer.
//   - Error reporting goes through `set_err` + `ng_last_error()`, which
//     uses thread_local storage so two threads racing on the same model
//     pointer don't clobber each other's message.
//   - Predictions must be safe to call concurrently on a single model,
//     so the wrapper doesn't take a lock — `predict_row` reads only
//     const state from the bundle.

#define NEUROGRAPH_RT_BUILDING 1
#include "neurograph/version.hpp"
#include "neurograph/capi/neurograph_rt.h"

#include "neurograph/runtime/bundle.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

// Thread-local storage for ng_last_error. std::string has a stable
// c_str() inside the same thread; callers are told not to share the
// pointer across threads, so TLS is enough.
thread_local std::string g_last_err;

void set_err(const char* msg) noexcept {
    try {
        g_last_err = msg ? msg : "";
    } catch (...) {
        // If string assignment threw (OOM), fall back to empty so
        // ng_last_error still returns a valid pointer.
        g_last_err.clear();
    }
}

void set_err(const std::string& m) noexcept { set_err(m.c_str()); }

// Hidden wrapper holding the bundle plus any derived data the C ABI
// surfaces as pointers (version string, class labels). C callers see
// `ng_model*` which is this struct reinterpret-cast — safe because
// ng_model is declared but never defined in the header.
struct NgModelImpl {
    neurograph::runtime::ModelBundle bundle;
    std::string                       version_cached;   // pinned storage for ng_model_version
};

} // namespace

// The opaque `ng_model` struct the header forward-declares is the
// same type as NgModelImpl — we just expose it under the tag name.
struct ng_model : NgModelImpl {};

namespace {

ng_model* to_handle(std::unique_ptr<NgModelImpl> p) noexcept {
    // NgModelImpl → ng_model is a safe upcast because ng_model inherits
    // from NgModelImpl with no added state. `release()` transfers
    // ownership to the caller, who eventually hands it back via
    // ng_model_free.
    return static_cast<ng_model*>(p.release());
}

const NgModelImpl* as_impl(const ng_model* m) noexcept {
    return static_cast<const NgModelImpl*>(m);
}

NgModelImpl* as_impl(ng_model* m) noexcept {
    return static_cast<NgModelImpl*>(m);
}

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

NG_EXPORT ng_status ng_model_load(const char* bundle_dir, ng_model** out) {
    if (!out) {
        set_err("ng_model_load: out pointer is NULL");
        return NG_ERR_BAD_ARG;
    }
    *out = nullptr;
    if (!bundle_dir) {
        set_err("ng_model_load: bundle_dir is NULL");
        return NG_ERR_BAD_ARG;
    }
    try {
        auto result = neurograph::runtime::load_bundle(bundle_dir);
        if (!result.ok()) {
            set_err(result.error().message);
            // Classify: "not a directory" / "cannot open" → IO, else
            // corrupt payload. Good enough for the customer-facing
            // status. ng_last_error carries the exact reason.
            const auto& m = result.error().message;
            if (m.find("not a directory") != std::string::npos ||
                m.find("cannot open")     != std::string::npos) {
                return NG_ERR_IO;
            }
            return NG_ERR_CORRUPT;
        }
        auto impl = std::make_unique<NgModelImpl>();
        impl->bundle = std::move(result).value();

        // Pull the version string out of the manifest once; the raw
        // pointer we return later must outlive the call, so pin it on
        // the handle.
        if (impl->bundle.manifest.is_object()) {
            const auto& mf = impl->bundle.manifest;
            if (mf.contains("neurograph_version") &&
                mf["neurograph_version"].is_string()) {
                impl->version_cached = mf["neurograph_version"].get<std::string>();
            }
        }
        *out = to_handle(std::move(impl));
        set_err("");
        return NG_OK;
    } catch (const std::exception& e) {
        set_err(std::string("ng_model_load: ") + e.what());
        return NG_ERR_INTERNAL;
    } catch (...) {
        set_err("ng_model_load: unknown exception");
        return NG_ERR_INTERNAL;
    }
}

NG_EXPORT void ng_model_free(ng_model* m) {
    // delete on NULL is a no-op per the standard; we rely on that.
    delete as_impl(m);
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

NG_EXPORT int ng_model_num_features(const ng_model* m) {
    if (!m) return 0;
    return as_impl(m)->bundle.n_features();
}

NG_EXPORT int ng_model_num_classes(const ng_model* m) {
    if (!m) return 0;
    return as_impl(m)->bundle.n_classes();   // already handles both variants
}

NG_EXPORT const char* ng_model_class_label(const ng_model* m, int idx) {
    if (!m) return nullptr;
    const auto& bundle = as_impl(m)->bundle;
    if (const auto* c = std::get_if<neurograph::runtime::LinearClassificationModel>(&bundle.model)) {
        if (idx < 0 || idx >= static_cast<int>(c->class_labels.size())) return nullptr;
        return c->class_labels[idx].c_str();
    }
    // P1-15: SON cls variant has the same label list shape.
    if (const auto* nc = std::get_if<neurograph::runtime::NeuralNetClassificationModel>(&bundle.model)) {
        if (idx < 0 || idx >= static_cast<int>(nc->class_labels.size())) return nullptr;
        return nc->class_labels[idx].c_str();
    }
    if (const auto* nm = std::get_if<neurograph::runtime::NativeModel>(&bundle.model)) {
        if (idx < 0 || idx >= static_cast<int>(nm->class_labels.size())) return nullptr;
        return nm->class_labels[idx].c_str();
    }
    return nullptr;
}

NG_EXPORT const char* ng_model_version(const ng_model* m) {
    if (!m) return "";
    return as_impl(m)->version_cached.c_str();
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

namespace {

// Fill `out_probs` / `out_pred` from a Prediction. Handles the size-1
// out_probs case (cls caller wants argmax prob only) and reg case.
void write_prediction(const ng_model* m,
                      const neurograph::runtime::Prediction& pr,
                      double* out_probs,
                      int*    out_pred) {
    const auto& bundle = as_impl(m)->bundle;
    if (bundle.is_classification()) {
        if (out_probs) {
            for (std::size_t k = 0; k < pr.probabilities.size(); ++k) {
                out_probs[k] = pr.probabilities[k];
            }
        }
        if (out_pred) *out_pred = pr.predicted_class;
    } else {
        if (out_probs) out_probs[0] = pr.value;
        // out_pred stays untouched for regression.
    }
}

} // namespace

NG_EXPORT ng_status ng_predict_json(const ng_model* m,
                                    const char*     json_row,
                                    double*         out_probs,
                                    int*            out_pred) {
    if (!m) {
        set_err("ng_predict_json: model is NULL");
        return NG_ERR_BAD_ARG;
    }
    if (!json_row) {
        set_err("ng_predict_json: json_row is NULL");
        return NG_ERR_BAD_ARG;
    }
    try {
        nlohmann::json j = nlohmann::json::parse(json_row);
        const auto pr = neurograph::runtime::predict_row(as_impl(m)->bundle, j);
        write_prediction(m, pr, out_probs, out_pred);
        set_err("");
        return NG_OK;
    } catch (const nlohmann::json::parse_error& e) {
        set_err(std::string("ng_predict_json: ") + e.what());
        return NG_ERR_BAD_ARG;
    } catch (const std::exception& e) {
        set_err(std::string("ng_predict_json: ") + e.what());
        return NG_ERR_INTERNAL;
    } catch (...) {
        set_err("ng_predict_json: unknown exception");
        return NG_ERR_INTERNAL;
    }
}

NG_EXPORT ng_status ng_predict_vec(const ng_model* m,
                                   const double*   features,
                                   int             feature_count,
                                   double*         out_probs,
                                   int*            out_pred) {
    if (!m) {
        set_err("ng_predict_vec: model is NULL");
        return NG_ERR_BAD_ARG;
    }
    if (!features || feature_count < 0) {
        set_err("ng_predict_vec: features buffer invalid");
        return NG_ERR_BAD_ARG;
    }
    try {
        const auto& bundle = as_impl(m)->bundle;
        const std::vector<double> x(features, features + feature_count);

        // Run the model directly, bypassing apply_pipeline — the caller
        // is telling us they already have a transformed vector. This is
        // the path latency-sensitive customers use (transform once,
        // predict many times on a cached vector).
        //
        // For linear models we hand-roll the matrix multiply for max
        // throughput. For SON variants (P1-15) we forward through
        // Backprop the same way `predict_row` does — there's no faster
        // shortcut without exposing the Backprop forward pass directly.
        neurograph::runtime::Prediction pr;
        if (const auto* c = std::get_if<neurograph::runtime::LinearClassificationModel>(&bundle.model)) {
            if (c->n_classes < 2) {
                set_err("ng_predict_vec: model has <2 classes");
                return NG_ERR_CORRUPT;
            }
            std::vector<double> z(c->n_classes, 0.0);
            const std::size_t D = std::min<std::size_t>(x.size(), c->n_features);
            for (int k = 0; k < c->n_classes; ++k) {
                double s = c->b[k];
                for (std::size_t d = 0; d < D; ++d) s += c->W[k][d] * x[d];
                z[k] = s;
            }
            const double zmax = *std::max_element(z.begin(), z.end());
            double       sum  = 0.0;
            for (double& v : z) { v = std::exp(v - zmax); sum += v; }
            if (sum > 0.0) for (double& v : z) v /= sum;
            int best = 0;
            for (std::size_t k = 1; k < z.size(); ++k) if (z[k] > z[best]) best = static_cast<int>(k);
            pr.predicted_class = best;
            pr.probabilities   = std::move(z);
        } else if (const auto* r = std::get_if<neurograph::runtime::LinearRegressionModel>(&bundle.model)) {
            double yhat = r->bias;
            const std::size_t D = std::min<std::size_t>(x.size(), r->n_features);
            for (std::size_t d = 0; d < D; ++d) yhat += r->w[d] * x[d];
            pr.value = yhat;
        } else {
            // P1-15 SON variants. Delegate to bundle.cpp's
            // `predict_vec` helper — keeps NeuralNetInferenceState
            // opaque to this TU (it's a PIMPL forward-declared in
            // bundle.hpp; only bundle.cpp has the full definition).
            pr = neurograph::runtime::predict_vec(bundle, x);
        }
        write_prediction(m, pr, out_probs, out_pred);
        set_err("");
        return NG_OK;
    } catch (const std::exception& e) {
        set_err(std::string("ng_predict_vec: ") + e.what());
        return NG_ERR_INTERNAL;
    } catch (...) {
        set_err("ng_predict_vec: unknown exception");
        return NG_ERR_INTERNAL;
    }
}

NG_EXPORT ng_status ng_predict_batch_vec(const ng_model* m,
                                         const double*   features_flat,
                                         int             batch_size,
                                         int             feature_count,
                                         double*         out_probs_flat,
                                         int*            out_preds) {
    if (!m) {
        set_err("ng_predict_batch_vec: model is NULL");
        return NG_ERR_BAD_ARG;
    }
    if (batch_size < 0 || feature_count < 0) {
        set_err("ng_predict_batch_vec: negative size");
        return NG_ERR_BAD_ARG;
    }
    if (batch_size > 0 && !features_flat) {
        set_err("ng_predict_batch_vec: features_flat is NULL");
        return NG_ERR_BAD_ARG;
    }
    const int n_out_per_row =
        ng_model_num_classes(m) > 0 ? ng_model_num_classes(m) : 1;

    for (int i = 0; i < batch_size; ++i) {
        double* probs_i = out_probs_flat
            ? (out_probs_flat + static_cast<std::size_t>(i) * n_out_per_row)
            : nullptr;
        int*    pred_i  = out_preds ? (out_preds + i) : nullptr;
        const auto* row = features_flat + static_cast<std::size_t>(i) * feature_count;
        const ng_status s = ng_predict_vec(m, row, feature_count, probs_i, pred_i);
        if (s != NG_OK) return s;   // err already set by ng_predict_vec
    }
    set_err("");
    return NG_OK;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

NG_EXPORT const char* ng_last_error(void) {
    return g_last_err.c_str();
}

NG_EXPORT const char* ng_rt_version(void) {
    return neurograph::kVersion;
}

} // extern "C"
