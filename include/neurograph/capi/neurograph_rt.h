/*
 * neurograph_rt.h — C ABI for in-process NeuroGraph inference.
 *
 * P1-08. Consumed by customers writing in C, C++, Rust, Go, C#, Java
 * (JNI), Python (ctypes), or anything else with FFI. The contract here
 * is the one surface that works everywhere; shelling out to
 * `predict.exe` (P1-07) has the same semantics but higher latency.
 *
 * Contract rules:
 *   - Pure C89. No _Bool, no stdint.h, no variadic, no C++ types in
 *     signatures. `int` is used for all sizes — matches the spec and
 *     avoids stdint fuss on ancient compilers. If a batch needs more
 *     than 2^31 rows, the caller chunks.
 *   - All functions are `extern "C"` and tagged `NG_EXPORT` so they
 *     end up in the DLL's export table.
 *   - ng_model is opaque. Callers get a pointer from ng_model_load,
 *     pass it to predict, release it with ng_model_free. Never peek.
 *   - Predict entry points are safe to call from multiple threads on
 *     the same model pointer — the underlying `predict_row` is pure.
 *     Load/free are NOT thread-safe w.r.t. each other on the same
 *     pointer.
 *   - Errors: non-zero ng_status. The reason string is available via
 *     `ng_last_error()`, which returns thread-local storage so two
 *     threads racing on the same model don't scribble over each
 *     other's last message. The pointer is valid until the next
 *     call on the same thread; copy it if you need to persist.
 *
 * Memory semantics:
 *   - Strings returned by `ng_model_class_label` / `ng_model_version`
 *     point into the model's internal storage; do not free them,
 *     do not use them after `ng_model_free`.
 *   - `ng_last_error()`'s return is thread-local storage the library
 *     owns; same rules — don't free, don't read from another thread.
 *   - All `out_*` buffers the caller supplies are filled by the
 *     library; the caller owns the allocation.
 */

#ifndef NEUROGRAPH_RT_H
#define NEUROGRAPH_RT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Export / import decoration. Callers consuming the library from a
 * different DLL on Windows should define NG_IMPORT to get the matching
 * __declspec(dllimport); otherwise symbols are treated as exports. */
#if defined(_WIN32) || defined(_WIN64)
#  if defined(NEUROGRAPH_RT_BUILDING)
#    define NG_EXPORT __declspec(dllexport)
#  elif defined(NEUROGRAPH_RT_STATIC)
#    define NG_EXPORT
#  else
#    define NG_EXPORT __declspec(dllimport)
#  endif
#else
#  if defined(NEUROGRAPH_RT_BUILDING)
#    define NG_EXPORT __attribute__((visibility("default")))
#  else
#    define NG_EXPORT
#  endif
#endif

/* Opaque handle. Callers never dereference this. */
typedef struct ng_model ng_model;

/* Status codes. Keep numeric values stable — customers switch on them. */
typedef enum ng_status_e {
    NG_OK              = 0,
    NG_ERR_IO          = 1,   /* file not found, permission denied, ... */
    NG_ERR_CORRUPT     = 2,   /* bundle exists but is malformed */
    NG_ERR_BAD_ARG     = 3,   /* caller passed NULL / wrong sizes */
    NG_ERR_INTERNAL    = 4    /* unexpected — should not happen, report */
} ng_status;

/* ---------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------- */

/* Load a bundle directory produced by the NeuroGraph orchestrator
 * (pipeline.ngpipe, model.ngm, manifest.json). On success *out is
 * filled with a non-NULL handle the caller must eventually release
 * with ng_model_free. On failure the handle is set to NULL and
 * ng_last_error() carries a message. */
NG_EXPORT ng_status ng_model_load(const char* bundle_dir, ng_model** out);

/* Release a handle. Safe on NULL — a no-op. Not thread-safe w.r.t.
 * concurrent predict calls on the same pointer; the caller is
 * responsible for quiescing predictions before free. */
NG_EXPORT void ng_model_free(ng_model* m);

/* ---------------------------------------------------------------------
 * Metadata
 *
 * All accessors are const. Return 0 / NULL when the model pointer
 * itself is NULL — callers never need to null-check the result unless
 * the handle was possibly bad.
 * ------------------------------------------------------------------- */

/* Number of features the model expects after preprocessing (i.e. the
 * width of the vector you get from `apply_pipeline`). */
NG_EXPORT int ng_model_num_features(const ng_model* m);

/* Number of classes for a classifier, 0 for regression models.
 * Use this to size `out_probs` in predict calls. */
NG_EXPORT int ng_model_num_classes(const ng_model* m);

/* Display label for class `idx`. Returns an empty string if no label
 * exists (some bundles use numeric IDs only). `idx` out of range
 * returns NULL. */
NG_EXPORT const char* ng_model_class_label(const ng_model* m, int idx);

/* NeuroGraph version recorded in the bundle manifest, or "" when the
 * manifest is missing. Useful for compatibility diagnostics. */
NG_EXPORT const char* ng_model_version(const ng_model* m);

/* ---------------------------------------------------------------------
 * Inference
 *
 * Two entry points:
 *
 *   ng_predict_json  — raw row as JSON text (column_name → value).
 *                      Preprocess pipeline is applied internally.
 *   ng_predict_vec   — caller has already transformed the row
 *                      (e.g. reusing one apply across many inferences).
 *
 * Output contract:
 *   - Classification: fill `out_probs[0..num_classes-1]` with the
 *     softmax distribution; write the argmax class id into `*out_pred`.
 *   - Regression:     write the scalar prediction into `out_probs[0]`;
 *     `out_pred` is untouched (may be NULL).
 *
 * Either `out_probs` or `out_pred` may be NULL if the caller doesn't
 * want that piece — the library writes only what's requested. In
 * particular `out_probs` sized 1 works for both cls (argmax prob only)
 * and reg (the value).
 *
 * Thread safety: these functions read only immutable model state and
 * use stack-allocated scratch, so concurrent calls on the same model
 * pointer are safe. Separate `out_*` buffers per thread.
 * ------------------------------------------------------------------- */

NG_EXPORT ng_status ng_predict_json(const ng_model* m,
                                    const char*     json_row,
                                    double*         out_probs,
                                    int*            out_pred);

NG_EXPORT ng_status ng_predict_vec(const ng_model* m,
                                   const double*   features,
                                   int             feature_count,
                                   double*         out_probs,
                                   int*            out_pred);

/* Batched version. `features_flat` is row-major with
 * `batch_size * feature_count` doubles. `out_probs_flat` is row-major
 * with `batch_size * num_classes` doubles (or `batch_size * 1` for
 * regression). `out_preds` has `batch_size` ints (may be NULL for
 * regression). Stops on the first row that errors and returns that
 * row's status; successfully-processed rows before it are filled. */
NG_EXPORT ng_status ng_predict_batch_vec(const ng_model* m,
                                         const double*   features_flat,
                                         int             batch_size,
                                         int             feature_count,
                                         double*         out_probs_flat,
                                         int*            out_preds);

/* ---------------------------------------------------------------------
 * Diagnostics
 * ------------------------------------------------------------------- */

/* Thread-local last-error string. Returns "" when no error has been
 * recorded on this thread. Never NULL. */
NG_EXPORT const char* ng_last_error(void);

/* Semantic version of the loaded library. Stable across schema-
 * compatible builds — read at runtime if you need to feature-gate.
 * Returns a statically-allocated string; never NULL. */
NG_EXPORT const char* ng_rt_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEUROGRAPH_RT_H */
