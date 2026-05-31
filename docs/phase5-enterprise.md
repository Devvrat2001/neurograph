# Phase 5 — Enterprise & Scale: Detailed Engineering Specs

> **Goal of Phase 5:** Turn the product from "works for one seat on one
> workstation" into "works for teams, on big data, in regulated environments,
> optionally in the cloud." Scope is **ongoing and demand-driven** —
> prioritize tickets in whatever order customers are willing to pay for them.
>
> **Duration estimate:** 12+ months of steady engineering work. Ticket
> sizes range from S (days) to XL (multiple months).
>
> **Entry criteria:** Phases 1–3 in production with paying customers.
> Phase 4 mostly done (text + timeseries). Real feedback driving which
> Phase 5 tickets to do in what order.
>
> **Exit criteria:** there is no "exit" for Phase 5 — this phase defines
> the steady-state engineering work for as long as the product is alive.
> Cut scope into named releases (v2.0, v3.0) aligned with Phase 5 tickets
> that customers are asking for.

---

## How to use this document

Each ticket is self-contained like the prior phase docs. Unlike Phases
1–4, **do not try to do these in order** — customer demand dictates
sequencing. A ticket stays as "ready to pick up" until a paying customer
signals need.

### Track assignment

Tracks are orthogonal here, not sequential. Each is its own product line:

| Track | Tickets | Owner archetype |
|-------|---------|-----------------|
| **A — Scale** | P5-01, P5-10 | Systems / CUDA engineer |
| **B — Trust & Explainability** | P5-02, P5-03, P5-04 | ML engineer with stats background |
| **C — AutoML depth** | P5-05 | ML engineer with Bayesian optimization experience |
| **D — Cloud SaaS** | P5-06 | Full-stack + devops — likely a small sub-team |
| **E — Governance** | P5-07, P5-08, P5-09 | Backend + security engineer |

---

## Cross-cutting conventions (new in Phase 5)

- **Multi-tenancy awareness:** any data-touching feature must carry an
  implicit tenant/user context. Introduce a `SecurityContext` struct early
  (P5-08 or P5-09) and thread it through.
- **Backwards compatibility:** once we have customers, bundles / API
  responses get a `schema_version` and must deserialize old versions.
- **Observability:** every new long-running operation emits a
  `operation_id` and progress events via the same infrastructure as
  Phase 3's REST server logs.
- **Feature flags:** new Phase 5 features default **off** behind env vars
  or config flags. Accept the complexity — this is how enterprise ships.
- **Legal:** explainability + governance features will trigger questions
  from compliance teams. Document each feature's data handling in a
  `docs/compliance/<feature>.md` as it lands.

---

# Phase 5 Tickets

## P5-01 — Multi-GPU Data-Parallel Training

- **Size:** L  •  **Deps:** Phase 2 GPU path stable  •  **Track:** A

### Goal
Train one model across N GPUs on one machine via synchronous
data-parallelism (each GPU sees a shard of the batch; gradients averaged via
all-reduce; identical weights after each step).

### Why it matters
Single-GPU training caps out around a few hours of wall-clock. Customers
with bigger datasets start hitting that wall. 4× GPU speedup = 4× faster
iteration = 4× more experiments = better models.

### Files
- `include/neurograph/gpu/multi_gpu_context.hpp` — NEW
- `src/gpu/multi_gpu_context.cpp` — NEW
- `src/gpu/nccl_allreduce.cpp` — NEW (gate behind `NEUROGRAPH_USE_NCCL` CMake flag)
- `include/neurograph/training/training_engine.hpp` — modify: opt-in dispatch
- `tests/integration/test_multi_gpu.cpp` — NEW (skipped if < 2 GPUs)

### API
```cpp
namespace neurograph::gpu {

struct MultiGpuConfig {
    std::vector<int> device_ids;   // e.g. {0,1,2,3}
    bool use_nccl = true;          // fallback: peer-to-peer cudaMemcpyAsync
    int  gradient_accumulation_steps = 1;
};

class MultiGpuContext {
public:
    explicit MultiGpuContext(const MultiGpuConfig& cfg);
    ~MultiGpuContext();

    int world_size() const;

    // Broadcast initial weights from rank 0 to all.
    void broadcast_weights(core::Network& net);
    // All-reduce gradients (sum then divide by world_size).
    void allreduce_gradients(std::vector<float>& grad_buf);
};

} // namespace
```

### Algorithm notes
- **NCCL** when available — lightweight, GPU-aware. Ring all-reduce.
- **Fallback** over `cudaMemcpyPeerAsync` for single-node setups without NCCL
  — simple tree reduction. Slower but works.
- **Batch sharding:** each rank gets `batch / world_size` samples. If batch
  not divisible, rank 0 takes the remainder.
- **Gradient accumulation:** support in the training loop so large effective
  batch sizes are reachable even on modest memory.
- **Out of scope for this ticket:** distributed multi-node (MPI/torchrun
  style). Customers who need it go through P5-06 cloud SaaS instead.

### Acceptance criteria
- [ ] 2-GPU training on MNIST converges to same test accuracy as single-GPU
      (within RNG noise) in half the wall-clock.
- [ ] Scaling efficiency ≥ 75% at 4 GPUs on CIFAR-10 (throughput vs 1 GPU).
- [ ] Weights bit-identical across ranks after each step.
- [ ] Clean error when device_ids invalid (GPU not present, etc).

### Test plan
- Integration: 2-GPU CIFAR-10 run; compare metrics to 1-GPU baseline.
- Perf: throughput scaling chart at 1, 2, 4 GPUs.

---

## P5-02 — SHAP-Style Feature Attribution

- **Size:** L  •  **Deps:** P1-14  •  **Track:** B

### Goal
Compute SHAP values (approximate, via Integrated Gradients or Kernel SHAP)
for individual predictions. Expose via a CLI tool + C-ABI + Python binding.
Produce a visual "why did the model predict this" explanation in the
workbench.

### Why it matters
Regulated industries (finance, healthcare) require per-prediction
explanations. This is the #1 feature that blocks enterprise sales in these
verticals.

### Files
- `include/neurograph/explain/shap.hpp` — NEW
- `src/explain/integrated_gradients.cpp` — NEW
- `src/explain/kernel_shap.cpp` — NEW (slower, more accurate; for tabular)
- `include/neurograph/capi/neurograph_rt.h` — modify: add `ng_explain_*`
- `tests/unit/test_shap.cpp` — NEW

### API
```cpp
namespace neurograph::explain {

struct Attribution {
    std::vector<double> per_feature;  // same length as network's input
    std::vector<std::string> feature_names;
    double                    baseline_prediction;
    double                    actual_prediction;
    std::string               method;  // "integrated_gradients" | "kernel_shap"
};

struct ExplainOptions {
    int  ig_steps        = 32;    // Integrated Gradients
    int  ks_samples      = 200;   // Kernel SHAP
    std::string method   = "auto";  // auto: IG for neural, KS for tabular
};

Attribution explain_prediction(const core::Network& net,
                               const std::vector<double>& input,
                               const std::vector<double>& baseline,
                               const ExplainOptions& opts = {});

// Global explanation: mean(|shap|) across a dataset.
std::vector<std::pair<std::string, double>>
global_feature_importance(const core::Network& net,
                          const std::vector<std::vector<double>>& samples,
                          const std::vector<std::string>& feature_names);

} // namespace
```

### Algorithm notes
- **Integrated Gradients:** exact for differentiable models; linear in
  number of steps (32 is a good default). Works end-to-end with our backward
  pass.
- **Kernel SHAP:** slower but handles any black box. Needed for edge cases
  where IG doesn't fit (e.g. post-quantization models).
- **Baseline selection:** default to zero vector; allow user to pass training-set
  mean for more meaningful attributions.

### Acceptance criteria
- [ ] IG attributions sum to `f(x) - f(baseline)` (completeness property).
- [ ] Attributions symmetric for symmetric inputs (sanity check on trivial
      linear model).
- [ ] Matches SHAP library output within 5% on titanic test samples.
- [ ] 100 ms per explanation on titanic-sized model.

### Test plan
- Unit: completeness check on a linear model (exact).
- Unit: SHAP golden parity.
- Integration: workbench "Explain prediction" button renders bar chart.

---

## P5-03 — Permutation Importance

- **Size:** S  •  **Deps:** P1-14  •  **Track:** B

### Goal
For each feature, measure the drop in a target metric when that feature is
randomly permuted across the validation set. Report as feature importance.

### Why it matters
Simpler, model-agnostic cousin of SHAP. Cheap to compute. Customers who
don't need per-prediction explanations often find this is enough.

### Files
- `include/neurograph/explain/permutation_importance.hpp` — NEW
- `src/explain/permutation_importance.cpp` — NEW
- `tests/unit/test_permutation_importance.cpp` — NEW

### API
```cpp
namespace neurograph::explain {

struct PermutationImportance {
    std::vector<std::pair<std::string, double>> importances;  // sorted desc
    double baseline_metric;
    std::string metric_name;
};

PermutationImportance compute_permutation_importance(
    const core::Network& net,
    const std::vector<std::vector<double>>& X_val,
    const std::vector<std::vector<double>>& y_val,
    const std::vector<std::string>& feature_names,
    int n_repeats = 5,
    uint64_t seed = 42);

} // namespace
```

### Acceptance criteria
- [ ] Deterministic given seed.
- [ ] On synthetic data where feature k is pure noise, importance of k ≈ 0.
- [ ] Scales linearly in (n_features × n_repeats).

### Test plan
- Unit: synthetic "one real feature, many junk features" → real feature ranks
  first.

---

## P5-04 — Attention-Map Visualization

- **Size:** M  •  **Deps:** P2-07, P4-04  •  **Track:** B

### Goal
Render attention / gradient maps overlaid on inputs:
- **CNN:** Grad-CAM on the last conv layer — heatmap over the input image.
- **Transformer:** per-head attention weights between tokens — heatmap
  grid.

Exposed in the workbench result panel and via C-ABI for embedded use.

### Why it matters
Customers who work with images/text expect to see these. They're the default
way people develop intuition for what the model is attending to.

### Files
- `include/neurograph/explain/gradcam.hpp` — NEW
- `include/neurograph/explain/attention_viz.hpp` — NEW
- `src/explain/gradcam.cpp` — NEW
- `src/explain/attention_viz.cpp` — NEW
- `tests/unit/test_gradcam.cpp` — NEW

### API
```cpp
namespace neurograph::explain {

struct GradCam {
    tensor::Tensor4D heatmap;   // [1, 1, H, W] — normalized 0..1
    int              predicted_class;
};

GradCam grad_cam(const core::Network& net,
                 const tensor::Tensor4D& input,
                 int target_class = -1 /* argmax */);

struct AttentionMap {
    std::vector<std::vector<std::vector<double>>> weights; // [block, head, T*T]
    std::vector<std::string> tokens;
};

AttentionMap attention_map(const core::Network& net,
                           const std::vector<int>& token_ids);

} // namespace
```

### Acceptance criteria
- [ ] Grad-CAM heatmap sum normalized to 1.0.
- [ ] On a CIFAR cat image, attention concentrates on the cat (manual
      inspection gate — spot-check in PR).
- [ ] Attention map sums to 1 per-row (softmax property preserved).

### Test plan
- Unit: sum-to-one checks.
- Manual: spot-check visualizations on a fixture for each modality.

---

## P5-05 — Bayesian Hyperparameter Search

- **Size:** L  •  **Deps:** P1-14  •  **Track:** C

### Goal
Sit on top of `AutoPipeline` and search over hyperparameters (LR, batch size,
dropout, architecture depth) with a Bayesian surrogate (Gaussian process
or Tree Parzen Estimator). Beats the current one-shot auto-config on hard
datasets.

### Why it matters
For customers willing to spend compute, HPO typically finds models that beat
default auto-config by 2–5%. This is the "give me a better model and I'll
wait" knob.

### Files
- `include/neurograph/hpo/search.hpp` — NEW
- `src/hpo/bayesian.cpp` — NEW
- `src/hpo/tpe.cpp` — NEW
- `include/neurograph/pipeline/auto_pipeline.hpp` — modify: accept
  `HpoOptions`
- `tests/integration/test_hpo.cpp` — NEW

### API
```cpp
namespace neurograph::hpo {

enum class ParamType { Real, Integer, Categorical };

struct ParamSpec {
    std::string name;
    ParamType   type;
    double      low = 0, high = 1;       // Real / Integer
    std::vector<std::string> choices;    // Categorical
    bool        log_scale = false;       // Real only
};

struct SearchConfig {
    std::vector<ParamSpec> params;
    std::string            method = "tpe";   // "tpe" | "gp" | "random"
    int                    n_trials = 50;
    int                    seed = 42;
    int                    timeout_seconds = 3600;
    int                    parallel = 1;     // run N trials concurrently
};

struct TrialResult {
    std::unordered_map<std::string, std::variant<double,int,std::string>> params;
    double                 primary_metric;
    int                    trial_id;
    std::chrono::seconds   duration;
    bool                   pruned = false;
};

using ObjectiveFn = std::function<double(const std::unordered_map<std::string,
                                         std::variant<double,int,std::string>>&)>;

struct SearchResult {
    TrialResult                best;
    std::vector<TrialResult>   all;
    std::string                summary;
};

SearchResult run_search(const SearchConfig& cfg, ObjectiveFn objective);

} // namespace
```

### Acceptance criteria
- [ ] TPE converges to within 2% of the best trial found by exhaustive grid
      search on a 3-parameter fixture within 30 trials.
- [ ] Supports early-stopping ("pruning") of bad trials — median-stopping rule.
- [ ] Parallel trials produce same distribution as serial (tested with seeded
      RNG).
- [ ] Time budget respected.

### Test plan
- Integration: titanic HPO run, best trial beats default AutoConfig.
- Unit: acquisition function correctness on synthetic GP target.

---

## P5-06 — Cloud-Hosted SaaS Mode

- **Size:** XL  •  **Deps:** Phase 3 deployment infra  •  **Track:** D

### Goal
A web application where users upload a dataset, we spin up a cloud training
job, and deliver the trained model + metrics. Self-serve, credit-card billing.

### Why it matters
Removes the single biggest adoption friction — local GPU hardware. Customers
who would never spec a workstation will happily pay per-training-run.

### Scope (sub-tickets; this is a whole product line)
- P5-06-A: Web frontend (React or similar — not C++). Upload dataset,
  track job, download bundle.
- P5-06-B: Orchestration backend. Enqueue jobs, schedule on GPU workers,
  report status.
- P5-06-C: Worker runtime. Pulls a job, runs AutoPipeline, uploads bundle.
  Reuses the Docker image from P3-04.
- P5-06-D: Object storage integration (S3 / GCS / Azure Blob) for datasets,
  models, logs.
- P5-06-E: Auth & billing (Stripe). Free tier: N training-minutes/month.
- P5-06-F: Quotas & fair-share scheduler. Prevent one user from hogging.
- P5-06-G: Data residency options (EU-only, US-only buckets).

### Files
New top-level `cloud/` tree:
```
cloud/
├── frontend/           # React/Next.js — not in this repo's purview to detail
├── api/                # Go / Python service — route requests to workers
├── worker/             # Thin C++ wrapper around neurograph_server + lifecycle
├── infra/              # Terraform / helm charts
└── docs/
```

### Acceptance criteria (phase-level)
- [ ] User signs up, uploads titanic.csv via web UI, receives trained bundle
      within 5 minutes.
- [ ] 99.9% uptime over a rolling 90-day window.
- [ ] SOC2 Type I compliant within 6 months of launch.
- [ ] Support tickets tracked; < 24 h first-response median.

### Test plan
- Unlike other tickets: full QA + security audit before each release.
- Load test: 100 concurrent training jobs without degradation.

---

## P5-07 — Model Registry / MLflow-Style Lineage

- **Size:** M  •  **Deps:** P3-01  •  **Track:** E

### Goal
A local (and optionally remote) registry that tracks every trained model
produced by the product: metadata, metrics, parent dataset hash, training
config, user, timestamp. Query via CLI, REST, or workbench panel.

### Why it matters
Teams lose track of which model is in production and how it was trained.
A registry makes model management a one-command answer ("which model is
serving prod?").

### Files
- `include/neurograph/registry/model_registry.hpp` — NEW
- `src/registry/sqlite_backend.cpp` — NEW
- `src/registry/remote_backend.cpp` — NEW (HTTP API to Phase 3 server)
- `vcpkg.json` — add `sqlite3`
- `tests/unit/test_registry.cpp` — NEW

### API
```cpp
namespace neurograph::registry {

struct ModelEntry {
    std::string id;                       // uuid
    std::string name;
    std::string version;
    std::chrono::system_clock::time_point created_at;
    std::string created_by;               // from SecurityContext
    std::string dataset_sha256;
    std::string model_sha256;
    std::unordered_map<std::string, double> metrics;
    nlohmann::json config;
    std::string stage;                    // "dev", "staging", "production"
};

class ModelRegistry {
public:
    virtual ~ModelRegistry() = default;
    virtual std::string register_model(const ModelEntry& e) = 0;
    virtual std::optional<ModelEntry> get(const std::string& id) = 0;
    virtual std::vector<ModelEntry> list(std::string_view name = "") = 0;
    virtual bool promote(const std::string& id, const std::string& stage) = 0;
    virtual bool delete_entry(const std::string& id) = 0;
};

// Local SQLite implementation.
std::unique_ptr<ModelRegistry> make_sqlite_registry(const std::filesystem::path& db);

// Remote HTTP client implementation.
std::unique_ptr<ModelRegistry> make_remote_registry(const std::string& base_url,
                                                     const std::string& api_key);

} // namespace
```

### Acceptance criteria
- [ ] Registering, fetching, listing, promoting, deleting all work.
- [ ] Query "which model is in `production` for name=`churn-v2`?" returns
      one answer atomically.
- [ ] Audit trail: stage transitions append to a history table, never
      overwrite.

### Test plan
- Unit: SQLite CRUD.
- Integration: register 10 models, promote one, list filtered.

---

## P5-08 — Audit Log & Compliance

- **Size:** M  •  **Deps:** P5-07  •  **Track:** E

### Goal
Immutable audit log of every data-plane action: dataset upload, model train,
model promote, model download, prediction request. Exportable as signed
JSONL for compliance audits.

### Why it matters
SOC2 / HIPAA / GDPR require tamper-evident audit logs. Without this,
enterprise sales stall at the security review.

### Files
- `include/neurograph/audit/audit_log.hpp` — NEW
- `src/audit/hash_chain.cpp` — NEW
- `src/audit/audit_log.cpp` — NEW
- `tests/unit/test_audit_log.cpp` — NEW

### API
```cpp
namespace neurograph::audit {

struct AuditEvent {
    std::string id;
    std::chrono::system_clock::time_point timestamp;
    std::string actor;        // user id from SecurityContext
    std::string action;       // "model.train", "model.promote", ...
    std::string object;       // target entity id
    nlohmann::json details;
    std::string prev_hash;    // hash chain for tamper evidence
    std::string hash;
};

class AuditLog {
public:
    virtual ~AuditLog() = default;
    virtual void append(const AuditEvent& e) = 0;
    virtual std::vector<AuditEvent> query(std::string_view actor,
                                          std::chrono::system_clock::time_point since) = 0;
    virtual bool verify_chain(std::chrono::system_clock::time_point since) = 0;
    virtual std::string export_signed_jsonl(const std::filesystem::path& out) = 0;
};

} // namespace
```

### Algorithm notes
- **Hash chain:** each event includes `prev_hash = SHA256(prev)`. Tampering
  with any event breaks the chain from that point forward.
- **Signing:** final export signed with an Ed25519 key (reuse infra from
  license system P1-11). Verifier needs only the public key.
- **Storage:** SQLite with WAL mode. No DELETE permission in the app; log
  is append-only. Retention/deletion handled by a separate admin tool.
- **Data subject rights (GDPR):** deletion endpoint marks the event's
  `details` as REDACTED but keeps the hash chain intact.

### Acceptance criteria
- [ ] Hash chain verify passes on a clean log.
- [ ] Tampering any byte of any event breaks verify on subsequent events.
- [ ] Signed export re-verifies with the public key.
- [ ] Append-only: no API path allows overwriting events.

### Test plan
- Unit: append, verify, tamper, verify-fail.
- Unit: export + round-trip verify.

---

## P5-09 — RBAC for Teams

- **Size:** M  •  **Deps:** P3-07, P5-07  •  **Track:** E

### Goal
Roles + permissions model covering: viewer, trainer, reviewer, admin.
Scoped per project / team. Applied at REST server (P3-01) and registry
(P5-07).

### Why it matters
Teams > 3 people need role separation. "Our ML engineer can train; our
compliance officer can promote; our analyst can only view."

### Files
- `include/neurograph/auth/rbac.hpp` — NEW
- `src/auth/rbac.cpp` — NEW
- `include/neurograph/server/security_context.hpp` — NEW
- `tests/unit/test_rbac.cpp` — NEW

### API
```cpp
namespace neurograph::auth {

enum class Role { Viewer, Trainer, Reviewer, Admin };

struct SecurityContext {
    std::string user_id;
    std::string team_id;
    std::vector<Role> roles;
};

enum class Permission {
    DatasetRead, DatasetWrite,
    ModelRead, ModelTrain, ModelPromote, ModelDelete,
    AuditRead, AdminWrite,
};

bool has_permission(const SecurityContext& ctx, Permission p,
                    std::string_view object_team_id = {});

class RbacPolicy {
public:
    virtual ~RbacPolicy() = default;
    virtual std::vector<Role> roles_for(const std::string& user,
                                        const std::string& team) const = 0;
    virtual void grant(const std::string& user, const std::string& team,
                       Role r) = 0;
    virtual void revoke(const std::string& user, const std::string& team,
                        Role r) = 0;
};

} // namespace
```

### Acceptance criteria
- [ ] Viewer denied on ModelTrain / ModelPromote / ModelDelete.
- [ ] Trainer allowed on ModelTrain but denied on ModelPromote.
- [ ] Cross-team actions blocked unless user is Admin in both teams.
- [ ] Policy changes audit-logged (P5-08).

### Test plan
- Unit: matrix test of (role × permission).
- Integration: REST endpoints deny wrong-role requests with 403.

---

## P5-10 — Quantization (int8) for Edge

- **Size:** M  •  **Deps:** Phase 2 done, Phase 3 ONNX path done  •  **Track:** A

### Goal
Post-training quantization of Phase 1/2 models to int8. Calibrated via a
small activation-range calibration pass on a user-provided subset of their
training data. Shrinks model size 4× and speeds up CPU inference 2–3×.

### Why it matters
Edge deployment (IoT, mobile, low-power servers) needs int8. This also
helps customers who want the fastest possible on-prem CPU inference.

### Files
- `include/neurograph/quantize/quantize.hpp` — NEW
- `src/quantize/calibrator.cpp` — NEW
- `src/quantize/int8_engine.cpp` — NEW
- `tests/unit/test_quantize.cpp` — NEW

### API
```cpp
namespace neurograph::quantize {

struct QuantizeOptions {
    enum class Scheme { Symmetric, Asymmetric };
    Scheme                      scheme          = Scheme::Symmetric;
    int                         calibration_samples = 128;
    std::vector<std::vector<double>> calibration_data;
    bool                        per_channel = true;  // for conv weights
};

struct QuantizedModel {
    std::filesystem::path bundle_path;
    double                size_reduction_ratio;
    double                accuracy_loss_pct;       // vs float baseline
};

tl::expected<QuantizedModel, std::string>
quantize_model(const runtime::ModelBundle& float_bundle,
               const QuantizeOptions& opts,
               const std::filesystem::path& out_dir);

} // namespace
```

### Acceptance criteria
- [ ] On titanic MLP: int8 model ≥ 95% of float accuracy; 4× smaller.
- [ ] On MNIST CNN: int8 model ≥ 98% of float accuracy; 3.5× CPU inference
      speedup on a modern laptop.
- [ ] ONNX export of quantized model loads in onnxruntime and produces same
      predictions.
- [ ] Runs on CPUs without AVX (portable int8 kernels, not AVX-VNNI only).

### Test plan
- Unit: per-tensor and per-channel quantize/dequantize round-trip.
- Integration: titanic + MNIST accuracy + speed gates.

---

# Phase 5 cadence (non-linear)

Phase 5 ships as a series of **named releases**. Example sequencing (adjust
to customer demand):

| Release | Contents | Driver |
|---------|----------|--------|
| v2.0 | P5-01 (Multi-GPU) + P5-03 (Permutation importance) | Ship once first customer hits single-GPU wall |
| v2.1 | P5-07 (Registry) + P5-08 (Audit log) | Triggered by first enterprise security review |
| v2.2 | P5-02 (SHAP) + P5-04 (Attention viz) | Triggered by regulated industry customer |
| v3.0 | P5-06 (Cloud SaaS MVP) | Strategic bet; large investment |
| v3.1 | P5-05 (HPO) + P5-10 (Quantization) | Performance-oriented customers |
| v3.2 | P5-09 (RBAC) + enhancements to P5-07 | Team-tier pricing launch |

# Definition of Done (per release, not per phase)

Phase 5 has no global DoD. Each release has its own gate:

- [ ] Tickets in scope all merged with green CI.
- [ ] Performance budgets met (if performance ticket).
- [ ] Security review passed (if governance/security ticket).
- [ ] Docs updated for the new surface area.
- [ ] Existing customers upgraded with zero data loss.
- [ ] At least one paying customer committed to using the new feature.

---

# Ongoing engineering standards

Beyond ticket work, Phase 5 requires continuous investment in:

- **Dependency hygiene.** Monthly `vcpkg` / Python dep updates. CVE
  response within 7 days for High, 30 days for Medium.
- **Performance regression guard.** Every PR runs `tests/perf/` against
  a baseline; 5% regression fails the PR.
- **Customer support runbooks.** Documented in `docs/runbooks/`. Every
  new feature adds a "How to diagnose and fix when this breaks" doc
  before GA.
- **Compatibility matrix.** Supported Python / CUDA / OS versions
  published and tested in CI. Drop support only with one minor-version
  deprecation notice.

---

*End of phase docs. Back to [`ROADMAP.md`](ROADMAP.md) for the overall
product picture.*
