# NeuroGraph — Commercial Product Roadmap

> **Source of truth** for the plan to turn NeuroGraph into a commercial
> "upload-data → get-trained-model" product. Phase 1 is specified in full
> engineering detail; later phases are broken down to ticket-level scope and
> will be refined as we approach them.

---

## 1. Product vision

A desktop (later cloud) application where a **non-ML customer** can:

1. Drop a dataset (folder of files / spreadsheet) onto the app.
2. Get an automatic quality report and task diagnosis.
3. Train a neural network that is automatically designed and configured
   for their data.
4. Get honest metrics, a baseline comparison, and a learning-curve read on
   whether more data would help.
5. Export the trained model as a **standalone artifact** (.exe, .dll/.so,
   ONNX, REST endpoint, Python wheel) their own application can use.

The trained model's task is **fully defined by the shape and organization of
the training data** — folder structure, column choices, target types. The
user never writes a config file or picks an architecture.

---

## 2. Phases at a glance

| Phase | Name | Duration | Ship-ready outcome |
|---|---|---|---|
| **1** | Tabular MVP | 6–8 weeks | First sellable product: tabular classification + regression, wizard UI, data validation, baseline comparison, standalone export, installer |
| **2** | Images | 6–8 weeks | Add image classification/regression via CNN; GPU training; PNG/JPEG/archive support |
| **3** | Deployment | 4–6 weeks | REST server, ONNX export, Python wheel, Docker image |
| **4** | Text & sequences | 8–10 weeks | Transformer-lite for text classification; time-series forecasting |
| **5** | Enterprise & scale | ongoing | Distributed training, explainability, hyperparameter search, cloud-hosted SaaS |

Phase 1 alone produces a sellable product.

---

## 3. System architecture (target end state)

```
┌───────────────────────────────────────────────────────────┐
│                    UI LAYER                               │
│   Wizard workbench  |  CLI  |  REST server  |  Python API │
└────────────────┬──────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────┐
│               ORCHESTRATION LAYER                         │
│   AutoPipeline:                                           │
│     Dataset → DataQualityReport → TaskDetector            │
│       → PreprocessingPipeline → ArchitectureSelector      │
│       → AutoConfig → Trainer → Evaluator → ModelExporter  │
└────────────────┬──────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────┐
│                 CORE ENGINE                               │
│   Network (MLP / CNN / Transformer) | Neuron | Synapse    │
│   Self-organizing topology | Optimizer | Loss | Backprop  │
└────────────────┬──────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────┐
│              RUNTIME BACKENDS                             │
│              CPU     |     CUDA     |     ONNX runtime    │
└───────────────────────────────────────────────────────────┘
```

New top-level `neurograph::pipeline` namespace will hold the orchestration
layer. Core engine stays where it is today.

---

## 4. Data contracts (used across all phases)

These types are the **backbone of every phase**. Define them in Phase 1 and
do not change them lightly — downstream code depends on their shape.

### 4.1 `TaskKind`

```cpp
// include/neurograph/pipeline/task.hpp
namespace neurograph::pipeline {

enum class TaskKind {
    BinaryClassification,
    MulticlassClassification,
    MultilabelClassification,
    Regression,
    MultiOutputRegression,
    // Phase 4+:
    SequenceClassification,
    SequenceToSequence,
    TimeSeriesForecast,
    Clustering,
    AnomalyDetection,
    Unknown
};

enum class InputModality {
    Tabular,
    Image,
    Text,
    Sequence,
    Audio,
    Mixed
};

struct TaskSpec {
    TaskKind kind = TaskKind::Unknown;
    InputModality modality = InputModality::Tabular;
    int num_classes = 0;             // 0 for regression
    int output_dim = 1;
    std::vector<std::string> class_labels;  // human-readable
    std::vector<std::string> feature_names; // for explainability
};

} // namespace
```

### 4.2 `DatasetHandle`

A uniform handle over any loaded dataset. Replaces the direct `vector<Sample>`
passing we do today.

```cpp
// include/neurograph/pipeline/dataset.hpp
struct DatasetHandle {
    TaskSpec spec;
    std::vector<training::Sample> samples;   // existing type
    std::vector<std::string> column_names;   // for tabular
    ColumnSchema schema;                     // per-column dtype + stats
    std::optional<ImageMetadata> image_meta; // when modality == Image
    std::string source_uri;                  // for reproducibility
    std::string data_hash;                   // sha256 of inputs + targets
};
```

### 4.3 `DataQualityReport`

See P1-01 below.

### 4.4 `TrainingResult`

```cpp
struct TrainingResult {
    TaskSpec task;
    EvaluationMetrics model_metrics;     // NN
    EvaluationMetrics baseline_metrics;  // trivial baseline
    LearningCurve learning_curve;
    std::vector<EpochResult> history;
    std::string model_snapshot_path;
    std::string manifest_path;           // reproducibility
};
```

---

## 5. Phase 1 — Tabular MVP (6–8 weeks)

**Goal:** a sellable product for tabular classification + regression. Customer
downloads the app, drags a CSV, clicks through 4 wizard steps, exports a
working model.

### Phase 1 ticket list

| # | Ticket | Size | Depends on |
|---|---|---|---|
| P1-01 | Data quality scanner | M | — |
| P1-02 | Full preprocessing pipeline | L | P1-01 |
| P1-03 | Baseline model comparison | S | — |
| P1-04 | Classification metrics suite | M | — |
| P1-05 | Regression metrics suite | S | — |
| P1-06 | Learning curve analysis | M | P1-04, P1-05 |
| P1-07 | Standalone inference binary | L | — |
| P1-08 | C-ABI shared library | M | P1-07 |
| P1-09 | Wizard UI | L | P1-01, P1-04 |
| P1-10 | Windows installer | M | P1-07, P1-08 |
| P1-11 | License key system | M | — |
| P1-12 | Model card generator | M | P1-04, P1-05, P1-06 |
| P1-13 | Reproducibility manifest | S | — |
| P1-14 | `AutoPipeline` orchestrator | M | P1-01 through P1-06 |

**Sizing**: S = 1–2 days, M = 3–5 days, L = 1–2 weeks. Total ≈ 7 weeks for
one engineer, 4 weeks for two working in parallel on the non-dependent
tracks.

Parallel tracks:
- **Track A (core)**: P1-01 → P1-02 → P1-14
- **Track B (evaluation)**: P1-03, P1-04, P1-05, P1-06
- **Track C (export)**: P1-07 → P1-08 → P1-10
- **Track D (UI & polish)**: P1-09, P1-11, P1-12, P1-13

Full ticket specs for Phase 1 live in [`phase1-tabular-mvp.md`](phase1-tabular-mvp.md).

### Phase 1 Definition of Done

- [ ] A fresh Windows install can classify Iris, predict Boston housing
      prices, detect churn on a real SaaS dataset, all without touching a
      single hyperparameter.
- [ ] Data quality report flags at least these six scenarios (see P1-01).
- [ ] Every training run produces: model metrics, baseline metrics, learning
      curve, confusion matrix (or residual plot), and a model card PDF.
- [ ] Exported `.exe` runs on a machine with no NeuroGraph installed and
      produces identical predictions to the training-time model (bit-for-bit
      or within 1e-6 tolerance).
- [ ] Installer signed & tested on Windows 10 and 11.
- [ ] All tickets have unit tests; CI runs them on every commit.

---

## 6. Phase 2 — Images (6–8 weeks)

### Phase 2 ticket list

| # | Ticket | Size | Notes |
|---|---|---|---|
| P2-01 | `ConvNeuron` type: 2D convolution with shared weights | L | Extend `neuron.hpp`; new synapse layout (kernel tensor) |
| P2-02 | Pooling layer (max, avg, global avg) | M | New `PoolingBlock` in `network.hpp` |
| P2-03 | Image batching & tensor plumbing | M | Today samples are flat `vector<double>`; we need `Tensor4D` [N,C,H,W] |
| P2-04 | PNG + JPEG loader via `stb_image` | S | Replace BMP-only loader |
| P2-05 | ZIP archive extraction (`miniz`) | S | For large image datasets |
| P2-06 | Adaptive CNN architecture builder | L | Heuristic: `small_cnn` (≤28x28), `medium_cnn` (≤128x128), `large_cnn`. Self-organizing growth at filter level. |
| P2-07 | GPU backprop integration | L | Wire existing CUDA kernels into `backprop.cpp`. Currently the kernels in `cuda/kernels/` aren't called from training. |
| P2-08 | Architecture selector in `AutoPipeline` | M | Pick MLP vs CNN from `InputModality` and image dims |
| P2-09 | Expanded image augmentation | S | Cutout, MixUp, CutMix, color jitter |
| P2-10 | Object detection "light" (classification + bounding box regression) | L | **Optional, only if customer demand** |

### Key design decisions for Phase 2

**Convolution representation**: keep the `Neuron` abstraction — a conv neuron
has a 2D kernel as its weight tensor and a receptive field (x,y,w,h) in the
input feature map. This preserves the "self-organizing neurons" identity.
Self-organization can grow new filters (neurons) in the same layer.

**Tensor flow**: introduce `Tensor` as a lightweight view over contiguous
memory with shape metadata. Forward/backward passes for conv/pool layers
operate on tensors; dense layers keep their existing per-neuron path.

**GPU integration**: the blocker today is that backprop is entirely CPU.
Minimum path: port `compute_gradients_for_hidden_layer()` and the weight
update step to CUDA, dispatch batched. Use cuBLAS for dense matmul; keep
custom kernels for the self-org paths.

### Phase 2 Definition of Done
- [ ] CIFAR-10 reaches ≥70% test accuracy out of the box (auto-config only).
- [ ] MNIST training goes from ~30s (current CPU) to ≤3s on an RTX 3060.
- [ ] User can drop a folder of 50k JPEGs organized into class subfolders
      and get a trained classifier without manually decompressing.
- [ ] Exports still work (standalone .exe runs image inference).

---

## 7. Phase 3 — Deployment (4–6 weeks)

### Phase 3 ticket list

| # | Ticket | Size | Notes |
|---|---|---|---|
| P3-01 | REST inference server (`/predict`, `/health`, `/model/info`) | M | Reuse existing HTTP infra in `net/` if any; else `cpp-httplib` |
| P3-02 | ONNX export | L | Map our topology to ONNX ops; supports standard activations, Dense, Conv, Pool |
| P3-03 | Python bindings (`pybind11`) | M | `pip install neurograph`; expose `AutoPipeline.fit()`, `predict()` |
| P3-04 | Docker image (CPU + CUDA variants) | S | Multi-stage build, <500MB CPU image |
| P3-05 | gRPC server (optional, for high-throughput) | M | Using `grpc-cpp` |
| P3-06 | Model versioning & A/B routing in server | M | Multiple models loaded, route by header |
| P3-07 | Server auth (API keys, basic JWT) | S | Minimum viable; customers can put it behind their own gateway |
| P3-08 | Prometheus metrics endpoint | S | Request rate, latency, model version |

### Phase 3 Definition of Done
- [ ] `docker run neurograph/server -v model.ngmodel:/model` serves predictions
      via REST on port 8080.
- [ ] `pip install neurograph` works on Python 3.9–3.12 on Windows, Linux,
      macOS.
- [ ] ONNX exports verified against `onnxruntime` (same predictions ±1e-5).

---

## 8. Phase 4 — Text & sequences (8–10 weeks)

### Phase 4 ticket list

| # | Ticket | Size | Notes |
|---|---|---|---|
| P4-01 | BPE tokenizer (train + apply) | L | `sentencepiece`-style; integrate with existing `EmbeddingTable` |
| P4-02 | Text data loader (`.txt`, `.jsonl`, folder-of-files) | M | Labels via folder or separate CSV |
| P4-03 | Embedding + positional encoding pipeline | M | Slot into `backprop.cpp` forward pass |
| P4-04 | Multi-head self-attention layer | L | Core Transformer op |
| P4-05 | Transformer encoder block (attention + FFN + layer norm) | L | Stackable |
| P4-06 | Small-Transformer architecture (1–4 layers, 64–256 dim) | M | Sized for short texts, runs on one GPU |
| P4-07 | Sliding-window time-series dataset adapter | M | Turns a date-indexed CSV into supervised sequences |
| P4-08 | Seasonality + trend feature extraction | S | Fourier features, lag features |
| P4-09 | Sequence metrics (BLEU, ROUGE — if seq2seq) | M | Only if we do generation |
| P4-10 | Text augmentation (synonym swap, back-translation) | S | Optional |

### Phase 4 Definition of Done
- [ ] Sentiment classification on IMDB: ≥85% with auto-config.
- [ ] Airline-passenger time-series forecast: MAPE competitive with
      `statsmodels.SARIMAX`.

---

## 9. Phase 5 — Enterprise & scale (ongoing)

### Phase 5 ticket list (non-exhaustive; priorities set by customer demand)

| # | Ticket | Size |
|---|---|---|
| P5-01 | Multi-GPU data-parallel training | L |
| P5-02 | SHAP-style feature attribution | L |
| P5-03 | Permutation importance | S |
| P5-04 | Attention-map visualization (for CNN/Transformer) | M |
| P5-05 | Bayesian hyperparameter search over auto-config | L |
| P5-06 | Cloud-hosted SaaS mode (upload data, train remotely) | XL |
| P5-07 | Model registry / MLflow-style lineage | M |
| P5-08 | Audit log & compliance (GDPR data handling, model cards) | M |
| P5-09 | RBAC for teams | M |
| P5-10 | Quantization (int8) for edge deployment | M |

---

## 10. Cross-cutting engineering standards

### 10.1 Repo layout (target, after refactor)

```
neurograph/
├── include/neurograph/
│   ├── core/            # existing: neuron, synapse, network, topology
│   ├── training/        # existing: backprop, loss, optimizer, augmentation
│   ├── pipeline/        # NEW: auto_pipeline, data_quality, task_detector,
│   │                    #      preprocessor, architecture_selector, baseline,
│   │                    #      learning_curve, metrics (classification, regression)
│   ├── export/          # NEW: model_exporter, model_card, onnx_exporter, capi
│   ├── server/          # NEW (Phase 3): rest_server, grpc_server
│   ├── license/         # NEW (Phase 1): license_manager
│   ├── snapshot/        # existing
│   └── util/            # existing
├── src/                 # mirrors include/
├── python/              # NEW (Phase 3): pybind11 bindings
├── installer/           # NEW (Phase 1): windows/, linux/, macos/
├── docker/              # NEW (Phase 3)
├── docs/                # this file + phase docs + architecture
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/             # NEW: end-to-end with real sample datasets
```

### 10.2 Code style
- C++20, MSVC 2022 + GCC 11+ + Clang 14+ all supported.
- `.clang-format` committed at repo root — Google style with 4-space indent
  and 100-char line limit.
- Header/source split strict: no implementation in headers unless `inline`
  is load-bearing (e.g., templates, header-only data types).
- Use `namespace neurograph::<module>` — flat one level.
- Error handling: `util::Result<T>` everywhere in new code; no exceptions
  at API boundaries.
- No raw `new`/`delete`; `std::unique_ptr` by default, `shared_ptr` only for
  true shared ownership.

### 10.3 Testing policy
- **Unit tests required** for every new function with business logic.
  Trivial getters exempt.
- **Integration tests** for every ticket that crosses ≥2 modules.
- **E2E tests** with real small datasets (bundled, ≤1MB each) for every
  phase Definition-of-Done item.
- Target: ≥70% line coverage on non-UI code by end of Phase 1,
  ≥80% by end of Phase 2.
- Use `gtest` (already integrated). Name tests `Module.BehaviorDescribed`,
  not `Module.FunctionNameTest`.

### 10.4 CI/CD

Phase 1 minimum:
- GitHub Actions with Windows + Linux runners.
- Build + unit tests on every push.
- Integration tests on every PR to main.
- Release workflow: tag → build signed installer → upload to Releases page.

Phase 3 adds:
- Docker image publish on release.
- PyPI publish on release.

### 10.5 Licensing (of our product)
- Source: dual-licensed AGPLv3 + commercial, or fully proprietary — **decide
  before Phase 1 release**. This affects whether the code stays public.
- Third-party: audit all dependencies for compatible licenses. Existing deps
  (Dear ImGui, gtest, spdlog, fmt, nlohmann/json, libsodium, stb) are all
  compatible with commercial distribution.

### 10.6 Versioning
- Semantic versioning: `MAJOR.MINOR.PATCH`.
- 0.x during Phase 1 (no stability promise).
- 1.0.0 shipped at the Phase 1 Definition-of-Done.
- Public API stability promised from 1.0 onward for the C ABI, Python API,
  and REST API. Internal C++ API may break on minor versions.

### 10.7 Reproducibility
Every training run saves a `manifest.json`:
```json
{
  "neurograph_version": "1.0.3",
  "git_sha": "abc123",
  "platform": "windows-x86_64",
  "data_hash": "sha256:...",
  "data_source": "./customer_data.csv",
  "task_spec": { ... },
  "preprocessing": { ... },
  "architecture": { ... },
  "hyperparameters": { ... },
  "seed": 42,
  "training_time_seconds": 143.2,
  "metrics": { ... }
}
```

### 10.8 Telemetry (opt-in, Phase 1)
- Off by default. User must tick "Help improve NeuroGraph" at install time.
- Captures: anonymous install ID, crashes (stack traces with file paths
  scrubbed), feature usage counts (which wizard steps were used), training
  wall-clock time, dataset size bucket (never actual data).
- No dataset contents, filenames, or user identity ever sent.
- Opt-out at any time via Settings.

---

## 11. Go-to-market notes (not engineering, but scoping-relevant)

**Target customer for Phase 1**: small-to-mid-size businesses (SMBs) with an
analyst who can use Excel but not Python. Use cases:
- Customer churn prediction
- Lead scoring
- Product-fit classification
- Simple pricing models
- Fraud flagging

**Pricing model (affects product requirements)**:
- Per-seat annual license ($500–$2000/year per user, typical for this
  tier) — drives the need for node-locked licensing (P1-11).
- Lifetime license with annual update subscription — also supported by
  same license system.
- **Avoid usage-based pricing in Phase 1** — requires telemetry
  infrastructure we don't have.

**Support story**: in-app "Report Problem" button that bundles the manifest
+ recent logs into a zip for email. No live chat or cloud dashboard in
Phase 1.

---

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Self-organizing topology doesn't match fixed-arch accuracy on standard benchmarks | Med | High | Benchmark against sklearn/LightGBM on tabular, ResNet on images. If we lose by >5%, add fixed-architecture fallback. |
| GPU backprop integration takes longer than planned (Phase 2) | High | Med | Start P2-07 in parallel with P2-01; don't block other tickets on it. |
| ONNX export coverage gaps (Phase 3) | Med | Med | Test each op as it's added; publish coverage table. |
| Non-ML customers still find wizard too technical | Med | High | User-test wizard with 3 non-ML users before end of Phase 1. |
| License enforcement cracked within days of release | High | Low | Accept it — target market is SMBs who buy legitimately. |
| Windows-only hurts adoption | Med | Med | Port to Linux in Phase 2 (we already support it at build level; just need installer). macOS in Phase 3. |

---

## 13. How to use this document

- **Starting work on a ticket**: open the matching `phase*-<name>.md`
  document, find the ticket ID, treat its acceptance criteria as the contract.
- **Adding a new requirement**: propose it as a ticket in the appropriate
  phase doc; update the ticket table in this file.
- **Finishing a phase**: walk the Definition of Done checklist; nothing ships
  until every box is checked.

## Phase specification documents

Every phase has a dedicated spec document with full ticket-level engineering
detail (acceptance criteria, APIs, test plans). Read these in order when
starting work on a phase:

- [`phase1-tabular-mvp.md`](phase1-tabular-mvp.md) — 14 tickets for the
  Tabular MVP (first sellable release).
- [`phase2-images.md`](phase2-images.md) — 10 tickets adding CNN, image
  loaders, GPU training.
- [`phase3-deployment.md`](phase3-deployment.md) — 8 tickets for REST,
  ONNX, Python wheel, Docker, gRPC, metrics.
- [`phase4-text-sequences.md`](phase4-text-sequences.md) — 10 tickets for
  BPE, embeddings, Transformer, time-series.
- [`phase5-enterprise.md`](phase5-enterprise.md) — 10 tickets for
  multi-GPU, explainability, HPO, SaaS, governance.

Next read: [`phase1-tabular-mvp.md`](phase1-tabular-mvp.md) for the detailed
specs of the 14 Phase 1 tickets.
