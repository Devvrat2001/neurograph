# Phase 3 — Deployment: Detailed Engineering Specs

> **Goal of Phase 3:** Turn the trained model bundle into every shape a
> customer might need: a REST server, an ONNX file, a Python wheel, a
> Docker image, optionally a gRPC server, with versioning, auth, and
> metrics. Phase 1 already ships `.exe` + C-ABI DLL — this phase adds the
> rest.
>
> **Duration estimate:** 4–6 weeks with one engineer. ONNX export (P3-02)
> is the single long-pole ticket.
>
> **Entry criteria:** Phase 1 DoD met, ideally Phase 2 done too — ONNX
> exporters and the REST server both need to cover conv + pool ops.
>
> **Exit criteria:**
> - `docker run neurograph/server -v model:/model -p 8080:8080` serves
>   predictions via REST with authentication.
> - `pip install neurograph` works on Python 3.9–3.12, Windows/Linux/macOS.
> - ONNX exports of Phase 1 MLP + Phase 2 CNN models verified against
>   `onnxruntime`: predictions match within 1e-5.

---

## How to use this document

Same ticket format as `phase1-tabular-mvp.md`. Self-contained specs.

### Track assignment

| Track | Tickets | Owner archetype |
|-------|---------|-----------------|
| **A — Server** | P3-01, P3-05, P3-06, P3-07, P3-08 | Backend C++ engineer comfortable with cpp-httplib / gRPC |
| **B — Interop** | P3-02, P3-03 | ML/Python engineer who can read Python, knows ONNX spec |
| **C — Packaging** | P3-04 | DevOps-oriented, Docker-literate |

All tracks can start day 1. P3-06 waits for P3-01; P3-05 waits for P3-01 to
finalize the predict wire format.

---

## Cross-cutting conventions (new in Phase 3)

- **Wire formats:** REST uses JSON (nlohmann). gRPC uses protobuf. Both
  share the same semantic schema — define it once in `neurograph.proto`
  and hand-write the JSON encoder/decoder to match.
- **Versioning:** server and Python wheel version = `neurograph_lib`
  version + build SHA. Exposed in `/model/info`.
- **Logging:** server uses structured JSON logs via `spdlog`. Each request
  gets a `request_id` (UUID v4) propagated through logs and response
  headers.
- **Config:** server configured via CLI flags first, env vars second,
  config file third. Flags win, envs override file.
- **Testing:** every server endpoint has an integration test that spins up
  the server on a random port and hits it over HTTP.

---

# Phase 3 Tickets

## P3-01 — REST Inference Server

- **Size:** M  •  **Deps:** P1-07 (bundle format)  •  **Track:** A

### Goal
Standalone server binary `neurograph_server` that loads one model bundle on
startup and exposes:
- `POST /predict` — predict one or many rows, JSON in/out.
- `GET  /health` — 200 when model loaded and responsive.
- `GET  /model/info` — metadata: version, feature names, class labels, input
  schema.
- `GET  /metrics` — Prometheus text format (filled by P3-08).

### Why it matters
Many customer stacks consume model predictions via HTTP. Shipping a server
lets them deploy NeuroGraph models behind nginx or inside a Kubernetes pod
without writing integration code.

### Files
- `include/neurograph/server/rest_server.hpp` — NEW
- `src/server/rest_server.cpp` — NEW
- `src/server/server_main.cpp` — NEW (the binary)
- `include/neurograph/server/request_context.hpp` — NEW
- `vcpkg.json` — add `cpp-httplib`
- `CMakeLists.txt` — add `neurograph_server` executable
- `tests/integration/test_rest_server.cpp` — NEW

### API
```cpp
namespace neurograph::server {

struct ServerConfig {
    std::filesystem::path model_dir;
    std::string  host              = "0.0.0.0";
    int          port              = 8080;
    int          threads            = 0;        // 0 = hardware_concurrency()
    std::size_t  max_batch          = 256;
    int          request_timeout_ms = 30000;
    std::string  api_key_file;                  // empty → no auth (P3-07)
    bool         enable_metrics     = true;     // P3-08
};

class RestServer {
public:
    explicit RestServer(ServerConfig cfg);
    ~RestServer();

    // Blocking. Returns on signal or fatal error.
    void run();

    // Stops the listen loop from another thread.
    void shutdown();
};

} // namespace
```

### Wire format (`POST /predict`)
```json
Request:
  { "rows": [ { "age": 42, "income": 50000, "region": "east" }, ... ] }
  OR (batched tensor form for images):
  { "tensors": [ { "shape": [1,3,32,32], "data": [0.12, 0.08, ...] } ] }

Response:
  {
    "request_id": "uuid",
    "predictions": [
      { "class": "churn", "probabilities": { "churn": 0.81, "stay": 0.19 } },
      ...
    ],
    "model_version": "1.2.3",
    "latency_ms": 3.1
  }
```

### Acceptance criteria
- [ ] Handles 1k concurrent requests with p99 < 50 ms on a titanic-sized
      model (CPU inference).
- [ ] Graceful shutdown on SIGINT — in-flight requests complete, no new ones
      accepted.
- [ ] Corrupt JSON → 400 with error detail, no crash.
- [ ] Missing required feature → 422 with which feature is missing.
- [ ] Model-reload endpoint (`POST /admin/reload`) swaps the bundle without
      restarting the process (used by P3-06).

### Test plan
- Integration: spin up on random port, hit each endpoint, assert status +
  body schema.
- Load test: `tests/perf/test_server_load.cpp` — 10k req in 30 s, track p99.

---

## P3-02 — ONNX Export

- **Size:** L  •  **Deps:** P1-07, ideally P2-01/P2-02  •  **Track:** B

### Goal
Serialize a trained Network to an ONNX graph. Cover all layers we can
produce: Dense, Conv2D, Pool (Max/Avg/Global), activations (ReLU, Sigmoid,
Tanh, Softmax), BatchNorm, Dropout (no-op at inference), Flatten. Validated
against `onnxruntime`.

### Why it matters
ONNX is the portable format every downstream tool speaks — Triton,
ML.NET, Android NNAPI, iOS CoreML via converters, cloud inference services.
Without ONNX we're only as portable as our C ABI.

### Files
- `include/neurograph/export/onnx_exporter.hpp` — NEW
- `src/export/onnx_exporter.cpp` — NEW
- `src/export/onnx_ops.hpp` — NEW (per-op converters)
- `vcpkg.json` — add `onnx` + `protobuf`
- `tests/integration/test_onnx_export.cpp` — NEW (requires Python +
  onnxruntime in CI)

### API
```cpp
namespace neurograph::export_onnx {

struct OnnxExportOptions {
    int opset_version       = 17;  // last opset that covers all our ops
    bool include_metadata   = true;
    bool simplify_graph     = true;  // fold constants
    bool validate_after_export = true;  // run onnx.checker.check_model
    std::string producer_name    = "NeuroGraph";
    std::string producer_version = "1.0.0";
};

struct OnnxExportResult {
    std::filesystem::path path;
    std::size_t           num_ops = 0;
    std::size_t           file_size_bytes = 0;
    std::vector<std::string> unsupported_ops;  // empty on success
};

tl::expected<OnnxExportResult, std::string>
export_to_onnx(const core::Network& net,
               const preprocess::FittedPipeline* pipeline,  // nullable
               const std::filesystem::path& out,
               const OnnxExportOptions& opts = {});

// Coverage audit: report which of our ops we can emit at the given opset.
struct OnnxCoverage {
    int  opset;
    std::vector<std::string> supported;
    std::vector<std::string> unsupported;
};
OnnxCoverage report_coverage(int opset = 17);

} // namespace
```

### Algorithm notes
- **Graph building:** walk the network in topological order, emitting ONNX
  nodes. Map our internal activation enum to the ONNX op name.
- **Pipeline embedding:** emit a prefix subgraph that reproduces the
  scaling / one-hot / impute stage. Without this, callers would need a
  separate Python pipeline — defeats the purpose.
- **Shape inference:** use `onnx::shape_inference::InferShapes` before save
  so downstream runtimes get static shapes when possible.
- **Unsupported ops:** if found (e.g. a future custom op), fail early with
  a list rather than produce a broken model. The `unsupported_ops` field in
  the result lets the UI surface a precise message.

### Acceptance criteria
- [ ] MLP exported from the titanic bundle runs through `onnxruntime`,
      predictions match workbench within 1e-5 on all 10 held-out rows.
- [ ] CNN exported from an MNIST bundle (once Phase 2 lands) verified same.
- [ ] Exported `.onnx` files pass `onnx.checker.check_model`.
- [ ] Export + validation < 5 s for models up to 10 MB.
- [ ] `report_coverage` lists at least: Dense, Conv, MaxPool, AvgPool,
      GlobalAveragePool, Relu, Sigmoid, Tanh, Softmax, BatchNormalization,
      Flatten, Reshape.

### Test plan
- Integration: per-layer round-trip — build a one-layer network, export,
  load in onnxruntime, compare outputs on random inputs.
- Integration: whole-bundle parity on titanic + mnist.

---

## P3-03 — Python Bindings (pybind11)

- **Size:** M  •  **Deps:** P1-07, P1-14  •  **Track:** B

### Goal
A `neurograph` Python package installable via `pip install neurograph` that
exposes the high-level API: train a model from a pandas DataFrame, predict,
save/load bundle, export to ONNX. Not a full SDK — the wrapper is
intentionally thin.

### Why it matters
Python is how 95% of data scientists evaluate tools. Even if the GUI is the
primary product, a missing `pip install` turns off technical evaluators
before they try the GUI.

### Files
- `python/neurograph/__init__.py` — NEW
- `python/neurograph/_core.cpp` — NEW (pybind11 module)
- `python/setup.py` — NEW
- `python/pyproject.toml` — NEW (PEP 517)
- `python/tests/test_roundtrip.py` — NEW
- `CMakeLists.txt` — add `pybind11` discovery, `neurograph_py` module
- `.github/workflows/wheel.yml` — NEW (cibuildwheel matrix)

### API (Python-side)
```python
import neurograph as ng
import pandas as pd

df = pd.read_csv("titanic.csv")
model = ng.AutoPipeline.fit(df, target="survived", time_budget_seconds=60)

# Inference
preds = model.predict(df.drop(columns=["survived"]))   # returns DataFrame
probs = model.predict_proba(df.drop(columns=["survived"]))

# Persistence
model.save("titanic.bundle")
loaded = ng.load("titanic.bundle")

# Export
model.to_onnx("titanic.onnx")

# Metadata
print(model.metrics)     # dict
print(model.task_spec)   # dataclass
```

### Acceptance criteria
- [ ] `pip install neurograph` works on Python 3.9 / 3.10 / 3.11 / 3.12
      on Windows, Linux (manylinux 2_28), macOS (10.15+ intel + arm64).
- [ ] `ng.AutoPipeline.fit(df, ...)` matches the C++ path's metrics within
      the RNG-seed tolerance (same numbers if seed pinned).
- [ ] Round-trip: fit → save → load → predict identical predictions.
- [ ] Wheel size < 50 MB CPU-only, < 300 MB with CUDA runtime bundled.
- [ ] Type stubs (`.pyi`) for IDE completion.

### Test plan
- Python: smoke test that fits on titanic, checks accuracy > baseline.
- Python: round-trip save/load.
- Python: ONNX export + onnxruntime verification from Python-fit model.

---

## P3-04 — Docker Image

- **Size:** S  •  **Deps:** P3-01 (needs the server binary)  •  **Track:** C

### Goal
Two Docker images: `neurograph/server:cpu` and `neurograph/server:cuda-12`.
Both take a bundle via volume mount, run the server on port 8080, support
`--env API_KEY=...` for auth.

### Why it matters
Docker is how everyone deploys to prod in 2026. Publishing images to Docker
Hub / GHCR gives customers a 30-second integration.

### Files
- `docker/Dockerfile.cpu` — NEW
- `docker/Dockerfile.cuda` — NEW
- `docker/entrypoint.sh` — NEW
- `docker/README.md` — NEW (usage examples)
- `.github/workflows/docker.yml` — NEW (publish on tag)

### Image contract
```
$ docker run --rm -v $(pwd)/mybundle:/model -p 8080:8080 \
      -e API_KEY=secret neurograph/server:cpu

$ curl -H "Authorization: Bearer secret" \
      -d '{"rows":[{"age":42}]}' \
      http://localhost:8080/predict
```

### Acceptance criteria
- [ ] CPU image < 500 MB uncompressed (use `distroless` or `alpine` + static).
- [ ] CUDA image < 3 GB; includes runtime libs but not dev tools.
- [ ] Multi-arch CPU image (amd64 + arm64) via `docker buildx`.
- [ ] HEALTHCHECK configured so orchestrators detect crashed servers.
- [ ] Signed with cosign, provenance via SLSA.

### Test plan
- Build + run in CI, curl `/health`, assert 200.
- Run the image against a real bundle fixture and compare predictions.

---

## P3-05 — gRPC Server

- **Size:** M  •  **Deps:** P3-01  •  **Track:** A

### Goal
A `neurograph_grpc` binary exposing the same `Predict` and `Model.Info`
methods via gRPC. Shares the inference core with `rest_server` — only the
transport is different.

### Why it matters
For 10k+ QPS use cases (ad-tech scoring, fraud pipelines) REST's JSON
overhead matters. gRPC + protobuf halves payload size and doubles throughput.
Optional — ship if any customer asks.

### Files
- `proto/neurograph.proto` — NEW
- `include/neurograph/server/grpc_server.hpp` — NEW
- `src/server/grpc_server.cpp` — NEW
- `src/server/grpc_main.cpp` — NEW
- `vcpkg.json` — add `grpc`

### Proto sketch
```protobuf
service Predictor {
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc BatchPredict(stream PredictRequest) returns (stream PredictResponse);
    rpc GetInfo(InfoRequest) returns (InfoResponse);
}
```

### Acceptance criteria
- [ ] Unary RPC matches REST output byte-for-byte on the same inputs.
- [ ] Streaming RPC: 100k requests over a single channel, p99 < 20 ms.
- [ ] Proto file versioned; breaking changes bump major version in path
      (`neurograph.v1.Predictor`).

### Test plan
- Integration: gRPC client test in C++ + one in Python (grpc-python).
- Perf: benchmark vs REST on identical payloads.

---

## P3-06 — Model Versioning & A/B Routing

- **Size:** M  •  **Deps:** P3-01  •  **Track:** A

### Goal
The server can load multiple model bundles simultaneously (e.g. v1 and v2)
and route requests based on a header (`X-Model-Version: v2`) or a sticky
hash of a user ID. Supports canary rollouts (5% to v2, 95% to v1).

### Why it matters
Real deployments never cut over from one model to another — they ramp. This
is table stakes for any customer with a running production pipeline.

### Files
- `include/neurograph/server/model_registry.hpp` — NEW
- `src/server/model_registry.cpp` — NEW
- `src/server/rest_server.cpp` — modify: consult registry per-request
- `tests/integration/test_ab_routing.cpp` — NEW

### API
```cpp
namespace neurograph::server {

struct RoutingRule {
    std::string                       version_tag;
    double                            weight = 1.0;   // for hash routing
    std::optional<std::string>        header_match;   // exact match on X-Model-Version
};

class ModelRegistry {
public:
    void load_model(const std::string& tag,
                    const std::filesystem::path& bundle_dir);
    void unload_model(const std::string& tag);
    void set_routing(std::vector<RoutingRule> rules);

    // Pick which bundle answers this request.
    const runtime::ModelBundle* route(const http::Headers& hdrs,
                                      std::string_view user_id) const;

    std::vector<std::string> list_loaded() const;
};

} // namespace
```

### Acceptance criteria
- [ ] `POST /admin/models` loads a bundle at a given tag; `DELETE /admin/models/{tag}` unloads.
- [ ] Weight 0.1 vs 0.9 → observed distribution over 10k requests within 1% of rule.
- [ ] `X-Model-Version` header always wins over weight-based routing.
- [ ] Routes atomically: no request sees a half-loaded bundle.

### Test plan
- Integration: load two bundles, send 10k requests with pinned user IDs,
  assert same user always routes to same version.

---

## P3-07 — Server Auth

- **Size:** S  •  **Deps:** P3-01  •  **Track:** A

### Goal
Two auth modes:
1. **API key** — static bearer token compared constant-time.
2. **JWT HS256** — token signed with a shared secret; inspect claims for
   `exp`, `sub`, optional `scopes`.

No OAuth2, no OIDC — customers who need those already have an API gateway.

### Why it matters
Any server exposed to a network needs auth. Minimum viable keeps us from
shipping an embarrassing default-open product.

### Files
- `include/neurograph/server/auth.hpp` — NEW
- `src/server/auth.cpp` — NEW
- `src/server/rest_server.cpp` — modify: auth middleware before route dispatch
- `tests/unit/test_auth.cpp` — NEW

### API
```cpp
namespace neurograph::server {

enum class AuthMode { None, ApiKey, JwtHs256 };

struct AuthConfig {
    AuthMode    mode = AuthMode::None;
    std::string api_key;          // ApiKey mode
    std::string jwt_secret;       // JwtHs256 mode
    std::vector<std::string> required_scopes;  // empty = any
};

struct AuthResult {
    bool                        ok = false;
    std::string                 principal;   // username / sub
    std::vector<std::string>    scopes;
    std::string                 error_detail;
};

AuthResult authorize(const AuthConfig& cfg,
                     std::string_view authorization_header);

} // namespace
```

### Acceptance criteria
- [ ] API-key mode uses `CRYPTO_memcmp` (constant-time) to avoid timing
      attacks.
- [ ] Missing / malformed header → 401 with standardized error body.
- [ ] Expired JWT (exp in past) → 401.
- [ ] Missing required scope → 403.
- [ ] Keys come from env var or file — never hardcoded, never in logs.

### Test plan
- Unit: all failure modes return the right HTTP code.
- Integration: server boots with auth on, rejects no-auth and accepts valid.

---

## P3-08 — Prometheus Metrics

- **Size:** S  •  **Deps:** P3-01  •  **Track:** A

### Goal
`GET /metrics` returns Prometheus text format with:
- `neurograph_http_requests_total{endpoint, method, status}`
- `neurograph_http_request_duration_seconds_bucket` (histogram)
- `neurograph_inference_duration_seconds_bucket`
- `neurograph_inference_errors_total{reason}`
- `neurograph_model_version_info{version,loaded_at}`
- process-level: memory RSS, open FDs, CPU user/system time.

### Why it matters
Observability is non-negotiable. Ops teams pipe Prometheus → Grafana and
expect the basics out of the box.

### Files
- `include/neurograph/server/metrics.hpp` — NEW
- `src/server/metrics.cpp` — NEW
- `vcpkg.json` — add `prometheus-cpp`
- `tests/integration/test_metrics_endpoint.cpp` — NEW

### API
```cpp
namespace neurograph::server {

class MetricsCollector {
public:
    MetricsCollector();

    void observe_http(const std::string& endpoint, const std::string& method,
                      int status, double duration_s);
    void observe_inference(double duration_s, bool success, std::string_view reason);
    void set_model_version(const std::string& version);

    // Serialize to Prometheus text format.
    std::string serialize() const;
};

} // namespace
```

### Acceptance criteria
- [ ] `/metrics` scrapable by a real Prometheus instance (tested in CI via
      `prom/prometheus` in a sidecar).
- [ ] Histogram buckets cover 1 ms – 10 s realistically (standard widths).
- [ ] Per-endpoint label cardinality capped — e.g. path templates like
      `/predict` not full URI with IDs.
- [ ] No perf regression > 1% on the P3-01 load test.

### Test plan
- Integration: run server, scrape `/metrics`, validate text format.
- Integration: make 100 requests, assert `_total` counter jumped by 100.

---

# Phase 3 milestones

| Week | Milestones |
|------|-----------|
| 1 | P3-01 server shell routes /health and /predict; P3-02 ONNX per-op converters land for Dense + activations |
| 2 | P3-02 ONNX cover conv/pool/batchnorm; P3-03 pybind wheel builds locally |
| 3 | P3-03 wheels pass CI on 3 OS × 4 Python; P3-04 CPU Docker builds |
| 4 | P3-06 A/B routing; P3-07 auth; P3-04 CUDA Docker |
| 5 | P3-08 metrics; P3-05 gRPC (if pursued); perf + polish |
| 6 | RC; publish wheels to TestPyPI; publish Docker images |

# Definition of Done for Phase 3

- [ ] All non-optional tickets (P3-01 through P3-04, P3-06 through P3-08) merged.
- [ ] Docker Hub has `neurograph/server:cpu` and `:cuda-12` on every release tag.
- [ ] PyPI has `neurograph` wheels for 3 OSes × 4 Python versions.
- [ ] ONNX parity test green in CI.
- [ ] REST server load test p99 < 50 ms on titanic model, 1k concurrent connections.
- [ ] Docs: `docs/deployment.md` covering all deploy paths.

---

*Next doc: [`phase4-text-sequences.md`](phase4-text-sequences.md) — text
classification, time-series, small-Transformer.*
