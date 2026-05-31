# Phase 2 — Images: Detailed Engineering Specs

> **Goal of Phase 2:** Teach NeuroGraph to train on images. A user drops a
> folder of JPEG/PNG files (or a ZIP of them) organized into class subfolders,
> clicks **Train**, and gets a working image classifier — trained on the GPU
> in minutes instead of hours, exported as a standalone runtime like Phase 1.
>
> **Duration estimate:** 6–8 weeks with one senior C++/CUDA engineer, or ~10
> weeks with an average C++ developer who is willing to climb the CUDA
> learning curve on P2-07.
>
> **Entry criteria:** Phase 1 Definition of Done fully ticked. `AutoPipeline`
> runs end-to-end on tabular. Workbench wizard ships.
>
> **Exit criteria:**
> - CIFAR-10 reaches ≥70% test accuracy out-of-the-box (auto-config only).
> - MNIST training drops from ~30s (current CPU) to ≤3s on an RTX 3060.
> - A 50k-JPEG class-folder dataset trains end-to-end with zero manual
>   decompression and zero hyperparameter tuning.
> - Exported `predict.exe` and `neurograph_rt.dll` both handle image input.

---

## How to use this document

Same structure as `phase1-tabular-mvp.md`: every ticket lists metadata,
goal, why-it-matters, files, API, algorithm notes, acceptance criteria, and
test plan. Self-contained — you can open a ticket and start.

### Track assignment

| Track | Tickets | Owner archetype |
|-------|---------|-----------------|
| **A — Tensor & Layer Algebra** | P2-01, P2-02, P2-03 | C++ engineer comfortable with strided memory, maybe ex-graphics |
| **B — I/O & Data** | P2-04, P2-05, P2-09 | Generalist |
| **C — GPU** | P2-07 | CUDA-literate engineer (only blocker is that this needs CUDA expertise, not just building CUDA code) |
| **D — Auto & UX wiring** | P2-06, P2-08 | Same person who owned P1-14 (`AutoPipeline`) |
| **E — Stretch** | P2-10 | Only if a pilot customer asks for it |

Tracks A and B can start day 1. Track C starts once P2-01 merges (needs the new tensor types). Track D starts once P2-03 + P2-06 have at least a skeleton.

---

## Cross-cutting conventions (new in Phase 2)

- **Tensor dtype:** all new image/CNN code is `float` (FP32). Switch to
  `double` is too wasteful on GPU memory. Dense/tabular code remains `double`
  — we bridge at layer boundaries.
- **Layout:** NCHW everywhere (batch, channels, height, width). Row-major.
  Documented in `include/neurograph/tensor/layout.hpp` so no one has to guess.
- **Memory:** host tensors via `std::vector<float>` + shape. Device tensors
  via a thin RAII `CudaBuffer<float>`. Don't roll a full tensor library — we
  only need what CNN forward/backward requires.
- **Kernel naming:** CUDA kernels go in `cuda/kernels/` and are named
  `<layer>_<direction>_kernel`, e.g. `conv2d_forward_kernel`.
- **Epsilons:** `kBNEpsilon = 1e-5f`, `kSoftmaxEpsilon = 1e-9f` — centralize
  in `include/neurograph/tensor/numerics.hpp`.
- **Deterministic mode:** a `NG_DETERMINISTIC=1` env var forces cuDNN-less
  code paths and serial reductions. Required for reproducibility manifests
  on GPU — atomics in backward conv are the usual culprit.
- **Perf tests:** add `tests/perf/` target. Runs on CI only when a GPU is
  available (skip otherwise). Budgets are asserted, not logged.

---

# Phase 2 Tickets

## P2-01 — ConvNeuron + Conv2D Layer

- **Size:** L  •  **Deps:** none  •  **Track:** A

### Goal
Introduce a `ConvNeuron` concept and a `Conv2DLayer` that performs strided 2D
convolution with bias, padding, and shared weights. Forward + backward on
CPU first (GPU comes in P2-07). Preserve the "every neuron is a first-class
object" identity so self-organization (split, prune, mutate) still works —
but at the filter level, not per-output-pixel.

### Why it matters
CNN filters are the single biggest accuracy win over MLPs for images.
Without shared weights, a 64×64 image's first layer needs millions of
parameters — unusable on commodity hardware. This ticket is the foundation
for everything image-related.

### Files
- `include/neurograph/tensor/tensor.hpp` — NEW (lightweight host tensor)
- `include/neurograph/tensor/layout.hpp` — NEW (NCHW helpers, im2col indexer)
- `include/neurograph/core/conv_neuron.hpp` — NEW
- `include/neurograph/core/conv_layer.hpp` — NEW
- `src/core/conv_neuron.cpp` — NEW
- `src/core/conv_layer.cpp` — NEW
- `tests/unit/test_conv_layer.cpp` — NEW
- `include/neurograph/core/network.hpp` — modify: add `LayerVariant` and layer list alongside the existing per-neuron graph
- `CMakeLists.txt` — add new sources

### API
```cpp
// include/neurograph/tensor/tensor.hpp
namespace neurograph::tensor {

struct Shape4D { int n, c, h, w; std::size_t size() const { return std::size_t(n)*c*h*w; } };

class Tensor4D {
public:
    Tensor4D() = default;
    explicit Tensor4D(Shape4D s) : shape_(s), data_(s.size(), 0.0f) {}

    const Shape4D& shape() const { return shape_; }
    float* data()             { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Flat index — use the inline helpers in layout.hpp for readability.
    float& at(int n, int c, int h, int w);
    const float& at(int n, int c, int h, int w) const;

    void zero();
    void fill(float v);
private:
    Shape4D shape_{};
    std::vector<float> data_;
};

} // namespace
```

```cpp
// include/neurograph/core/conv_layer.hpp
namespace neurograph::core {

struct ConvConfig {
    int in_channels;
    int out_channels;   // = number of filters / ConvNeurons
    int kernel_size;    // square kernels only in v1
    int stride    = 1;
    int padding   = 0;
    bool bias     = true;
    ActivationType activation = ActivationType::ReLU;
    uint64_t init_seed = 0;  // Kaiming init
};

class ConvNeuron {
public:
    ConvNeuron(int in_channels, int kernel_size, uint64_t seed);

    // Weights: [in_channels, k, k]. Bias: 1 scalar.
    std::vector<float>& weights();
    float& bias();

    // Growth / self-org hooks — mirror existing Neuron API so the
    // self-org driver doesn't need branch-per-type logic. Splitting a
    // filter = clone with noise; pruning = remove from layer list.
    std::unique_ptr<ConvNeuron> clone_with_noise(std::mt19937& rng, float noise_stddev) const;

private:
    std::vector<float> w_;  // contiguous [in_ch * k * k]
    float b_ = 0.0f;
    int in_channels_, kernel_size_;
};

class Conv2DLayer {
public:
    explicit Conv2DLayer(const ConvConfig& cfg);

    // Forward: x [N,Cin,H,W] → y [N,Cout,Hout,Wout].
    // Hout = (H + 2*P - K) / S + 1; same for Wout.
    void forward(const tensor::Tensor4D& x, tensor::Tensor4D& y) const;

    // Backward: given upstream dy, compute dx, dW, db.
    void backward(const tensor::Tensor4D& x,
                  const tensor::Tensor4D& dy,
                  tensor::Tensor4D& dx,
                  std::vector<float>& dW,
                  std::vector<float>& db) const;

    void apply_gradients(const std::vector<float>& dW,
                         const std::vector<float>& db,
                         float lr);

    // Self-org
    void grow_filters(int n_new, std::mt19937& rng);
    void prune_filters(const std::vector<int>& filter_ids);
    int  num_filters() const;

    const ConvConfig& config() const { return cfg_; }

private:
    ConvConfig cfg_;
    std::vector<std::unique_ptr<ConvNeuron>> filters_;
};

} // namespace
```

### Algorithm notes
- **Forward via im2col → GEMM:** classic trick. Reshape input patches into
  a `[Cin*K*K, N*Hout*Wout]` matrix, weights into `[Cout, Cin*K*K]`, multiply,
  reshape back. Use a plain triple-loop GEMM for CPU; we'll swap in cuBLAS on
  GPU.
- **Backward:** two GEMMs (dW from x^T · dy, dx from dy · W) plus a col2im
  scatter. Write a straightforward implementation first; optimize later only
  if perf tests demand.
- **Init:** Kaiming-normal with fan-in = `Cin * K * K`.
- **Padding:** zero-pad only in v1. Reflect/replicate can wait.
- **ActivationType:** reuse existing `neurograph::core::ActivationType`. Apply
  in-place on `y` after the GEMM.

### Acceptance criteria
- [ ] Forward output matches PyTorch `torch.nn.functional.conv2d` within
      1e-5 on `tests/fixtures/conv/golden.npz` (check in a precomputed fixture).
- [ ] Backward gradients pass numerical gradient check (central-difference)
      with tolerance 1e-3 on a 4×4×3×3 random input.
- [ ] Shape calculation correct for stride 1–4, padding 0–2, kernel 1–7.
- [ ] `grow_filters(1)` appends one filter initialized with noise around an
      existing filter's weights — output shape grows by 1 channel, existing
      outputs unchanged.
- [ ] `prune_filters({0, 2})` removes filters 0 and 2; output shape drops
      by 2.
- [ ] Forward throughput ≥ 80 MFLOP/s on one CPU core for `[1,3,32,32]`
      with 32 filters (weak gate — im2col GEMM alone gets there).

### Test plan
- Unit: shape calc, forward vs PyTorch golden, gradient check, serialization
  round-trip (once P1-13 manifest lands, but just save/load here).
- Integration: a 2-conv + pool + dense model trains on a 500-sample MNIST
  subset to ≥80% train accuracy on CPU.

---

## P2-02 — Pooling Layers

- **Size:** M  •  **Deps:** P2-01  •  **Track:** A

### Goal
Max, average, and global average pooling with arbitrary window/stride, both
forward and backward on CPU.

### Why it matters
Every standard CNN architecture uses pooling to reduce spatial dimensions.
Global average pooling replaces the flatten+dense pattern at the end of
networks like ResNet and dramatically reduces parameters — we want it
available so the auto-architecture builder (P2-06) can use it.

### Files
- `include/neurograph/core/pooling_layer.hpp` — NEW
- `src/core/pooling_layer.cpp` — NEW
- `tests/unit/test_pooling.cpp` — NEW

### API
```cpp
namespace neurograph::core {

enum class PoolKind { Max, Avg, GlobalAvg };

struct PoolConfig {
    PoolKind kind         = PoolKind::Max;
    int      kernel_size  = 2;     // ignored when GlobalAvg
    int      stride       = 2;     // ignored when GlobalAvg
    int      padding      = 0;
};

class PoolingLayer {
public:
    explicit PoolingLayer(const PoolConfig& cfg);

    void forward(const tensor::Tensor4D& x, tensor::Tensor4D& y);
    // Backward uses the cached argmax indices for max-pool; for avg-pool
    // we recompute the uniform 1/K^2 distribution.
    void backward(const tensor::Tensor4D& x,
                  const tensor::Tensor4D& dy,
                  tensor::Tensor4D& dx);
private:
    PoolConfig cfg_;
    std::vector<int> argmax_cache_;  // populated in forward, used in backward
};

} // namespace
```

### Acceptance criteria
- [ ] Max-pool forward/backward match PyTorch on the same fixture file used
      by P2-01.
- [ ] Global avg-pool output shape is `[N, C, 1, 1]` for any input spatial
      size.
- [ ] Forward + backward round-trip preserves sum of dy / count-of-inputs in
      avg mode (sanity check).
- [ ] Max-pool backward routes the full gradient to exactly one input cell
      per window (ties broken by lowest index, reproducibly).

### Test plan
- Unit: 4×4 → 2×2 max pool by hand; global-avg on `[1,3,5,5]` gives a 3-vector
  equal to per-channel means.
- Integration folded into P2-01's model test.

---

## P2-03 — Tensor Plumbing & Image Batching

- **Size:** M  •  **Deps:** P2-01  •  **Track:** A

### Goal
Retrofit `TrainingEngine` and `Sample` to carry image data as `Tensor4D` (or
a view of one), not flat `vector<double>`. Add a batch collator that packs
multiple samples into a single `[N,C,H,W]` tensor. Keep tabular code path
untouched — tabular samples remain `vector<double>`.

### Why it matters
The current `Sample::features` is a 1D vector with no shape info — conv
layers can't use it without recomputing shape every call. This ticket makes
tensor-shape the contract so CNN layers never have to guess.

### Files
- `include/neurograph/training/image_sample.hpp` — NEW
- `include/neurograph/training/batch_collator.hpp` — NEW
- `src/training/batch_collator.cpp` — NEW
- `include/neurograph/training/training_engine.hpp` — modify: add
  `std::variant<TabularBatch, ImageBatch>` dispatch
- `src/training/training_engine.cpp` — modify: collate + route
- `tests/unit/test_batch_collator.cpp` — NEW

### API
```cpp
namespace neurograph::training {

struct ImageSample {
    tensor::Shape4D     shape;    // (1, C, H, W)
    std::vector<float>  pixels;   // length = C*H*W
    std::vector<double> target;   // one-hot or regression target
    std::string         label;
};

struct ImageBatch {
    tensor::Tensor4D                  images;    // [B, C, H, W]
    std::vector<std::vector<double>>  targets;   // [B, target_dim]
    std::vector<std::string>          labels;
};

class BatchCollator {
public:
    BatchCollator(int batch_size, int channels, int height, int width);

    // Returns an empty batch when `samples` is empty. Throws if a sample's
    // shape disagrees with the collator's declared shape — that's a
    // dataset-loader bug, not a data bug.
    ImageBatch collate(const std::vector<ImageSample>& samples) const;
};

} // namespace
```

### Acceptance criteria
- [ ] `collate(3 samples)` produces a `[3,C,H,W]` tensor; in-memory layout
      is contiguous row-major NCHW.
- [ ] Shape mismatch throws `std::invalid_argument` with a message naming
      the offending sample index.
- [ ] Zero overhead vs the raw `vector<float>` baseline for batch size 1
      (spot-check: within 5% on a perf test).
- [ ] Existing tabular training tests still pass unchanged.

### Test plan
- Unit: collate 3 fixed-shape samples, assert indexed pixel values.
- Unit: mixed-shape input throws.
- Integration: P2-01's model trains correctly when driven through the
  collator instead of raw vectors.

---

## P2-04 — PNG + JPEG Loader

- **Size:** S  •  **Deps:** P2-03  •  **Track:** B

### Goal
Replace the existing BMP-only image loader with `stb_image` (already vendored
for many projects — add via vcpkg as `stb`). Support PNG, JPEG, and BMP.
Produce `ImageSample` objects with pixel values normalized to `[0, 1]`.

### Why it matters
Real customer image datasets are JPEGs 99% of the time. BMP-only blocks
every non-toy use case.

### Files
- `include/neurograph/training/image_loader.hpp` — NEW (replaces BMP-only)
- `src/training/image_loader.cpp` — NEW
- `vcpkg.json` — add `stb` dependency
- `CMakeLists.txt` — link stb
- `tests/unit/test_image_loader.cpp` — NEW

### API
```cpp
namespace neurograph::training {

struct ImageLoadOptions {
    int     target_width   = 0;    // 0 = keep native
    int     target_height  = 0;
    int     target_channels = 3;   // 1=grayscale, 3=RGB, 4=RGBA
    bool    normalize_0_1  = true; // /255.0
    bool    mean_subtract  = false;
    std::array<float, 3> mean{0.485f, 0.456f, 0.406f}; // ImageNet defaults
};

tl::expected<ImageSample, std::string>
load_image(const std::filesystem::path& path,
           const ImageLoadOptions& opts = {},
           const std::string& label = "",
           const std::vector<double>& target = {});

// Load all images in a folder. Labels derived from immediate parent folder
// names: `root/class_a/img1.jpg` → label = "class_a".
tl::expected<std::vector<ImageSample>, std::string>
load_image_folder(const std::filesystem::path& root,
                  const ImageLoadOptions& opts = {});

} // namespace
```

### Algorithm notes
- Use `stbi_load` for decode; `stbir_resize_uint8` for resize. Both header-only.
- Class label → target: callers pass a class-label table; loader emits
  one-hot `target` vectors. Keep this logic out of stb-land.
- Corrupt files: skip with a warning, don't abort the whole load.

### Acceptance criteria
- [ ] Loads PNG, JPEG, BMP fixtures from `tests/fixtures/images/` correctly
      — pixel sum matches a hand-computed value ±1.
- [ ] Resizing to 32×32 produces tensors identical (±1 in uint8 space) to
      a Python reference using PIL.LANCZOS or `stb`'s default.
- [ ] Folder loader assigns labels alphabetically (so labels are stable
      across runs).
- [ ] Corrupted file logged as warning, loader continues with remaining files.
- [ ] 10k-image folder loads in < 5 s on NVMe SSD.

### Test plan
- Unit: each format in a 2×2 fixture; grayscale mode on an RGB image.
- Integration: load MNIST as PNGs (convert fixture once, check in), verify
  first/last pixel match baseline.

---

## P2-05 — ZIP Archive Extraction

- **Size:** S  •  **Deps:** P2-04  •  **Track:** B

### Goal
Accept a `.zip` file as a dataset source. Extract on the fly to a temp dir,
feed through P2-04's folder loader. Delete temp dir when training finishes or
fails.

### Why it matters
Every public dataset ships as a ZIP. Customers exporting from cloud buckets
get ZIPs. Forcing users to unzip first is a trivial but high-friction step
that the wizard should eliminate.

### Files
- `include/neurograph/training/zip_extractor.hpp` — NEW
- `src/training/zip_extractor.cpp` — NEW
- `vcpkg.json` — add `miniz` dependency (single header, lightweight)
- `tests/unit/test_zip_extractor.cpp` — NEW

### API
```cpp
namespace neurograph::training {

struct ZipExtractResult {
    std::filesystem::path root;   // points inside a unique temp dir
    std::size_t n_files = 0;
    std::size_t total_bytes = 0;
};

tl::expected<ZipExtractResult, std::string>
extract_zip(const std::filesystem::path& zip_path,
            const std::filesystem::path& target_dir = {});  // empty → temp

// RAII cleanup helper. Destructor recursively removes the dir.
class ScopedExtracted {
public:
    explicit ScopedExtracted(ZipExtractResult r);
    ~ScopedExtracted();
    const std::filesystem::path& root() const;
private:
    ZipExtractResult r_;
};

} // namespace
```

### Acceptance criteria
- [ ] Extracts a 1000-file archive in < 2 s on NVMe.
- [ ] Zip-slip attack (paths containing `..`) refused with an error.
- [ ] Temp dir auto-cleaned on `ScopedExtracted` destruction, even on
      exceptions.
- [ ] Handles ZIPs > 4 GB (zip64) if the extractor supports it — miniz does.

### Test plan
- Unit: extract `tests/fixtures/zips/small.zip` → count matches.
- Unit: extract a malicious zip with `../foo` → returns error, no files
  written.
- Integration: `AutoPipeline` can point at a ZIP as dataset source.

---

## P2-06 — Adaptive CNN Architecture Builder

- **Size:** L  •  **Deps:** P2-01, P2-02, P2-03  •  **Track:** D

### Goal
Given image dimensions and number of classes, synthesize a reasonable CNN
topology automatically. Three presets by input size:
- **small_cnn** (≤32×32): 2 conv blocks + dense head (MNIST, CIFAR sized)
- **medium_cnn** (≤128×128): 4 conv blocks + global avg pool + dense head
- **large_cnn** (>128×128): 6 conv blocks with downsampling + global avg pool

Self-organization then adds/removes filters within each layer according to
the existing growth/prune heuristics.

### Why it matters
Non-experts cannot design CNN architectures. This is the "auto" part of
auto-ML for images. Simple preset heuristics get 90% of the benefit; we don't
need NAS here.

### Files
- `include/neurograph/pipeline/cnn_builder.hpp` — NEW
- `src/pipeline/cnn_builder.cpp` — NEW
- `tests/unit/test_cnn_builder.cpp` — NEW
- `tests/integration/test_cnn_autobuild.cpp` — NEW

### API
```cpp
namespace neurograph::pipeline {

struct CnnBlueprint {
    std::vector<core::ConvConfig>    conv_layers;
    std::vector<core::PoolConfig>    pool_layers;   // pools slotted between convs
    // Final head: global avg pool → flatten → dense(num_classes)
    int   final_dense_units;
    bool  use_global_avg_pool = true;
    std::string preset_name;
};

CnnBlueprint build_cnn(int channels, int height, int width,
                       int num_classes,
                       const training::SupervisedConfig& training_cfg);

// Materialize the blueprint into an actual Network.
std::unique_ptr<core::Network>
instantiate_cnn(const CnnBlueprint& bp, uint64_t seed);

} // namespace
```

### Algorithm notes (the presets)

| Preset | Trigger | Channels progression | Notes |
|--------|---------|---------------------|-------|
| small  | H*W ≤ 1024 | 32 → 64 → dense(128) → softmax | MNIST / small icons |
| medium | ≤ 16384 | 32 → 64 → 128 → 256 → GAP → softmax | Roughly ResNet-8 capacity |
| large  | > 16384 | 32 → 64 → 128 → 256 → 512 → 512 → GAP → softmax | Adds strided downsampling every other block |

All convs: 3×3, padding=1, ReLU, followed by 2×2 max pool every block until
spatial dim would drop below 4. Final dense only in small preset; medium/large
use GAP + softmax directly.

### Acceptance criteria
- [ ] Deterministic given fixed seed (same architecture, same initial weights).
- [ ] `build_cnn(1, 28, 28, 10, ...)` picks `small`, `build_cnn(3, 32, 32, 10, ...)`
      picks `small`, `build_cnn(3, 64, 64, 10, ...)` picks `medium`,
      `build_cnn(3, 224, 224, 1000, ...)` picks `large`.
- [ ] Instantiated CNN passes a forward call on a random `[2, C, H, W]`
      tensor without shape error.
- [ ] Self-organization step runs on the instantiated CNN without crashing
      (grow_filters at some conv layer).

### Test plan
- Unit: preset selection from shape.
- Integration: full train-for-5-epochs on MNIST via `instantiate_cnn` +
  auto_supervised_config; assert val accuracy > 90% (easy target).

---

## P2-07 — GPU Backprop Integration

- **Size:** L  •  **Deps:** P2-01, P2-02  •  **Track:** C

### Goal
Make the existing CUDA kernels in `cuda/kernels/` actually used by the
training loop. Today they compile but aren't called. Wire up forward,
backward, and weight-update on the GPU for dense + conv + pool layers.
Fall back to CPU when CUDA isn't available.

### Why it matters
A RTX 3060 is ~30× faster than one CPU core on CNN training. Without GPU
support, a CIFAR-10 run takes an hour; with it, 2 minutes. This ticket is
the single biggest user-visible perf win in Phase 2.

### Files
- `include/neurograph/gpu/device_context.hpp` — NEW (CUDA stream + workspace)
- `include/neurograph/gpu/device_tensor.hpp` — NEW (`CudaBuffer<float>`)
- `cuda/kernels/conv2d_forward.cu` — NEW
- `cuda/kernels/conv2d_backward.cu` — NEW
- `cuda/kernels/pool_forward.cu` — NEW
- `cuda/kernels/pool_backward.cu` — NEW
- `src/gpu/gpu_trainer.cpp` — NEW (drives an epoch on GPU)
- `include/neurograph/training/training_engine.hpp` — modify: pick GPU path
  when `device == Device::Gpu` and CUDA is available
- `tests/unit/test_gpu_parity.cpp` — NEW (CPU vs GPU forward/backward within tol)

### API
```cpp
namespace neurograph::gpu {

enum class Device { Cpu, Gpu, Auto };

class DeviceContext {
public:
    static DeviceContext& instance();
    bool cuda_available() const;
    int  device_id() const;
    cudaStream_t stream() const;
    size_t free_memory_bytes() const;
};

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t n);
    ~CudaBuffer();

    CudaBuffer(CudaBuffer&&) noexcept;
    CudaBuffer& operator=(CudaBuffer&&) noexcept;
    CudaBuffer(const CudaBuffer&) = delete;

    T* data();
    size_t size() const;

    void copy_from_host(const T* host, size_t n);
    void copy_to_host(T* host, size_t n) const;

private:
    T* ptr_ = nullptr;
    size_t n_ = 0;
};

} // namespace
```

### Algorithm notes
- **cuBLAS** for all dense/im2col-GEMM paths. Single `cublasHandle_t` per
  thread, stored in `DeviceContext`.
- **Custom kernels** for pool forward/backward and for conv backward's col2im
  step (the pattern doesn't fit a BLAS call cleanly). One thread per output
  element, shared-memory tile for pool.
- **Mixed precision deferred:** stay FP32 in Phase 2. FP16/TF32 can come later
  if needed.
- **Deterministic mode** (`NG_DETERMINISTIC=1`): use `-sort`-style reductions
  (single-stream, no atomicAdd in backward) at 2-3× perf cost. The switch
  lives in `DeviceContext`.
- **Fallback:** at `TrainingEngine` init, probe `cuda_available()`. If false
  or user requested CPU, run existing CPU path unchanged.

### Acceptance criteria
- [ ] GPU forward output matches CPU forward within 1e-4 on a random
      `[16, 3, 32, 32]` tensor through a 3-conv + pool + dense network.
- [ ] GPU backward gradients match CPU gradients within 1e-3 same setup.
- [ ] MNIST training on one RTX 3060: ≤ 3 seconds per epoch for the small_cnn
      preset (batch 128).
- [ ] CIFAR-10 training on one RTX 3060: ≤ 20 s per epoch for small_cnn.
- [ ] `NG_DETERMINISTIC=1` gives bit-identical loss across runs (tolerance 0).
- [ ] Runs cleanly on a machine without CUDA: detects absence, logs once,
      silently uses CPU.

### Test plan
- Unit: CPU-vs-GPU parity on each layer type.
- Perf: `tests/perf/test_gpu_throughput.cpp` — gated on CUDA availability.
- Integration: MNIST train-to-accuracy budget test; must hit ≥97% val in ≤ 1 minute.

---

## P2-08 — Architecture Selector in AutoPipeline

- **Size:** M  •  **Deps:** P2-06  •  **Track:** D

### Goal
Extend P1-14's `AutoPipeline` to branch on `InputModality`. For images: run
P2-06 CNN builder; for tabular: continue with the existing MLP auto-config.
Choose the right loss, metrics, and augmentation defaults for each.

### Why it matters
Without this, the wizard's "Train" button only works on tabular. This is the
glue that makes the whole CNN path reachable from the user's perspective.

### Files
- `src/pipeline/auto_pipeline.cpp` — modify: add image branch
- `include/neurograph/pipeline/modality_detector.hpp` — NEW (small helper:
  is this folder of images? or a CSV?)
- `src/pipeline/modality_detector.cpp` — NEW
- `tests/integration/test_auto_pipeline_images.cpp` — NEW

### API
```cpp
namespace neurograph::pipeline {

enum class DatasetSourceKind { Csv, ImageFolder, Zip, Unknown };

DatasetSourceKind detect_source_kind(const std::filesystem::path& path);

} // namespace
```

Modifications to `AutoPipeline::run`:
- Inspect input path. CSV → existing flow. Folder or zip → CNN flow:
  1. Extract zip if needed (P2-05).
  2. Load images (P2-04) into ImageSamples.
  3. Profile dataset: shape, class count, class balance.
  4. Build CNN (P2-06).
  5. Train on GPU if available (P2-07).
  6. Evaluate with classification metrics (P1-04).

### Acceptance criteria
- [ ] Point `AutoPipeline::run` at a folder of JPEGs → trained image
      classifier bundle produced.
- [ ] Point at a CSV → existing tabular flow unchanged (regression test).
- [ ] Point at a ZIP → extracted, trained, bundle produced; temp dir cleaned.
- [ ] Respects `AutoOptions::time_budget_seconds` on the image path (scales
      epochs, not just samples per epoch).

### Test plan
- Integration: three flavors of input (CSV, folder, zip), each produces a
  working bundle.
- Regression: the existing titanic integration test still passes.

---

## P2-09 — Expanded Image Augmentation

- **Size:** S  •  **Deps:** P2-03  •  **Track:** B

### Goal
Beyond the existing rotate/shift/noise, add: horizontal flip, cutout,
MixUp, CutMix, color jitter (brightness/contrast/saturation). Turn on
sensibly by default depending on domain (e.g. digit recognition → no h-flip).

### Why it matters
Modern augmentations routinely deliver +3–5% accuracy on image benchmarks
with no architectural changes. This is cheap performance the user gets for
free.

### Files
- `include/neurograph/training/image_augmentation.hpp` — extend existing or NEW
- `src/training/image_augmentation.cpp` — extend
- `tests/unit/test_image_aug.cpp` — NEW / extend

### API
```cpp
namespace neurograph::training {

struct ImageAugConfig {
    bool  horizontal_flip = false;
    bool  cutout          = false;
    int   cutout_size     = 8;       // pixels, square
    bool  mixup           = false;
    float mixup_alpha     = 0.2f;
    bool  cutmix          = false;
    float cutmix_alpha    = 1.0f;
    bool  color_jitter    = false;
    float brightness      = 0.2f;    // ±frac
    float contrast        = 0.2f;
    float saturation      = 0.2f;
};

// Applies augmentation in-place to a batch. MixUp/CutMix modify targets
// too — the batch's targets vector becomes soft labels.
void augment_batch(ImageBatch& batch, const ImageAugConfig& cfg,
                   std::mt19937& rng);

} // namespace
```

### Acceptance criteria
- [ ] Each augmentation preserves tensor shape.
- [ ] MixUp/CutMix produce soft targets that sum to 1 per sample (within 1e-6).
- [ ] `horizontal_flip=false` default in auto-config for digit datasets
      (domain heuristic from P1-14 → P2-08).
- [ ] No nondeterminism: same `rng` state → identical augmentation.

### Test plan
- Unit: each augmentation on a single fixture image; assert shape,
  value-range, and for MixUp the label math.

---

## P2-10 — Object Detection Lite (Optional)

- **Size:** L  •  **Deps:** P2-06, P2-07  •  **Track:** E

### Goal
Support a restricted object-detection task: classification + single bounding
box per image. Think "find the face in this photo" not "find all cars in
this street scene." Full YOLO/Faster-RCNN is out of scope.

### Why it matters
This is **optional** — ship only if a Phase-2 pilot customer explicitly asks.
Including the scaffolding in the plan so we know what it would take.

### Files
- `include/neurograph/pipeline/detection.hpp` — NEW
- `src/pipeline/detection.cpp` — NEW
- `include/neurograph/training/detection_loss.hpp` — NEW (combined
  cross-entropy + IoU loss)
- `tests/integration/test_detection.cpp` — NEW

### API
```cpp
namespace neurograph::pipeline {

struct BoundingBox { float x, y, w, h; };  // normalized 0..1

struct DetectionSample {
    training::ImageSample image;
    int                   class_id;
    BoundingBox           bbox;
};

struct DetectionPrediction {
    int         class_id;
    float       class_confidence;
    BoundingBox bbox;
    float       bbox_iou_with_target;  // only at eval time
};

} // namespace
```

### Acceptance criteria
- [ ] Synthetic 64×64 dataset (coloured square on noise) trains to ≥90%
      classification + mean IoU > 0.7 in 2 minutes on GPU.
- [ ] Inference runtime exports the bbox via C-ABI.

### Test plan
- Unit: loss math on fixed logits + box.
- Integration: synthetic dataset train-to-accuracy gate.

---

# Phase 2 milestones

| Week | Milestones |
|------|-----------|
| 1 | P2-01 Conv2DLayer forward passes PyTorch parity; P2-03 tensor plumbing merges; P2-04 image loader lands |
| 2 | P2-01 Conv backward gradient check passes; P2-02 pool forward+backward; P2-05 ZIP extractor |
| 3 | P2-06 builder ships small/medium/large presets; P2-08 MODALITY branch wires CSV/folder/zip; MNIST trains CPU end-to-end |
| 4 | P2-07 GPU forward parity; P2-09 augmentations |
| 5 | P2-07 GPU backward parity + perf gate; CIFAR-10 baseline run |
| 6 | End-to-end on 50k-image class folder; wizard UX pass for images |
| 7 | Perf polish; bug-fix; MSI update |
| 8 | RC + pilot-customer trial |

# Definition of Done for Phase 2

- [ ] All tickets P2-01 through P2-09 merged; P2-10 only if a customer asked.
- [ ] CIFAR-10 ≥ 70% test accuracy via auto-config only.
- [ ] MNIST ≤ 3 s/epoch on RTX 3060.
- [ ] Drag-and-drop a 50k-JPEG folder → trained model in ≤ 10 minutes.
- [ ] CPU-vs-GPU parity tests pass in CI.
- [ ] Exports (`predict.exe`, `neurograph_rt.dll`) handle images.
- [ ] Docs: `docs/user-guide-images.md` for end users; model card template
      updated to include sample image predictions.

---

*Next doc: [`phase3-deployment.md`](phase3-deployment.md) — ONNX, Python
wheel, REST server, Docker.*
