# Phase 4 — Text & Sequences: Detailed Engineering Specs

> **Goal of Phase 4:** Extend the product to the third major modality —
> sequential data. Two customer stories:
> 1. **Text classification** — drop a folder of `.txt` files organized by
>    class, get a working classifier. Internally: BPE tokenizer → embedding
>    → small Transformer encoder → softmax.
> 2. **Time-series forecasting** — drop a date-indexed CSV, predict the next
>    N steps. Internally: sliding-window adapter + seasonality features →
>    small Transformer or MLP regressor.
>
> **Duration estimate:** 8–10 weeks with one experienced engineer who has
> touched Transformers before. ~14 weeks if learning from scratch.
>
> **Entry criteria:** Phase 2 GPU path stable. Phase 3 deploy paths shipped
> (needed so Phase 4 models export the same way).
>
> **Exit criteria:**
> - IMDB sentiment classification: ≥85% test accuracy via auto-config only.
> - Airline-passenger monthly forecast: MAPE within 10% of `statsmodels.SARIMAX`.
> - `predict.exe`, C-ABI DLL, REST server, and ONNX all handle the new
>   model types.

---

## How to use this document

Same ticket format as prior phase docs. Every ticket is self-contained.

### Track assignment

| Track | Tickets | Owner archetype |
|-------|---------|-----------------|
| **A — Tokenization** | P4-01, P4-02 | C++/Python polyglot; must understand BPE algorithmics |
| **B — Transformer core** | P4-03, P4-04, P4-05, P4-06 | ML-engineer with Transformer implementation experience |
| **C — Time-series** | P4-07, P4-08 | Stats/forecasting background |
| **D — Polish** | P4-09, P4-10 | Generalist |

---

## Cross-cutting conventions (new in Phase 4)

- **Token IDs:** `int32_t` throughout. 2^31 tokens is more than any BPE
  vocab we'll ever ship.
- **Padding token:** always ID 0; no real token gets ID 0.
- **Unknown token:** always ID 1; reserved.
- **Sequence tensors:** shape `[B, T]` for token IDs, `[B, T, D]` after
  embedding. Row-major, same NCHW-style contiguous layout as Phase 2.
- **Masks:** separate `[B, T]` int tensor of 0/1. Passed alongside the
  sequence tensor; don't bake padding into the data.
- **Max sequence length:** default 512. Exposed as a pipeline parameter so
  customers with short docs can drop to 128 and save compute.

---

# Phase 4 Tickets

## P4-01 — BPE Tokenizer

- **Size:** L  •  **Deps:** none  •  **Track:** A

### Goal
Train and apply a Byte Pair Encoding tokenizer. Produce a vocabulary file
and a deterministic encode/decode path. No external dependency on
SentencePiece; we want a single portable C++ implementation that can be
embedded in the predict runtime.

### Why it matters
Every text model starts with tokenization. Without our own BPE, we'd have
to bundle a Python runtime or a second binary. A native C++ tokenizer is
small, fast, and keeps the predict runtime self-contained.

### Files
- `include/neurograph/text/bpe.hpp` — NEW
- `include/neurograph/text/vocab.hpp` — NEW
- `src/text/bpe_trainer.cpp` — NEW
- `src/text/bpe_encoder.cpp` — NEW
- `tests/unit/test_bpe.cpp` — NEW

### API
```cpp
namespace neurograph::text {

struct BpeTrainConfig {
    int         vocab_size           = 8000;
    int         min_pair_freq        = 2;
    std::string unk_token            = "<unk>";
    std::string pad_token            = "<pad>";
    std::string bos_token            = "<bos>";
    std::string eos_token            = "<eos>";
    // Normalize: lowercase + NFKC. Off by default; UX sets it for English.
    bool        lowercase            = false;
    bool        unicode_normalize    = false;
    uint64_t    seed                 = 42;
};

class BpeVocab {
public:
    static BpeVocab load(const std::filesystem::path& file);
    void            save(const std::filesystem::path& file) const;

    int id_for(std::string_view token) const;   // returns UNK_ID if missing
    const std::string& token_for(int id) const;
    int size() const;
    int pad_id() const; int unk_id() const; int bos_id() const; int eos_id() const;
};

class BpeTokenizer {
public:
    BpeTokenizer(BpeVocab vocab, std::vector<std::pair<std::string,std::string>> merges);

    std::vector<int> encode(std::string_view text,
                            bool add_special_tokens = true) const;
    std::string      decode(const std::vector<int>& ids,
                            bool skip_special_tokens = true) const;

    // Batched variants for the training loop.
    void encode_batch(const std::vector<std::string>& texts,
                      std::vector<std::vector<int>>& out) const;
};

// Trainer: iterate corpus, merge frequent pairs until vocab full.
std::pair<BpeVocab, std::vector<std::pair<std::string,std::string>>>
train_bpe(const std::vector<std::string>& corpus,
          const BpeTrainConfig& cfg,
          std::function<void(int merge_idx, int total)> progress = {});

} // namespace
```

### Algorithm notes
- **Byte-level BPE** — start from UTF-8 bytes, not unicode codepoints. Handles
  any language cleanly without a pre-tokenizer. Matches the GPT-2 / Llama
  family — our embeddings transfer-learn more easily if customers want to
  warm-start.
- **Merge table:** store as a dense `unordered_map<pair<int,int>, int>`.
  Encode uses a priority queue over pair positions — O(N log N) per token.
- **Deterministic:** merges applied in training-time order (lowest index first
  for ties). Seed the tie-breaking with `cfg.seed`.

### Acceptance criteria
- [ ] Training on a 1 MB corpus in < 10 seconds, 8k vocab.
- [ ] Encode-decode round-trip preserves input exactly (modulo lowercase if
      enabled) for 1000 random English + CJK + emoji strings.
- [ ] Serialized vocab + merges file < 2 MB for 32k vocab.
- [ ] Deterministic: identical corpus + seed → identical vocab.
- [ ] Thread-safe `encode` (training loop uses many threads).

### Test plan
- Unit: round-trip on mixed-language fixture.
- Unit: known-vocab fixture → known token IDs.
- Perf: encode throughput ≥ 50 MB/s on one core.

---

## P4-02 — Text Data Loader

- **Size:** M  •  **Deps:** P4-01  •  **Track:** A

### Goal
Load text datasets in three shapes:
1. Folder of `.txt` files organized by class subfolder (like image loader).
2. A `.jsonl` file with `{"text": "...", "label": "..."}` per line.
3. A 2-column CSV with text + label.

Produce `TextSample` objects; train the tokenizer on the corpus; emit
padded/truncated sequence batches.

### Why it matters
Different customers have different dataset shapes. Supporting all three
common formats is the minimum for avoiding "I don't know how to feed my data
in" support tickets.

### Files
- `include/neurograph/text/text_loader.hpp` — NEW
- `src/text/text_loader.cpp` — NEW
- `include/neurograph/text/batch_collator.hpp` — NEW
- `src/text/batch_collator.cpp` — NEW
- `tests/unit/test_text_loader.cpp` — NEW

### API
```cpp
namespace neurograph::text {

struct TextSample {
    std::string text;
    std::string label;           // class name, possibly empty for regression
    std::vector<int> token_ids;  // filled by encode pass
    std::vector<double> target;  // one-hot / regression
};

struct TextBatch {
    std::vector<std::vector<int>>    token_ids;  // [B][T] — each row padded to T
    std::vector<std::vector<int>>    mask;       // [B][T], 1 for real, 0 for pad
    std::vector<std::vector<double>> targets;
    int                               max_len;
};

struct TextLoaderOptions {
    int   max_sequence_length = 512;
    bool  lowercase           = false;
    bool  shuffle             = true;
    uint64_t seed             = 42;
};

tl::expected<std::vector<TextSample>, std::string>
load_text_folder(const std::filesystem::path& root, const TextLoaderOptions& opts);

tl::expected<std::vector<TextSample>, std::string>
load_text_jsonl(const std::filesystem::path& file, const TextLoaderOptions& opts);

tl::expected<std::vector<TextSample>, std::string>
load_text_csv(const std::filesystem::path& file,
              const std::string& text_col, const std::string& label_col,
              const TextLoaderOptions& opts);

// Apply the tokenizer to every sample in-place.
void encode_all(std::vector<TextSample>& samples, const BpeTokenizer& tok);

// Collate into a padded batch.
TextBatch collate_text(const std::vector<TextSample>& batch,
                       int pad_id, int max_len);

} // namespace
```

### Acceptance criteria
- [ ] Each loader produces identical samples when given an equivalent
      dataset in different formats (content parity check).
- [ ] `max_sequence_length` truncates; shorter samples get `mask=0` in the
      padding positions.
- [ ] UTF-8 handled throughout — sample loader never splits in the middle of
      a codepoint.
- [ ] 100k-sample jsonl loads in < 3 s.

### Test plan
- Unit: each format on a 10-sample fixture; assert parity across formats.
- Unit: mask correctness on a mixed-length batch.

---

## P4-03 — Embedding + Positional Encoding

- **Size:** M  •  **Deps:** P4-02  •  **Track:** B

### Goal
Token embedding table + sinusoidal or learnable positional encoding. Forward
and backward on CPU + GPU. Plugs into the existing training pipeline as a
first layer for text.

### Why it matters
Embeddings are the input-side geometry of every text model. Without a clean
embedding layer, the Transformer work in P4-04/05 has nothing to consume.

### Files
- `include/neurograph/core/embedding_layer.hpp` — NEW
- `src/core/embedding_layer.cpp` — NEW
- `cuda/kernels/embedding_forward.cu` — NEW (gather)
- `cuda/kernels/embedding_backward.cu` — NEW (scatter-add)
- `tests/unit/test_embedding.cpp` — NEW

### API
```cpp
namespace neurograph::core {

enum class PositionalKind { Sinusoidal, Learned };

struct EmbeddingConfig {
    int            vocab_size;
    int            embed_dim;
    int            max_position     = 512;
    PositionalKind positional       = PositionalKind::Sinusoidal;
    bool           scale_embeddings = true;  // multiply by sqrt(d)
    uint64_t       init_seed        = 42;
};

class EmbeddingLayer {
public:
    explicit EmbeddingLayer(const EmbeddingConfig& cfg);

    // x:    [B, T]    (int32 token ids)
    // mask: [B, T]    (optional; ignored rows contribute zero)
    // y:    [B, T, D]
    void forward(const std::vector<std::vector<int>>& x,
                 tensor::Tensor3D& y) const;

    // dy: [B, T, D] → accumulate into dW_tokens [vocab, D] and
    // dW_positions [max_pos, D] (if learned).
    void backward(const std::vector<std::vector<int>>& x,
                  const tensor::Tensor3D& dy,
                  std::vector<float>& dW_tok,
                  std::vector<float>& dW_pos);

    void apply_gradients(const std::vector<float>& dW_tok,
                         const std::vector<float>& dW_pos,
                         float lr);
};

} // namespace
```

### Acceptance criteria
- [ ] Embedding forward matches a reference numpy implementation to 1e-6.
- [ ] Sinusoidal positions: `PE[t, 2i] = sin(t / 10000^(2i/D))`, verified
      against reference.
- [ ] Backward: gradient of same-id token is sum of gradients across all
      positions it appears in (classic embedding backward check).
- [ ] GPU scatter-add deterministic under `NG_DETERMINISTIC=1` (serial
      reduction).

### Test plan
- Unit: forward against reference.
- Unit: grad check via central differences on a tiny example.

---

## P4-04 — Multi-Head Self-Attention

- **Size:** L  •  **Deps:** P4-03  •  **Track:** B

### Goal
Implement multi-head scaled dot-product self-attention. Forward + backward,
CPU + GPU. Support causal masking (for future autoregressive use) and
padding masking.

### Why it matters
This is the core op of the Transformer. Every subsequent text/seq model
is built on this one primitive.

### Files
- `include/neurograph/core/attention_layer.hpp` — NEW
- `src/core/attention_layer.cpp` — NEW
- `cuda/kernels/attention_forward.cu` — NEW
- `cuda/kernels/attention_backward.cu` — NEW
- `tests/unit/test_attention.cpp` — NEW

### API
```cpp
namespace neurograph::core {

struct AttentionConfig {
    int  embed_dim;
    int  num_heads;
    bool causal      = false;
    float dropout    = 0.0f;
    uint64_t init_seed = 42;
};

class MultiHeadAttention {
public:
    explicit MultiHeadAttention(const AttentionConfig& cfg);

    // x:    [B, T, D]
    // mask: [B, T]  — 0 for pad positions
    // out:  [B, T, D]
    void forward(const tensor::Tensor3D& x,
                 const std::vector<std::vector<int>>& mask,
                 tensor::Tensor3D& out) const;

    void backward(const tensor::Tensor3D& x,
                  const tensor::Tensor3D& dout,
                  tensor::Tensor3D& dx,
                  // dW_Q, dW_K, dW_V, dW_O — one gradient per projection
                  std::vector<float>& dW_q, std::vector<float>& dW_k,
                  std::vector<float>& dW_v, std::vector<float>& dW_o);

    void apply_gradients(/* same */ float lr);
};

} // namespace
```

### Algorithm notes
- **Scaling factor:** 1/√d_head.
- **Mask:** add a large negative number (−1e9) to masked logits pre-softmax.
- **Flash-attention deferred:** bog-standard O(T²·D) for v1. Fast enough
  for T ≤ 512 on a single GPU; we can optimize later if needed.
- **Dropout:** deterministic under `NG_DETERMINISTIC=1` (Philox counter-based
  RNG keyed by batch+layer).

### Acceptance criteria
- [ ] Forward matches a PyTorch reference to 1e-5 on `[2, 16, 64]` with 4 heads.
- [ ] Backward gradients pass central-difference check.
- [ ] Causal mask: output at position t is independent of inputs at t+1..T.
- [ ] Padding mask: output at padded positions has no effect on unmasked
      positions' outputs.

### Test plan
- Unit: PyTorch parity on one fixture.
- Unit: masking behavior unit-tested.
- Unit: gradient check.

---

## P4-05 — Transformer Encoder Block

- **Size:** L  •  **Deps:** P4-04  •  **Track:** B

### Goal
Stackable encoder block = LayerNorm → MHA → residual → LayerNorm →
feed-forward (2-layer MLP with GELU) → residual. Pre-norm variant (more
stable training than post-norm for small models).

### Why it matters
Single-attention layers rarely do much on their own — the residual +
feed-forward pattern is what makes Transformers learn.

### Files
- `include/neurograph/core/transformer_block.hpp` — NEW
- `src/core/transformer_block.cpp` — NEW
- `include/neurograph/core/layer_norm.hpp` — NEW
- `src/core/layer_norm.cpp` — NEW
- `tests/unit/test_transformer_block.cpp` — NEW

### API
```cpp
namespace neurograph::core {

struct TransformerBlockConfig {
    int   embed_dim;
    int   num_heads;
    int   ffn_hidden;            // typically 4 × embed_dim
    float dropout = 0.1f;
    uint64_t init_seed = 42;
};

class TransformerBlock {
public:
    explicit TransformerBlock(const TransformerBlockConfig& cfg);

    void forward(const tensor::Tensor3D& x,
                 const std::vector<std::vector<int>>& mask,
                 tensor::Tensor3D& out) const;

    void backward(/* same shape conventions */);
    void apply_gradients(float lr);
};

} // namespace
```

### Acceptance criteria
- [ ] One block trains to low loss on the toy copy task (input sequence =
      output sequence after pooling) within 1000 steps.
- [ ] Stacking 4 blocks doesn't introduce NaN on long training runs
      (residual + pre-norm stability).
- [ ] LayerNorm matches PyTorch reference to 1e-6.

### Test plan
- Unit: forward shape + numerical parity on a tiny example.
- Integration: the toy copy task trains to expected loss.

---

## P4-06 — Small-Transformer Architecture

- **Size:** M  •  **Deps:** P4-05  •  **Track:** B

### Goal
Auto-build a Transformer-classifier network from dataset profile. Presets
by size:
- **tiny** (1 block, 64 dim, 2 heads) — tiny / simple tasks.
- **small** (2 blocks, 128 dim, 4 heads) — default for text classification.
- **medium** (4 blocks, 256 dim, 8 heads) — >100k samples, longer texts.

Classification head = mean pooling over non-pad positions → dense → softmax.

### Why it matters
Auto-config must synthesize the model. We do not ask users to pick head/dim/depth.

### Files
- `include/neurograph/pipeline/transformer_builder.hpp` — NEW
- `src/pipeline/transformer_builder.cpp` — NEW
- `tests/integration/test_transformer_autobuild.cpp` — NEW

### API
```cpp
namespace neurograph::pipeline {

struct TransformerBlueprint {
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_blocks;
    int max_position;
    int num_classes;
    std::string preset_name;
};

TransformerBlueprint build_text_transformer(int vocab_size,
                                            int num_samples,
                                            int max_sequence_length,
                                            int num_classes);

std::unique_ptr<core::Network>
instantiate_transformer(const TransformerBlueprint& bp, uint64_t seed);

} // namespace
```

### Acceptance criteria
- [ ] Preset picked correctly by dataset size thresholds.
- [ ] Forward-pass shape correct on `[B, T]` input.
- [ ] Parameter count within 10% of the reference for each preset (e.g. small
      ≈ 1.2 M params).

### Test plan
- Unit: preset selection.
- Integration: 5-epoch SST-2 subset training reaches ≥70% val (easy gate).

---

## P4-07 — Sliding-Window Time-Series Adapter

- **Size:** M  •  **Deps:** P1-02  •  **Track:** C

### Goal
Given a date-indexed CSV with one or more numeric columns, produce supervised
samples via sliding window. Each window = `(x_past[L], y_future[H])` where L
is the lookback length and H is the forecast horizon.

### Why it matters
Time-series forecasting is the other high-value non-tabular tabular task.
Framing it as supervised windows lets us reuse the rest of the pipeline —
just needs this adapter.

### Files
- `include/neurograph/timeseries/window_adapter.hpp` — NEW
- `src/timeseries/window_adapter.cpp` — NEW
- `tests/unit/test_window_adapter.cpp` — NEW

### API
```cpp
namespace neurograph::timeseries {

struct WindowConfig {
    int     lookback_length   = 24;    // e.g. 24 hours / 24 months
    int     forecast_horizon  = 1;
    int     stride            = 1;
    bool    include_target_in_past = true;   // autoregressive
    std::vector<std::string> feature_columns;
    std::string              target_column;
    std::string              date_column;     // optional
    double  val_fraction  = 0.2;   // last 20% for validation (temporal split!)
};

struct TimeSeriesDataset {
    std::vector<std::vector<double>> x_train, x_val;
    std::vector<std::vector<double>> y_train, y_val;
    std::vector<std::string>         feature_names;  // expanded per lag
};

tl::expected<TimeSeriesDataset, std::string>
windowize_csv(const std::filesystem::path& csv, const WindowConfig& cfg);

} // namespace
```

### Algorithm notes
- **Temporal split, not random.** Last X% by date → validation. Random split
  leaks future info and trivially overfits — a classic ts pitfall.
- **Feature flattening:** [L lookback, F features] → 1D vector of L*F. Keeps
  downstream training code unchanged.
- **Missing date handling:** warn but accept. Use index position if no date
  column given.

### Acceptance criteria
- [ ] 1000-row dataset with stride 1, L=24, H=1 → 976 samples.
- [ ] Temporal split: every val date strictly later than every train date.
- [ ] Handles multiple exogenous features + 1 target; feature names include
      lag suffixes like `temp_lag_12`.

### Test plan
- Unit: airline-passengers fixture → known number of windows.
- Unit: val is temporally after train.

---

## P4-08 — Seasonality + Trend Features

- **Size:** S  •  **Deps:** P4-07  •  **Track:** C

### Goal
Compute classic time-series features automatically and append them:
- Fourier harmonics of the date index (seasonality).
- Lag differences.
- Rolling mean/stddev.
- One-hot of day-of-week, month, quarter if a date column exists.

### Why it matters
Pure neural nets often struggle on time series without hand-engineered
seasonality features. These cheap additions typically double out-of-the-box
accuracy.

### Files
- `include/neurograph/timeseries/seasonal_features.hpp` — NEW
- `src/timeseries/seasonal_features.cpp` — NEW
- `tests/unit/test_seasonal_features.cpp` — NEW

### API
```cpp
namespace neurograph::timeseries {

struct SeasonalFeatureConfig {
    int  num_fourier_terms  = 4;     // gives 2*N columns (sin+cos)
    std::vector<int> period_days = {7, 30, 365}; // weekly, monthly, yearly
    std::vector<int> rolling_windows = {7, 28};
    std::vector<int> lag_diffs       = {1, 7};
    bool include_dow_month_quarter = true;
};

void augment_with_seasonal(std::vector<std::vector<double>>& rows,
                           std::vector<std::string>& feature_names,
                           const SeasonalFeatureConfig& cfg,
                           const std::vector<std::chrono::sys_days>& dates);

} // namespace
```

### Acceptance criteria
- [ ] Adding seasonal features improves MAPE by ≥20% on the airline-passenger
      benchmark vs no seasonal features (same model).
- [ ] Features are deterministic functions of the date — no random state.

### Test plan
- Unit: hand-compute features for a 10-date series; assert exact match.
- Integration: airline-passengers benchmark asserts MAPE gate.

---

## P4-09 — Sequence Metrics (BLEU / ROUGE) — Optional

- **Size:** M  •  **Deps:** P4-01, P4-05  •  **Track:** D

### Goal
BLEU-4, ROUGE-L, and optionally ChrF metrics for sequence-to-sequence
outputs. Only relevant if we ship seq2seq in Phase 4 (we might not in v1 —
classification-only could be enough).

### Why it matters
Only matters if seq2seq ships. Spec here so scope is captured if a customer
demand materializes.

### Files
- `include/neurograph/eval/sequence_metrics.hpp` — NEW
- `src/eval/sequence_metrics.cpp` — NEW
- `tests/unit/test_sequence_metrics.cpp` — NEW

### API
```cpp
namespace neurograph::eval {

struct SequenceMetrics {
    double bleu4       = 0.0;
    double rouge_l     = 0.0;
    double chrf        = 0.0;
    double length_ratio = 0.0;  // pred / ref average length
};

SequenceMetrics evaluate_sequences(
    const std::vector<std::vector<int>>& predictions,
    const std::vector<std::vector<int>>& references);

} // namespace
```

### Acceptance criteria
- [ ] Matches `sacrebleu` within 1e-4 on the sacrebleu test vectors.
- [ ] Handles identical inputs → BLEU=1.0.

### Test plan
- Unit: sacrebleu golden parity.

---

## P4-10 — Text Augmentation — Optional

- **Size:** S  •  **Deps:** P4-01  •  **Track:** D

### Goal
Cheap text augmentation strategies for classifiers:
- Synonym replacement (needs a thesaurus — use WordNet English only).
- Random word deletion.
- Random word swap.
- Back-translation (requires an external translation model — defer unless
  customer demand).

### Why it matters
Modest accuracy boost on small labelled datasets (< 10k samples). Optional;
we can ship classification without it.

### Files
- `include/neurograph/text/augmentation.hpp` — NEW
- `src/text/augmentation.cpp` — NEW
- `tests/unit/test_text_aug.cpp` — NEW

### API
```cpp
namespace neurograph::text {

struct TextAugConfig {
    double synonym_replace_prob = 0.1;
    double random_delete_prob   = 0.1;
    double random_swap_prob     = 0.0;
    uint64_t seed = 42;
};

std::string augment_text(const std::string& text, const TextAugConfig& cfg,
                         std::mt19937& rng);

} // namespace
```

### Acceptance criteria
- [ ] Deterministic given rng state.
- [ ] Augmented output is close in length to input (within ±20%).
- [ ] Does not break tokenization (round-trip OK afterward).

### Test plan
- Unit: deterministic output check.

---

# Phase 4 milestones

| Week | Milestones |
|------|-----------|
| 1 | P4-01 BPE trainer + encoder (single-thread) |
| 2 | P4-01 BPE batched + GPU; P4-02 text loaders all three formats |
| 3 | P4-03 embedding layer CPU+GPU; P4-07 windowing |
| 4 | P4-04 attention forward parity |
| 5 | P4-04 attention backward; P4-05 transformer block |
| 6 | P4-06 builder ships tiny/small/medium; SST-2 smoke test |
| 7 | P4-08 seasonal features; airline forecast gate met |
| 8 | IMDB 85% gate met; P4-09 / P4-10 if time |
| 9 | Export paths: ONNX covers attention ops (update P3-02 coverage) |
| 10 | RC; pilot customer test |

# Definition of Done for Phase 4

- [ ] IMDB sentiment ≥ 85% test accuracy via auto-config only.
- [ ] Airline-passengers MAPE within 10% of `statsmodels.SARIMAX`.
- [ ] Exports work: ONNX export of a small Transformer verified under
      onnxruntime.
- [ ] Workbench wizard accepts: text folder, `.jsonl`, text CSV, time-series CSV.
- [ ] Docs: `docs/user-guide-text.md`, `docs/user-guide-timeseries.md`.

---

*Next doc: [`phase5-enterprise.md`](phase5-enterprise.md) — multi-GPU,
explainability, hyperparameter search, cloud SaaS.*
