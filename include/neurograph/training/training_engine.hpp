#pragma once

#include "neurograph/core/network.hpp"
#include "neurograph/training/data_loader.hpp"
#include "neurograph/training/io_manager.hpp"
#include "neurograph/training/backprop.hpp"
#include "neurograph/training/loss.hpp"
#include "neurograph/training/optimizer.hpp"
#include "neurograph/training/embedding.hpp"
#include "neurograph/training/augmentation.hpp"
#include "neurograph/util/result.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

namespace neurograph::training {

struct SupervisedConfig {
    int epochs = 100;
    size_t batch_size = 32;
    double test_ratio = 0.2;
    LossType loss_type = LossType::CROSS_ENTROPY;
    std::string optimizer_name = "adam";
    OptimizerConfig optimizer_config;

    // Self-organization interleaving
    bool interleave_self_org = false;
    int self_org_interval = 10;    // every N epochs
    int self_org_steps = 1;
    /// P1-20: when true, the training loop calls
    /// `Network::growth_only_self_organize` instead of the full
    /// `Network::self_organize`. The growth-only variant runs only the
    /// phases that don't fight a concurrent supervised gradient
    /// (prune-synapses + prune-dead-neurons + grow-stressed). Required
    /// for stable interleaving with backprop. Defaults to false to
    /// preserve legacy `load_dataset`-driven training behavior.
    bool self_org_growth_only = false;

    // Learning rate schedule (forwarded to OptimizerConfig)
    std::string schedule_name = "cosine";   // "constant", "cosine", "exponential"
    double min_lr = 1e-6;
    double decay_factor = 0.95;
    int warmup_epochs = 5;

    // Dropout
    double dropout_rate = 0.2;

    // Batch normalization
    bool batch_norm = true;

    // Early stopping
    int patience = 20;
    double min_delta = 0.001;

    // Embedding for categorical features
    EmbeddingConfig embedding;

    // Data augmentation
    AugmentationConfig augmentation;
};

struct LRFinderResult {
    std::vector<double> learning_rates;
    std::vector<double> losses;
    double suggested_lr = 0.001;
};

struct EpochResult {
    int epoch = 0;
    double train_loss = 0.0;
    double train_accuracy = 0.0;
    double val_loss = 0.0;
    double val_accuracy = 0.0;
    double epoch_duration_ms = 0.0;
};

using TrainingCallback = std::function<void(const EpochResult&)>;

class TrainingEngine {
public:
    explicit TrainingEngine(core::Network& network);

    /// Load dataset from CSV, set up IO neurons, normalize data
    util::Result<void> load_dataset(const std::string& csv_path,
                                     int num_target_cols = 1,
                                     bool has_header = true);

    /// Load MNIST from IDX format files
    util::Result<void> load_mnist(const std::string& images_path,
                                   const std::string& labels_path,
                                   int max_samples = 0);

    /// Load images from a folder structure (subfolders = classes, .bmp files)
    util::Result<void> load_image_folder(const std::string& root_path,
                                          int target_width, int target_height,
                                          int max_samples = 0);

    /// Sentinel for `load_prepared_dataset`'s `hidden_layer_size`
    /// parameter — picks a heuristic size based on `num_features`.
    /// See the function's doc comment for the formula.
    static constexpr int kAutoHiddenSize = -1;

    /// Load pre-split, pre-normalized data produced by the preprocess pipeline.
    /// Bypasses CSV loading and normalization — the caller has already done both.
    /// norm_stats are stored as identity transforms so predict() is a pass-through.
    /// seed governs the initial random weight wiring only.
    ///
    /// `hidden_layer_size` (P1-18):
    ///   - `kAutoHiddenSize` (default): allocate `clamp(2 * num_features, 8, 64)`
    ///     ReLU hidden neurons fully-connected between input and output.
    ///     Glorot-uniform initial weights for stable gradient flow.
    ///   - `0`: legacy behavior — no hidden layer, inputs wired directly
    ///     to outputs. Tests that explicitly want a single linear layer
    ///     pass 0; production callers leave the default.
    ///   - `> 0`: allocate exactly that many ReLU hidden neurons.
    util::Result<void> load_prepared_dataset(
        std::vector<Sample> train_samples,
        std::vector<Sample> val_samples,
        int num_features,
        int num_outputs,
        const std::vector<std::string>& class_labels = {},
        std::uint64_t seed = 42,
        int hidden_layer_size = kAutoHiddenSize);

    /// Run LR finder: sweep learning rate from min_lr to max_lr over num_steps
    /// batches, recording loss at each step. Returns suggested LR.
    LRFinderResult find_learning_rate(double min_lr = 1e-7, double max_lr = 1.0,
                                       int num_steps = 100);

    /// Run supervised training loop (blocking). Call from a background thread.
    util::Result<void> train(const SupervisedConfig& config,
                              TrainingCallback callback = nullptr);

    /// Stop training (thread-safe)
    void request_stop();

    /// Inference: return raw output activations
    std::vector<double> predict(const std::vector<double>& features);

    /// Inference: return predicted class index
    int predict_class(const std::vector<double>& features);

    /// Get predicted class label
    std::string predict_label(const std::vector<double>& features);

    // State queries
    bool is_training() const { return training_.load(); }
    int current_epoch() const { return current_epoch_.load(); }
    bool has_dataset() const { return !all_data_.empty(); }
    int num_features() const { return io_manager_ ? io_manager_->num_features() : 0; }
    int num_outputs() const { return io_manager_ ? io_manager_->num_outputs() : 0; }
    int num_samples() const { return static_cast<int>(all_data_.size()); }
    int num_train_samples() const { return static_cast<int>(split_.train.size()); }
    int num_test_samples() const { return static_cast<int>(split_.test.size()); }

    const std::vector<EpochResult>& history() const { return history_; }
    const std::vector<std::string>& class_labels() const;
    const std::vector<NormStats>& norm_stats() const { return norm_stats_; }

    IOManager* io_manager() { return io_manager_.get(); }

private:
    core::Network& network_;
    std::unique_ptr<IOManager> io_manager_;
    std::unique_ptr<Backprop> backprop_;
    std::unique_ptr<LossFunction> loss_;
    std::unique_ptr<Optimizer> optimizer_;

    std::vector<Sample> all_data_;
    DataSplit split_;
    std::vector<NormStats> norm_stats_;

    std::unique_ptr<Augmentor> augmentor_;

    std::vector<EpochResult> history_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> training_{false};
    std::atomic<int> current_epoch_{0};

    // Run one epoch of training
    EpochResult run_epoch(int epoch_num, const std::vector<DataBatch>& batches);

    // Evaluate on a dataset (forward pass only)
    void evaluate(const std::vector<Sample>& data, double& loss, double& accuracy);
};

} // namespace neurograph::training
