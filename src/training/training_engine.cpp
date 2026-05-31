#include "neurograph/training/training_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>
#include <unordered_map>

namespace neurograph::training {

TrainingEngine::TrainingEngine(core::Network& network)
    : network_(network) {}

util::Result<void> TrainingEngine::load_dataset(const std::string& csv_path,
                                                  int num_target_cols,
                                                  bool has_header) {
    auto result = DataLoader::load_csv(csv_path, num_target_cols, has_header, true);
    if (!result.ok())
        return util::Error("Failed to load CSV: " + result.error().message);

    all_data_ = std::move(result.value());
    if (all_data_.empty())
        return util::Error("Dataset is empty");

    int num_features = static_cast<int>(all_data_[0].features.size());
    int num_outputs = static_cast<int>(all_data_[0].target.size());

    if (num_features == 0)
        return util::Error("No features found in dataset");
    if (num_outputs == 0)
        return util::Error("No targets found in dataset");

    // Compute normalization stats and normalize
    norm_stats_ = DataLoader::compute_stats(all_data_);
    DataLoader::normalize_minmax(all_data_, norm_stats_);

    // Set up IO manager with input/output neurons
    io_manager_ = std::make_unique<IOManager>(network_);
    auto io_result = io_manager_->ensure_io_neurons(
        num_features, num_outputs, DataLoader::last_class_labels());
    if (!io_result.ok())
        return util::Error("Failed to create IO neurons: " + io_result.error().message);

    // Connect input neurons to existing hidden neurons if no connections exist
    // and connect hidden neurons to output neurons
    auto& input_ids = io_manager_->input_ids();
    auto& output_ids = io_manager_->output_ids();

    // Gather hidden neuron IDs (neurons that are neither input nor output)
    std::unordered_set<core::NeuronId> io_set;
    for (auto& id : input_ids) io_set.insert(id);
    for (auto& id : output_ids) io_set.insert(id);

    std::vector<core::NeuronId> hidden_ids;
    for (auto& [id, _] : network_.neurons()) {
        if (!io_set.count(id)) hidden_ids.push_back(id);
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> wt(0.01, 0.5);

    if (hidden_ids.empty()) {
        // No hidden neurons — connect inputs directly to outputs
        for (auto& in_id : input_ids) {
            for (auto& out_id : output_ids) {
                network_.connect(in_id, out_id, wt(rng));
            }
        }
    } else {
        // Connect each input to a subset of hidden neurons
        int max_per_input = std::max(1, static_cast<int>(hidden_ids.size()) / 2);
        for (auto& in_id : input_ids) {
            std::shuffle(hidden_ids.begin(), hidden_ids.end(), rng);
            int count = std::min(max_per_input, static_cast<int>(hidden_ids.size()));
            for (int i = 0; i < count; ++i) {
                // Only connect if no existing connection
                network_.connect(in_id, hidden_ids[i], wt(rng));
            }
        }

        // Connect hidden neurons to outputs
        for (auto& hid_id : hidden_ids) {
            for (auto& out_id : output_ids) {
                network_.connect(hid_id, out_id, wt(rng));
            }
        }
    }

    // Split data
    split_ = DataLoader::train_test_split(all_data_, 0.2, true, 42);

    history_.clear();
    return util::Result<void>::success();
}

util::Result<void> TrainingEngine::load_mnist(const std::string& images_path,
                                               const std::string& labels_path,
                                               int max_samples) {
    auto result = DataLoader::load_mnist_idx(images_path, labels_path, max_samples);
    if (!result.ok())
        return util::Error("Failed to load MNIST: " + result.error().message);

    all_data_ = std::move(result.value());
    if (all_data_.empty())
        return util::Error("MNIST dataset is empty");

    // MNIST pixels are already 0-1 from the loader, skip normalization
    // but still set up dummy stats for inference
    int num_features = 784;
    int num_outputs = 10;
    norm_stats_.resize(num_features);
    for (auto& s : norm_stats_) { s.min_val = 0.0; s.max_val = 1.0; s.mean = 0.0; s.stddev = 1.0; }

    io_manager_ = std::make_unique<IOManager>(network_);
    auto io_result = io_manager_->ensure_io_neurons(
        num_features, num_outputs, DataLoader::last_class_labels());
    if (!io_result.ok())
        return util::Error("Failed to create IO neurons: " + io_result.error().message);

    // Wire up the network
    auto& input_ids = io_manager_->input_ids();
    auto& output_ids = io_manager_->output_ids();

    std::unordered_set<core::NeuronId> io_set;
    for (auto& id : input_ids) io_set.insert(id);
    for (auto& id : output_ids) io_set.insert(id);

    std::vector<core::NeuronId> hidden_ids;
    for (auto& [id, _] : network_.neurons()) {
        if (!io_set.count(id)) hidden_ids.push_back(id);
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> wt(0.01, 0.3);

    if (hidden_ids.empty()) {
        // Direct input→output connections
        for (auto& in_id : input_ids) {
            for (auto& out_id : output_ids) {
                network_.connect(in_id, out_id, wt(rng));
            }
        }
    } else {
        // Fully connected: all inputs -> all hidden, all hidden -> all outputs
        for (auto& in_id : input_ids) {
            for (auto& hid_id : hidden_ids)
                network_.connect(in_id, hid_id, wt(rng));
        }
        for (auto& hid_id : hidden_ids) {
            for (auto& out_id : output_ids)
                network_.connect(hid_id, out_id, wt(rng));
        }
    }

    split_ = DataLoader::train_test_split(all_data_, 0.2, true, 42);
    history_.clear();
    return util::Result<void>::success();
}

util::Result<void> TrainingEngine::load_image_folder(const std::string& root_path,
                                                      int target_width, int target_height,
                                                      int max_samples) {
    auto result = DataLoader::load_image_folder(root_path, target_width, target_height, max_samples);
    if (!result.ok())
        return util::Error("Failed to load image folder: " + result.error().message);

    all_data_ = std::move(result.value());
    if (all_data_.empty())
        return util::Error("No images found in folder");

    int num_features = target_width * target_height;
    int num_outputs = 0;
    if (!all_data_.empty()) num_outputs = static_cast<int>(all_data_[0].target.size());

    // Already normalized to [0,1] by the loader
    norm_stats_.resize(num_features);
    for (auto& s : norm_stats_) { s.min_val = 0.0; s.max_val = 1.0; s.mean = 0.0; s.stddev = 1.0; }

    io_manager_ = std::make_unique<IOManager>(network_);
    auto io_result = io_manager_->ensure_io_neurons(
        num_features, num_outputs, DataLoader::last_class_labels());
    if (!io_result.ok())
        return util::Error("Failed to create IO neurons: " + io_result.error().message);

    // Wire up (same as MNIST — direct input→output, self-org adds hidden later)
    auto& input_ids = io_manager_->input_ids();
    auto& output_ids = io_manager_->output_ids();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> wt(0.01, 0.3);

    for (auto& in_id : input_ids) {
        for (auto& out_id : output_ids) {
            network_.connect(in_id, out_id, wt(rng));
        }
    }

    split_ = DataLoader::train_test_split(all_data_, 0.2, true, 42);
    history_.clear();
    return util::Result<void>::success();
}

LRFinderResult TrainingEngine::find_learning_rate(double min_lr, double max_lr,
                                                   int num_steps) {
    LRFinderResult result;

    if (!io_manager_ || all_data_.empty()) {
        result.suggested_lr = 0.001;
        return result;
    }

    // ---- Save all synapse weights so we can restore them afterwards ----
    std::unordered_map<core::SynapseId, double> saved_weights;
    for (auto& [sid, syn] : network_.synapses()) {
        saved_weights[sid] = syn.weight();
    }

    // ---- Create temporary optimizer (Adam) and backprop ----
    OptimizerConfig opt_cfg;
    opt_cfg.learning_rate = min_lr;
    opt_cfg.schedule_name = "constant";
    auto temp_optimizer = make_optimizer("adam", opt_cfg);

    auto temp_backprop = std::make_unique<Backprop>(network_, *io_manager_);
    temp_backprop->set_dropout_rate(0.0);
    temp_backprop->set_batch_norm(false);
    temp_backprop->set_training_mode(true);
    temp_backprop->rebuild_topo_order();

    auto temp_loss = make_loss(LossType::CROSS_ENTROPY);

    // ---- Shuffle training data and make batches of size 32 ----
    std::vector<Sample> shuffled = split_.train.empty() ? all_data_ : split_.train;
    std::mt19937 rng(42);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    auto batches = DataLoader::make_batches(shuffled, 32);

    if (batches.empty()) {
        // Restore weights and return default
        for (auto& [sid, w] : saved_weights) {
            auto* syn = network_.get_synapse(sid);
            if (syn) syn->set_weight(w);
        }
        result.suggested_lr = 0.001;
        return result;
    }

    // ---- Sweep learning rate ----
    double min_loss = 1e30;
    int batch_idx = 0;

    for (int step = 0; step < num_steps; ++step) {
        // Compute current lr on exponential schedule
        double lr = min_lr * std::pow(max_lr / min_lr,
                                       static_cast<double>(step) / (num_steps - 1));
        temp_optimizer->set_learning_rate(lr);

        // Pick the next batch (wrap around if we run out)
        auto& batch = batches[batch_idx % batches.size()];
        batch_idx++;

        // Compute gradients for this batch
        GradientMap batch_grads;
        double batch_loss = 0.0;

        for (auto& sample : batch.samples) {
            double sample_loss = 0.0;
            auto grads = temp_backprop->compute_gradients(sample, *temp_loss, sample_loss);
            Backprop::accumulate(batch_grads, grads);
            batch_loss += sample_loss;
        }

        double avg_loss = batch_loss / static_cast<double>(batch.samples.size());
        Backprop::scale(batch_grads, 1.0 / static_cast<double>(batch.samples.size()));

        // Apply optimizer step
        temp_optimizer->step(network_, batch_grads);

        // Record (lr, loss)
        result.learning_rates.push_back(lr);
        result.losses.push_back(avg_loss);

        if (avg_loss < min_loss) min_loss = avg_loss;

        // Stop if loss is diverging (> 4x minimum)
        if (avg_loss > 4.0 * min_loss && min_loss < 1e29) break;
    }

    // ---- Find suggested LR: where smoothed loss decreased fastest ----
    if (result.losses.size() > 2) {
        // Simple exponential moving average for smoothing
        std::vector<double> smoothed(result.losses.size());
        smoothed[0] = result.losses[0];
        double beta = 0.9;
        for (size_t i = 1; i < result.losses.size(); ++i) {
            smoothed[i] = beta * smoothed[i - 1] + (1.0 - beta) * result.losses[i];
        }

        // Find steepest negative slope
        double best_slope = 0.0;
        size_t best_idx = 0;
        for (size_t i = 1; i < smoothed.size(); ++i) {
            double slope = smoothed[i] - smoothed[i - 1];
            if (slope < best_slope) {
                best_slope = slope;
                best_idx = i;
            }
        }

        // Divide by 10 for safety margin
        result.suggested_lr = result.learning_rates[best_idx] / 10.0;
    } else {
        result.suggested_lr = 0.001;
    }

    // ---- Restore all synapse weights ----
    for (auto& [sid, w] : saved_weights) {
        auto* syn = network_.get_synapse(sid);
        if (syn) syn->set_weight(w);
    }

    return result;
}

util::Result<void> TrainingEngine::train(const SupervisedConfig& config,
                                          TrainingCallback callback) {
    if (!io_manager_)
        return util::Error("No dataset loaded. Call load_dataset() first.");

    training_ = true;
    stop_requested_ = false;
    current_epoch_ = 0;

    loss_ = make_loss(config.loss_type);

    // Forward schedule settings from SupervisedConfig into OptimizerConfig
    OptimizerConfig opt_cfg = config.optimizer_config;
    opt_cfg.schedule_name = config.schedule_name;
    opt_cfg.min_lr = config.min_lr;
    opt_cfg.decay_factor = config.decay_factor;
    opt_cfg.warmup_epochs = config.warmup_epochs;
    optimizer_ = make_optimizer(config.optimizer_name, opt_cfg);
    backprop_ = std::make_unique<Backprop>(network_, *io_manager_);
    backprop_->set_dropout_rate(config.dropout_rate);
    backprop_->set_batch_norm(config.batch_norm);
    backprop_->set_training_mode(true);
    backprop_->rebuild_topo_order();

    // Set up data augmentation
    if (config.augmentation.enabled) {
        augmentor_ = std::make_unique<Augmentor>(config.augmentation);
    } else {
        augmentor_.reset();
    }

    if (backprop_->topo_order().empty()) {
        training_ = false;
        return util::Error("No path from input to output neurons. Cannot train.");
    }

    // Register IO neurons as protected (self-org must never prune them)
    {
        std::unordered_set<core::NeuronId> io_set;
        for (auto& id : io_manager_->input_ids()) io_set.insert(id);
        for (auto& id : io_manager_->output_ids()) io_set.insert(id);
        network_.set_protected_neurons(io_set);
    }

    // Early stopping state
    double best_val_loss = 1e30;
    int patience_counter = 0;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        if (stop_requested_.load()) break;
        current_epoch_ = epoch;

        // Update learning rate for this epoch
        optimizer_->update_lr(epoch, config.epochs);

        // Shuffle training data each epoch
        std::mt19937 rng(epoch + 42);
        std::shuffle(split_.train.begin(), split_.train.end(), rng);
        auto batches = DataLoader::make_batches(split_.train, config.batch_size);

        auto result = run_epoch(epoch, batches);

        // Evaluate on validation set (disable dropout)
        backprop_->set_training_mode(false);
        evaluate(split_.test, result.val_loss, result.val_accuracy);
        backprop_->set_training_mode(true);

        // Report loss to network for adaptive topology decisions
        network_.report_training_loss(result.val_loss);

        history_.push_back(result);
        if (callback) callback(result);

        // Interleave self-organization (now with adaptive topology).
        // P1-20: route through `growth_only_self_organize` when the
        // caller has opted into it — that variant skips the rewire +
        // Hebbian-strengthen + activation-mutation phases that
        // overwrite weights the optimizer has been tuning.
        if (config.interleave_self_org &&
            epoch > 0 && (epoch % config.self_org_interval == 0)) {
            if (config.self_org_growth_only) {
                network_.growth_only_self_organize(config.self_org_steps);
            } else {
                network_.self_organize(config.self_org_steps);
            }
            backprop_->rebuild_topo_order();
        }

        // Early stopping
        if (result.val_loss < best_val_loss - config.min_delta) {
            best_val_loss = result.val_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= config.patience) break;
        }
    }

    training_ = false;
    return util::Result<void>::success();
}

EpochResult TrainingEngine::run_epoch(int epoch_num,
                                       const std::vector<DataBatch>& batches) {
    auto start = std::chrono::high_resolution_clock::now();

    double total_loss = 0.0;
    int total_correct = 0;
    int total_samples = 0;

    for (auto& batch : batches) {
        if (stop_requested_.load()) break;

        GradientMap batch_grads;
        double batch_loss = 0.0;

        for (auto& sample : batch.samples) {
            // Apply data augmentation if enabled
            const Sample& effective_sample = augmentor_
                ? [&]() -> const Sample& {
                      thread_local Sample augmented;
                      augmented = augmentor_->augment(sample);
                      return augmented;
                  }()
                : sample;

            double sample_loss = 0.0;
            auto grads = backprop_->compute_gradients(effective_sample, *loss_, sample_loss);
            Backprop::accumulate(batch_grads, grads);
            batch_loss += sample_loss;

            // Check accuracy (argmax match)
            auto predicted = io_manager_->read_outputs();
            int pred_class = static_cast<int>(
                std::max_element(predicted.begin(), predicted.end()) - predicted.begin());
            int true_class = static_cast<int>(
                std::max_element(effective_sample.target.begin(), effective_sample.target.end()) - effective_sample.target.begin());
            if (pred_class == true_class) total_correct++;
            total_samples++;
        }

        // Average gradients over batch
        double batch_size = static_cast<double>(batch.samples.size());
        Backprop::scale(batch_grads, 1.0 / batch_size);

        // Update weights
        optimizer_->step(network_, batch_grads);

        total_loss += batch_loss;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    EpochResult result;
    result.epoch = epoch_num;
    result.train_loss = (total_samples > 0) ? total_loss / total_samples : 0.0;
    result.train_accuracy = (total_samples > 0)
        ? static_cast<double>(total_correct) / total_samples : 0.0;
    result.epoch_duration_ms = duration;
    return result;
}

void TrainingEngine::evaluate(const std::vector<Sample>& data,
                               double& loss_out, double& accuracy_out) {
    if (!backprop_ || !loss_ || data.empty()) {
        loss_out = 0.0;
        accuracy_out = 0.0;
        return;
    }

    double total_loss = 0.0;
    int correct = 0;

    for (auto& sample : data) {
        auto state = backprop_->forward(sample.features);
        auto predicted = io_manager_->read_outputs();

        total_loss += loss_->compute(predicted, sample.target);

        int pred_class = static_cast<int>(
            std::max_element(predicted.begin(), predicted.end()) - predicted.begin());
        int true_class = static_cast<int>(
            std::max_element(sample.target.begin(), sample.target.end()) - sample.target.begin());
        if (pred_class == true_class) correct++;
    }

    loss_out = total_loss / data.size();
    accuracy_out = static_cast<double>(correct) / data.size();
}

void TrainingEngine::request_stop() {
    stop_requested_ = true;
}

std::vector<double> TrainingEngine::predict(const std::vector<double>& features) {
    if (!io_manager_ || !backprop_) return {};

    // Normalize input using saved stats
    std::vector<double> normalized = features;
    for (size_t i = 0; i < normalized.size() && i < norm_stats_.size(); ++i) {
        double range = norm_stats_[i].max_val - norm_stats_[i].min_val;
        if (range < 1e-12) range = 1.0;
        normalized[i] = (normalized[i] - norm_stats_[i].min_val) / range;
    }

    backprop_->set_training_mode(false);
    backprop_->forward(normalized);
    return io_manager_->read_outputs();
}

int TrainingEngine::predict_class(const std::vector<double>& features) {
    auto out = predict(features);
    if (out.empty()) return -1;
    return static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
}

std::string TrainingEngine::predict_label(const std::vector<double>& features) {
    int cls = predict_class(features);
    auto& labels = class_labels();
    if (cls >= 0 && cls < static_cast<int>(labels.size()))
        return labels[cls];
    return "class_" + std::to_string(cls);
}

const std::vector<std::string>& TrainingEngine::class_labels() const {
    if (io_manager_) return io_manager_->class_labels();
    static std::vector<std::string> empty;
    return empty;
}

util::Result<void> TrainingEngine::load_prepared_dataset(
    std::vector<Sample> train_samples,
    std::vector<Sample> val_samples,
    int num_features,
    int num_outputs,
    const std::vector<std::string>& class_labels,
    std::uint64_t seed,
    int hidden_layer_size) {

    if (train_samples.empty())
        return util::Error("load_prepared_dataset: train set is empty");
    if (val_samples.empty())
        return util::Error("load_prepared_dataset: val set is empty");
    if (num_features < 1)
        return util::Error("load_prepared_dataset: num_features must be >= 1");
    if (num_outputs < 1)
        return util::Error("load_prepared_dataset: num_outputs must be >= 1");
    if (hidden_layer_size < 0 && hidden_layer_size != kAutoHiddenSize)
        return util::Error("load_prepared_dataset: hidden_layer_size must be "
                           ">= 0 or kAutoHiddenSize (-1)");

    // P1-18: resolve auto sizing now so the rest of the function deals
    // with a concrete count. clamp(2 * features, 8, 64) is a tabular-MLP
    // rule-of-thumb baseline — see the doc on the function for why
    // those specific bounds.
    int hidden_size = hidden_layer_size;
    if (hidden_size == kAutoHiddenSize) {
        hidden_size = std::clamp(2 * num_features, 8, 64);
    }

    // Merge into all_data_ so num_samples() / num_train_samples() are accurate.
    all_data_.clear();
    all_data_.reserve(train_samples.size() + val_samples.size());
    all_data_.insert(all_data_.end(), train_samples.begin(), train_samples.end());
    all_data_.insert(all_data_.end(), val_samples.begin(), val_samples.end());

    split_.train = std::move(train_samples);
    split_.test  = std::move(val_samples);

    // Data is already normalized — store identity stats so predict() is a
    // pass-through: normalized[i] = (x[i] - 0) / (1 - 0) = x[i].
    norm_stats_.assign(num_features, NormStats{0.0, 1.0, 0.0, 1.0});

    // Set up IO neurons.
    io_manager_ = std::make_unique<IOManager>(network_);
    auto io_r = io_manager_->ensure_io_neurons(num_features, num_outputs, class_labels);
    if (!io_r.ok())
        return util::Error("load_prepared_dataset: " + io_r.error().message);

    const auto& input_ids  = io_manager_->input_ids();
    const auto& output_ids = io_manager_->output_ids();

    // P1-18: when a hidden layer is requested, allocate ReLU neurons
    // and fully connect input→hidden + hidden→output. Glorot-uniform
    // weights — symmetric around zero with variance scaled by the
    // fan-in/fan-out so the initial pre-activations don't saturate
    // the loss-function's softmax.
    std::mt19937 rng(static_cast<unsigned>(seed));

    if (hidden_size > 0) {
        // Glorot bounds for input→hidden layer.
        const double bound_in = std::sqrt(
            6.0 / (static_cast<double>(num_features) + hidden_size));
        std::uniform_real_distribution<double> wt_in(-bound_in, bound_in);

        // Glorot bounds for hidden→output layer.
        const double bound_out = std::sqrt(
            6.0 / (static_cast<double>(hidden_size) + num_outputs));
        std::uniform_real_distribution<double> wt_out(-bound_out, bound_out);

        std::vector<core::NeuronId> hidden_ids;
        hidden_ids.reserve(hidden_size);

        // Lay hidden neurons out in a column between input (x=0)
        // and output (x=10) so the visualizer renders a coherent
        // graph rather than a tangle.
        const double total_height = std::min(12.0, hidden_size * 1.0);
        const double hid_spacing = (hidden_size > 1)
            ? total_height / (hidden_size - 1) : 0.0;
        for (int h = 0; h < hidden_size; ++h) {
            auto id = core::NeuronId::generate();
            core::Neuron neuron(id, "hidden_" + std::to_string(h));
            const double y = (h - hidden_size / 2.0) * hid_spacing;
            neuron.set_position(core::Vec3{5.0, y, 0.0});
            // ReLU on hidden so the network has nonlinearity. Sigmoid
            // would saturate; LINEAR would collapse the layer to a
            // single linear transform mathematically equivalent to
            // direct input→output wiring.
            neuron.set_activation_type(core::ActivationType::RELU);
            neuron.set_threshold(0.0);
            auto add_r = network_.add_neuron(std::move(neuron));
            if (!add_r.ok())
                return util::Error("load_prepared_dataset: hidden alloc: "
                                   + add_r.error().message);
            hidden_ids.push_back(id);
        }

        for (const auto& in_id : input_ids) {
            for (const auto& hid_id : hidden_ids) {
                network_.connect(in_id, hid_id, wt_in(rng));
            }
        }
        for (const auto& hid_id : hidden_ids) {
            for (const auto& out_id : output_ids) {
                network_.connect(hid_id, out_id, wt_out(rng));
            }
        }
    } else {
        // Legacy direct-wiring path. Preserved so callers that
        // explicitly opt out of the hidden layer (hidden_layer_size=0,
        // typically tests asserting on the linear-layer behavior) get
        // the exact pre-P1-18 weight distribution.
        std::uniform_real_distribution<double> wt(0.01, 0.5);
        for (const auto& in_id : input_ids) {
            for (const auto& out_id : output_ids) {
                network_.connect(in_id, out_id, wt(rng));
            }
        }
    }

    history_.clear();
    return util::Result<void>::success();
}

} // namespace neurograph::training
