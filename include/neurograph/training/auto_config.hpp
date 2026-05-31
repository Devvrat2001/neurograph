#pragma once

#include "neurograph/training/training_engine.hpp"
#include "neurograph/training/data_loader.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace neurograph::training {

/// Summary of a loaded dataset's characteristics — used to drive auto-config.
struct DatasetProfile {
    int num_samples = 0;
    int num_features = 0;
    int num_outputs = 0;
    bool is_image = false;       // MNIST / image-folder
    bool is_classification = false; // one-hot target (outputs > 1)
    int image_width = 0;
    int image_height = 0;

    // Rough "size bucket" — drives most heuristics.
    // tiny: <= 500, small: <= 5k, medium: <= 50k, large: > 50k
    enum class SizeBucket { Tiny, Small, Medium, Large };
    SizeBucket size_bucket() const {
        if (num_samples <= 500)   return SizeBucket::Tiny;
        if (num_samples <= 5000)  return SizeBucket::Small;
        if (num_samples <= 50000) return SizeBucket::Medium;
        return SizeBucket::Large;
    }
};

/// Inspect an already-loaded TrainingEngine and build a profile. Call after
/// load_dataset / load_mnist / load_image_folder.
inline DatasetProfile profile_dataset(const TrainingEngine& engine,
                                      bool is_mnist = false,
                                      int image_w = 0, int image_h = 0) {
    DatasetProfile p;
    p.num_samples = engine.num_samples();
    p.num_features = engine.num_features();
    p.num_outputs = engine.num_outputs();
    p.is_classification = p.num_outputs > 1;

    if (is_mnist) {
        p.is_image = true;
        p.image_width = 28;
        p.image_height = 28;
    } else if (image_w > 0 && image_h > 0 &&
               p.num_features == image_w * image_h) {
        p.is_image = true;
        p.image_width = image_w;
        p.image_height = image_h;
    } else if (p.num_features > 0) {
        // Heuristic: if features form a perfect square and are "large enough",
        // treat as grayscale image (helps when loaded via generic paths).
        int side = static_cast<int>(std::round(std::sqrt(
            static_cast<double>(p.num_features))));
        if (side * side == p.num_features && side >= 16) {
            p.is_image = true;
            p.image_width = side;
            p.image_height = side;
        }
    }
    return p;
}

/// Produce a SupervisedConfig tuned for this dataset. Aims for fast
/// convergence with minimal wasted compute — larger datasets get fewer
/// epochs and bigger batches, image data enables augmentation, tiny
/// datasets get more regularization and longer patience.
inline SupervisedConfig auto_supervised_config(const DatasetProfile& p) {
    SupervisedConfig c;

    // Loss follows output shape: softmax+CE for multi-class one-hot,
    // MSE otherwise (regression or single output).
    c.loss_type = p.is_classification ? LossType::CROSS_ENTROPY : LossType::MSE;
    c.optimizer_name = "adam";  // Adam is a safe default across domains

    // --- Size-bucket heuristics ---
    switch (p.size_bucket()) {
        case DatasetProfile::SizeBucket::Tiny:
            c.epochs = 150;
            c.batch_size = 8;
            c.optimizer_config.learning_rate = 0.01;
            c.warmup_epochs = 5;
            c.dropout_rate = 0.1;    // less dropout, too aggressive on tiny data
            c.patience = 40;
            break;
        case DatasetProfile::SizeBucket::Small:
            c.epochs = 80;
            c.batch_size = 16;
            c.optimizer_config.learning_rate = 0.005;
            c.warmup_epochs = 3;
            c.dropout_rate = 0.2;
            c.patience = 25;
            break;
        case DatasetProfile::SizeBucket::Medium:
            c.epochs = 40;
            c.batch_size = 32;
            c.optimizer_config.learning_rate = 0.001;
            c.warmup_epochs = 2;
            c.dropout_rate = 0.25;
            c.patience = 15;
            break;
        case DatasetProfile::SizeBucket::Large:
            c.epochs = 20;
            c.batch_size = 64;
            c.optimizer_config.learning_rate = 0.001;
            c.warmup_epochs = 1;
            c.dropout_rate = 0.3;
            c.patience = 10;
            break;
    }

    // --- Schedule ---
    // Cosine annealing pairs well with warmup + Adam; covers most cases.
    c.schedule_name = "cosine";
    c.min_lr = 1e-6;
    c.optimizer_config.schedule_name = c.schedule_name;
    c.optimizer_config.min_lr = c.min_lr;
    c.optimizer_config.warmup_epochs = c.warmup_epochs;

    // --- Batch norm ---
    // Always helpful for deeper / larger feature spaces.
    c.batch_norm = (p.num_features >= 20);

    // --- Augmentation (images only) ---
    if (p.is_image) {
        c.augmentation.enabled = true;
        c.augmentation.image_width = p.image_width;
        c.augmentation.image_height = p.image_height;
        // Horizontal flip is wrong for digit recognition (MNIST) since
        // mirrored digits aren't the same class. Leave it off unless we
        // know the domain is natural imagery.
        c.augmentation.horizontal_flip = false;
        c.augmentation.random_rotation = true;
        c.augmentation.random_shift = true;
        c.augmentation.random_noise = true;
        c.augmentation.noise_stddev = 0.05;
    } else {
        c.augmentation.enabled = false;
    }

    // --- Self-organization ---
    // Interleave by default. Tiny datasets want frequent restructuring;
    // large datasets get fewer cycles (each cycle is relatively expensive).
    c.interleave_self_org = true;
    switch (p.size_bucket()) {
        case DatasetProfile::SizeBucket::Tiny:   c.self_org_interval = 5;  break;
        case DatasetProfile::SizeBucket::Small:  c.self_org_interval = 8;  break;
        case DatasetProfile::SizeBucket::Medium: c.self_org_interval = 10; break;
        case DatasetProfile::SizeBucket::Large:  c.self_org_interval = 15; break;
    }

    return c;
}

/// Suggested adaptive-topology tuning for the same dataset. Returned as a
/// partial overlay — caller applies it to the network's current TopologyConfig.
struct AutoTopology {
    bool enable_neuron_pruning = true;
    bool enable_neuron_growth = true;
    bool enable_dynamic_rewiring = true;
    bool use_stall_counter = true;
    int growth_patience = 10;
    double growth_split_ratio = 0.3;
    int max_hidden_neurons = 0;   // 0 = unlimited
};

inline AutoTopology auto_topology_config(const DatasetProfile& p) {
    AutoTopology t;

    // Cap hidden neurons to keep memory bounded — proportional to problem
    // complexity. Tiny problems shouldn't be allowed to balloon.
    int capacity = static_cast<int>(std::ceil(
        std::sqrt(static_cast<double>(p.num_features) *
                  std::max(1, p.num_outputs))));
    switch (p.size_bucket()) {
        case DatasetProfile::SizeBucket::Tiny:
            t.max_hidden_neurons = std::max(16,  capacity * 2);
            t.growth_patience = 5;
            t.growth_split_ratio = 0.4;
            break;
        case DatasetProfile::SizeBucket::Small:
            t.max_hidden_neurons = std::max(32,  capacity * 3);
            t.growth_patience = 8;
            t.growth_split_ratio = 0.3;
            break;
        case DatasetProfile::SizeBucket::Medium:
            t.max_hidden_neurons = std::max(128, capacity * 4);
            t.growth_patience = 10;
            t.growth_split_ratio = 0.25;
            break;
        case DatasetProfile::SizeBucket::Large:
            t.max_hidden_neurons = std::max(256, capacity * 4);
            t.growth_patience = 15;
            t.growth_split_ratio = 0.2;
            break;
    }
    // Always gate growth on loss for auto — it's the conservative default.
    t.use_stall_counter = true;
    return t;
}

/// Human-readable one-line description of what was chosen — handy for UI
/// confirmation messages.
inline std::string describe_profile(const DatasetProfile& p) {
    const char* bucket = "tiny";
    switch (p.size_bucket()) {
        case DatasetProfile::SizeBucket::Tiny:   bucket = "tiny";   break;
        case DatasetProfile::SizeBucket::Small:  bucket = "small";  break;
        case DatasetProfile::SizeBucket::Medium: bucket = "medium"; break;
        case DatasetProfile::SizeBucket::Large:  bucket = "large";  break;
    }
    std::string kind = p.is_image ? "image" : (p.is_classification ? "tabular-classification" : "tabular-regression");
    return std::to_string(p.num_samples) + " samples, " +
           std::to_string(p.num_features) + " features, " +
           std::to_string(p.num_outputs) + " outputs (" +
           bucket + " " + kind + ")";
}

} // namespace neurograph::training
