#pragma once

#include "neurograph/util/result.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace neurograph::training {

struct Sample {
    std::vector<double> features;
    std::vector<double> target;   // one-hot for classification, raw for regression
    std::string label;            // original class label (if classification)
};

struct DataBatch {
    std::vector<Sample> samples;
    size_t batch_index = 0;
};

struct DataSplit {
    std::vector<Sample> train;
    std::vector<Sample> test;
};

struct NormStats {
    double mean = 0.0;
    double stddev = 1.0;
    double min_val = 0.0;
    double max_val = 1.0;
};

class DataLoader {
public:
    /// Load CSV file. Last num_target_cols columns are targets.
    /// If auto_one_hot is true and target is non-numeric, it's one-hot encoded.
    static util::Result<std::vector<Sample>> load_csv(
        const std::string& path,
        int num_target_cols = 1,
        bool has_header = true,
        bool auto_one_hot = true);

    /// Split data into train and test sets
    static DataSplit train_test_split(
        const std::vector<Sample>& data,
        double test_ratio = 0.2,
        bool shuffle = true,
        uint64_t seed = 42);

    /// Create batches from data
    static std::vector<DataBatch> make_batches(
        const std::vector<Sample>& data,
        size_t batch_size);

    /// Compute per-feature normalization statistics
    static std::vector<NormStats> compute_stats(const std::vector<Sample>& data);

    /// Normalize features using z-score: (x - mean) / stddev
    static void normalize_zscore(std::vector<Sample>& data,
                                  const std::vector<NormStats>& stats);

    /// Normalize features to [0, 1]: (x - min) / (max - min)
    static void normalize_minmax(std::vector<Sample>& data,
                                  const std::vector<NormStats>& stats);

    /// Load MNIST IDX format files (images + labels)
    /// images_path: path to *-images-idx3-ubyte file
    /// labels_path: path to *-labels-idx1-ubyte file
    /// max_samples: limit samples (0 = load all)
    static util::Result<std::vector<Sample>> load_mnist_idx(
        const std::string& images_path,
        const std::string& labels_path,
        int max_samples = 0);

    /// Load images from a folder structure (each subdirectory = one class).
    /// Only .bmp files are supported (24-bit and 32-bit uncompressed).
    /// Images are resized to target_width x target_height using nearest-neighbor,
    /// converted to grayscale, and normalized to [0, 1].
    static util::Result<std::vector<Sample>> load_image_folder(
        const std::string& root_path,
        int target_width,
        int target_height,
        int max_samples = 0);

    /// Resize a grayscale image using nearest-neighbor interpolation.
    /// Input and output are flattened row-major pixel arrays.
    static std::vector<double> resize_image(
        const std::vector<double>& pixels,
        int src_w, int src_h,
        int dst_w, int dst_h);

    /// Get class labels discovered during CSV loading
    static const std::vector<std::string>& last_class_labels();

    /// Set class labels manually (for MNIST etc.)
    static void set_class_labels(const std::vector<std::string>& labels);

private:
    /// Read pixel data from a BMP file (24-bit or 32-bit uncompressed).
    /// Returns RGB pixels in top-to-bottom, left-to-right order.
    static std::vector<uint8_t> read_bmp_pixels(
        const std::string& path,
        int& width, int& height, int& channels);
};

} // namespace neurograph::training
