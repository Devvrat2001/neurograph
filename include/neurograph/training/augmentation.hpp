#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace neurograph::training {

struct AugmentationConfig {
    bool enabled = false;
    bool random_rotation = true;       // small rotation +-15 degrees
    bool random_shift = true;          // translate +-2 pixels
    bool random_noise = true;          // Gaussian noise
    bool horizontal_flip = false;      // off by default for digits
    double noise_stddev = 0.05;
    int image_width = 28;              // for 2D augmentations
    int image_height = 28;
};

// Forward declaration — Sample is defined in data_loader.hpp.
// We only need features/target/label, so we work with the concrete type.
struct Sample;

class Augmentor {
public:
    explicit Augmentor(AugmentationConfig config)
        : config_(std::move(config)) {}

    /// Returns a new sample with augmented features.
    /// Target and label are copied unchanged.
    Sample augment(const Sample& original) const {
        Sample result = original;  // copy

        const size_t expected = static_cast<size_t>(config_.image_width) *
                                static_cast<size_t>(config_.image_height);
        bool is_image = (result.features.size() == expected);

        // Apply 2D augmentations only when dimensions match
        if (is_image) {
            if (config_.random_rotation)
                apply_rotation(result.features);
            if (config_.random_shift)
                apply_shift(result.features);
            if (config_.horizontal_flip)
                apply_flip(result.features);
        }

        // Noise is applied regardless of image dimensions
        if (config_.random_noise)
            apply_noise(result.features);

        return result;
    }

private:
    AugmentationConfig config_;

    // Thread-safe RNG accessor
    static std::mt19937& rng() {
        thread_local std::mt19937 gen{std::random_device{}()};
        return gen;
    }

    // ---- Rotation (bilinear interpolation) ----
    void apply_rotation(std::vector<double>& features) const {
        const int w = config_.image_width;
        const int h = config_.image_height;

        std::uniform_real_distribution<double> angle_dist(-15.0, 15.0);
        double angle_deg = angle_dist(rng());
        double angle_rad = angle_deg * (3.14159265358979323846 / 180.0);
        double cos_a = std::cos(angle_rad);
        double sin_a = std::sin(angle_rad);

        double cx = (w - 1) * 0.5;
        double cy = (h - 1) * 0.5;

        std::vector<double> rotated(features.size(), 0.0);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Inverse rotation: find source coordinates
                double dx = x - cx;
                double dy = y - cy;
                double src_x = cos_a * dx + sin_a * dy + cx;
                double src_y = -sin_a * dx + cos_a * dy + cy;

                rotated[y * w + x] = bilinear_sample(features, w, h, src_x, src_y);
            }
        }

        features = std::move(rotated);
    }

    // Bilinear interpolation from flat image buffer
    static double bilinear_sample(const std::vector<double>& img,
                                  int w, int h,
                                  double x, double y) {
        if (x < 0.0 || x >= w - 1e-6 || y < 0.0 || y >= h - 1e-6)
            return 0.0;

        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));
        int x1 = std::min(x0 + 1, w - 1);
        int y1 = std::min(y0 + 1, h - 1);

        double fx = x - x0;
        double fy = y - y0;

        double v00 = img[y0 * w + x0];
        double v10 = img[y0 * w + x1];
        double v01 = img[y1 * w + x0];
        double v11 = img[y1 * w + x1];

        return v00 * (1 - fx) * (1 - fy)
             + v10 * fx       * (1 - fy)
             + v01 * (1 - fx) * fy
             + v11 * fx       * fy;
    }

    // ---- Shift (translate +-2 pixels) ----
    void apply_shift(std::vector<double>& features) const {
        const int w = config_.image_width;
        const int h = config_.image_height;

        std::uniform_int_distribution<int> shift_dist(-2, 2);
        int dx = shift_dist(rng());
        int dy = shift_dist(rng());

        if (dx == 0 && dy == 0) return;

        std::vector<double> shifted(features.size(), 0.0);

        for (int y = 0; y < h; ++y) {
            int src_y = y - dy;
            if (src_y < 0 || src_y >= h) continue;
            for (int x = 0; x < w; ++x) {
                int src_x = x - dx;
                if (src_x < 0 || src_x >= w) continue;
                shifted[y * w + x] = features[src_y * w + src_x];
            }
        }

        features = std::move(shifted);
    }

    // ---- Gaussian noise ----
    void apply_noise(std::vector<double>& features) const {
        std::normal_distribution<double> noise_dist(0.0, config_.noise_stddev);

        for (auto& v : features) {
            v += noise_dist(rng());
            v = std::clamp(v, 0.0, 1.0);
        }
    }

    // ---- Horizontal flip (50% probability) ----
    void apply_flip(std::vector<double>& features) const {
        std::uniform_int_distribution<int> coin(0, 1);
        if (coin(rng()) == 0) return;  // 50% chance to skip

        const int w = config_.image_width;
        const int h = config_.image_height;

        for (int y = 0; y < h; ++y) {
            int row_start = y * w;
            for (int x = 0; x < w / 2; ++x) {
                std::swap(features[row_start + x],
                          features[row_start + (w - 1 - x)]);
            }
        }
    }
};

} // namespace neurograph::training
