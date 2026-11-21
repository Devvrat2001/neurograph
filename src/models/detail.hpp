#pragma once

// Internal helpers shared by the model implementations. Not installed.

#include "neurograph/models/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>

namespace neurograph::models::detail {

/// In-place softmax over `n` values, stable for large logits.
inline void softmax(double* z, int n) {
    double m = z[0];
    for (int i = 1; i < n; ++i) m = std::max(m, z[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { z[i] = std::exp(z[i] - m); sum += z[i]; }
    const double inv = 1.0 / sum;
    for (int i = 0; i < n; ++i) z[i] *= inv;
}

inline double sigmoid(double x) {
    if (x >= 0.0) { const double e = std::exp(-x); return 1.0 / (1.0 + e); }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

/// Probabilities are clipped here before taking logs so one confident
/// mistake cannot produce an infinite loss.
constexpr double kProbFloor = 1e-15;

/// Wall-clock budget tracker shared by the trainers.
class Deadline {
public:
    explicit Deadline(double limit_seconds)
        : start_(std::chrono::steady_clock::now()), limit_(limit_seconds) {}

    double elapsed() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_).count();
    }
    bool expired() const { return limit_ > 0.0 && elapsed() >= limit_; }

private:
    std::chrono::steady_clock::time_point start_;
    double limit_;
};

inline bool is_cancelled(const FitControl& ctl) {
    return ctl.cancelled && ctl.cancelled();
}

inline void report(const FitControl& ctl, double frac, const std::string& msg) {
    if (ctl.progress) ctl.progress(std::clamp(frac, 0.0, 1.0), msg);
}

// Factories implemented by each family's translation unit.
std::unique_ptr<Model> mlp_from_json(const nlohmann::json& j);
std::unique_ptr<Model> gbt_from_json(const nlohmann::json& j);
std::unique_ptr<Model> blend_from_json(const nlohmann::json& j);

} // namespace neurograph::models::detail
