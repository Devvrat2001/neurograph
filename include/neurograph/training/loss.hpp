#pragma once

#include <vector>
#include <string_view>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace neurograph::training {

enum class LossType : uint8_t { MSE = 0, CROSS_ENTROPY = 1 };

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double compute(const std::vector<double>& predicted,
                           const std::vector<double>& target) const = 0;
    virtual std::vector<double> gradient(const std::vector<double>& predicted,
                                         const std::vector<double>& target) const = 0;
    virtual std::string_view name() const = 0;
    virtual LossType type() const = 0;
};

class MSELoss : public LossFunction {
public:
    double compute(const std::vector<double>& predicted,
                   const std::vector<double>& target) const override {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            double d = predicted[i] - target[i];
            sum += d * d;
        }
        return sum / static_cast<double>(predicted.size());
    }

    std::vector<double> gradient(const std::vector<double>& predicted,
                                  const std::vector<double>& target) const override {
        double n = static_cast<double>(predicted.size());
        std::vector<double> grad(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i)
            grad[i] = 2.0 * (predicted[i] - target[i]) / n;
        return grad;
    }

    std::string_view name() const override { return "mse"; }
    LossType type() const override { return LossType::MSE; }
};

/// Softmax utility
inline std::vector<double> softmax(const std::vector<double>& logits) {
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> exps(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_val);
        sum += exps[i];
    }
    for (auto& e : exps) e /= sum;
    return exps;
}

class CrossEntropyLoss : public LossFunction {
public:
    double compute(const std::vector<double>& predicted,
                   const std::vector<double>& target) const override {
        auto probs = softmax(predicted);
        double loss = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (target[i] > 0.0)
                loss -= target[i] * std::log(std::max(probs[i], 1e-12));
        }
        return loss;
    }

    /// Combined softmax + cross-entropy gradient: (softmax(predicted) - target)
    std::vector<double> gradient(const std::vector<double>& predicted,
                                  const std::vector<double>& target) const override {
        auto probs = softmax(predicted);
        std::vector<double> grad(probs.size());
        for (size_t i = 0; i < probs.size(); ++i)
            grad[i] = probs[i] - target[i];
        return grad;
    }

    std::string_view name() const override { return "cross_entropy"; }
    LossType type() const override { return LossType::CROSS_ENTROPY; }
};

inline std::unique_ptr<LossFunction> make_loss(LossType type) {
    switch (type) {
        case LossType::MSE: return std::make_unique<MSELoss>();
        case LossType::CROSS_ENTROPY: return std::make_unique<CrossEntropyLoss>();
        default: return std::make_unique<MSELoss>();
    }
}

} // namespace neurograph::training
