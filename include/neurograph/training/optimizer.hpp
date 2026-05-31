#pragma once

#include "neurograph/core/network.hpp"
#include "neurograph/training/backprop.hpp"
#include <unordered_map>
#include <string_view>
#include <memory>
#include <cmath>
#include <string>
#include <algorithm>

namespace neurograph::training {

struct OptimizerConfig {
    double learning_rate = 0.001;
    double momentum = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 0.0;
    double gradient_clip = 5.0;   // max gradient norm, 0 = disabled

    // Learning rate schedule
    std::string schedule_name = "cosine";   // "constant", "cosine", "exponential"
    double min_lr = 1e-6;
    double decay_factor = 0.95;
    int warmup_epochs = 5;
};

// ---------------------------------------------------------------------------
// Learning rate schedule hierarchy
// ---------------------------------------------------------------------------

class LearningRateSchedule {
public:
    virtual ~LearningRateSchedule() = default;
    /// Return the learning rate for the given epoch / total_epochs.
    virtual double compute_lr(double initial_lr, int epoch, int total_epochs) const = 0;
};

class ConstantSchedule : public LearningRateSchedule {
public:
    double compute_lr(double initial_lr, int /*epoch*/, int /*total_epochs*/) const override {
        return initial_lr;
    }
};

class CosineAnnealingSchedule : public LearningRateSchedule {
public:
    explicit CosineAnnealingSchedule(double min_lr) : min_lr_(min_lr) {}

    double compute_lr(double initial_lr, int epoch, int total_epochs) const override {
        if (total_epochs <= 1) return initial_lr;
        double progress = static_cast<double>(epoch) / static_cast<double>(total_epochs - 1);
        progress = std::clamp(progress, 0.0, 1.0);
        return min_lr_ + 0.5 * (initial_lr - min_lr_) * (1.0 + std::cos(3.14159265358979323846 * progress));
    }

private:
    double min_lr_;
};

class ExponentialDecaySchedule : public LearningRateSchedule {
public:
    explicit ExponentialDecaySchedule(double decay_factor) : decay_factor_(decay_factor) {}

    double compute_lr(double initial_lr, int epoch, int /*total_epochs*/) const override {
        return initial_lr * std::pow(decay_factor_, epoch);
    }

private:
    double decay_factor_;
};

/// Wraps any schedule, linearly ramping LR from 0 to initial_lr over
/// warmup_epochs, then delegating to the inner schedule.
class WarmupSchedule : public LearningRateSchedule {
public:
    WarmupSchedule(int warmup_epochs, std::unique_ptr<LearningRateSchedule> inner)
        : warmup_epochs_(warmup_epochs), inner_(std::move(inner)) {}

    double compute_lr(double initial_lr, int epoch, int total_epochs) const override {
        if (epoch < warmup_epochs_) {
            // Linear ramp from 0 to initial_lr
            return initial_lr * static_cast<double>(epoch + 1) / static_cast<double>(warmup_epochs_);
        }
        // After warmup, shift epoch so the inner schedule starts at 0
        int adjusted_epoch = epoch - warmup_epochs_;
        int adjusted_total = std::max(1, total_epochs - warmup_epochs_);
        return inner_->compute_lr(initial_lr, adjusted_epoch, adjusted_total);
    }

private:
    int warmup_epochs_;
    std::unique_ptr<LearningRateSchedule> inner_;
};

/// Factory: build a schedule (optionally wrapped with warmup) from config.
inline std::unique_ptr<LearningRateSchedule> make_schedule(const OptimizerConfig& cfg) {
    std::unique_ptr<LearningRateSchedule> base;
    if (cfg.schedule_name == "constant")
        base = std::make_unique<ConstantSchedule>();
    else if (cfg.schedule_name == "exponential")
        base = std::make_unique<ExponentialDecaySchedule>(cfg.decay_factor);
    else  // default: cosine
        base = std::make_unique<CosineAnnealingSchedule>(cfg.min_lr);

    if (cfg.warmup_epochs > 0)
        return std::make_unique<WarmupSchedule>(cfg.warmup_epochs, std::move(base));
    return base;
}

// ---------------------------------------------------------------------------
// Optimizer base & implementations
// ---------------------------------------------------------------------------

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(core::Network& network, const GradientMap& gradients) = 0;
    virtual void reset() = 0;
    virtual std::string_view name() const = 0;

    /// Adjust the internal learning rate for the current epoch using the schedule.
    void update_lr(int epoch, int total_epochs) {
        if (schedule_)
            current_lr_ = schedule_->compute_lr(initial_lr_, epoch, total_epochs);
    }

    /// Manually override the current learning rate (e.g. for LR finder).
    virtual void set_learning_rate(double lr) { current_lr_ = lr; }

protected:
    void init_schedule(const OptimizerConfig& cfg) {
        initial_lr_ = cfg.learning_rate;
        current_lr_ = cfg.learning_rate;
        schedule_ = make_schedule(cfg);
    }

    double current_lr_ = 0.001;

private:
    double initial_lr_ = 0.001;
    std::unique_ptr<LearningRateSchedule> schedule_;
};

class SGDOptimizer : public Optimizer {
public:
    explicit SGDOptimizer(OptimizerConfig config) : config_(config) {
        init_schedule(config);
    }

    void step(core::Network& network, const GradientMap& gradients) override {
        for (auto& [sid, grad] : gradients.weight_gradients) {
            auto* syn = network.get_synapse(sid);
            if (!syn) continue;

            double g = clip_gradient(grad);

            // Momentum
            velocity_[sid] = config_.momentum * velocity_[sid] + g;
            double update = current_lr_ * velocity_[sid];

            double new_weight = syn->weight() - update;
            if (config_.weight_decay > 0.0)
                new_weight -= config_.weight_decay * syn->weight();

            syn->set_weight(new_weight);
        }
    }

    void reset() override { velocity_.clear(); }
    std::string_view name() const override { return "sgd"; }

private:
    double clip_gradient(double g) const {
        if (config_.gradient_clip > 0.0)
            return std::clamp(g, -config_.gradient_clip, config_.gradient_clip);
        return g;
    }

    OptimizerConfig config_;
    std::unordered_map<core::SynapseId, double> velocity_;
};

class AdamOptimizer : public Optimizer {
public:
    explicit AdamOptimizer(OptimizerConfig config) : config_(config) {
        init_schedule(config);
    }

    void step(core::Network& network, const GradientMap& gradients) override {
        t_++;
        for (auto& [sid, grad] : gradients.weight_gradients) {
            auto* syn = network.get_synapse(sid);
            if (!syn) continue;

            double g = clip_gradient(grad);

            // Update biased first/second moment estimates
            m_[sid] = config_.beta1 * m_[sid] + (1.0 - config_.beta1) * g;
            v_[sid] = config_.beta2 * v_[sid] + (1.0 - config_.beta2) * g * g;

            // Bias-corrected estimates
            double m_hat = m_[sid] / (1.0 - std::pow(config_.beta1, t_));
            double v_hat = v_[sid] / (1.0 - std::pow(config_.beta2, t_));

            double update = current_lr_ * m_hat / (std::sqrt(v_hat) + config_.epsilon);

            double new_weight = syn->weight() - update;
            if (config_.weight_decay > 0.0)
                new_weight -= config_.weight_decay * syn->weight();

            syn->set_weight(new_weight);
        }
    }

    void reset() override {
        m_.clear();
        v_.clear();
        t_ = 0;
    }

    std::string_view name() const override { return "adam"; }

private:
    double clip_gradient(double g) const {
        if (config_.gradient_clip > 0.0)
            return std::clamp(g, -config_.gradient_clip, config_.gradient_clip);
        return g;
    }

    OptimizerConfig config_;
    std::unordered_map<core::SynapseId, double> m_;
    std::unordered_map<core::SynapseId, double> v_;
    int t_ = 0;
};

inline std::unique_ptr<Optimizer> make_optimizer(const std::string& name, OptimizerConfig config) {
    if (name == "adam") return std::make_unique<AdamOptimizer>(config);
    return std::make_unique<SGDOptimizer>(config);
}

} // namespace neurograph::training
