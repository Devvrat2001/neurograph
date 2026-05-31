#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <string_view>

#ifdef __CUDACC__
#define NG_HOST_DEVICE __host__ __device__
#else
#define NG_HOST_DEVICE
#endif

namespace neurograph::core {

// ============================================================
// Device-compatible activation functions (usable in CUDA kernels)
// ============================================================

/// Activation function IDs for GPU kernel dispatch.
///
/// Stable integer values — `Neuron::to_json` round-trips this enum as
/// the int, so changing a value silently corrupts saved bundles.
/// Append-only.
enum class ActivationType : uint8_t {
    SIGMOID    = 0,
    RELU       = 1,
    TANH       = 2,
    STEP       = 3,
    LEAKY_RELU = 4,
    GELU       = 5,
    SWISH      = 6,
    ELU        = 7,
    /// Identity — output = input, derivative = 1. P1-17. Use on output
    /// neurons that feed `CrossEntropyLoss` (which applies softmax
    /// internally and requires raw logits) or on regression heads
    /// where you want unbounded scalar outputs.
    LINEAR     = 8
};

/// Device-callable activation: dispatches by ActivationType.
NG_HOST_DEVICE inline float device_activate(float x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:
            return 1.0f / (1.0f + expf(-x));
        case ActivationType::RELU:
            return x > 0.0f ? x : 0.0f;
        case ActivationType::TANH:
            return tanhf(x);
        case ActivationType::STEP:
            return x >= 0.0f ? 1.0f : 0.0f;
        case ActivationType::LEAKY_RELU:
            return x >= 0.0f ? x : 0.01f * x;
        case ActivationType::GELU: {
            constexpr float kSqrt2OverPi = 0.7978845608f; // sqrt(2/pi)
            float k = kSqrt2OverPi * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanhf(k));
        }
        case ActivationType::SWISH: {
            float sig = 1.0f / (1.0f + expf(-x));
            return x * sig;
        }
        case ActivationType::ELU:
            return x >= 0.0f ? x : (expf(x) - 1.0f);
        case ActivationType::LINEAR:
            return x;
        default:
            return x;
    }
}

/// Device-callable activation derivative (for learning).
NG_HOST_DEVICE inline float device_activate_derivative(float x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: {
            float s = 1.0f / (1.0f + expf(-x));
            return s * (1.0f - s);
        }
        case ActivationType::RELU:
            return x > 0.0f ? 1.0f : 0.0f;
        case ActivationType::TANH: {
            float t = tanhf(x);
            return 1.0f - t * t;
        }
        case ActivationType::STEP:
            return 0.0f; // not differentiable
        case ActivationType::LEAKY_RELU:
            return x >= 0.0f ? 1.0f : 0.01f;
        case ActivationType::GELU: {
            constexpr float kSqrt2OverPi = 0.7978845608f;
            float k = kSqrt2OverPi * (x + 0.044715f * x * x * x);
            float tanh_k = tanhf(k);
            return 0.5f * (1.0f + tanh_k)
                 + 0.5f * x * (1.0f - tanh_k * tanh_k)
                   * kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);
        }
        case ActivationType::SWISH: {
            float sig = 1.0f / (1.0f + expf(-x));
            float sw = x * sig;
            return sw + sig * (1.0f - sw);
        }
        case ActivationType::ELU: {
            if (x >= 0.0f) return 1.0f;
            float e = expf(x) - 1.0f; // elu(x)
            return e + 1.0f;           // elu(x) + alpha
        }
        case ActivationType::LINEAR:
            return 1.0f;
        default:
            return 1.0f;
    }
}

// ============================================================
// CPU-side activation function interface (strategy pattern)
// ============================================================

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    virtual double compute(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::string_view name() const = 0;
    virtual ActivationType type() const = 0;
};

class SigmoidActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double derivative(double x) const override {
        double s = compute(x);
        return s * (1.0 - s);
    }
    std::string_view name() const override { return "sigmoid"; }
    ActivationType type() const override { return ActivationType::SIGMOID; }
};

class ReLUActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        return x > 0.0 ? x : 0.0;
    }
    double derivative(double x) const override {
        return x > 0.0 ? 1.0 : 0.0;
    }
    std::string_view name() const override { return "relu"; }
    ActivationType type() const override { return ActivationType::RELU; }
};

class TanhActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        return std::tanh(x);
    }
    double derivative(double x) const override {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
    std::string_view name() const override { return "tanh"; }
    ActivationType type() const override { return ActivationType::TANH; }
};

class StepActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        return x >= 0.0 ? 1.0 : 0.0;
    }
    double derivative(double x) const override {
        return 0.0;
    }
    std::string_view name() const override { return "step"; }
    ActivationType type() const override { return ActivationType::STEP; }
};

class LeakyReLUActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        return x >= 0.0 ? x : 0.01 * x;
    }
    double derivative(double x) const override {
        return x >= 0.0 ? 1.0 : 0.01;
    }
    std::string_view name() const override { return "leaky_relu"; }
    ActivationType type() const override { return ActivationType::LEAKY_RELU; }
};

class GELUActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        constexpr double kSqrt2OverPi = 0.7978845608028654;
        double k = kSqrt2OverPi * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + std::tanh(k));
    }
    double derivative(double x) const override {
        constexpr double kSqrt2OverPi = 0.7978845608028654;
        double k = kSqrt2OverPi * (x + 0.044715 * x * x * x);
        double tanh_k = std::tanh(k);
        return 0.5 * (1.0 + tanh_k)
             + 0.5 * x * (1.0 - tanh_k * tanh_k)
               * kSqrt2OverPi * (1.0 + 3.0 * 0.044715 * x * x);
    }
    std::string_view name() const override { return "gelu"; }
    ActivationType type() const override { return ActivationType::GELU; }
};

class SwishActivation : public ActivationFunction {
public:
    double compute(double x) const override {
        return x / (1.0 + std::exp(-x));
    }
    double derivative(double x) const override {
        double sig = 1.0 / (1.0 + std::exp(-x));
        double sw = x * sig;
        return sw + sig * (1.0 - sw);
    }
    std::string_view name() const override { return "swish"; }
    ActivationType type() const override { return ActivationType::SWISH; }
};

class ELUActivation : public ActivationFunction {
    static constexpr double alpha_ = 1.0;
public:
    double compute(double x) const override {
        return x >= 0.0 ? x : alpha_ * (std::exp(x) - 1.0);
    }
    double derivative(double x) const override {
        return x >= 0.0 ? 1.0 : compute(x) + alpha_;
    }
    std::string_view name() const override { return "elu"; }
    ActivationType type() const override { return ActivationType::ELU; }
};

/// Identity activation. P1-17. Use on output neurons feeding
/// CrossEntropyLoss (which expects raw logits) or on regression
/// heads. compute(x) = x, derivative(x) = 1.
class LinearActivation : public ActivationFunction {
public:
    double compute(double x) const override { return x; }
    double derivative(double /*x*/) const override { return 1.0; }
    std::string_view name() const override { return "linear"; }
    ActivationType type() const override { return ActivationType::LINEAR; }
};

/// Factory: create activation function by type.
std::unique_ptr<ActivationFunction> make_activation(ActivationType type);

/// Factory: create activation function by name.
std::unique_ptr<ActivationFunction> make_activation(std::string_view name);

/// Parse a string name into an ActivationType enum value.
ActivationType parse_activation_type(std::string_view name);

/// Convert an ActivationType enum value to its string representation.
std::string_view activation_type_str(ActivationType type);

} // namespace neurograph::core
