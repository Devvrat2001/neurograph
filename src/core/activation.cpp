#include "neurograph/core/activation.hpp"

#include <stdexcept>

namespace neurograph::core {

std::unique_ptr<ActivationFunction> make_activation(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:    return std::make_unique<SigmoidActivation>();
        case ActivationType::RELU:       return std::make_unique<ReLUActivation>();
        case ActivationType::TANH:       return std::make_unique<TanhActivation>();
        case ActivationType::STEP:       return std::make_unique<StepActivation>();
        case ActivationType::LEAKY_RELU: return std::make_unique<LeakyReLUActivation>();
        case ActivationType::GELU:       return std::make_unique<GELUActivation>();
        case ActivationType::SWISH:      return std::make_unique<SwishActivation>();
        case ActivationType::ELU:        return std::make_unique<ELUActivation>();
        case ActivationType::LINEAR:     return std::make_unique<LinearActivation>();
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

std::unique_ptr<ActivationFunction> make_activation(std::string_view name) {
    return make_activation(parse_activation_type(name));
}

ActivationType parse_activation_type(std::string_view name) {
    if (name == "sigmoid")    return ActivationType::SIGMOID;
    if (name == "relu")       return ActivationType::RELU;
    if (name == "tanh")       return ActivationType::TANH;
    if (name == "step")       return ActivationType::STEP;
    if (name == "leaky_relu") return ActivationType::LEAKY_RELU;
    if (name == "gelu")       return ActivationType::GELU;
    if (name == "swish" || name == "silu") return ActivationType::SWISH;
    if (name == "elu")        return ActivationType::ELU;
    // P1-17. "identity" is accepted as an alias since it's the more
    // common ML term for this function; "linear" is the canonical name
    // (matches `activation_type_str(LINEAR)`).
    if (name == "linear" || name == "identity") return ActivationType::LINEAR;
    throw std::invalid_argument("Unknown activation name: " + std::string(name));
}

std::string_view activation_type_str(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:    return "sigmoid";
        case ActivationType::RELU:       return "relu";
        case ActivationType::TANH:       return "tanh";
        case ActivationType::STEP:       return "step";
        case ActivationType::LEAKY_RELU: return "leaky_relu";
        case ActivationType::GELU:       return "gelu";
        case ActivationType::SWISH:      return "swish";
        case ActivationType::ELU:        return "elu";
        case ActivationType::LINEAR:     return "linear";
        default:                         return "unknown";
    }
}

} // namespace neurograph::core
