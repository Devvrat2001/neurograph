#pragma once

#include "neurograph/core/network.hpp"
#include "neurograph/command/command_types.hpp"
#include "neurograph/util/result.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <functional>

namespace neurograph::module {

/// Module capability flags — declare what the module provides.
enum class ModuleCapability : uint32_t {
    NONE             = 0,
    CUSTOM_COMMANDS  = 1 << 0,   // Module handles custom command types
    NEURON_PROCESSOR = 1 << 1,   // Module processes neuron activations
    DATA_SOURCE      = 1 << 2,   // Module provides external data input
    DATA_SINK        = 1 << 3,   // Module consumes network output
    ANALYZER         = 1 << 4,   // Module analyzes network state
};

inline ModuleCapability operator|(ModuleCapability a, ModuleCapability b) {
    return static_cast<ModuleCapability>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline ModuleCapability operator&(ModuleCapability a, ModuleCapability b) {
    return static_cast<ModuleCapability>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline bool has_capability(ModuleCapability flags, ModuleCapability cap) {
    return (flags & cap) != ModuleCapability::NONE;
}

/// Module metadata — returned by the module to identify itself.
struct ModuleInfo {
    std::string name;           // Unique module name (e.g., "vision_processor")
    std::string version;        // Semantic version string
    std::string description;    // Human-readable description
    std::string author;
    ModuleCapability capabilities = ModuleCapability::NONE;
};

/// Abstract interface that all NeuroGraph modules must implement.
/// Modules are loaded as shared libraries (.dll/.so) and accessed
/// through this interface via C factory functions.
class ModuleInterface {
public:
    virtual ~ModuleInterface() = default;

    /// Return module metadata.
    virtual ModuleInfo info() const = 0;

    /// Initialize the module. Called once after loading.
    /// The network reference allows the module to register neurons,
    /// synapses, or inspect network state.
    virtual util::Result<void> initialize(core::Network& network) = 0;

    /// Shut down the module. Called before unloading.
    /// The module should clean up any resources it holds.
    virtual util::Result<void> shutdown() = 0;

    /// Called each network tick/cycle. Modules can process data,
    /// update neuron states, feed input, or collect output.
    virtual void on_tick(core::Network& network, uint64_t tick) = 0;

    /// Handle a custom command directed at this module.
    /// Only called if the module declares CUSTOM_COMMANDS capability.
    virtual util::Result<nlohmann::json> handle_command(
        const std::string& command_name,
        const nlohmann::json& payload) {
        return util::Error("Module does not handle commands");
    }

    /// Query module status. Returns module-specific state as JSON.
    virtual nlohmann::json status() const {
        return nlohmann::json{{"status", "ok"}};
    }
};

/// C-linkage factory function type. Every module shared library must export:
///   extern "C" neurograph::module::ModuleInterface* neurograph_create_module();
///   extern "C" void neurograph_destroy_module(neurograph::module::ModuleInterface*);
using CreateModuleFunc = ModuleInterface* (*)();
using DestroyModuleFunc = void (*)(ModuleInterface*);

} // namespace neurograph::module
