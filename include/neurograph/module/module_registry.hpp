#pragma once

#include "neurograph/module/module_interface.hpp"
#include "neurograph/util/result.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>

namespace neurograph::module {

/// Entry for a loaded module in the registry.
struct ModuleEntry {
    ModuleInfo info;
    ModuleInterface* instance = nullptr;
    DestroyModuleFunc destroy = nullptr;
    void* library_handle = nullptr;     // Platform-specific handle (HMODULE / void*)
    std::string file_path;              // Path to the shared library
    bool initialized = false;
};

/// Central registry that tracks all loaded modules.
/// Provides lookup by name, lifecycle management, and tick dispatch.
class ModuleRegistry {
public:
    ModuleRegistry() = default;
    ~ModuleRegistry();

    // Non-copyable
    ModuleRegistry(const ModuleRegistry&) = delete;
    ModuleRegistry& operator=(const ModuleRegistry&) = delete;

    /// Register a module entry (called by ModuleLoader after loading).
    util::Result<void> register_module(ModuleEntry entry);

    /// Unregister and destroy a module by name.
    util::Result<void> unregister_module(const std::string& name);

    /// Initialize all registered modules.
    util::Result<void> initialize_all(core::Network& network);

    /// Shut down all registered modules.
    void shutdown_all();

    /// Dispatch a tick to all initialized modules.
    void tick_all(core::Network& network, uint64_t tick);

    /// Route a custom command to the appropriate module.
    util::Result<nlohmann::json> dispatch_command(
        const std::string& module_name,
        const std::string& command_name,
        const nlohmann::json& payload);

    /// Get a module by name.
    ModuleInterface* get_module(const std::string& name) const;

    /// Check if a module is loaded.
    bool has_module(const std::string& name) const;

    /// Get all loaded module names.
    std::vector<std::string> module_names() const;

    /// Get info for all modules.
    std::vector<ModuleInfo> all_info() const;

    /// Number of loaded modules.
    size_t size() const noexcept { return modules_.size(); }

private:
    std::unordered_map<std::string, ModuleEntry> modules_;
};

} // namespace neurograph::module
