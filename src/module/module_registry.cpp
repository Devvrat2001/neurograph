#include "neurograph/module/module_registry.hpp"
#include "neurograph/util/logger.hpp"

namespace neurograph::module {

ModuleRegistry::~ModuleRegistry() {
    shutdown_all();
}

util::Result<void> ModuleRegistry::register_module(ModuleEntry entry) {
    const auto& name = entry.info.name;
    if (modules_.count(name)) {
        return util::Error("Module already registered: " + name);
    }

    NG_LOG_INFO("Registered module: {} v{}", name, entry.info.version);
    modules_[name] = std::move(entry);
    return util::Result<void>::success();
}

util::Result<void> ModuleRegistry::unregister_module(const std::string& name) {
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        return util::Error("Module not found: " + name);
    }

    auto& entry = it->second;

    // Shut down if initialized
    if (entry.initialized && entry.instance) {
        auto result = entry.instance->shutdown();
        if (result.is_error()) {
            NG_LOG_WARN("Module {} shutdown error: {}", name, result.error().message);
        }
    }

    // Destroy the instance
    if (entry.instance && entry.destroy) {
        entry.destroy(entry.instance);
    }

    NG_LOG_INFO("Unregistered module: {}", name);
    modules_.erase(it);
    return util::Result<void>::success();
}

util::Result<void> ModuleRegistry::initialize_all(core::Network& network) {
    bool has_errors = false;
    for (auto& [name, entry] : modules_) {
        if (entry.initialized || !entry.instance) continue;

        auto result = entry.instance->initialize(network);
        if (result.ok()) {
            entry.initialized = true;
            NG_LOG_INFO("Initialized module: {}", name);
        } else {
            NG_LOG_ERROR("Failed to initialize module {}: {}", name, result.error().message);
            has_errors = true;
        }
    }
    return has_errors ? util::Result<void>(util::Error("Some modules failed to initialize"))
                      : util::Result<void>::success();
}

void ModuleRegistry::shutdown_all() {
    for (auto& [name, entry] : modules_) {
        if (!entry.initialized || !entry.instance) continue;

        auto result = entry.instance->shutdown();
        if (result.is_error()) {
            NG_LOG_WARN("Module {} shutdown error: {}", name, result.error().message);
        }
        entry.initialized = false;
    }
}

void ModuleRegistry::tick_all(core::Network& network, uint64_t tick) {
    for (auto& [name, entry] : modules_) {
        if (!entry.initialized || !entry.instance) continue;
        entry.instance->on_tick(network, tick);
    }
}

util::Result<nlohmann::json> ModuleRegistry::dispatch_command(
    const std::string& module_name,
    const std::string& command_name,
    const nlohmann::json& payload) {

    auto it = modules_.find(module_name);
    if (it == modules_.end()) {
        return util::Error("Module not found: " + module_name);
    }

    auto& entry = it->second;
    if (!entry.initialized || !entry.instance) {
        return util::Error("Module not initialized: " + module_name);
    }

    if (!has_capability(entry.info.capabilities, ModuleCapability::CUSTOM_COMMANDS)) {
        return util::Error("Module does not support custom commands: " + module_name);
    }

    return entry.instance->handle_command(command_name, payload);
}

ModuleInterface* ModuleRegistry::get_module(const std::string& name) const {
    auto it = modules_.find(name);
    return (it != modules_.end()) ? it->second.instance : nullptr;
}

bool ModuleRegistry::has_module(const std::string& name) const {
    return modules_.count(name) > 0;
}

std::vector<std::string> ModuleRegistry::module_names() const {
    std::vector<std::string> names;
    names.reserve(modules_.size());
    for (const auto& [name, _] : modules_) {
        names.push_back(name);
    }
    return names;
}

std::vector<ModuleInfo> ModuleRegistry::all_info() const {
    std::vector<ModuleInfo> infos;
    infos.reserve(modules_.size());
    for (const auto& [_, entry] : modules_) {
        infos.push_back(entry.info);
    }
    return infos;
}

} // namespace neurograph::module
