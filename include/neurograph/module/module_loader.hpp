#pragma once

#include "neurograph/module/module_registry.hpp"
#include "neurograph/util/result.hpp"

#include <string>
#include <filesystem>

namespace neurograph::module {

/// Loads module shared libraries (.dll on Windows, .so on Linux)
/// and registers them with a ModuleRegistry.
///
/// Each shared library must export:
///   extern "C" ModuleInterface* neurograph_create_module();
///   extern "C" void neurograph_destroy_module(ModuleInterface*);
class ModuleLoader {
public:
    explicit ModuleLoader(ModuleRegistry& registry);
    ~ModuleLoader() = default;

    /// Load a single module from a shared library path.
    util::Result<std::string> load(const std::string& path);

    /// Unload a module by name. Shuts down, unregisters, and frees the library.
    util::Result<void> unload(const std::string& name);

    /// Scan a directory for module libraries and load all found.
    /// On Windows, scans for *.dll; on Linux, scans for *.so.
    std::vector<std::string> load_directory(const std::string& dir_path);

private:
    /// Platform-specific: open a shared library, returning handle.
    static void* open_library(const std::string& path);

    /// Platform-specific: get a function pointer from a library handle.
    static void* get_symbol(void* handle, const char* name);

    /// Platform-specific: close a shared library handle.
    static void close_library(void* handle);

    /// Platform-specific: get the last error message.
    static std::string last_error();

    ModuleRegistry& registry_;
};

} // namespace neurograph::module
