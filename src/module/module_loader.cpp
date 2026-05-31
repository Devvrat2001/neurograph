#include "neurograph/module/module_loader.hpp"
#include "neurograph/util/logger.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace neurograph::module {

ModuleLoader::ModuleLoader(ModuleRegistry& registry)
    : registry_(registry) {}

util::Result<std::string> ModuleLoader::load(const std::string& path) {
    // Check file exists
    if (!std::filesystem::exists(path)) {
        return util::Error("Module file not found: " + path);
    }

    // Open the shared library
    void* handle = open_library(path);
    if (!handle) {
        return util::Error("Failed to load library " + path + ": " + last_error());
    }

    // Look up factory functions
    auto create_fn = reinterpret_cast<CreateModuleFunc>(
        get_symbol(handle, "neurograph_create_module"));
    if (!create_fn) {
        close_library(handle);
        return util::Error("Missing neurograph_create_module in: " + path);
    }

    auto destroy_fn = reinterpret_cast<DestroyModuleFunc>(
        get_symbol(handle, "neurograph_destroy_module"));
    if (!destroy_fn) {
        close_library(handle);
        return util::Error("Missing neurograph_destroy_module in: " + path);
    }

    // Create the module instance
    ModuleInterface* instance = create_fn();
    if (!instance) {
        close_library(handle);
        return util::Error("neurograph_create_module returned null in: " + path);
    }

    // Get module info
    ModuleInfo info = instance->info();
    std::string name = info.name;

    NG_LOG_INFO("Loaded module: {} v{} from {}", name, info.version, path);

    // Register it
    ModuleEntry entry;
    entry.info = std::move(info);
    entry.instance = instance;
    entry.destroy = destroy_fn;
    entry.library_handle = handle;
    entry.file_path = path;
    entry.initialized = false;

    auto reg_result = registry_.register_module(std::move(entry));
    if (reg_result.is_error()) {
        destroy_fn(instance);
        close_library(handle);
        return util::Error(reg_result.error().message);
    }

    return name;
}

util::Result<void> ModuleLoader::unload(const std::string& name) {
    // The registry handles shutdown and destroy.
    // We need to get the library handle before unregistering.
    auto* mod = registry_.get_module(name);
    if (!mod) {
        return util::Error("Module not loaded: " + name);
    }

    // Unregister (shuts down + destroys instance)
    auto result = registry_.unregister_module(name);
    if (result.is_error()) {
        return result;
    }

    // Note: We don't close the library handle here because the registry
    // already destroyed the instance. Closing the library while the
    // vtable might still be referenced could cause issues. In practice,
    // the library stays loaded until process exit. For full cleanup,
    // the library handle is stored in the ModuleEntry.

    NG_LOG_INFO("Unloaded module: {}", name);
    return util::Result<void>::success();
}

std::vector<std::string> ModuleLoader::load_directory(const std::string& dir_path) {
    std::vector<std::string> loaded;

    if (!std::filesystem::is_directory(dir_path)) {
        NG_LOG_WARN("Module directory not found: {}", dir_path);
        return loaded;
    }

#ifdef _WIN32
    const std::string extension = ".dll";
#else
    const std::string extension = ".so";
#endif

    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != extension) continue;

        auto result = load(entry.path().string());
        if (result.ok()) {
            loaded.push_back(result.value());
        } else {
            NG_LOG_WARN("Failed to load {}: {}", entry.path().string(), result.error().message);
        }
    }

    NG_LOG_INFO("Loaded {} modules from {}", loaded.size(), dir_path);
    return loaded;
}

// ============================================================
// Platform-specific implementations
// ============================================================

#ifdef _WIN32

void* ModuleLoader::open_library(const std::string& path) {
    return static_cast<void*>(LoadLibraryA(path.c_str()));
}

void* ModuleLoader::get_symbol(void* handle, const char* name) {
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle), name));
}

void ModuleLoader::close_library(void* handle) {
    FreeLibrary(static_cast<HMODULE>(handle));
}

std::string ModuleLoader::last_error() {
    DWORD err = GetLastError();
    if (err == 0) return "Unknown error";

    LPSTR buf = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buf), 0, nullptr);

    std::string msg(buf, size);
    LocalFree(buf);
    return msg;
}

#else

void* ModuleLoader::open_library(const std::string& path) {
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void* ModuleLoader::get_symbol(void* handle, const char* name) {
    return dlsym(handle, name);
}

void ModuleLoader::close_library(void* handle) {
    dlclose(handle);
}

std::string ModuleLoader::last_error() {
    const char* err = dlerror();
    return err ? std::string(err) : "Unknown error";
}

#endif

} // namespace neurograph::module
