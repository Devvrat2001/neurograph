#include "neurograph/reporting/manifest.hpp"

#include <nlohmann/json.hpp>
#include <sodium.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <intrin.h>
#elif defined(__unix__) || defined(__APPLE__)
  #include <sys/utsname.h>
#endif

namespace neurograph::reporting {

namespace {

// ---- libsodium init ------------------------------------------------------
//
// sodium_init() is idempotent (subsequent calls return 1, not -1) but we
// guard with a flag so we only pay the cost once per process. Throwing
// from the static initializer would crash the loader, so init lazily.

void ensure_sodium_init() {
    static const bool ok = []() {
        if (sodium_init() < 0) {
            throw std::runtime_error("manifest: sodium_init() failed");
        }
        return true;
    }();
    (void)ok;
}

std::string hex_of(const unsigned char* bytes, std::size_t n) {
    static const char kHex[] = "0123456789abcdef";
    std::string out(n * 2, '\0');
    for (std::size_t i = 0; i < n; ++i) {
        out[2 * i + 0] = kHex[bytes[i] >> 4];
        out[2 * i + 1] = kHex[bytes[i] & 0x0F];
    }
    return out;
}

constexpr int kManifestSchemaVersion = 1;

constexpr const char* kArtifactDataset  = "dataset.csv";
constexpr const char* kArtifactPipeline = "pipeline.ngpipe";
constexpr const char* kArtifactModel    = "model.ngm";

// Streaming SHA-256: 64 KiB chunks → ~30 MB/s on a slow HDD without
// blowing the heap on multi-GB files.
constexpr std::size_t kChunkBytes = 64 * 1024;

// Verify a single artifact: returns true if the file matches `expected`,
// false on any mismatch / missing file / read error. Empty `expected`
// is treated as "not tracked" → returns true (skip).
bool verify_one(const std::filesystem::path& path, const std::string& expected) {
    if (expected.empty()) return true;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) return false;
    try {
        return sha256_of_file(path) == expected;
    } catch (const std::exception&) {
        return false;
    }
}

#if defined(_WIN32)
std::string detect_os_windows() {
    OSVERSIONINFOEXW vi{};
    vi.dwOSVersionInfoSize = sizeof(vi);
    // GetVersionExW is deprecated for app gating but still fine for a
    // best-effort version string. Microsoft's recommended API
    // (RtlGetVersion via ntdll) is not part of the public WinSDK.
#pragma warning(push)
#pragma warning(disable : 4996)
    if (GetVersionExW(reinterpret_cast<OSVERSIONINFOW*>(&vi))) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Windows %lu.%lu.%lu",
                      vi.dwMajorVersion, vi.dwMinorVersion, vi.dwBuildNumber);
        return buf;
    }
#pragma warning(pop)
    return "Windows (unknown version)";
}

std::string detect_cpu_windows() {
    // EAX=0x80000002..0x80000004 returns the 48-byte processor brand
    // string. Skip if extended CPUID isn't supported.
    int regs[4]{};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return {};

    std::array<char, 49> brand{};
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(brand.data() + i * 16, regs, sizeof(regs));
    }
    brand.back() = '\0';
    // Trim leading spaces — Intel pads the brand with them.
    std::string s(brand.data());
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    while (!s.empty() && s.back()  == '\0') s.pop_back();
    return s;
}
#endif

} // namespace

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

std::string sha256_of_bytes(const void* data, std::size_t n_bytes) {
    ensure_sodium_init();
    unsigned char digest[crypto_hash_sha256_BYTES];
    crypto_hash_sha256(digest,
                       reinterpret_cast<const unsigned char*>(data),
                       n_bytes);
    return hex_of(digest, crypto_hash_sha256_BYTES);
}

std::string sha256_of_file(const std::filesystem::path& path) {
    ensure_sodium_init();
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("sha256_of_file: cannot open " + path.string());
    }
    crypto_hash_sha256_state st;
    crypto_hash_sha256_init(&st);
    std::vector<char> buf(kChunkBytes);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize n = f.gcount();
        if (n > 0) {
            crypto_hash_sha256_update(
                &st,
                reinterpret_cast<const unsigned char*>(buf.data()),
                static_cast<unsigned long long>(n));
        }
    }
    if (f.bad()) {
        throw std::runtime_error("sha256_of_file: read failed for " + path.string());
    }
    unsigned char digest[crypto_hash_sha256_BYTES];
    crypto_hash_sha256_final(&st, digest);
    return hex_of(digest, crypto_hash_sha256_BYTES);
}

// ---------------------------------------------------------------------------
// Host info + timestamp
// ---------------------------------------------------------------------------

void populate_host_info(Manifest& m) {
#if defined(_WIN32)
    m.os  = detect_os_windows();
    m.cpu = detect_cpu_windows();
#elif defined(__unix__) || defined(__APPLE__)
    utsname u{};
    if (uname(&u) == 0) {
        m.os = std::string(u.sysname) + " " + u.release;
    }
    // CPU brand on POSIX requires reading /proc/cpuinfo or sysctl;
    // both add platform branches we don't need yet. Leave empty.
#endif
    // GPU detection deliberately deferred until P1-08 wires CUDA
    // device enumeration. Empty string → "unknown" in the manifest.
    if (m.gpu.empty()) m.gpu.clear();
}

std::string current_iso8601_utc() {
    using namespace std::chrono;
    const auto now    = system_clock::now();
    const auto secs   = time_point_cast<seconds>(now);
    const std::time_t t = system_clock::to_time_t(secs);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

// ---------------------------------------------------------------------------
// JSON I/O
// ---------------------------------------------------------------------------

namespace {

nlohmann::json to_json(const Manifest& m) {
    return {
        {"schema_version",     kManifestSchemaVersion},
        {"neurograph_version", m.neurograph_version},
        {"build_git_sha",      m.build_git_sha},
        {"dataset_sha256",     m.dataset_sha256},
        {"pipeline_sha256",    m.pipeline_sha256},
        {"model_sha256",       m.model_sha256},
        {"rng_seed",           m.rng_seed},
        {"os",                 m.os},
        {"cpu",                m.cpu},
        {"gpu",                m.gpu},
        {"timestamp_iso8601",  m.timestamp_iso8601},
        {"pipeline_config",    m.pipeline_config},
        {"training_config",    m.training_config},
    };
}

Manifest from_json(const nlohmann::json& j) {
    if (!j.contains("schema_version")) {
        throw std::runtime_error("manifest: missing schema_version");
    }
    const int v = j.at("schema_version").get<int>();
    if (v != kManifestSchemaVersion) {
        throw std::runtime_error(
            "manifest: unsupported schema_version " + std::to_string(v));
    }
    Manifest m;
    m.neurograph_version = j.value("neurograph_version", "");
    m.build_git_sha      = j.value("build_git_sha",      "");
    m.dataset_sha256     = j.value("dataset_sha256",     "");
    m.pipeline_sha256    = j.value("pipeline_sha256",    "");
    m.model_sha256       = j.value("model_sha256",       "");
    m.rng_seed           = j.value("rng_seed",           std::uint64_t{0});
    m.os                 = j.value("os",                 "");
    m.cpu                = j.value("cpu",                "");
    m.gpu                = j.value("gpu",                "");
    m.timestamp_iso8601  = j.value("timestamp_iso8601",  "");
    if (j.contains("pipeline_config")) m.pipeline_config = j.at("pipeline_config");
    if (j.contains("training_config")) m.training_config = j.at("training_config");
    return m;
}

} // namespace

void save_manifest(const Manifest& m, const std::filesystem::path& path) {
    const std::string body = to_json(m).dump(2);
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("save_manifest: cannot open " + path.string());
    }
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (!f.good()) {
        throw std::runtime_error("save_manifest: write failed for " + path.string());
    }
}

Manifest load_manifest(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("load_manifest: cannot open " + path.string());
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("load_manifest: invalid JSON: ") + e.what());
    }
    return from_json(j);
}

bool verify_against(const Manifest& m, const std::filesystem::path& bundle_dir) {
    if (!verify_one(bundle_dir / kArtifactDataset,  m.dataset_sha256))  return false;
    if (!verify_one(bundle_dir / kArtifactPipeline, m.pipeline_sha256)) return false;
    if (!verify_one(bundle_dir / kArtifactModel,    m.model_sha256))    return false;
    return true;
}

} // namespace neurograph::reporting
