// P1-11 -- License + trial persistence layer.
//
// Splits cleanly from license.cpp so the cryptographic core stays
// portable + dependency-free (just libsodium) while this file pulls
// in the platform-specific blob protection (DPAPI on Windows, plain
// pepper-XOR on POSIX).
//
// Wire format (both license.dat and trial.dat):
//
//     magic[4] = "NGLS"
//     version  = u8 (1)
//     payload_len = u32 LE
//     payload  = bytes (semantic content -- see file_kind comments)
//
// On Windows the whole framed block is then handed to CryptProtectData
// before being written out, and the file header above is BEFORE the
// CryptProtectData call (so we can refuse to even decrypt a file with
// the wrong magic). On POSIX we XOR-obfuscate with kPepper -- this is
// not security, just enough that "delete trial.dat" isn't a one-line
// reset for an aimed adversary; Phase 1 doesn't ship POSIX, so this
// path exists only so unit tests can run on Linux CI.

#include "neurograph/licensing/license.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <dpapi.h>
  #include <shlobj.h>
  // crypt32 is linked from CMakeLists.txt on Windows; no #pragma comment
  // (those don't compose with the multi-target CMake build).
#endif

namespace neurograph::licensing {

namespace {

// ---- Framing -------------------------------------------------------------

constexpr std::array<unsigned char, 4> kMagic = {'N','G','L','S'};
constexpr std::uint8_t kFileVersion = 1;

// Build-time pepper for the POSIX obfuscation fallback. NOT a secret --
// the binary is shipped to every customer. The point is to make a
// hex-editor reset of trial.dat take more than two seconds of thought.
constexpr std::array<unsigned char, 16> kPepper = {
    0x4e, 0x47, 0x4c, 0x53, 0x70, 0x65, 0x70, 0x70,
    0x65, 0x72, 0x76, 0x31, 0x32, 0x30, 0x32, 0x36,
};

void put_u32_le(std::vector<unsigned char>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFF));
    }
}

bool get_u32_le(const std::vector<unsigned char>& in, std::size_t& pos,
                std::uint32_t& out) {
    if (pos + 4 > in.size()) return false;
    out = 0;
    for (int i = 0; i < 4; ++i) {
        out |= static_cast<std::uint32_t>(in[pos + i]) << (8 * i);
    }
    pos += 4;
    return true;
}

std::vector<unsigned char> frame(const std::vector<unsigned char>& payload) {
    std::vector<unsigned char> out;
    out.reserve(payload.size() + 9);
    out.insert(out.end(), kMagic.begin(), kMagic.end());
    out.push_back(kFileVersion);
    put_u32_le(out, static_cast<std::uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

util::Result<std::vector<unsigned char>>
unframe(const std::vector<unsigned char>& blob) {
    if (blob.size() < 9) {
        return util::Error("license: file too short to contain header");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), blob.begin())) {
        return util::Error("license: bad magic (not a NeuroGraph license file)");
    }
    if (blob[4] != kFileVersion) {
        return util::Error("license: unsupported file version");
    }
    std::size_t pos = 5;
    std::uint32_t plen = 0;
    if (!get_u32_le(blob, pos, plen)) {
        return util::Error("license: truncated header");
    }
    if (pos + plen != blob.size()) {
        return util::Error("license: payload length mismatch");
    }
    return std::vector<unsigned char>(blob.begin() + pos, blob.end());
}

// ---- Blob protection ------------------------------------------------------

#ifdef _WIN32

// DPAPI wrapper. Returns an empty result on failure rather than
// throwing -- the caller can map a failed decrypt to "file is corrupt
// or was minted by a different Windows user".
util::Result<std::vector<unsigned char>>
dpapi_protect(const std::vector<unsigned char>& plain) {
    DATA_BLOB in{};
    in.cbData = static_cast<DWORD>(plain.size());
    in.pbData = const_cast<BYTE*>(plain.data());

    DATA_BLOB out{};
    if (!CryptProtectData(&in, L"NeuroGraph license envelope",
                          nullptr, nullptr, nullptr,
                          0,   // CurrentUser scope
                          &out)) {
        return util::Error("license: CryptProtectData failed");
    }
    std::vector<unsigned char> r(out.pbData, out.pbData + out.cbData);
    LocalFree(out.pbData);
    return r;
}

util::Result<std::vector<unsigned char>>
dpapi_unprotect(const std::vector<unsigned char>& ciphertext) {
    DATA_BLOB in{};
    in.cbData = static_cast<DWORD>(ciphertext.size());
    in.pbData = const_cast<BYTE*>(ciphertext.data());

    DATA_BLOB out{};
    if (!CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr,
                            0, &out)) {
        return util::Error("license: CryptUnprotectData failed (file copied "
                           "from another machine, another user account, "
                           "or has been corrupted)");
    }
    std::vector<unsigned char> r(out.pbData, out.pbData + out.cbData);
    LocalFree(out.pbData);
    return r;
}

#else

// POSIX fallback: obfuscation only. Streamed XOR with the pepper.
std::vector<unsigned char> xor_pepper(const std::vector<unsigned char>& src) {
    std::vector<unsigned char> r(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        r[i] = src[i] ^ kPepper[i % kPepper.size()];
    }
    return r;
}

util::Result<std::vector<unsigned char>>
dpapi_protect(const std::vector<unsigned char>& plain) {
    return xor_pepper(plain);
}

util::Result<std::vector<unsigned char>>
dpapi_unprotect(const std::vector<unsigned char>& ciphertext) {
    return xor_pepper(ciphertext);
}

#endif

// ---- File I/O ------------------------------------------------------------

util::Result<std::vector<unsigned char>>
read_all(const std::filesystem::path& p) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
        return util::Error("license: file does not exist: " + p.string());
    }
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        return util::Error("license: failed to open " + p.string());
    }
    f.seekg(0, std::ios::end);
    auto n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(static_cast<std::size_t>(n));
    if (n > 0) {
        f.read(reinterpret_cast<char*>(buf.data()), n);
    }
    return buf;
}

util::Result<void>
write_all(const std::filesystem::path& p,
          const std::vector<unsigned char>& bytes) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    // Atomic-ish write: write to a sibling .tmp then rename. Avoids a
    // half-written license.dat surviving a crash mid-write, which would
    // brick the install.
    auto tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return util::Error("license: cannot open for write: " + tmp.string());
        if (!bytes.empty()) {
            f.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
        }
        if (!f) return util::Error("license: write failed: " + tmp.string());
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        // On Windows, rename onto an existing file fails; try remove first.
        std::filesystem::remove(p, ec);
        std::filesystem::rename(tmp, p, ec);
        if (ec) return util::Error("license: rename failed: " + ec.message());
    }
    return util::Result<void>::success();
}

// ---- Default paths -------------------------------------------------------

std::filesystem::path appdata_dir() {
#ifdef _WIN32
    PWSTR raw = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &raw))) {
        std::filesystem::path p(raw);
        CoTaskMemFree(raw);
        return p / "NeuroGraph";
    }
    // Fall back to APPDATA env var if SHGetKnownFolderPath fails -- e.g.
    // service accounts that don't have the shell32 shim available.
    if (const char* env = std::getenv("APPDATA")) {
        return std::filesystem::path(env) / "NeuroGraph";
    }
    return std::filesystem::temp_directory_path() / "NeuroGraph";
#else
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
        return std::filesystem::path(xdg) / "neurograph";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".config" / "neurograph";
    }
    return std::filesystem::temp_directory_path() / "neurograph";
#endif
}

// ---- License envelope content -------------------------------------------
//
// We persist the original key string (UTF-8) -- not the parsed License
// fields -- and re-verify it on every load. That makes the on-disk
// blob trivial to format and means a forged blob still has to forge a
// valid Ed25519 signature against the embedded public key.

std::vector<unsigned char> build_license_payload(const std::string& key) {
    std::vector<unsigned char> p;
    p.reserve(key.size() + 4);
    put_u32_le(p, static_cast<std::uint32_t>(key.size()));
    p.insert(p.end(), key.begin(), key.end());
    return p;
}

util::Result<std::string>
parse_license_payload(const std::vector<unsigned char>& p) {
    std::size_t pos = 0;
    std::uint32_t klen = 0;
    if (!get_u32_le(p, pos, klen)) {
        return util::Error("license: truncated key field");
    }
    if (pos + klen != p.size()) {
        return util::Error("license: key length mismatch");
    }
    return std::string(reinterpret_cast<const char*>(p.data() + pos), klen);
}

// ---- Trial envelope content ---------------------------------------------

constexpr std::size_t kTrialPayloadBytes = 16;  // two i64 timestamps

void put_i64_le(std::vector<unsigned char>& out, std::int64_t v) {
    auto u = static_cast<std::uint64_t>(v);
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<unsigned char>((u >> (8 * i)) & 0xFF));
    }
}

bool get_i64_le(const std::vector<unsigned char>& in, std::size_t& pos,
                std::int64_t& out) {
    if (pos + 8 > in.size()) return false;
    std::uint64_t u = 0;
    for (int i = 0; i < 8; ++i) {
        u |= static_cast<std::uint64_t>(in[pos + i]) << (8 * i);
    }
    out = static_cast<std::int64_t>(u);
    pos += 8;
    return true;
}

std::vector<unsigned char> build_trial_payload(const TrialState& t) {
    std::vector<unsigned char> p;
    p.reserve(kTrialPayloadBytes);
    put_i64_le(p, std::chrono::duration_cast<std::chrono::seconds>(
                       t.started_at.time_since_epoch()).count());
    put_i64_le(p, std::chrono::duration_cast<std::chrono::seconds>(
                       t.high_watermark.time_since_epoch()).count());
    return p;
}

util::Result<TrialState>
parse_trial_payload(const std::vector<unsigned char>& p) {
    if (p.size() != kTrialPayloadBytes) {
        return util::Error("license: trial file payload wrong size");
    }
    std::size_t pos = 0;
    std::int64_t started = 0, hwm = 0;
    if (!get_i64_le(p, pos, started)) return util::Error("license: trial start truncated");
    if (!get_i64_le(p, pos, hwm))     return util::Error("license: trial hwm truncated");
    TrialState t;
    t.started_at     = std::chrono::system_clock::time_point(std::chrono::seconds(started));
    t.high_watermark = std::chrono::system_clock::time_point(std::chrono::seconds(hwm));
    return t;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::filesystem::path default_state_path() {
    return appdata_dir() / "license.dat";
}

std::filesystem::path default_trial_path() {
    return appdata_dir() / "trial.dat";
}

util::Result<void>
persist_license(const License& l, const std::filesystem::path& state_path) {
    auto cipher = dpapi_protect(build_license_payload(l.key));
    if (cipher.is_error()) return cipher.error();
    return write_all(state_path, frame(cipher.value()));
}

namespace {

// Shared decrypt + parse step. Returns the raw key string ready for
// verification; the caller decides whether to verify against the
// embedded production key or a caller-supplied test key.
util::Result<std::string>
load_key_string(const std::filesystem::path& state_path) {
    auto raw = read_all(state_path);
    if (raw.is_error()) return raw.error();

    auto cipher = unframe(raw.value());
    if (cipher.is_error()) return cipher.error();

    auto plain = dpapi_unprotect(cipher.value());
    if (plain.is_error()) return plain.error();

    return parse_license_payload(plain.value());
}

} // namespace

util::Result<License>
load_persisted_license(const std::filesystem::path& state_path) {
    auto key = load_key_string(state_path);
    if (key.is_error()) return key.error();
    // Re-verify against the embedded public key on every load. A
    // tampered blob therefore still has to forge an Ed25519 signature.
    return verify_key(key.value());
}

util::Result<License>
load_persisted_license_with(const std::filesystem::path& state_path,
                            const std::vector<unsigned char>& public_key) {
    auto key = load_key_string(state_path);
    if (key.is_error()) return key.error();
    return verify_key_with(key.value(), public_key);
}

util::Result<License>
activate(const std::string& key, const std::filesystem::path& state_path) {
    auto v = verify_key(key);
    if (v.is_error()) return v.error();
    if (!is_active(v.value())) {
        return util::Error("license: key has expired");
    }
    auto p = persist_license(v.value(), state_path);
    if (p.is_error()) return p.error();
    return v;
}

std::optional<License>
current_license(const std::filesystem::path& state_path) {
    auto r = load_persisted_license(state_path);
    if (r.is_error())          return std::nullopt;
    if (!is_active(r.value())) return std::nullopt;
    return r.value();
}

void clear_persisted_license(const std::filesystem::path& state_path) {
    std::error_code ec;
    std::filesystem::remove(state_path, ec);
    // Ignore ec -- "file already gone" is a successful clear.
}

// ---- Trial ---------------------------------------------------------------

util::Result<TrialState>
load_or_init_trial_at(const std::filesystem::path& trial_path,
                      std::chrono::system_clock::time_point now) {
    auto raw = read_all(trial_path);
    if (raw.is_error()) {
        // First run -- mint a fresh trial. Both timestamps start at now.
        TrialState t{now, now};
        auto cipher = dpapi_protect(build_trial_payload(t));
        if (cipher.is_error()) return cipher.error();
        auto w = write_all(trial_path, frame(cipher.value()));
        if (w.is_error()) return w.error();
        return t;
    }

    auto cipher = unframe(raw.value());
    if (cipher.is_error()) return cipher.error();
    auto plain = dpapi_unprotect(cipher.value());
    if (plain.is_error()) return plain.error();
    auto parsed = parse_trial_payload(plain.value());
    if (parsed.is_error()) return parsed.error();

    TrialState t = parsed.value();
    // Anti-rollback: high_watermark only ever moves forward.
    if (now > t.high_watermark) {
        t.high_watermark = now;
        auto cipher2 = dpapi_protect(build_trial_payload(t));
        if (cipher2.is_error()) return cipher2.error();
        auto w = write_all(trial_path, frame(cipher2.value()));
        if (w.is_error()) return w.error();
    }
    return t;
}

util::Result<TrialState>
load_or_init_trial(const std::filesystem::path& trial_path) {
    return load_or_init_trial_at(trial_path, std::chrono::system_clock::now());
}

bool trial_active(const TrialState& t, std::chrono::hours length) {
    // Use high_watermark as "effective now". Backdating the system
    // clock cannot make trial_active() return true again once it's
    // returned false -- the watermark only moves forward.
    return t.high_watermark < t.started_at + length;
}

int trial_days_remaining(const TrialState& t, std::chrono::hours length) {
    const auto deadline = t.started_at + length;
    if (t.high_watermark >= deadline) return 0;
    const auto remaining = deadline - t.high_watermark;
    const auto days = std::chrono::duration_cast<std::chrono::hours>(remaining).count() / 24;
    // Round up partial days so "5h left" displays as "1 day", not "0".
    const auto extra_hours = std::chrono::duration_cast<std::chrono::hours>(remaining).count() % 24;
    int total = static_cast<int>(days);
    if (extra_hours > 0) total += 1;
    if (total < 0) total = 0;
    return total;
}

License make_trial_license(const TrialState& t, std::chrono::hours length) {
    License l;
    l.key      = "TRIAL";
    l.holder   = "Trial user";
    l.tier     = Tier::Trial;
    l.expires  = t.started_at + length;
    l.features = {};
    return l;
}

} // namespace neurograph::licensing
