#include "neurograph/licensing/license.hpp"

#include <sodium.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace neurograph::licensing {

namespace {

// ---- libsodium init ------------------------------------------------------

void ensure_sodium_init() {
    static const bool ok = []() {
        if (sodium_init() < 0) {
            throw std::runtime_error("license: sodium_init() failed");
        }
        return true;
    }();
    (void)ok;
}

// ---- Wire format ---------------------------------------------------------
//
// Versioned packed binary so we can evolve later without breaking issued
// keys. All multi-byte integers are little-endian (matches the host on
// every platform we ship to; we serialize/deserialize byte-by-byte so
// host endianness doesn't actually matter for cross-platform).
//
//   1 byte   : schema version (=1)
//   1 byte   : tier
//   8 bytes  : expires_unix (int64, LE)
//   2 bytes  : holder length (uint16, LE)
//   N bytes  : holder UTF-8
//   2 bytes  : feature count (uint16, LE)
//   for each feature:
//     2 bytes : length (uint16, LE)
//     M bytes : feature name UTF-8
//   followed by 64 bytes of detached Ed25519 signature.

constexpr std::uint8_t  kPayloadVersion = 1;
constexpr std::size_t   kSigBytes       = crypto_sign_BYTES;        // 64
constexpr std::size_t   kPubKeyBytes    = crypto_sign_PUBLICKEYBYTES; // 32
constexpr std::size_t   kSecKeyBytes    = crypto_sign_SECRETKEYBYTES; // 64

void put_u16(std::vector<unsigned char>& out, std::uint16_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xFF));
    out.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}

void put_i64(std::vector<unsigned char>& out, std::int64_t v) {
    auto u = static_cast<std::uint64_t>(v);
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<unsigned char>((u >> (8 * i)) & 0xFF));
    }
}

void put_str(std::vector<unsigned char>& out, const std::string& s) {
    if (s.size() > 0xFFFF) {
        throw std::runtime_error("license: string field exceeds 65535 bytes");
    }
    put_u16(out, static_cast<std::uint16_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

bool get_u16(const std::vector<unsigned char>& in, std::size_t& pos,
             std::uint16_t& out) {
    if (pos + 2 > in.size()) return false;
    out = static_cast<std::uint16_t>(in[pos]) |
          (static_cast<std::uint16_t>(in[pos + 1]) << 8);
    pos += 2;
    return true;
}

bool get_i64(const std::vector<unsigned char>& in, std::size_t& pos,
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

bool get_str(const std::vector<unsigned char>& in, std::size_t& pos,
             std::string& out) {
    std::uint16_t n = 0;
    if (!get_u16(in, pos, n)) return false;
    if (pos + n > in.size())  return false;
    out.assign(reinterpret_cast<const char*>(in.data() + pos), n);
    pos += n;
    return true;
}

std::vector<unsigned char> serialize_payload(const LicensePayload& p) {
    std::vector<unsigned char> out;
    out.reserve(64 + p.holder.size());
    out.push_back(kPayloadVersion);
    out.push_back(static_cast<unsigned char>(p.tier));
    put_i64(out, std::chrono::duration_cast<std::chrono::seconds>(
                     p.expires.time_since_epoch())
                     .count());
    put_str(out, p.holder);
    if (p.features.size() > 0xFFFF) {
        throw std::runtime_error("license: too many features");
    }
    put_u16(out, static_cast<std::uint16_t>(p.features.size()));
    for (const auto& f : p.features) put_str(out, f);
    return out;
}

bool deserialize_payload(const std::vector<unsigned char>& bytes,
                          std::size_t end,
                          LicensePayload& out) {
    std::size_t pos = 0;
    if (pos + 2 > end) return false;
    if (bytes[pos++] != kPayloadVersion) return false;
    const auto tier_byte = bytes[pos++];
    if (tier_byte > static_cast<std::uint8_t>(Tier::Enterprise)) return false;
    out.tier = static_cast<Tier>(tier_byte);

    std::int64_t exp = 0;
    if (!get_i64(bytes, pos, exp)) return false;
    out.expires = std::chrono::system_clock::time_point(
        std::chrono::seconds(exp));

    if (!get_str(bytes, pos, out.holder)) return false;

    std::uint16_t n_features = 0;
    if (!get_u16(bytes, pos, n_features)) return false;
    out.features.clear();
    out.features.reserve(n_features);
    for (std::uint16_t i = 0; i < n_features; ++i) {
        std::string f;
        if (!get_str(bytes, pos, f)) return false;
        out.features.push_back(std::move(f));
    }
    return pos == end;
}

// ---- Base32 (RFC 4648, no padding, uppercase) ----------------------------
//
// Custom impl rather than pulling in another vcpkg dep. Ed25519 needs
// 64-byte sigs so the encoded key string is ~140 chars; that's fine.

constexpr const char* kB32Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

int b32_decode_one(char c) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= '2' && c <= '7') return 26 + (c - '2');
    return -1;
}

std::string base32_encode(const unsigned char* data, std::size_t n) {
    std::string out;
    out.reserve((n * 8 + 4) / 5);
    int    buffer = 0;
    int    bits   = 0;
    for (std::size_t i = 0; i < n; ++i) {
        buffer = (buffer << 8) | data[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            const int idx = (buffer >> bits) & 0x1F;
            out.push_back(kB32Alphabet[idx]);
        }
    }
    if (bits > 0) {
        const int idx = (buffer << (5 - bits)) & 0x1F;
        out.push_back(kB32Alphabet[idx]);
    }
    return out;
}

bool base32_decode(const std::string& s, std::vector<unsigned char>& out) {
    out.clear();
    out.reserve((s.size() * 5) / 8);
    int buffer = 0;
    int bits   = 0;
    for (char c : s) {
        if (c == '-' || c == ' ') continue;
        const int v = b32_decode_one(c);
        if (v < 0) return false;
        buffer = (buffer << 5) | v;
        bits += 5;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((buffer >> bits) & 0xFF));
        }
    }
    // Trailing partial group (< 8 bits remaining) is dropped — this is
    // standard for unpadded base32 of byte-aligned input.
    return true;
}

std::string group_with_dashes(const std::string& s, std::size_t group = 4) {
    std::string out;
    out.reserve(s.size() + s.size() / group);
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i > 0 && i % group == 0) out.push_back('-');
        out.push_back(s[i]);
    }
    return out;
}

// ---- Embedded public key -------------------------------------------------
//
// Production builds replace this with a real ops-issued public key
// (typically via a generated header included here). For now it's all
// zeros so verify_key() consistently fails on production-style calls
// during development, but the rest of the app can wire up the API.

const std::vector<unsigned char>& placeholder_public_key() {
    static const std::vector<unsigned char> k(kPubKeyBytes, 0);
    return k;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Keypair generate_keypair() {
    ensure_sodium_init();
    Keypair kp;
    kp.public_key.resize(kPubKeyBytes);
    kp.secret_key.resize(kSecKeyBytes);
    crypto_sign_keypair(kp.public_key.data(), kp.secret_key.data());
    return kp;
}

util::Result<std::string>
issue_key(const LicensePayload& p,
          const std::vector<unsigned char>& secret_key) {
    ensure_sodium_init();
    if (secret_key.size() != kSecKeyBytes) {
        return util::Error("license: secret key wrong size");
    }
    std::vector<unsigned char> payload;
    try {
        payload = serialize_payload(p);
    } catch (const std::exception& e) {
        return util::Error(std::string("license: ") + e.what());
    }

    std::vector<unsigned char> sig(kSigBytes);
    crypto_sign_detached(sig.data(), nullptr,
                         payload.data(), payload.size(),
                         secret_key.data());

    std::vector<unsigned char> framed;
    framed.reserve(payload.size() + sig.size());
    framed.insert(framed.end(), payload.begin(), payload.end());
    framed.insert(framed.end(), sig.begin(),     sig.end());

    return group_with_dashes(base32_encode(framed.data(), framed.size()));
}

util::Result<License>
verify_key_with(const std::string& key,
                const std::vector<unsigned char>& public_key) {
    ensure_sodium_init();
    if (public_key.size() != kPubKeyBytes) {
        return util::Error("license: public key wrong size");
    }
    std::vector<unsigned char> framed;
    if (!base32_decode(key, framed)) {
        return util::Error("license: invalid characters in key");
    }
    if (framed.size() < kSigBytes + 12) {
        // 12 = minimum payload size: ver+tier+8(expires)+2(holder len 0)+...
        return util::Error("license: key too short");
    }
    const std::size_t payload_len = framed.size() - kSigBytes;
    const unsigned char* payload  = framed.data();
    const unsigned char* sig      = framed.data() + payload_len;

    if (crypto_sign_verify_detached(sig, payload, payload_len,
                                    public_key.data()) != 0) {
        return util::Error("license: signature verification failed");
    }

    LicensePayload p;
    if (!deserialize_payload(framed, payload_len, p)) {
        return util::Error("license: payload decode failed");
    }

    License lic;
    lic.key      = key;
    lic.holder   = std::move(p.holder);
    lic.tier     = p.tier;
    lic.expires  = p.expires;
    lic.features = std::move(p.features);
    return lic;
}

util::Result<License> verify_key(const std::string& key) {
    return verify_key_with(key, embedded_public_key());
}

const std::vector<unsigned char>& embedded_public_key() {
    return placeholder_public_key();
}

bool is_active(const License& l) {
    return l.expires > std::chrono::system_clock::now();
}

bool has_feature(const License& l, std::string_view feature) {
    return std::any_of(l.features.begin(), l.features.end(),
                       [feature](const std::string& f) { return f == feature; });
}

} // namespace neurograph::licensing
