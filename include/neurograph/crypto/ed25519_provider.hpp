#pragma once

#include <vector>
#include <cstdint>
#include <array>

namespace neurograph::crypto {

/// Size constants for Ed25519 (libsodium).
constexpr size_t ED25519_PUBLIC_KEY_SIZE = 32;
constexpr size_t ED25519_SECRET_KEY_SIZE = 64;
constexpr size_t ED25519_SIGNATURE_SIZE = 64;
constexpr size_t ED25519_SEED_SIZE = 32;

using PublicKey = std::array<uint8_t, ED25519_PUBLIC_KEY_SIZE>;
using SecretKey = std::array<uint8_t, ED25519_SECRET_KEY_SIZE>;
using Signature = std::array<uint8_t, ED25519_SIGNATURE_SIZE>;

/// Key pair for Ed25519 signing.
struct KeyPair {
    PublicKey public_key;
    SecretKey secret_key;
};

/// Thin wrapper over libsodium's Ed25519 functions.
/// Handles sodium_init, key generation, signing, and verification.
class Ed25519Provider {
public:
    Ed25519Provider();

    /// Generate a new random Ed25519 key pair.
    KeyPair generate_keypair() const;

    /// Generate a key pair from a 32-byte seed (deterministic).
    KeyPair keypair_from_seed(const std::array<uint8_t, ED25519_SEED_SIZE>& seed) const;

    /// Sign a message with the secret key. Returns 64-byte signature.
    Signature sign(const std::vector<uint8_t>& message,
                   const SecretKey& secret_key) const;

    /// Verify a signature against a message and public key.
    bool verify(const std::vector<uint8_t>& message,
                const Signature& signature,
                const PublicKey& public_key) const;
};

} // namespace neurograph::crypto
