#pragma once

#include "neurograph/crypto/ed25519_provider.hpp"
#include "neurograph/util/result.hpp"
#include <string>
#include <unordered_set>
#include <vector>
#include <optional>

namespace neurograph::crypto {

/// Manages Ed25519 key pairs: generation, storage, loading, and master key authorization.
class KeyManager {
public:
    explicit KeyManager(const Ed25519Provider& provider);

    /// Generate a new master key pair and store it.
    KeyPair generate_master_key();

    /// Load a key pair from files (public_key_path, secret_key_path).
    /// Files contain raw 32/64-byte keys.
    util::Result<KeyPair> load_keypair(const std::string& public_key_path,
                                       const std::string& secret_key_path) const;

    /// Save a key pair to files.
    util::Result<void> save_keypair(const KeyPair& kp,
                                    const std::string& public_key_path,
                                    const std::string& secret_key_path) const;

    /// Register a public key as an authorized master key.
    void authorize_master(const PublicKey& public_key);

    /// Revoke a previously authorized master key.
    void revoke_master(const PublicKey& public_key);

    /// Check if a public key is an authorized master.
    bool is_authorized_master(const PublicKey& public_key) const;

    /// Get all authorized master public keys.
    const std::vector<PublicKey>& authorized_masters() const noexcept {
        return authorized_masters_;
    }

    /// Get the current master key pair (if generated/loaded).
    const std::optional<KeyPair>& master_keypair() const noexcept {
        return master_keypair_;
    }

    /// Set the master key pair directly.
    void set_master_keypair(KeyPair kp);

private:
    const Ed25519Provider& provider_;
    std::optional<KeyPair> master_keypair_;
    std::vector<PublicKey> authorized_masters_;
};

} // namespace neurograph::crypto
