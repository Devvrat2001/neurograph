#pragma once

#include "neurograph/command/command_types.hpp"
#include "neurograph/crypto/ed25519_provider.hpp"
#include "neurograph/core/neuron_id.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <vector>
#include <optional>
#include <string>

namespace neurograph::command {

class Command {
public:
    Command(CommandType type, nlohmann::json payload,
            PropagationMode mode = PropagationMode::BFS);

    // --- Identity ---
    const std::string& id() const noexcept { return id_; }
    CommandType type() const noexcept { return type_; }

    // --- Cryptographic ---
    const crypto::PublicKey& issuer_public_key() const noexcept {
        return issuer_public_key_;
    }
    const crypto::Signature& signature() const noexcept { return signature_; }

    void set_signature(const crypto::Signature& sig,
                       const crypto::PublicKey& pub_key);
    bool is_signed() const noexcept { return is_signed_; }

    // --- Timing ---
    std::chrono::system_clock::time_point timestamp() const noexcept {
        return timestamp_;
    }
    bool is_expired() const noexcept;

    /// Set command expiry duration (default: 5 minutes).
    void set_expiry(std::chrono::seconds duration) { expiry_ = duration; }

    // --- Propagation ---
    PropagationMode propagation_mode() const noexcept { return propagation_mode_; }
    uint32_t ttl() const noexcept { return ttl_; }
    void set_ttl(uint32_t ttl) { ttl_ = ttl; }
    void decrement_ttl() { if (ttl_ > 0) --ttl_; }

    std::optional<core::NeuronId> target_neuron() const { return target_neuron_; }
    void set_target_neuron(core::NeuronId id) { target_neuron_ = std::move(id); }

    // --- Payload ---
    const nlohmann::json& payload() const noexcept { return payload_; }
    nlohmann::json& payload() noexcept { return payload_; }

    // --- Serialization ---
    /// Returns the canonical byte representation for signing.
    /// Includes id, type, timestamp, ttl, payload — excludes signature.
    std::vector<uint8_t> serialize_for_signing() const;

    nlohmann::json to_json() const;
    static Command from_json(const nlohmann::json& j);

private:
    std::string id_;
    CommandType type_;
    crypto::PublicKey issuer_public_key_{};
    crypto::Signature signature_{};
    bool is_signed_ = false;
    std::chrono::system_clock::time_point timestamp_;
    std::chrono::seconds expiry_{300}; // 5 minutes default
    uint32_t ttl_ = 256;
    nlohmann::json payload_;
    std::optional<core::NeuronId> target_neuron_;
    PropagationMode propagation_mode_;
};

} // namespace neurograph::command
