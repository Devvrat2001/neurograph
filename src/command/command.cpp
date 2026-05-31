#include "neurograph/command/command.hpp"
#include "neurograph/util/uuid.hpp"

#include <sstream>
#include <iomanip>

namespace neurograph::command {

Command::Command(CommandType type, nlohmann::json payload, PropagationMode mode)
    : id_(util::generate_uuid())
    , type_(type)
    , timestamp_(std::chrono::system_clock::now())
    , payload_(std::move(payload))
    , propagation_mode_(mode) {}

void Command::set_signature(const crypto::Signature& sig,
                             const crypto::PublicKey& pub_key) {
    signature_ = sig;
    issuer_public_key_ = pub_key;
    is_signed_ = true;
}

bool Command::is_expired() const noexcept {
    auto now = std::chrono::system_clock::now();
    return (now - timestamp_) > expiry_;
}

std::vector<uint8_t> Command::serialize_for_signing() const {
    // Build a deterministic canonical representation:
    // id | type(1 byte) | timestamp(8 bytes) | ttl(4 bytes) | mode(1 byte) | payload_json
    std::vector<uint8_t> data;

    // ID string
    data.insert(data.end(), id_.begin(), id_.end());

    // Command type (1 byte)
    data.push_back(static_cast<uint8_t>(type_));

    // Timestamp as epoch microseconds (8 bytes, big-endian)
    auto epoch_us = std::chrono::duration_cast<std::chrono::microseconds>(
        timestamp_.time_since_epoch()).count();
    for (int i = 7; i >= 0; --i) {
        data.push_back(static_cast<uint8_t>((epoch_us >> (i * 8)) & 0xFF));
    }

    // TTL (4 bytes, big-endian)
    for (int i = 3; i >= 0; --i) {
        data.push_back(static_cast<uint8_t>((ttl_ >> (i * 8)) & 0xFF));
    }

    // Propagation mode (1 byte)
    data.push_back(static_cast<uint8_t>(propagation_mode_));

    // Target neuron ID (if present)
    if (target_neuron_) {
        const auto& tid = target_neuron_->value();
        data.insert(data.end(), tid.begin(), tid.end());
    }

    // Payload as canonical JSON string
    std::string payload_str = payload_.dump(-1, ' ', false,
                                            nlohmann::json::error_handler_t::replace);
    data.insert(data.end(), payload_str.begin(), payload_str.end());

    return data;
}

nlohmann::json Command::to_json() const {
    nlohmann::json j = {
        {"id", id_},
        {"type", static_cast<int>(type_)},
        {"type_name", std::string(command_type_str(type_))},
        {"propagation_mode", static_cast<int>(propagation_mode_)},
        {"ttl", ttl_},
        {"payload", payload_},
        {"is_signed", is_signed_}
    };

    // Timestamp as ISO 8601
    auto epoch = std::chrono::system_clock::to_time_t(timestamp_);
    std::ostringstream ts;
    ts << std::put_time(std::gmtime(&epoch), "%Y-%m-%dT%H:%M:%SZ");
    j["timestamp"] = ts.str();

    if (target_neuron_) {
        j["target_neuron"] = target_neuron_->value();
    }

    if (is_signed_) {
        // Encode public key and signature as hex
        auto to_hex = [](const auto& arr) {
            std::ostringstream hex;
            hex << std::hex << std::setfill('0');
            for (auto b : arr) hex << std::setw(2) << static_cast<int>(b);
            return hex.str();
        };
        j["issuer_public_key"] = to_hex(issuer_public_key_);
        j["signature"] = to_hex(signature_);
    }

    return j;
}

Command Command::from_json(const nlohmann::json& j) {
    auto type = static_cast<CommandType>(j.at("type").get<int>());
    auto mode = static_cast<PropagationMode>(j.value("propagation_mode", 0));
    auto payload = j.value("payload", nlohmann::json::object());

    Command cmd(type, std::move(payload), mode);

    // Restore id (overwrite the generated one)
    // Access private field via a small trick: reconstruct with the right id
    // Since id_ is set in constructor, we need to set it explicitly
    if (j.contains("id")) {
        cmd.id_ = j["id"].get<std::string>();
    }

    if (j.contains("ttl")) cmd.ttl_ = j["ttl"].get<uint32_t>();

    if (j.contains("target_neuron")) {
        cmd.target_neuron_ = core::NeuronId(j["target_neuron"].get<std::string>());
    }

    return cmd;
}

} // namespace neurograph::command
