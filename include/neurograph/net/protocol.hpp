#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace neurograph::net {

/// Wire protocol for inter-NeuroGraph communication.
///
/// Message format (binary):
///   [4 bytes] payload length (big-endian uint32)
///   [N bytes] JSON payload (UTF-8)
///
/// Message types:
///   - HANDSHAKE: Initial connection setup, exchange network IDs
///   - SIGNAL:    Send activation signals to remote neurons
///   - COMMAND:   Forward a signed command to a remote network
///   - QUERY:     Request state from a remote network
///   - RESPONSE:  Reply to a query
///   - HEARTBEAT: Keep-alive ping
///   - DISCONNECT: Graceful shutdown

enum class MessageType : uint8_t {
    HANDSHAKE  = 0,
    SIGNAL     = 1,
    COMMAND    = 2,
    QUERY      = 3,
    RESPONSE   = 4,
    HEARTBEAT  = 5,
    DISCONNECT = 6,
};

std::string message_type_to_string(MessageType type);
MessageType message_type_from_string(const std::string& str);

/// A protocol message exchanged between NeuroGraph instances.
struct Message {
    MessageType type;
    std::string sender_id;      // Network ID of sender
    std::string target_id;      // Network ID of target (or "*" for broadcast)
    uint64_t sequence = 0;      // Monotonic sequence number
    uint64_t timestamp = 0;     // Unix timestamp ms
    nlohmann::json payload;     // Type-specific data

    nlohmann::json to_json() const;
    static Message from_json(const nlohmann::json& j);
};

/// Serialize a Message to wire format (length-prefixed JSON).
std::vector<uint8_t> serialize_message(const Message& msg);

/// Deserialize a Message from a complete JSON buffer.
/// The caller is responsible for reading the 4-byte length prefix
/// and providing exactly that many bytes.
Message deserialize_message(const uint8_t* data, size_t len);

/// Read a 4-byte big-endian length prefix from a buffer.
uint32_t read_length_prefix(const uint8_t* data);

/// Write a 4-byte big-endian length prefix into a buffer.
void write_length_prefix(uint8_t* dest, uint32_t length);

/// Create a handshake message.
Message make_handshake(const std::string& network_id, const std::string& network_name);

/// Create a signal message (batch of neuron activations).
Message make_signal(const std::string& sender_id,
                    const std::vector<std::pair<std::string, double>>& signals);

/// Create a heartbeat message.
Message make_heartbeat(const std::string& sender_id);

/// Create a disconnect message.
Message make_disconnect(const std::string& sender_id, const std::string& reason);

} // namespace neurograph::net
