#include "neurograph/net/protocol.hpp"
#include <chrono>
#include <stdexcept>

namespace neurograph::net {

std::string message_type_to_string(MessageType type) {
    switch (type) {
        case MessageType::HANDSHAKE:  return "HANDSHAKE";
        case MessageType::SIGNAL:     return "SIGNAL";
        case MessageType::COMMAND:    return "COMMAND";
        case MessageType::QUERY:      return "QUERY";
        case MessageType::RESPONSE:   return "RESPONSE";
        case MessageType::HEARTBEAT:  return "HEARTBEAT";
        case MessageType::DISCONNECT: return "DISCONNECT";
    }
    return "UNKNOWN";
}

MessageType message_type_from_string(const std::string& str) {
    if (str == "HANDSHAKE")  return MessageType::HANDSHAKE;
    if (str == "SIGNAL")     return MessageType::SIGNAL;
    if (str == "COMMAND")    return MessageType::COMMAND;
    if (str == "QUERY")      return MessageType::QUERY;
    if (str == "RESPONSE")   return MessageType::RESPONSE;
    if (str == "HEARTBEAT")  return MessageType::HEARTBEAT;
    if (str == "DISCONNECT") return MessageType::DISCONNECT;
    return MessageType::HEARTBEAT; // fallback
}

static uint64_t now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

nlohmann::json Message::to_json() const {
    return {
        {"type", message_type_to_string(type)},
        {"sender_id", sender_id},
        {"target_id", target_id},
        {"sequence", sequence},
        {"timestamp", timestamp},
        {"payload", payload},
    };
}

Message Message::from_json(const nlohmann::json& j) {
    Message msg;
    msg.type = message_type_from_string(j.at("type").get<std::string>());
    msg.sender_id = j.at("sender_id").get<std::string>();
    msg.target_id = j.value("target_id", "*");
    msg.sequence = j.value("sequence", uint64_t{0});
    msg.timestamp = j.value("timestamp", uint64_t{0});
    msg.payload = j.value("payload", nlohmann::json::object());
    return msg;
}

std::vector<uint8_t> serialize_message(const Message& msg) {
    std::string json_str = msg.to_json().dump();
    uint32_t len = static_cast<uint32_t>(json_str.size());

    std::vector<uint8_t> buf(4 + len);
    write_length_prefix(buf.data(), len);
    std::memcpy(buf.data() + 4, json_str.data(), len);
    return buf;
}

Message deserialize_message(const uint8_t* data, size_t len) {
    std::string json_str(reinterpret_cast<const char*>(data), len);
    auto j = nlohmann::json::parse(json_str);
    return Message::from_json(j);
}

uint32_t read_length_prefix(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
            static_cast<uint32_t>(data[3]);
}

void write_length_prefix(uint8_t* dest, uint32_t length) {
    dest[0] = static_cast<uint8_t>((length >> 24) & 0xFF);
    dest[1] = static_cast<uint8_t>((length >> 16) & 0xFF);
    dest[2] = static_cast<uint8_t>((length >> 8) & 0xFF);
    dest[3] = static_cast<uint8_t>(length & 0xFF);
}

Message make_handshake(const std::string& network_id, const std::string& network_name) {
    Message msg;
    msg.type = MessageType::HANDSHAKE;
    msg.sender_id = network_id;
    msg.target_id = "*";
    msg.timestamp = now_ms();
    msg.payload = {
        {"network_name", network_name},
        {"protocol_version", "1.0"},
    };
    return msg;
}

Message make_signal(const std::string& sender_id,
                    const std::vector<std::pair<std::string, double>>& signals) {
    nlohmann::json signal_array = nlohmann::json::array();
    for (const auto& [neuron_id, value] : signals) {
        signal_array.push_back({{"neuron_id", neuron_id}, {"value", value}});
    }

    Message msg;
    msg.type = MessageType::SIGNAL;
    msg.sender_id = sender_id;
    msg.target_id = "*";
    msg.timestamp = now_ms();
    msg.payload = {{"signals", signal_array}};
    return msg;
}

Message make_heartbeat(const std::string& sender_id) {
    Message msg;
    msg.type = MessageType::HEARTBEAT;
    msg.sender_id = sender_id;
    msg.target_id = "*";
    msg.timestamp = now_ms();
    return msg;
}

Message make_disconnect(const std::string& sender_id, const std::string& reason) {
    Message msg;
    msg.type = MessageType::DISCONNECT;
    msg.sender_id = sender_id;
    msg.target_id = "*";
    msg.timestamp = now_ms();
    msg.payload = {{"reason", reason}};
    return msg;
}

} // namespace neurograph::net
