#include <gtest/gtest.h>
#include "neurograph/net/protocol.hpp"

using namespace neurograph::net;

// ============================================================
// MessageType Tests
// ============================================================

TEST(MessageType, ToStringRoundtrip) {
    EXPECT_EQ(message_type_to_string(MessageType::HANDSHAKE), "HANDSHAKE");
    EXPECT_EQ(message_type_to_string(MessageType::SIGNAL), "SIGNAL");
    EXPECT_EQ(message_type_to_string(MessageType::COMMAND), "COMMAND");
    EXPECT_EQ(message_type_to_string(MessageType::QUERY), "QUERY");
    EXPECT_EQ(message_type_to_string(MessageType::RESPONSE), "RESPONSE");
    EXPECT_EQ(message_type_to_string(MessageType::HEARTBEAT), "HEARTBEAT");
    EXPECT_EQ(message_type_to_string(MessageType::DISCONNECT), "DISCONNECT");
}

TEST(MessageType, FromString) {
    EXPECT_EQ(message_type_from_string("HANDSHAKE"), MessageType::HANDSHAKE);
    EXPECT_EQ(message_type_from_string("SIGNAL"), MessageType::SIGNAL);
    EXPECT_EQ(message_type_from_string("DISCONNECT"), MessageType::DISCONNECT);
}

// ============================================================
// Message Tests
// ============================================================

TEST(Message, JsonRoundtrip) {
    Message msg;
    msg.type = MessageType::SIGNAL;
    msg.sender_id = "net-001";
    msg.target_id = "net-002";
    msg.sequence = 42;
    msg.timestamp = 1234567890;
    msg.payload = {{"key", "value"}};

    auto j = msg.to_json();
    auto restored = Message::from_json(j);

    EXPECT_EQ(restored.type, MessageType::SIGNAL);
    EXPECT_EQ(restored.sender_id, "net-001");
    EXPECT_EQ(restored.target_id, "net-002");
    EXPECT_EQ(restored.sequence, 42);
    EXPECT_EQ(restored.timestamp, 1234567890);
    EXPECT_EQ(restored.payload["key"], "value");
}

TEST(Message, DefaultTargetId) {
    nlohmann::json j = {
        {"type", "HEARTBEAT"},
        {"sender_id", "net-001"},
    };
    auto msg = Message::from_json(j);
    EXPECT_EQ(msg.target_id, "*");
}

// ============================================================
// Serialization Tests
// ============================================================

TEST(Protocol, SerializeDeserialize) {
    Message original;
    original.type = MessageType::SIGNAL;
    original.sender_id = "test-net";
    original.target_id = "*";
    original.sequence = 1;
    original.payload = {
        {"signals", {{{"neuron_id", "abc"}, {"value", 0.75}}}}
    };

    auto wire = serialize_message(original);

    // Check length prefix
    ASSERT_GE(wire.size(), 4);
    uint32_t len = read_length_prefix(wire.data());
    EXPECT_EQ(len, wire.size() - 4);

    // Deserialize
    auto restored = deserialize_message(wire.data() + 4, len);
    EXPECT_EQ(restored.type, MessageType::SIGNAL);
    EXPECT_EQ(restored.sender_id, "test-net");
    EXPECT_EQ(restored.payload["signals"][0]["neuron_id"], "abc");
    EXPECT_DOUBLE_EQ(restored.payload["signals"][0]["value"], 0.75);
}

TEST(Protocol, LengthPrefix) {
    uint8_t buf[4];

    write_length_prefix(buf, 0);
    EXPECT_EQ(read_length_prefix(buf), 0u);

    write_length_prefix(buf, 256);
    EXPECT_EQ(read_length_prefix(buf), 256u);

    write_length_prefix(buf, 0x01020304);
    EXPECT_EQ(read_length_prefix(buf), 0x01020304u);

    write_length_prefix(buf, 0xFFFFFFFF);
    EXPECT_EQ(read_length_prefix(buf), 0xFFFFFFFFu);
}

// ============================================================
// Factory Function Tests
// ============================================================

TEST(Protocol, MakeHandshake) {
    auto msg = make_handshake("net-001", "MyNetwork");
    EXPECT_EQ(msg.type, MessageType::HANDSHAKE);
    EXPECT_EQ(msg.sender_id, "net-001");
    EXPECT_EQ(msg.target_id, "*");
    EXPECT_EQ(msg.payload["network_name"], "MyNetwork");
    EXPECT_EQ(msg.payload["protocol_version"], "1.0");
    EXPECT_GT(msg.timestamp, 0u);
}

TEST(Protocol, MakeSignal) {
    std::vector<std::pair<std::string, double>> signals = {
        {"neuron-a", 0.5},
        {"neuron-b", 0.9},
    };

    auto msg = make_signal("net-001", signals);
    EXPECT_EQ(msg.type, MessageType::SIGNAL);
    EXPECT_EQ(msg.sender_id, "net-001");

    auto& sigs = msg.payload["signals"];
    ASSERT_EQ(sigs.size(), 2);
    EXPECT_EQ(sigs[0]["neuron_id"], "neuron-a");
    EXPECT_DOUBLE_EQ(sigs[0]["value"], 0.5);
    EXPECT_EQ(sigs[1]["neuron_id"], "neuron-b");
    EXPECT_DOUBLE_EQ(sigs[1]["value"], 0.9);
}

TEST(Protocol, MakeHeartbeat) {
    auto msg = make_heartbeat("net-001");
    EXPECT_EQ(msg.type, MessageType::HEARTBEAT);
    EXPECT_EQ(msg.sender_id, "net-001");
    EXPECT_GT(msg.timestamp, 0u);
}

TEST(Protocol, MakeDisconnect) {
    auto msg = make_disconnect("net-001", "shutting down");
    EXPECT_EQ(msg.type, MessageType::DISCONNECT);
    EXPECT_EQ(msg.sender_id, "net-001");
    EXPECT_EQ(msg.payload["reason"], "shutting down");
}

// ============================================================
// Client/Server Integration Test (loopback)
// ============================================================
// Note: Full TCP integration tests require actual socket connections.
// These are tested in the main.cpp demo. Here we test the protocol layer only.

TEST(Protocol, LargePayload) {
    Message msg;
    msg.type = MessageType::SIGNAL;
    msg.sender_id = "net-001";

    // Create a large payload with many signals
    nlohmann::json signals = nlohmann::json::array();
    for (int i = 0; i < 1000; ++i) {
        signals.push_back({
            {"neuron_id", "neuron-" + std::to_string(i)},
            {"value", static_cast<double>(i) / 1000.0}
        });
    }
    msg.payload = {{"signals", signals}};

    auto wire = serialize_message(msg);
    uint32_t len = read_length_prefix(wire.data());
    auto restored = deserialize_message(wire.data() + 4, len);

    EXPECT_EQ(restored.payload["signals"].size(), 1000);
}
