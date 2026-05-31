#pragma once

#include "neurograph/net/protocol.hpp"
#include "neurograph/net/network_server.hpp" // for SocketType, MessageHandler etc.
#include "neurograph/util/result.hpp"

#include <string>
#include <thread>
#include <atomic>
#include <mutex>

namespace neurograph::net {

/// TCP client that connects to a remote NeuroGraph server.
/// Handles handshake, message send/receive, and reconnection.
class NetworkClient {
public:
    explicit NetworkClient(const std::string& network_id,
                          const std::string& network_name = "NeuroGraph");
    ~NetworkClient();

    // Non-copyable
    NetworkClient(const NetworkClient&) = delete;
    NetworkClient& operator=(const NetworkClient&) = delete;

    /// Connect to a remote server.
    util::Result<void> connect(const std::string& host, uint16_t port);

    /// Disconnect from the server.
    void disconnect();

    /// Check if connected.
    bool is_connected() const noexcept { return connected_.load(); }

    /// Send a message to the remote server.
    util::Result<void> send(const Message& msg);

    /// Get the remote peer ID (available after handshake).
    std::string remote_id() const;

    /// Set handler for incoming messages.
    void on_message(MessageHandler handler) { on_message_ = std::move(handler); }

    /// Set handler for connection established.
    void on_connect(ConnectionHandler handler) { on_connect_ = std::move(handler); }

    /// Set handler for disconnection.
    void on_disconnect(ConnectionHandler handler) { on_disconnect_ = std::move(handler); }

private:
    void read_loop();
    bool send_raw(const std::vector<uint8_t>& data);
    void close_socket();

    std::string network_id_;
    std::string network_name_;
    SocketType socket_ = INVALID_SOCK;
    std::atomic<bool> connected_{false};

    std::string remote_id_;
    mutable std::mutex mutex_;
    std::thread read_thread_;

    MessageHandler on_message_;
    ConnectionHandler on_connect_;
    ConnectionHandler on_disconnect_;
};

} // namespace neurograph::net
