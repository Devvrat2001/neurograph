#pragma once

#include "neurograph/net/protocol.hpp"
#include "neurograph/util/result.hpp"

#include <string>
#include <functional>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketType = SOCKET;
constexpr SocketType INVALID_SOCK = INVALID_SOCKET;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
using SocketType = int;
constexpr SocketType INVALID_SOCK = -1;
#endif

namespace neurograph::net {

/// Callback for received messages.
using MessageHandler = std::function<void(const std::string& peer_id, const Message& msg)>;

/// Callback for connection events.
using ConnectionHandler = std::function<void(const std::string& peer_id)>;

/// Information about a connected peer.
struct PeerInfo {
    std::string peer_id;
    std::string network_name;
    SocketType socket = INVALID_SOCK;
    std::string address;
    uint16_t port = 0;
    bool handshake_complete = false;
};

/// TCP server that listens for connections from other NeuroGraph instances.
/// Uses a simple threading model: one thread for accept, one thread per peer.
class NetworkServer {
public:
    /// Create a server with a unique network ID.
    explicit NetworkServer(const std::string& network_id,
                          const std::string& network_name = "NeuroGraph");
    ~NetworkServer();

    // Non-copyable
    NetworkServer(const NetworkServer&) = delete;
    NetworkServer& operator=(const NetworkServer&) = delete;

    /// Start listening on the given port.
    util::Result<void> start(uint16_t port);

    /// Stop the server and disconnect all peers.
    void stop();

    /// Check if server is running.
    bool is_running() const noexcept { return running_.load(); }

    /// Send a message to a specific peer.
    util::Result<void> send(const std::string& peer_id, const Message& msg);

    /// Broadcast a message to all connected peers.
    void broadcast(const Message& msg);

    /// Disconnect a specific peer.
    void disconnect_peer(const std::string& peer_id);

    /// Get connected peer info.
    std::vector<PeerInfo> connected_peers() const;

    /// Get the port the server is listening on.
    uint16_t port() const noexcept { return port_; }

    /// Set handler for incoming messages.
    void on_message(MessageHandler handler) { on_message_ = std::move(handler); }

    /// Set handler for new connections.
    void on_connect(ConnectionHandler handler) { on_connect_ = std::move(handler); }

    /// Set handler for disconnections.
    void on_disconnect(ConnectionHandler handler) { on_disconnect_ = std::move(handler); }

private:
    void accept_loop();
    void peer_read_loop(const std::string& peer_id);
    void handle_message(const std::string& peer_id, const Message& msg);
    bool send_raw(SocketType sock, const std::vector<uint8_t>& data);
    void close_socket(SocketType sock);

    std::string network_id_;
    std::string network_name_;
    uint16_t port_ = 0;
    SocketType listen_socket_ = INVALID_SOCK;
    std::atomic<bool> running_{false};

    std::thread accept_thread_;
    mutable std::mutex peers_mutex_;
    std::unordered_map<std::string, PeerInfo> peers_;
    std::unordered_map<std::string, std::thread> peer_threads_;

    MessageHandler on_message_;
    ConnectionHandler on_connect_;
    ConnectionHandler on_disconnect_;
};

} // namespace neurograph::net
