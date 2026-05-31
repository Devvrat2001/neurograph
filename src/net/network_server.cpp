#include "neurograph/net/network_server.hpp"
#include "neurograph/util/logger.hpp"

#include <cstring>
#include <algorithm>

#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#endif

namespace neurograph::net {

// ============================================================
// Platform helpers
// ============================================================

#ifdef _WIN32
struct WinsockInit {
    WinsockInit() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
    ~WinsockInit() { WSACleanup(); }
};
static WinsockInit winsock_init;
#endif

static void set_socket_reuse(SocketType sock) {
    int opt = 1;
#ifdef _WIN32
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
}

// ============================================================
// NetworkServer
// ============================================================

NetworkServer::NetworkServer(const std::string& network_id,
                             const std::string& network_name)
    : network_id_(network_id), network_name_(network_name) {}

NetworkServer::~NetworkServer() {
    stop();
}

util::Result<void> NetworkServer::start(uint16_t port) {
    if (running_.load()) {
        return util::Error("Server already running");
    }

    listen_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_socket_ == INVALID_SOCK) {
        return util::Error("Failed to create socket");
    }

    set_socket_reuse(listen_socket_);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(listen_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_socket(listen_socket_);
        listen_socket_ = INVALID_SOCK;
        return util::Error("Failed to bind to port " + std::to_string(port));
    }

    if (listen(listen_socket_, 16) != 0) {
        close_socket(listen_socket_);
        listen_socket_ = INVALID_SOCK;
        return util::Error("Failed to listen on port " + std::to_string(port));
    }

    port_ = port;
    running_.store(true);
    accept_thread_ = std::thread(&NetworkServer::accept_loop, this);

    NG_LOG_INFO("NetworkServer started on port {}", port);
    return util::Result<void>::success();
}

void NetworkServer::stop() {
    if (!running_.load()) return;

    running_.store(false);

    // Close the listen socket to unblock accept()
    if (listen_socket_ != INVALID_SOCK) {
        close_socket(listen_socket_);
        listen_socket_ = INVALID_SOCK;
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    // Disconnect all peers
    std::vector<std::string> peer_ids;
    {
        std::lock_guard lock(peers_mutex_);
        for (auto& [id, _] : peers_) {
            peer_ids.push_back(id);
        }
    }

    for (const auto& id : peer_ids) {
        disconnect_peer(id);
    }

    // Join all peer threads
    for (auto& [id, thread] : peer_threads_) {
        if (thread.joinable()) thread.join();
    }
    peer_threads_.clear();

    NG_LOG_INFO("NetworkServer stopped");
}

util::Result<void> NetworkServer::send(const std::string& peer_id, const Message& msg) {
    std::lock_guard lock(peers_mutex_);
    auto it = peers_.find(peer_id);
    if (it == peers_.end()) {
        return util::Error("Peer not found: " + peer_id);
    }

    auto data = serialize_message(msg);
    if (!send_raw(it->second.socket, data)) {
        return util::Error("Failed to send to peer: " + peer_id);
    }
    return util::Result<void>::success();
}

void NetworkServer::broadcast(const Message& msg) {
    auto data = serialize_message(msg);
    std::lock_guard lock(peers_mutex_);
    for (auto& [id, peer] : peers_) {
        if (peer.handshake_complete) {
            send_raw(peer.socket, data);
        }
    }
}

void NetworkServer::disconnect_peer(const std::string& peer_id) {
    SocketType sock = INVALID_SOCK;
    {
        std::lock_guard lock(peers_mutex_);
        auto it = peers_.find(peer_id);
        if (it == peers_.end()) return;
        sock = it->second.socket;
        peers_.erase(it);
    }

    if (sock != INVALID_SOCK) {
        // Send disconnect message before closing
        auto msg = make_disconnect(network_id_, "server disconnect");
        auto data = serialize_message(msg);
        send_raw(sock, data);
        close_socket(sock);
    }

    if (on_disconnect_) {
        on_disconnect_(peer_id);
    }
}

std::vector<PeerInfo> NetworkServer::connected_peers() const {
    std::lock_guard lock(peers_mutex_);
    std::vector<PeerInfo> result;
    result.reserve(peers_.size());
    for (const auto& [_, info] : peers_) {
        result.push_back(info);
    }
    return result;
}

// ============================================================
// Private methods
// ============================================================

void NetworkServer::accept_loop() {
    while (running_.load()) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif

        SocketType client_sock = accept(listen_socket_,
                                        reinterpret_cast<sockaddr*>(&client_addr),
                                        &addr_len);

        if (client_sock == INVALID_SOCK) {
            if (!running_.load()) break; // shutdown
            NG_LOG_WARN("Accept failed, continuing...");
            continue;
        }

        // Generate a temporary peer ID until handshake
        char addr_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, addr_str, sizeof(addr_str));
        uint16_t client_port = ntohs(client_addr.sin_port);

        std::string temp_id = std::string(addr_str) + ":" + std::to_string(client_port);

        PeerInfo peer;
        peer.peer_id = temp_id;
        peer.socket = client_sock;
        peer.address = addr_str;
        peer.port = client_port;

        {
            std::lock_guard lock(peers_mutex_);
            peers_[temp_id] = peer;
        }

        // Send our handshake
        auto hs = make_handshake(network_id_, network_name_);
        auto data = serialize_message(hs);
        send_raw(client_sock, data);

        // Start read thread for this peer
        peer_threads_[temp_id] = std::thread(&NetworkServer::peer_read_loop, this, temp_id);

        NG_LOG_INFO("Accepted connection from {}", temp_id);
    }
}

void NetworkServer::peer_read_loop(const std::string& peer_id) {
    SocketType sock = INVALID_SOCK;
    {
        std::lock_guard lock(peers_mutex_);
        auto it = peers_.find(peer_id);
        if (it == peers_.end()) return;
        sock = it->second.socket;
    }

    while (running_.load()) {
        // Read 4-byte length prefix
        uint8_t len_buf[4];
        int total_read = 0;
        while (total_read < 4) {
            int n = recv(sock, reinterpret_cast<char*>(len_buf + total_read),
                        4 - total_read, 0);
            if (n <= 0) {
                NG_LOG_INFO("Peer {} disconnected", peer_id);
                goto cleanup;
            }
            total_read += n;
        }

        uint32_t msg_len = read_length_prefix(len_buf);
        if (msg_len > 16 * 1024 * 1024) { // 16MB limit
            NG_LOG_WARN("Message too large from {}: {} bytes", peer_id, msg_len);
            goto cleanup;
        }

        // Read message body
        std::vector<uint8_t> msg_buf(msg_len);
        total_read = 0;
        while (static_cast<uint32_t>(total_read) < msg_len) {
            int n = recv(sock, reinterpret_cast<char*>(msg_buf.data() + total_read),
                        static_cast<int>(msg_len - total_read), 0);
            if (n <= 0) {
                NG_LOG_INFO("Peer {} disconnected during message read", peer_id);
                goto cleanup;
            }
            total_read += n;
        }

        // Deserialize and handle
        try {
            auto msg = deserialize_message(msg_buf.data(), msg_len);
            handle_message(peer_id, msg);
        } catch (const std::exception& e) {
            NG_LOG_WARN("Invalid message from {}: {}", peer_id, e.what());
        }
    }

cleanup:
    {
        std::lock_guard lock(peers_mutex_);
        auto it = peers_.find(peer_id);
        if (it != peers_.end()) {
            close_socket(it->second.socket);
            peers_.erase(it);
        }
    }
    if (on_disconnect_) {
        on_disconnect_(peer_id);
    }
}

void NetworkServer::handle_message(const std::string& peer_id, const Message& msg) {
    if (msg.type == MessageType::HANDSHAKE) {
        // Update peer info with real network ID
        std::string real_id = msg.sender_id;
        std::lock_guard lock(peers_mutex_);
        auto it = peers_.find(peer_id);
        if (it != peers_.end()) {
            it->second.peer_id = real_id;
            it->second.network_name = msg.payload.value("network_name", "unknown");
            it->second.handshake_complete = true;
            NG_LOG_INFO("Handshake complete with {} ({})", real_id,
                        it->second.network_name);

            // Re-key under real ID if different
            if (real_id != peer_id) {
                PeerInfo info = std::move(it->second);
                peers_.erase(it);
                peers_[real_id] = std::move(info);
            }
        }
        if (on_connect_) {
            on_connect_(real_id);
        }
        return;
    }

    if (msg.type == MessageType::DISCONNECT) {
        NG_LOG_INFO("Peer {} sent disconnect: {}", peer_id,
                    msg.payload.value("reason", "no reason"));
        return;
    }

    // Forward to message handler
    if (on_message_) {
        on_message_(peer_id, msg);
    }
}

bool NetworkServer::send_raw(SocketType sock, const std::vector<uint8_t>& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        int n = ::send(sock, reinterpret_cast<const char*>(data.data() + total_sent),
                      static_cast<int>(data.size() - total_sent), 0);
        if (n <= 0) return false;
        total_sent += n;
    }
    return true;
}

void NetworkServer::close_socket(SocketType sock) {
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
}

} // namespace neurograph::net
