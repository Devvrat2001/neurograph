#include "neurograph/net/network_client.hpp"
#include "neurograph/util/logger.hpp"

#include <cstring>

#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#endif

namespace neurograph::net {

NetworkClient::NetworkClient(const std::string& network_id,
                             const std::string& network_name)
    : network_id_(network_id), network_name_(network_name) {}

NetworkClient::~NetworkClient() {
    disconnect();
}

util::Result<void> NetworkClient::connect(const std::string& host, uint16_t port) {
    if (connected_.load()) {
        return util::Error("Already connected");
    }

    socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_ == INVALID_SOCK) {
        return util::Error("Failed to create socket");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close_socket();
        return util::Error("Invalid host address: " + host);
    }

    if (::connect(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_socket();
        return util::Error("Failed to connect to " + host + ":" + std::to_string(port));
    }

    connected_.store(true);
    NG_LOG_INFO("Connected to {}:{}", host, port);

    // Send our handshake
    auto hs = make_handshake(network_id_, network_name_);
    auto data = serialize_message(hs);
    if (!send_raw(data)) {
        disconnect();
        return util::Error("Failed to send handshake");
    }

    // Start read loop
    read_thread_ = std::thread(&NetworkClient::read_loop, this);

    return util::Result<void>::success();
}

void NetworkClient::disconnect() {
    if (!connected_.load()) return;

    // Send disconnect message
    auto msg = make_disconnect(network_id_, "client disconnect");
    auto data = serialize_message(msg);
    send_raw(data); // best-effort

    connected_.store(false);
    close_socket();

    if (read_thread_.joinable()) {
        read_thread_.join();
    }

    std::string rid;
    {
        std::lock_guard lock(mutex_);
        rid = remote_id_;
        remote_id_.clear();
    }

    if (on_disconnect_ && !rid.empty()) {
        on_disconnect_(rid);
    }

    NG_LOG_INFO("Disconnected from remote");
}

util::Result<void> NetworkClient::send(const Message& msg) {
    if (!connected_.load()) {
        return util::Error("Not connected");
    }

    auto data = serialize_message(msg);
    if (!send_raw(data)) {
        return util::Error("Send failed");
    }
    return util::Result<void>::success();
}

std::string NetworkClient::remote_id() const {
    std::lock_guard lock(mutex_);
    return remote_id_;
}

// ============================================================
// Private
// ============================================================

void NetworkClient::read_loop() {
    while (connected_.load()) {
        // Read 4-byte length prefix
        uint8_t len_buf[4];
        int total_read = 0;
        while (total_read < 4) {
            int n = recv(socket_, reinterpret_cast<char*>(len_buf + total_read),
                        4 - total_read, 0);
            if (n <= 0) {
                if (connected_.load()) {
                    NG_LOG_INFO("Remote server disconnected");
                }
                connected_.store(false);
                return;
            }
            total_read += n;
        }

        uint32_t msg_len = read_length_prefix(len_buf);
        if (msg_len > 16 * 1024 * 1024) {
            NG_LOG_WARN("Message too large from server: {} bytes", msg_len);
            connected_.store(false);
            return;
        }

        // Read message body
        std::vector<uint8_t> msg_buf(msg_len);
        total_read = 0;
        while (static_cast<uint32_t>(total_read) < msg_len) {
            int n = recv(socket_, reinterpret_cast<char*>(msg_buf.data() + total_read),
                        static_cast<int>(msg_len - total_read), 0);
            if (n <= 0) {
                if (connected_.load()) {
                    NG_LOG_INFO("Remote disconnected during read");
                }
                connected_.store(false);
                return;
            }
            total_read += n;
        }

        try {
            auto msg = deserialize_message(msg_buf.data(), msg_len);

            // Handle handshake
            if (msg.type == MessageType::HANDSHAKE) {
                std::lock_guard lock(mutex_);
                remote_id_ = msg.sender_id;
                NG_LOG_INFO("Handshake from server: {} ({})",
                            msg.sender_id,
                            msg.payload.value("network_name", "unknown"));
                if (on_connect_) {
                    on_connect_(remote_id_);
                }
                continue;
            }

            if (msg.type == MessageType::DISCONNECT) {
                NG_LOG_INFO("Server sent disconnect: {}",
                            msg.payload.value("reason", "no reason"));
                connected_.store(false);
                return;
            }

            // Forward to handler
            if (on_message_) {
                std::string rid;
                {
                    std::lock_guard lock(mutex_);
                    rid = remote_id_;
                }
                on_message_(rid, msg);
            }
        } catch (const std::exception& e) {
            NG_LOG_WARN("Invalid message from server: {}", e.what());
        }
    }
}

bool NetworkClient::send_raw(const std::vector<uint8_t>& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        int n = ::send(socket_, reinterpret_cast<const char*>(data.data() + total_sent),
                      static_cast<int>(data.size() - total_sent), 0);
        if (n <= 0) return false;
        total_sent += n;
    }
    return true;
}

void NetworkClient::close_socket() {
    if (socket_ == INVALID_SOCK) return;
#ifdef _WIN32
    closesocket(socket_);
#else
    close(socket_);
#endif
    socket_ = INVALID_SOCK;
}

} // namespace neurograph::net
