#pragma once

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace neurograph::visualizer {

#ifdef _WIN32
using WsSocketType = SOCKET;
constexpr WsSocketType WS_INVALID_SOCK = INVALID_SOCKET;
#else
using WsSocketType = int;
constexpr WsSocketType WS_INVALID_SOCK = -1;
#endif

/// Minimal WebSocket server for the NeuroGraph visualizer.
class WsServer {
public:
    WsServer();
    ~WsServer();

    WsServer(const WsServer&) = delete;
    WsServer& operator=(const WsServer&) = delete;

    bool start(uint16_t port);
    void stop();
    bool send(const std::string& message);

    bool is_connected() const noexcept { return client_connected_.load(); }
    bool is_running() const noexcept { return running_.load(); }

    void on_message(std::function<void(const std::string&)> callback) { on_message_ = std::move(callback); }
    void on_connect(std::function<void()> callback) { on_connect_ = std::move(callback); }
    void on_disconnect(std::function<void()> callback) { on_disconnect_ = std::move(callback); }

    uint16_t port() const noexcept { return port_; }

private:
    void accept_loop();
    void client_loop(WsSocketType client_sock);
    bool do_handshake(WsSocketType sock);
    bool send_frame(WsSocketType sock, uint8_t opcode, const uint8_t* data, size_t len);
    bool read_exact(WsSocketType sock, uint8_t* buf, size_t len);

    static std::string compute_accept_key(const std::string& client_key);
    static std::vector<uint8_t> sha1(const uint8_t* data, size_t len);
    static std::string base64_encode(const uint8_t* data, size_t len);

    WsSocketType listen_sock_ = WS_INVALID_SOCK;
    WsSocketType client_sock_ = WS_INVALID_SOCK;
    std::atomic<bool> running_{false};
    std::atomic<bool> client_connected_{false};
    uint16_t port_ = 0;

    std::thread accept_thread_;
    std::thread client_thread_;
    std::mutex send_mutex_;

    std::function<void(const std::string&)> on_message_;
    std::function<void()> on_connect_;
    std::function<void()> on_disconnect_;

#ifdef _WIN32
    struct WinsockInit {
        WinsockInit() { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); }
        ~WinsockInit() { WSACleanup(); }
    };
    static WinsockInit winsock_init_;
#endif
};

} // namespace neurograph::visualizer
