#include "neurograph/visualizer/ws_server.hpp"
#include "neurograph/util/logger.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#ifdef _WIN32
#include <ws2tcpip.h>
#endif

// Portable shutdown constant
#ifdef _WIN32
#define WS_SHUT_RDWR SD_BOTH
#else
#define WS_SHUT_RDWR SHUT_RDWR
#endif

namespace neurograph::visualizer {

#ifdef _WIN32
WsServer::WinsockInit WsServer::winsock_init_;

static void close_socket(WsSocketType s) {
    if (s != WS_INVALID_SOCK) closesocket(s);
}
static int get_socket_error() { return WSAGetLastError(); }
#else
static void close_socket(WsSocketType s) {
    if (s != WS_INVALID_SOCK) ::close(s);
}
static int get_socket_error() { return errno; }
#endif

// ============================================================
// SHA-1 (RFC 3174)
// ============================================================

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

std::vector<uint8_t> WsServer::sha1(const uint8_t* data, size_t len) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    size_t ml = len * 8;
    size_t padded_len = ((len + 1 + 8 + 63) / 64) * 64;
    std::vector<uint8_t> msg(padded_len, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    for (int i = 0; i < 8; ++i)
        msg[padded_len - 1 - i] = static_cast<uint8_t>((ml >> (i * 8)) & 0xFF);

    for (size_t offset = 0; offset < padded_len; offset += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(msg[offset + i*4]) << 24) |
                   (static_cast<uint32_t>(msg[offset + i*4+1]) << 16) |
                   (static_cast<uint32_t>(msg[offset + i*4+2]) << 8) |
                   (static_cast<uint32_t>(msg[offset + i*4+3]));
        }
        for (int i = 16; i < 80; ++i)
            w[i] = rotl32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;             k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;             k = 0xCA62C1D6; }

            uint32_t temp = rotl32(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    std::vector<uint8_t> hash(20);
    for (int i = 0; i < 4; ++i) {
        hash[i]    = static_cast<uint8_t>((h0 >> (24 - i*8)) & 0xFF);
        hash[4+i]  = static_cast<uint8_t>((h1 >> (24 - i*8)) & 0xFF);
        hash[8+i]  = static_cast<uint8_t>((h2 >> (24 - i*8)) & 0xFF);
        hash[12+i] = static_cast<uint8_t>((h3 >> (24 - i*8)) & 0xFF);
        hash[16+i] = static_cast<uint8_t>((h4 >> (24 - i*8)) & 0xFF);
    }
    return hash;
}

// ============================================================
// Base64
// ============================================================

std::string WsServer::base64_encode(const uint8_t* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(4 * ((len + 2) / 3));
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (static_cast<uint32_t>(data[i]) << 16);
        if (i + 1 < len) n |= (static_cast<uint32_t>(data[i+1]) << 8);
        if (i + 2 < len) n |= static_cast<uint32_t>(data[i+2]);
        result += table[(n >> 18) & 0x3F];
        result += table[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? table[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? table[n & 0x3F] : '=';
    }
    return result;
}

// ============================================================
// WebSocket Accept Key
// ============================================================

std::string WsServer::compute_accept_key(const std::string& client_key) {
    const std::string magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string input = client_key + magic;
    auto hash = sha1(reinterpret_cast<const uint8_t*>(input.data()), input.size());
    return base64_encode(hash.data(), hash.size());
}

// ============================================================
// Constructor / Destructor
// ============================================================

WsServer::WsServer() = default;

WsServer::~WsServer() {
    stop();
}

// ============================================================
// Start / Stop
// ============================================================

bool WsServer::start(uint16_t port) {
    if (running_.load()) return false;

    listen_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock_ == WS_INVALID_SOCK) {
        NG_LOG_ERROR("WsServer: failed to create socket, err={}", get_socket_error());
        return false;
    }

    // Allow port reuse
    int opt = 1;
    setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        NG_LOG_ERROR("WsServer: failed to bind port {}, err={}", port, get_socket_error());
        close_socket(listen_sock_);
        listen_sock_ = WS_INVALID_SOCK;
        return false;
    }

    if (listen(listen_sock_, 1) != 0) {
        NG_LOG_ERROR("WsServer: failed to listen, err={}", get_socket_error());
        close_socket(listen_sock_);
        listen_sock_ = WS_INVALID_SOCK;
        return false;
    }

    port_ = port;
    running_ = true;
    accept_thread_ = std::thread(&WsServer::accept_loop, this);

    NG_LOG_INFO("WsServer: listening on port {} (ws://localhost:{})", port, port);
    return true;
}

void WsServer::stop() {
    if (!running_.load()) return;
    running_ = false;

    // Close client socket to unblock recv in client_loop
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        if (client_sock_ != WS_INVALID_SOCK) {
            shutdown(client_sock_, WS_SHUT_RDWR);
            close_socket(client_sock_);
            client_sock_ = WS_INVALID_SOCK;
        }
        client_connected_ = false;
    }

    // Close listener to unblock select() in accept_loop
    if (listen_sock_ != WS_INVALID_SOCK) {
        close_socket(listen_sock_);
        listen_sock_ = WS_INVALID_SOCK;
    }

    if (accept_thread_.joinable()) accept_thread_.join();
    if (client_thread_.joinable()) client_thread_.join();

    NG_LOG_INFO("WsServer: stopped");
}

// ============================================================
// Accept Loop
// ============================================================

void WsServer::accept_loop() {
    while (running_.load()) {
        // Guard against listen_sock_ being closed during stop()
        if (listen_sock_ == WS_INVALID_SOCK) break;

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_sock_, &fds);
        timeval tv{1, 0};

        int sel = select(static_cast<int>(listen_sock_) + 1, &fds, nullptr, nullptr, &tv);
        if (sel < 0) {
            // Socket was closed or error
            if (running_.load()) {
                NG_LOG_WARN("WsServer: accept select error={}", get_socket_error());
            }
            break;
        }
        if (sel == 0) continue; // timeout, check running_ and loop

        sockaddr_in client_addr{};
        int addr_len = sizeof(client_addr);
        WsSocketType new_sock = accept(listen_sock_,
            reinterpret_cast<sockaddr*>(&client_addr),
#ifdef _WIN32
            &addr_len);
#else
            reinterpret_cast<socklen_t*>(&addr_len));
#endif

        if (new_sock == WS_INVALID_SOCK) {
            NG_LOG_WARN("WsServer: accept failed, err={}", get_socket_error());
            continue;
        }

        // Set TCP_NODELAY — prevent Nagle buffering small frames
        int nodelay = 1;
        setsockopt(new_sock, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

        // Set larger send buffer (1MB) for big state messages
        int send_buf = 1024 * 1024;
        setsockopt(new_sock, SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&send_buf), sizeof(send_buf));

        // Do NOT set SO_RCVTIMEO — we use select() for timeout control instead
        // Setting SO_RCVTIMEO causes recv() to fail with WSAETIMEDOUT which
        // looks like a disconnect to read_exact()

        // If a client is already connected, check if it's still alive
        if (client_connected_.load()) {
            // Test if old connection is alive by checking socket state
            // If the old client hasn't sent data in a while, it may be dead
            // but we haven't noticed yet. Give the new connection a chance.
            fd_set test_fds;
            FD_ZERO(&test_fds);
            FD_SET(client_sock_, &test_fds);
            timeval zero_tv{0, 0};
            int test_sel = select(static_cast<int>(client_sock_) + 1, &test_fds, nullptr, nullptr, &zero_tv);

            if (test_sel > 0) {
                // Data or close is pending — read to check
                char test_buf[1];
                int peek = recv(client_sock_, test_buf, 1, MSG_PEEK);
                if (peek <= 0) {
                    // Old connection is dead, allow replacement
                    NG_LOG_INFO("WsServer: old connection is dead, accepting new client");
                } else {
                    // Old connection is alive and has data, reject new one
                    NG_LOG_INFO("WsServer: rejecting new connection — active client exists");
                    close_socket(new_sock);
                    continue;
                }
            } else {
                // No data pending — old connection may be idle but alive, reject
                NG_LOG_INFO("WsServer: rejecting new connection — active client exists");
                close_socket(new_sock);
                continue;
            }

            // Close old connection before accepting new one
            {
                std::lock_guard<std::mutex> lock(send_mutex_);
                if (client_sock_ != WS_INVALID_SOCK) {
                    shutdown(client_sock_, WS_SHUT_RDWR);
                    close_socket(client_sock_);
                    client_sock_ = WS_INVALID_SOCK;
                }
                client_connected_ = false;
            }
        }

        // If previous client thread is still cleaning up, wait for it
        if (client_thread_.joinable()) {
            client_thread_.join();
        }

        // Perform WebSocket handshake
        if (!do_handshake(new_sock)) {
            NG_LOG_WARN("WsServer: WebSocket handshake failed");
            close_socket(new_sock);
            continue;
        }

        // Store client — protect with mutex so send() doesn't race
        {
            std::lock_guard<std::mutex> lock(send_mutex_);
            client_sock_ = new_sock;
            client_connected_ = true;
        }

        NG_LOG_INFO("WsServer: client connected successfully");

        // Start client read loop
        client_thread_ = std::thread(&WsServer::client_loop, this, new_sock);

        // Give client_loop time to start reading before we send initial data
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Fire connect callback — this typically sends initial state
        if (on_connect_) {
            try {
                on_connect_();
            } catch (const std::exception& e) {
                NG_LOG_WARN("WsServer: on_connect callback error: {}", e.what());
            }
        }
    }
}

// ============================================================
// WebSocket Handshake
// ============================================================

bool WsServer::do_handshake(WsSocketType sock) {
    // Read HTTP upgrade request (up to 4KB)
    char buf[4096];
    int received = recv(sock, buf, sizeof(buf) - 1, 0);
    if (received <= 0) {
        NG_LOG_WARN("WsServer: handshake recv failed, err={}", get_socket_error());
        return false;
    }
    buf[received] = '\0';

    std::string request(buf);

    // Extract Sec-WebSocket-Key header
    std::string key_header = "Sec-WebSocket-Key: ";
    auto pos = request.find(key_header);
    if (pos == std::string::npos) {
        NG_LOG_WARN("WsServer: no Sec-WebSocket-Key in handshake request");
        return false;
    }

    auto start = pos + key_header.size();
    auto end = request.find("\r\n", start);
    if (end == std::string::npos) return false;

    std::string client_key = request.substr(start, end - start);
    while (!client_key.empty() && client_key.back() == ' ') client_key.pop_back();

    std::string accept_key = compute_accept_key(client_key);

    // Build and send HTTP 101 upgrade response
    std::string response =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + accept_key + "\r\n"
        "\r\n";

    int sent = ::send(sock, response.c_str(), static_cast<int>(response.size()), 0);
    if (sent != static_cast<int>(response.size())) {
        NG_LOG_WARN("WsServer: handshake response send failed, err={}", get_socket_error());
        return false;
    }

    NG_LOG_INFO("WsServer: WebSocket handshake completed");
    return true;
}

// ============================================================
// Client Read Loop
// ============================================================

void WsServer::client_loop(WsSocketType sock) {
    NG_LOG_INFO("WsServer: client read loop started");

    while (running_.load() && client_connected_.load()) {
        // Use select() with timeout — never block indefinitely on recv
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(sock, &read_fds);
        timeval tv{2, 0}; // 2 second check interval

        int sel = select(static_cast<int>(sock) + 1, &read_fds, nullptr, nullptr, &tv);
        if (sel < 0) {
            int err = get_socket_error();
            // On Windows, WSAENOTSOCK (10038) means socket was closed by stop()
            #ifdef _WIN32
            if (err == WSAENOTSOCK || err == WSAEINTR) break;
            #else
            if (err == EBADF || err == EINTR) break;
            #endif
            NG_LOG_WARN("WsServer: client select error={}", err);
            break;
        }
        if (sel == 0) {
            // Timeout — no data from client. This is normal. Just loop.
            continue;
        }

        // Data is available — read the 2-byte frame header
        uint8_t header[2];
        if (!read_exact(sock, header, 2)) {
            NG_LOG_INFO("WsServer: client read loop - failed to read frame header");
            break;
        }

        uint8_t opcode = header[0] & 0x0F;
        bool masked = (header[1] & 0x80) != 0;
        uint64_t payload_len = header[1] & 0x7F;

        // Extended payload length
        if (payload_len == 126) {
            uint8_t ext[2];
            if (!read_exact(sock, ext, 2)) break;
            payload_len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
        } else if (payload_len == 127) {
            uint8_t ext[8];
            if (!read_exact(sock, ext, 8)) break;
            payload_len = 0;
            for (int i = 0; i < 8; ++i)
                payload_len = (payload_len << 8) | ext[i];
        }

        // Masking key
        uint8_t mask[4] = {0};
        if (masked) {
            if (!read_exact(sock, mask, 4)) break;
        }

        // Payload size limit
        if (payload_len > 16 * 1024 * 1024) {
            NG_LOG_WARN("WsServer: payload too large ({}), disconnecting", payload_len);
            break;
        }

        // Read payload
        std::vector<uint8_t> payload(static_cast<size_t>(payload_len));
        if (payload_len > 0) {
            if (!read_exact(sock, payload.data(), static_cast<size_t>(payload_len))) {
                NG_LOG_INFO("WsServer: failed to read payload ({} bytes)", payload_len);
                break;
            }
        }

        // Unmask
        if (masked) {
            for (size_t i = 0; i < payload.size(); ++i)
                payload[i] ^= mask[i % 4];
        }

        // Dispatch by opcode
        if (opcode == 0x1) { // Text frame
            std::string msg(payload.begin(), payload.end());
            try {
                if (on_message_) on_message_(msg);
            } catch (const std::exception& e) {
                NG_LOG_WARN("WsServer: message handler threw: {}", e.what());
            }
        } else if (opcode == 0x8) { // Close frame
            NG_LOG_INFO("WsServer: received close frame from client");
            // Echo close frame back
            send_frame(sock, 0x8, payload.data(), payload.size());
            break;
        } else if (opcode == 0x9) { // Ping
            send_frame(sock, 0xA, payload.data(), payload.size());
        }
        // 0xA = pong, 0x0 = continuation — ignore
    }

    // Clean disconnect — protect with mutex to avoid racing with send()
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        client_connected_ = false;
        // Only close if this is still our active socket
        if (client_sock_ == sock) {
            shutdown(sock, WS_SHUT_RDWR);
            close_socket(sock);
            client_sock_ = WS_INVALID_SOCK;
        }
    }

    NG_LOG_INFO("WsServer: client read loop ended, client disconnected");

    if (on_disconnect_) {
        try {
            on_disconnect_();
        } catch (...) {}
    }
}

// ============================================================
// Send
// ============================================================

bool WsServer::send(const std::string& message) {
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (!client_connected_.load() || client_sock_ == WS_INVALID_SOCK) return false;
    return send_frame(client_sock_, 0x1,
        reinterpret_cast<const uint8_t*>(message.data()), message.size());
}

bool WsServer::send_frame(WsSocketType sock, uint8_t opcode, const uint8_t* data, size_t len) {
    std::vector<uint8_t> frame;

    // FIN + opcode
    frame.push_back(0x80 | opcode);

    // Payload length (server frames are never masked)
    if (len < 126) {
        frame.push_back(static_cast<uint8_t>(len));
    } else if (len < 65536) {
        frame.push_back(126);
        frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        frame.push_back(static_cast<uint8_t>(len & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; --i)
            frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xFF));
    }

    // Payload data
    if (data && len > 0) {
        frame.insert(frame.end(), data, data + len);
    }

    // Send the entire frame, looping on partial sends
    size_t total = frame.size();
    size_t sent = 0;
    while (sent < total) {
        // Don't try to send more than 64KB at a time to avoid blocking
        size_t chunk = std::min(total - sent, static_cast<size_t>(65536));
        int n = ::send(sock, reinterpret_cast<const char*>(frame.data() + sent),
                       static_cast<int>(chunk), 0);
        if (n <= 0) {
            NG_LOG_WARN("WsServer: send_frame failed at {}/{} bytes, err={}",
                        sent, total, get_socket_error());
            return false;
        }
        sent += n;
    }
    return true;
}

// ============================================================
// Helpers
// ============================================================

bool WsServer::read_exact(WsSocketType sock, uint8_t* buf, size_t len) {
    size_t received = 0;
    while (received < len) {
        if (!running_.load() || !client_connected_.load()) return false;

        int n = recv(sock, reinterpret_cast<char*>(buf + received),
                     static_cast<int>(len - received), 0);
        if (n <= 0) {
            // n == 0 means clean disconnect, n < 0 means error
            return false;
        }
        received += n;
    }
    return true;
}

} // namespace neurograph::visualizer
