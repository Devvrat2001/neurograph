#pragma once

#include "neurograph/visualizer/ws_server.hpp"
#include "neurograph/core/network.hpp"
#include "neurograph/snapshot/snapshot_registry.hpp"
#include "neurograph/util/result.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <random>

namespace neurograph::visualizer {

/// Bridges the C++ NeuroGraph engine to the HTML workbench via WebSocket.
///
/// Protocol (Server -> Client):
///   {"type":"state", "neurons":[...], "synapses":[...], "stats":{...}}
///   {"type":"event", "event":"...", "data":{...}}
///   {"type":"registry", "images":[...]}
///   {"type":"response", "status":"ok|error", "message":"..."}
///
/// Protocol (Client -> Server):
///   {"type":"command", "action":"...", ...}
///
/// Supported actions:
///   get_state, start_stream, stop_stream, step,
///   create_network, load_image, save_image, delete_image,
///   set_param, get_registry
///
class VisualizerBridge {
public:
    VisualizerBridge(core::Network& network, uint16_t port = 8765);
    ~VisualizerBridge();

    VisualizerBridge(const VisualizerBridge&) = delete;
    VisualizerBridge& operator=(const VisualizerBridge&) = delete;

    /// Start the WebSocket server and bridge.
    util::Result<void> start();

    /// Stop everything.
    void stop();

    /// Set the snapshot registry for neural image management.
    void set_registry(snapshot::SnapshotRegistry* registry) { registry_ = registry; }

    /// Send a one-time state update to the client.
    void send_state();

    /// Send an event to the client.
    void send_event(const std::string& event_type, const nlohmann::json& data = {});

    /// Send a response message (ok/error) to the client.
    void send_response(const std::string& status, const std::string& message);

    /// Start streaming state updates at the given interval.
    /// Each tick also runs one self-organization step.
    void start_streaming(std::chrono::milliseconds interval = std::chrono::milliseconds(100));

    /// Stop streaming.
    void stop_streaming();

    bool is_streaming() const noexcept { return streaming_.load(); }
    bool is_connected() const { return ws_server_.is_connected(); }

    /// Set callback for commands not handled internally.
    void on_command(std::function<void(const std::string& action, const nlohmann::json& params)> callback) {
        on_command_ = std::move(callback);
    }

    uint16_t port() const noexcept { return port_; }

private:
    void handle_message(const std::string& msg);
    void handle_command(const nlohmann::json& cmd);
    void stream_loop();

    // Command handlers
    void handle_create_network(const nlohmann::json& cmd);
    void handle_load_image(const nlohmann::json& cmd);
    void handle_save_image(const nlohmann::json& cmd);
    void handle_delete_image(const nlohmann::json& cmd);
    void handle_set_param(const nlohmann::json& cmd);

    // Serialization
    nlohmann::json serialize_state() const;
    nlohmann::json serialize_neuron(const core::Neuron& neuron) const;
    nlohmann::json serialize_synapse(const core::Synapse& synapse) const;
    nlohmann::json serialize_neuron_compact(const core::Neuron& neuron) const;
    nlohmann::json serialize_synapse_compact(const core::Synapse& synapse) const;
    nlohmann::json serialize_registry() const;

    core::Network& network_;
    WsServer ws_server_;
    uint16_t port_;

    snapshot::SnapshotRegistry* registry_ = nullptr;

    std::atomic<bool> streaming_{false};
    std::thread stream_thread_;
    std::chrono::milliseconds stream_interval_{100};

    std::function<void(const std::string&, const nlohmann::json&)> on_command_;

    mutable std::mutex network_mutex_;
};

} // namespace neurograph::visualizer
