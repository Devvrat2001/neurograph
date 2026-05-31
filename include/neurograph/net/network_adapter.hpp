#pragma once

#include "neurograph/net/network_server.hpp"
#include "neurograph/net/network_client.hpp"
#include "neurograph/net/protocol.hpp"
#include "neurograph/core/network.hpp"
#include "neurograph/command/command_signer.hpp"
#include "neurograph/command/command_propagator.hpp"
#include "neurograph/util/result.hpp"

#include <string>
#include <functional>
#include <vector>
#include <unordered_map>

namespace neurograph::net {

/// Bridges external network communication with the internal command system.
///
/// The NetworkAdapter:
/// - Receives SIGNAL messages and injects them as neuron activations
/// - Receives COMMAND messages and routes them through the command propagator
/// - Sends outgoing signals from designated "output" neurons to remote networks
/// - Manages neuron-to-remote mappings (which local neurons are mapped to remote ones)
class NetworkAdapter {
public:
    NetworkAdapter(core::Network& network,
                   command::CommandPropagator& propagator,
                   NetworkServer& server);
    ~NetworkAdapter() = default;

    /// Start processing incoming messages.
    void start();

    /// Add a client connection to a remote network.
    void add_remote(const std::string& name, NetworkClient& client);

    /// Map a local neuron to a remote network + neuron ID.
    /// When the local neuron fires, its activation is forwarded.
    void map_output(const core::NeuronId& local_neuron,
                    const std::string& remote_name,
                    const std::string& remote_neuron_id);

    /// Map a remote neuron to a local neuron.
    /// When signals arrive for this remote neuron, they're injected locally.
    void map_input(const std::string& remote_name,
                   const std::string& remote_neuron_id,
                   const core::NeuronId& local_neuron);

    /// Flush output neurons — check mapped outputs and send signals.
    void flush_outputs();

    /// Get the network ID used in protocol messages.
    const std::string& network_id() const;

    /// Stats for monitoring.
    struct AdapterStats {
        uint64_t signals_received = 0;
        uint64_t signals_sent = 0;
        uint64_t commands_received = 0;
        uint64_t commands_forwarded = 0;
        uint64_t errors = 0;
    };

    AdapterStats stats() const { return stats_; }

private:
    void handle_incoming(const std::string& peer_id, const Message& msg);
    void handle_signal(const std::string& peer_id, const Message& msg);
    void handle_command(const std::string& peer_id, const Message& msg);
    void handle_query(const std::string& peer_id, const Message& msg);

    core::Network& network_;
    command::CommandPropagator& propagator_;
    NetworkServer& server_;

    // Remote connections (name -> client reference)
    std::unordered_map<std::string, NetworkClient*> remotes_;

    // Output mappings: local neuron -> (remote_name, remote_neuron_id)
    struct OutputMapping {
        core::NeuronId local_neuron;
        std::string remote_name;
        std::string remote_neuron_id;
    };
    std::vector<OutputMapping> output_mappings_;

    // Input mappings: (remote_name + ":" + remote_neuron_id) -> local neuron
    std::unordered_map<std::string, core::NeuronId> input_mappings_;

    AdapterStats stats_;
};

} // namespace neurograph::net
