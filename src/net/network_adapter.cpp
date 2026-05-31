#include "neurograph/net/network_adapter.hpp"
#include "neurograph/command/command.hpp"
#include "neurograph/command/command_types.hpp"
#include "neurograph/util/logger.hpp"

namespace neurograph::net {

NetworkAdapter::NetworkAdapter(core::Network& network,
                               command::CommandPropagator& propagator,
                               NetworkServer& server)
    : network_(network), propagator_(propagator), server_(server) {}

void NetworkAdapter::start() {
    server_.on_message([this](const std::string& peer_id, const Message& msg) {
        handle_incoming(peer_id, msg);
    });

    // Also register on all existing remote clients
    for (auto& [name, client] : remotes_) {
        client->on_message([this](const std::string& peer_id, const Message& msg) {
            handle_incoming(peer_id, msg);
        });
    }

    NG_LOG_INFO("NetworkAdapter started, listening for messages");
}

void NetworkAdapter::add_remote(const std::string& name, NetworkClient& client) {
    remotes_[name] = &client;

    client.on_message([this](const std::string& peer_id, const Message& msg) {
        handle_incoming(peer_id, msg);
    });

    NG_LOG_INFO("Added remote network: {}", name);
}

void NetworkAdapter::map_output(const core::NeuronId& local_neuron,
                                const std::string& remote_name,
                                const std::string& remote_neuron_id) {
    output_mappings_.push_back({local_neuron, remote_name, remote_neuron_id});
    NG_LOG_DEBUG("Mapped output: {} -> {}:{}", local_neuron.value().substr(0, 8),
                 remote_name, remote_neuron_id.substr(0, 8));
}

void NetworkAdapter::map_input(const std::string& remote_name,
                               const std::string& remote_neuron_id,
                               const core::NeuronId& local_neuron) {
    std::string key = remote_name + ":" + remote_neuron_id;
    input_mappings_[key] = local_neuron;
    NG_LOG_DEBUG("Mapped input: {} -> {}", key, local_neuron.value().substr(0, 8));
}

void NetworkAdapter::flush_outputs() {
    // Group signals by remote name
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> grouped;

    for (const auto& mapping : output_mappings_) {
        const auto* neuron = network_.get_neuron(mapping.local_neuron);
        if (!neuron) continue;

        double activation = neuron->activation_level();
        if (activation > 0.0) {
            grouped[mapping.remote_name].emplace_back(
                mapping.remote_neuron_id, activation);
        }
    }

    // Send signals to each remote
    for (auto& [remote_name, signals] : grouped) {
        auto it = remotes_.find(remote_name);
        if (it == remotes_.end() || !it->second->is_connected()) continue;

        auto msg = make_signal(server_.port() > 0 ? "server" : "client", signals);
        auto result = it->second->send(msg);
        if (result.ok()) {
            stats_.signals_sent += signals.size();
        } else {
            stats_.errors++;
        }
    }
}

const std::string& NetworkAdapter::network_id() const {
    // Use the first neuron's ID prefix as a stand-in, or empty
    static std::string empty;
    return empty;
}

// ============================================================
// Message handling
// ============================================================

void NetworkAdapter::handle_incoming(const std::string& peer_id, const Message& msg) {
    switch (msg.type) {
        case MessageType::SIGNAL:
            handle_signal(peer_id, msg);
            break;
        case MessageType::COMMAND:
            handle_command(peer_id, msg);
            break;
        case MessageType::QUERY:
            handle_query(peer_id, msg);
            break;
        case MessageType::HEARTBEAT:
            // Acknowledged silently
            break;
        default:
            NG_LOG_WARN("Unhandled message type from {}: {}",
                        peer_id, message_type_to_string(msg.type));
            break;
    }
}

void NetworkAdapter::handle_signal(const std::string& peer_id, const Message& msg) {
    if (!msg.payload.contains("signals") || !msg.payload["signals"].is_array()) {
        stats_.errors++;
        return;
    }

    for (const auto& sig : msg.payload["signals"]) {
        std::string neuron_id = sig.value("neuron_id", "");
        double value = sig.value("value", 0.0);

        // Check if there's an input mapping for this signal
        std::string key = peer_id + ":" + neuron_id;
        auto it = input_mappings_.find(key);
        if (it != input_mappings_.end()) {
            // Inject activation into the mapped local neuron
            auto* neuron = network_.get_neuron(it->second);
            if (neuron) {
                neuron->activate(value);
                stats_.signals_received++;
            }
        } else {
            // Try direct neuron ID match
            auto* neuron = network_.get_neuron(core::NeuronId(neuron_id));
            if (neuron) {
                neuron->activate(value);
                stats_.signals_received++;
            }
        }
    }
}

void NetworkAdapter::handle_command(const std::string& peer_id, const Message& msg) {
    stats_.commands_received++;

    if (!msg.payload.contains("command")) {
        stats_.errors++;
        return;
    }

    try {
        auto cmd = command::Command::from_json(msg.payload["command"]);
        auto result = propagator_.propagate(cmd);
        stats_.commands_forwarded++;

        NG_LOG_INFO("Forwarded command from {}: type={}, visited={} neurons",
                    peer_id, std::string(command::command_type_str(cmd.type())),
                    result.neurons_visited);
    } catch (const std::exception& e) {
        NG_LOG_WARN("Failed to process command from {}: {}", peer_id, e.what());
        stats_.errors++;
    }
}

void NetworkAdapter::handle_query(const std::string& peer_id, const Message& msg) {
    std::string query_type = msg.payload.value("query_type", "");

    nlohmann::json response_payload;

    if (query_type == "stats") {
        auto net_stats = network_.stats();
        response_payload = {
            {"neuron_count", net_stats.neuron_count},
            {"synapse_count", net_stats.synapse_count},
            {"avg_activation", net_stats.avg_activation},
            {"avg_weight", net_stats.avg_weight},
        };
    } else if (query_type == "neuron") {
        std::string nid = msg.payload.value("neuron_id", "");
        const auto* neuron = network_.get_neuron(core::NeuronId(nid));
        if (neuron) {
            response_payload = neuron->to_json();
        } else {
            response_payload = {{"error", "neuron not found"}};
        }
    } else {
        response_payload = {{"error", "unknown query type: " + query_type}};
    }

    // Send response back via server
    Message resp;
    resp.type = MessageType::RESPONSE;
    resp.sender_id = msg.target_id;
    resp.target_id = msg.sender_id;
    resp.sequence = msg.sequence;
    resp.payload = response_payload;

    server_.send(peer_id, resp);
}

} // namespace neurograph::net
