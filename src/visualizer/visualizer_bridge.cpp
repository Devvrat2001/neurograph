#include "neurograph/visualizer/visualizer_bridge.hpp"
#include "neurograph/util/logger.hpp"

namespace neurograph::visualizer {

VisualizerBridge::VisualizerBridge(core::Network& network, uint16_t port)
    : network_(network), port_(port) {}

VisualizerBridge::~VisualizerBridge() {
    stop();
}

// ============================================================
// Start / Stop
// ============================================================

util::Result<void> VisualizerBridge::start() {
    ws_server_.on_message([this](const std::string& msg) {
        handle_message(msg);
    });

    ws_server_.on_connect([this]() {
        NG_LOG_INFO("VisualizerBridge: client connected, waiting for client to request data");
        // Do NOT auto-send state here — the client will request it via get_state
        // This avoids sending a huge payload before the client is ready.
        // Only send a small welcome message so the client knows it's connected.
        try {
            nlohmann::json welcome = {
                {"type", "welcome"},
                {"message", "NeuroGraph backend connected"},
                {"neuron_count", network_.stats().neuron_count},
                {"synapse_count", network_.stats().synapse_count}
            };
            ws_server_.send(welcome.dump());
        } catch (const std::exception& e) {
            NG_LOG_WARN("VisualizerBridge: error sending welcome: {}", e.what());
        }
    });

    ws_server_.on_disconnect([]() {
        NG_LOG_INFO("VisualizerBridge: client disconnected");
    });

    if (!ws_server_.start(port_)) {
        return util::Error("Failed to start WebSocket server on port " + std::to_string(port_));
    }

    return util::Result<void>::success();
}

void VisualizerBridge::stop() {
    stop_streaming();
    ws_server_.stop();
}

// ============================================================
// State Serialization
// ============================================================

nlohmann::json VisualizerBridge::serialize_neuron(const core::Neuron& neuron) const {
    nlohmann::json j;
    j["id"] = neuron.id().value();
    j["label"] = neuron.label();
    j["activation"] = neuron.activation_level();
    j["threshold"] = neuron.threshold();
    j["state"] = std::string(core::neuron_state_str(neuron.state()));
    j["activation_type"] = static_cast<int>(neuron.activation_type());

    if (neuron.position().has_value()) {
        j["x"] = neuron.position()->x;
        j["y"] = neuron.position()->y;
        j["z"] = neuron.position()->z;
    } else {
        j["x"] = 0.0;
        j["y"] = 0.0;
        j["z"] = 0.0;
    }

    j["incoming"] = neuron.incoming_synapses().size();
    j["outgoing"] = neuron.outgoing_synapses().size();

    // Activation history (last 30 values)
    auto& history = neuron.activation_history();
    nlohmann::json hist = nlohmann::json::array();
    size_t start = history.size() > 30 ? history.size() - 30 : 0;
    for (size_t i = start; i < history.size(); ++i) {
        hist.push_back(history[i]);
    }
    j["history"] = hist;

    // Determine type based on connections
    if (neuron.incoming_synapses().empty() && !neuron.outgoing_synapses().empty()) {
        j["type"] = "input";
    } else if (!neuron.incoming_synapses().empty() && neuron.outgoing_synapses().empty()) {
        j["type"] = "output";
    } else {
        j["type"] = "hidden";
    }

    return j;
}

nlohmann::json VisualizerBridge::serialize_synapse(const core::Synapse& synapse) const {
    return {
        {"id", synapse.id().value()},
        {"source", synapse.source_id().value()},
        {"target", synapse.target_id().value()},
        {"weight", synapse.weight()},
        {"type", synapse.type() == core::SynapseType::EXCITATORY ? "excitatory" : "inhibitory"},
        {"plasticity", synapse.plasticity()}
    };
}

nlohmann::json VisualizerBridge::serialize_neuron_compact(const core::Neuron& neuron) const {
    // Compact format: skip history, minimize field names
    nlohmann::json j;
    j["id"] = neuron.id().value();
    j["l"] = neuron.label();
    j["a"] = std::round(neuron.activation_level() * 10000.0) / 10000.0;
    j["t"] = std::round(neuron.threshold() * 1000.0) / 1000.0;
    j["s"] = static_cast<int>(neuron.state());

    if (neuron.position().has_value()) {
        j["x"] = std::round(neuron.position()->x * 100.0) / 100.0;
        j["y"] = std::round(neuron.position()->y * 100.0) / 100.0;
        j["z"] = std::round(neuron.position()->z * 100.0) / 100.0;
    } else {
        j["x"] = 0.0; j["y"] = 0.0; j["z"] = 0.0;
    }

    j["in"] = neuron.incoming_synapses().size();
    j["out"] = neuron.outgoing_synapses().size();

    // Type heuristic
    if (neuron.incoming_synapses().empty() && !neuron.outgoing_synapses().empty())
        j["tp"] = "i";  // input
    else if (!neuron.incoming_synapses().empty() && neuron.outgoing_synapses().empty())
        j["tp"] = "o";  // output
    else
        j["tp"] = "h";  // hidden

    return j;
}

nlohmann::json VisualizerBridge::serialize_synapse_compact(const core::Synapse& synapse) const {
    return {
        {"src", synapse.source_id().value()},
        {"tgt", synapse.target_id().value()},
        {"w", std::round(synapse.weight() * 1000.0) / 1000.0},
        {"tp", synapse.type() == core::SynapseType::EXCITATORY ? "e" : "i"}
    };
}

nlohmann::json VisualizerBridge::serialize_state() const {
    nlohmann::json state;
    state["type"] = "state";

    auto stats = network_.stats();
    state["stats"] = {
        {"neuron_count", stats.neuron_count},
        {"synapse_count", stats.synapse_count},
        {"avg_activation", stats.avg_activation},
        {"avg_weight", stats.avg_weight},
        {"organization_cycle", stats.organization_cycle}
    };

    // Use compact format to keep payload under 500KB
    // For large networks (>200 neurons), use compact serialization
    bool use_compact = stats.neuron_count > 200;

    nlohmann::json neurons_arr = nlohmann::json::array();
    for (const auto& [id, neuron] : network_.neurons()) {
        if (use_compact)
            neurons_arr.push_back(serialize_neuron_compact(neuron));
        else
            neurons_arr.push_back(serialize_neuron(neuron));
    }
    state["neurons"] = neurons_arr;

    // Cap synapses at 2000 for visualization — sending 10k+ synapses
    // creates multi-MB payloads that crash browser WebSocket connections
    constexpr size_t MAX_VIS_SYNAPSES = 2000;
    nlohmann::json synapses_arr = nlohmann::json::array();
    size_t syn_count = 0;
    for (const auto& [id, synapse] : network_.synapses()) {
        if (syn_count >= MAX_VIS_SYNAPSES) break;
        if (use_compact)
            synapses_arr.push_back(serialize_synapse_compact(synapse));
        else
            synapses_arr.push_back(serialize_synapse(synapse));
        syn_count++;
    }
    state["synapses"] = synapses_arr;
    state["compact"] = use_compact;
    state["synapses_capped"] = (stats.synapse_count > MAX_VIS_SYNAPSES);

    return state;
}

nlohmann::json VisualizerBridge::serialize_registry() const {
    nlohmann::json msg;
    msg["type"] = "registry";
    msg["images"] = nlohmann::json::array();

    if (!registry_) return msg;

    for (const auto& [name, entry] : registry_->entries()) {
        msg["images"].push_back({
            {"name", name},
            {"task_type", std::string(snapshot::task_type_str(entry.metadata.task_type))},
            {"task_name", entry.metadata.task_name},
            {"neuron_count", entry.metadata.neuron_count},
            {"synapse_count", entry.metadata.synapse_count},
            {"accuracy", entry.metadata.metrics.accuracy},
            {"epochs", entry.metadata.training.epochs_completed},
            {"tags", entry.metadata.tags},
            {"file_size", entry.file_size_bytes},
            {"description", entry.metadata.description}
        });
    }

    return msg;
}

// ============================================================
// Send
// ============================================================

void VisualizerBridge::send_state() {
    if (!ws_server_.is_connected()) return;
    auto json = serialize_state();
    ws_server_.send(json.dump());
}

void VisualizerBridge::send_event(const std::string& event_type, const nlohmann::json& data) {
    if (!ws_server_.is_connected()) return;
    nlohmann::json msg = {
        {"type", "event"},
        {"event", event_type},
        {"data", data},
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()}
    };
    ws_server_.send(msg.dump());
}

void VisualizerBridge::send_response(const std::string& status, const std::string& message) {
    if (!ws_server_.is_connected()) return;
    nlohmann::json msg = {
        {"type", "response"},
        {"status", status},
        {"message", message}
    };
    ws_server_.send(msg.dump());
}

// ============================================================
// Streaming
// ============================================================

void VisualizerBridge::start_streaming(std::chrono::milliseconds interval) {
    if (streaming_.load()) return;
    stream_interval_ = interval;
    streaming_ = true;
    stream_thread_ = std::thread(&VisualizerBridge::stream_loop, this);
    NG_LOG_INFO("VisualizerBridge: streaming started ({}ms interval)", interval.count());
}

void VisualizerBridge::stop_streaming() {
    streaming_ = false;
    if (stream_thread_.joinable()) stream_thread_.join();
}

void VisualizerBridge::stream_loop() {
    while (streaming_.load()) {
        if (ws_server_.is_connected()) {
            // Run one self-organization step per stream tick
            try {
                network_.self_organize(1);
                send_state();
            } catch (const std::exception& e) {
                NG_LOG_WARN("VisualizerBridge: stream error: {}", e.what());
            }
        } else {
            // Client disconnected — stop streaming automatically
            NG_LOG_INFO("VisualizerBridge: client disconnected, stopping stream");
            streaming_ = false;
            break;
        }
        std::this_thread::sleep_for(stream_interval_);
    }
}

// ============================================================
// Incoming Messages
// ============================================================

void VisualizerBridge::handle_message(const std::string& msg) {
    try {
        auto json = nlohmann::json::parse(msg);
        if (!json.contains("type")) return;

        std::string type = json["type"];
        if (type == "command") {
            handle_command(json);
        }
    } catch (const std::exception& e) {
        NG_LOG_WARN("VisualizerBridge: failed to parse message: {}", e.what());
    }
}

void VisualizerBridge::handle_command(const nlohmann::json& cmd) {
    std::string action = cmd.value("action", "");

    if (action == "get_state") {
        send_state();

    } else if (action == "start_stream") {
        int interval = cmd.value("interval", 100);
        start_streaming(std::chrono::milliseconds(interval));

    } else if (action == "stop_stream") {
        stop_streaming();

    } else if (action == "step") {
        network_.self_organize(1);
        send_state();
        send_event("epoch_complete", {{"cycle", network_.stats().organization_cycle}});

    } else if (action == "get_registry") {
        if (registry_) {
            registry_->scan(); // re-scan directory for any new files
            ws_server_.send(serialize_registry().dump());
        }

    } else if (action == "create_network") {
        handle_create_network(cmd);

    } else if (action == "load_image") {
        handle_load_image(cmd);

    } else if (action == "save_image") {
        handle_save_image(cmd);

    } else if (action == "delete_image") {
        handle_delete_image(cmd);

    } else if (action == "set_param") {
        handle_set_param(cmd);

    } else {
        // Forward unknown commands to the callback
        if (on_command_) {
            on_command_(action, cmd);
        }
    }
}

// ============================================================
// Command Handlers
// ============================================================

void VisualizerBridge::handle_create_network(const nlohmann::json& cmd) {
    int num_neurons = cmd.value("num_neurons", 50);
    int num_connections = cmd.value("num_connections", 100);
    std::string name = cmd.value("name", "untitled");
    std::string task_type = cmd.value("task_type", "custom");
    std::string description = cmd.value("description", "");
    double learning_rate = cmd.value("learning_rate", 0.01);

    NG_LOG_INFO("VisualizerBridge: creating network '{}' ({} neurons, {} connections)",
                name, num_neurons, num_connections);

    // Clear the existing network by creating a fresh one
    // We need to remove all existing neurons and synapses
    {
        std::vector<core::SynapseId> syn_ids;
        for (const auto& [id, _] : network_.synapses()) syn_ids.push_back(id);
        for (const auto& id : syn_ids) network_.remove_synapse(id);

        std::vector<core::NeuronId> neuron_ids;
        for (const auto& [id, _] : network_.neurons()) neuron_ids.push_back(id);
        for (const auto& id : neuron_ids) network_.remove_neuron(id);
    }

    // Create new neurons with random positions
    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> weight_dist(0.1, 1.0);
    std::uniform_int_distribution<int> type_dist(0, 3);

    std::vector<core::NeuronId> ids;
    for (int i = 0; i < num_neurons; ++i) {
        auto id = core::NeuronId::generate();
        auto neuron = core::Neuron(id, name + "_n" + std::to_string(i));
        neuron.set_position(core::Vec3{pos_dist(rng), pos_dist(rng), pos_dist(rng)});
        neuron.set_threshold(0.3 + 0.4 * (rng() % 100) / 100.0);
        neuron.set_activation_type(static_cast<core::ActivationType>(type_dist(rng)));
        network_.add_neuron(std::move(neuron));
        ids.push_back(id);
    }

    // Create random connections
    if (num_neurons > 1) {
        std::uniform_int_distribution<int> idx_dist(0, num_neurons - 1);
        for (int i = 0; i < num_connections; ++i) {
            int src = idx_dist(rng);
            int tgt = idx_dist(rng);
            if (src == tgt) continue;
            network_.connect(ids[src], ids[tgt], weight_dist(rng));
        }
    }

    // Set topology config
    core::TopologyConfig config;
    config.learning_rate = learning_rate;
    network_.set_topology_config(config);

    send_state();
    send_event("network_created", {{"name", name}, {"neurons", num_neurons}, {"connections", num_connections}});
    send_response("ok", "Network '" + name + "' created with " + std::to_string(num_neurons) + " neurons");
}

void VisualizerBridge::handle_load_image(const nlohmann::json& cmd) {
    std::string name = cmd.value("name", "");
    if (name.empty() || !registry_) {
        send_response("error", "No image name specified or registry not available");
        return;
    }

    auto result = registry_->load(name);
    if (!result.ok()) {
        send_response("error", "Failed to load '" + name + "': " + result.error().message);
        return;
    }

    NG_LOG_INFO("VisualizerBridge: loading neural image '{}'", name);

    // Clear existing network
    {
        std::vector<core::SynapseId> syn_ids;
        for (const auto& [id, _] : network_.synapses()) syn_ids.push_back(id);
        for (const auto& id : syn_ids) network_.remove_synapse(id);

        std::vector<core::NeuronId> neuron_ids;
        for (const auto& [id, _] : network_.neurons()) neuron_ids.push_back(id);
        for (const auto& id : neuron_ids) network_.remove_neuron(id);
    }

    // Copy loaded network into our reference network
    auto& loaded = result.value().network;

    // Add neurons from loaded network
    for (const auto& [id, neuron] : loaded.neurons()) {
        network_.add_neuron(core::Neuron(neuron));
    }

    // Add synapses from loaded network
    for (const auto& [id, synapse] : loaded.synapses()) {
        network_.add_synapse(core::Synapse(synapse));
    }

    send_state();
    send_event("image_loaded", {{"name", name}});
    send_response("ok", "Loaded neural image '" + name + "'");

    // Forward to callback if registered
    if (on_command_) on_command_("load_image", cmd);
}

void VisualizerBridge::handle_save_image(const nlohmann::json& cmd) {
    std::string name = cmd.value("name", "");
    if (name.empty()) {
        send_response("error", "No name specified for save");
        return;
    }

    if (!registry_) {
        send_response("error", "Registry not available");
        return;
    }

    std::string description = cmd.value("description", "");
    std::string task_type_str = cmd.value("task_type", "custom");
    bool overwrite = cmd.value("overwrite", false);

    // Parse task type
    snapshot::TaskType task = snapshot::parse_task_type(task_type_str);

    // Build metadata
    snapshot::SnapshotMetadata meta;
    meta.name = name;
    meta.description = description;
    meta.task_type = task;
    meta.task_name = task_type_str;

    // Parse tags
    if (cmd.contains("tags") && cmd["tags"].is_array()) {
        for (const auto& t : cmd["tags"]) {
            if (t.is_string()) meta.tags.push_back(t.get<std::string>());
        }
    }

    // Auto-fill training config from current topology
    auto tconfig = network_.topology_config();
    meta.training.learning_rate = tconfig.learning_rate;
    meta.training.self_org_iterations = static_cast<int>(network_.stats().organization_cycle);

    // Auto-fill metrics
    meta.metrics.avg_activation = network_.stats().avg_activation;
    meta.metrics.avg_weight = network_.stats().avg_weight;

    // If overwriting, remove old entry first
    if (overwrite && registry_->has(name)) {
        registry_->remove(name);
    }

    auto save_result = registry_->save(network_, meta);
    if (save_result.ok()) {
        NG_LOG_INFO("VisualizerBridge: saved neural image '{}'", name);
        send_event("image_saved", {{"name", name}});
        send_response("ok", "Saved '" + name + "'");

        // Re-scan and send updated registry
        registry_->scan();
        ws_server_.send(serialize_registry().dump());
    } else {
        send_response("error", "Save failed: " + save_result.error().message);
    }

    if (on_command_) on_command_("save_image", cmd);
}

void VisualizerBridge::handle_delete_image(const nlohmann::json& cmd) {
    std::string name = cmd.value("name", "");
    if (name.empty() || !registry_) {
        send_response("error", "No name specified or registry not available");
        return;
    }

    auto result = registry_->remove(name);
    if (result.ok()) {
        NG_LOG_INFO("VisualizerBridge: deleted neural image '{}'", name);
        send_event("image_deleted", {{"name", name}});
        send_response("ok", "Deleted '" + name + "'");

        // Re-scan and send updated registry
        registry_->scan();
        ws_server_.send(serialize_registry().dump());
    } else {
        send_response("error", "Delete failed: " + result.error().message);
    }
}

void VisualizerBridge::handle_set_param(const nlohmann::json& cmd) {
    std::string param = cmd.value("param", "");
    double value = cmd.value("value", 0.0);

    auto config = network_.topology_config();

    if (param == "learning_rate") {
        config.learning_rate = value;
    } else if (param == "pruning_threshold") {
        config.affinity_threshold = value; // use affinity threshold as proxy
    } else if (param == "affinity_threshold") {
        config.affinity_threshold = value;
    } else if (param == "max_total_weight") {
        config.max_total_weight = value;
    } else {
        NG_LOG_WARN("VisualizerBridge: unknown param '{}'", param);
        return;
    }

    network_.set_topology_config(config);
    NG_LOG_INFO("VisualizerBridge: set {} = {}", param, value);
}

} // namespace neurograph::visualizer
