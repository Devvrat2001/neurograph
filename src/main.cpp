#include "neurograph/core/network.hpp"
#include "neurograph/core/topology.hpp"
#include "neurograph/gpu/device_network.hpp"
#include "neurograph/crypto/ed25519_provider.hpp"
#include "neurograph/crypto/key_manager.hpp"
#include "neurograph/command/command.hpp"
#include "neurograph/command/command_signer.hpp"
#include "neurograph/command/command_verifier.hpp"
#include "neurograph/command/command_executor.hpp"
#include "neurograph/command/command_propagator.hpp"
#include "neurograph/persistence/neo4j_config.hpp"
#include "neurograph/persistence/neo4j_client.hpp"
#include "neurograph/persistence/neo4j_repository.hpp"
#include "neurograph/persistence/sync_manager.hpp"
#include "neurograph/module/module_interface.hpp"
#include "neurograph/module/module_registry.hpp"
#include "neurograph/module/module_loader.hpp"
#include "neurograph/net/protocol.hpp"
#include "neurograph/net/network_server.hpp"
#include "neurograph/net/network_client.hpp"
#include "neurograph/net/network_adapter.hpp"
#include "neurograph/snapshot/model_snapshot.hpp"
#include "neurograph/snapshot/snapshot_registry.hpp"
#include "neurograph/visualizer/visualizer_bridge.hpp"
#include "neurograph/util/logger.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <limits>
#ifdef _WIN32
#include <conio.h>
#endif

using namespace neurograph;

/// Creates a network with N neurons, randomly positioned in 3D space.
core::Network create_test_network(int num_neurons, int initial_connections) {
    core::Network net;
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> weight_dist(0.1, 1.0);
    std::uniform_int_distribution<int> type_dist(0, 3);

    // Create neurons
    std::vector<core::NeuronId> ids;
    for (int i = 0; i < num_neurons; ++i) {
        auto id = core::NeuronId::generate();
        auto neuron = core::Neuron(id, "neuron_" + std::to_string(i));
        neuron.set_position(core::Vec3{pos_dist(rng), pos_dist(rng), pos_dist(rng)});
        neuron.set_threshold(0.3 + 0.4 * (rng() % 100) / 100.0);
        neuron.set_activation_type(static_cast<core::ActivationType>(type_dist(rng)));
        net.add_neuron(std::move(neuron));
        ids.push_back(id);
    }

    // Create random initial connections
    std::uniform_int_distribution<int> idx_dist(0, num_neurons - 1);
    for (int i = 0; i < initial_connections; ++i) {
        int src = idx_dist(rng);
        int tgt = idx_dist(rng);
        if (src == tgt) continue;
        net.connect(ids[src], ids[tgt], weight_dist(rng));
    }

    return net;
}

void print_stats(const core::NetworkStats& stats) {
    std::cout << "  Neurons:        " << stats.neuron_count << "\n"
              << "  Synapses:       " << stats.synapse_count << "\n"
              << "  Avg Activation: " << std::fixed << std::setprecision(4)
              << stats.avg_activation << "\n"
              << "  Avg Weight:     " << stats.avg_weight << "\n"
              << "  Cycle:          " << stats.organization_cycle << "\n";
}

int main() {
    NG_LOG_INFO("=== NeuroGraph v0.1.0 ===");

    // GPU info is printed by CUDA kernels when first invoked
    NG_LOG_INFO("GPU acceleration enabled (CUDA).");

    // Create network
    constexpr int NUM_NEURONS = 1000;
    constexpr int INITIAL_CONNECTIONS = 2000;
    NG_LOG_INFO("Creating network with {} neurons and {} initial connections...",
                NUM_NEURONS, INITIAL_CONNECTIONS);

    auto net = create_test_network(NUM_NEURONS, INITIAL_CONNECTIONS);

    std::cout << "\n--- Initial State ---\n";
    print_stats(net.stats());

    // Generate random input signals
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> input_dist(0.0, 1.0);
    std::vector<double> inputs(NUM_NEURONS);
    for (auto& v : inputs) v = input_dist(rng);

    // CPU activation
    NG_LOG_INFO("Running CPU activation...");
    auto t0 = std::chrono::high_resolution_clock::now();
    net.activate_all(inputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    NG_LOG_INFO("CPU activation: {:.2f} ms", cpu_ms);

    std::cout << "\n--- After Activation ---\n";
    print_stats(net.stats());

    // CPU self-organization
    NG_LOG_INFO("Running CPU self-organization (5 iterations)...");
    core::TopologyConfig config;
    config.affinity_threshold = 0.25;
    config.learning_rate = 0.01;
    net.set_topology_config(config);

    t0 = std::chrono::high_resolution_clock::now();
    net.self_organize(5);
    t1 = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    NG_LOG_INFO("CPU self-organization: {:.2f} ms", cpu_ms);

    std::cout << "\n--- After Self-Organization ---\n";
    print_stats(net.stats());

    // GPU self-organization
    NG_LOG_INFO("Running GPU-accelerated self-organization (5 iterations)...");
    gpu::DeviceNetwork device_net;
    t0 = std::chrono::high_resolution_clock::now();
    device_net.self_organize_gpu(net, 5, 0.01f);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    NG_LOG_INFO("GPU self-organization: {:.2f} ms", gpu_ms);

    std::cout << "\n--- After GPU Self-Organization ---\n";
    print_stats(net.stats());

    // Demonstrate replication
    NG_LOG_INFO("Demonstrating neuron replication...");
    auto first_id = net.neurons().begin()->first;
    auto clone_result = net.replicate_neuron(first_id, 0.05, true);
    if (clone_result.ok()) {
        NG_LOG_INFO("Replicated neuron {} -> {}", first_id.value(),
                    clone_result.value().value());
    }

    std::cout << "\n--- Final State ---\n";
    print_stats(net.stats());

    // BFS traversal demo
    NG_LOG_INFO("BFS traversal from first neuron (first 10 visited):");
    int count = 0;
    net.bfs(first_id, [&](const core::NeuronId& id) {
        if (count < 10) {
            const auto* n = net.get_neuron(id);
            if (n) {
                std::cout << "  [" << count << "] " << n->label()
                          << " (activation=" << std::fixed << std::setprecision(3)
                          << n->activation_level() << ")\n";
            }
        }
        return ++count < 10;
    });

    // ============================================================
    // Phase 2: Command System Demo
    // ============================================================
    std::cout << "\n========================================\n";
    std::cout << "    COMMAND SYSTEM DEMO (Ed25519)\n";
    std::cout << "========================================\n\n";

    // Initialize cryptographic subsystem
    crypto::Ed25519Provider ed25519;
    crypto::KeyManager key_manager(ed25519);

    NG_LOG_INFO("Generating master key pair...");
    auto master_kp = key_manager.generate_master_key();
    NG_LOG_INFO("Master public key authorized.");

    // Create signer (master user), verifier, executor, propagator
    command::CommandSigner signer(ed25519, master_kp);
    command::CommandVerifier verifier(ed25519, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    // Demo 1: BROADCAST ACTIVATE - activate all neurons via signed command
    NG_LOG_INFO("--- Demo 1: Signed BROADCAST ACTIVATE ---");
    auto activate_cmd = signer.create_signed(
        command::CommandType::ACTIVATE,
        {{"input", 3.0}},
        command::PropagationMode::BROADCAST);

    auto prop_result = propagator.propagate(activate_cmd);
    NG_LOG_INFO("  Authenticated: {}", prop_result.authenticated ? "YES" : "NO");
    NG_LOG_INFO("  Neurons visited: {}, succeeded: {}, failed: {}",
                prop_result.neurons_visited, prop_result.neurons_succeeded,
                prop_result.neurons_failed);

    // Demo 2: TARGETED CREATE_NEURON - create a new neuron
    NG_LOG_INFO("--- Demo 2: Signed CREATE_NEURON ---");
    auto create_cmd = signer.create_signed(
        command::CommandType::CREATE_NEURON,
        {{"label", "command_created_neuron"},
         {"threshold", 0.4},
         {"position", {{"x", 0.0}, {"y", 0.0}, {"z", 0.0}}}},
        command::PropagationMode::TARGETED);
    // CREATE_NEURON is a global command, use executor directly
    auto create_result = executor.execute_global(create_cmd);
    if (create_result.success) {
        NG_LOG_INFO("  Created neuron: {}", create_result.data["neuron_id"].get<std::string>());
    }

    // Demo 3: BFS REPLICATE - replicate through the network
    NG_LOG_INFO("--- Demo 3: Signed BFS REPLICATE ---");
    auto replicate_cmd = signer.create_signed(
        command::CommandType::REPLICATE,
        {{"mutation_rate", 0.05}, {"inherit_connections", false}},
        command::PropagationMode::BFS);
    replicate_cmd.set_ttl(3); // Only replicate first 3 neurons in BFS
    signer.sign(replicate_cmd); // Re-sign after TTL change

    auto rep_result = propagator.propagate(replicate_cmd, first_id);
    NG_LOG_INFO("  Replicated {} neurons", rep_result.neurons_succeeded);

    std::cout << "\n--- After Command System Demo ---\n";
    print_stats(net.stats());

    // Demo 4: Unauthorized command should be rejected
    NG_LOG_INFO("--- Demo 4: Unauthorized command (should be rejected) ---");
    auto rogue_kp = ed25519.generate_keypair();
    command::CommandSigner rogue_signer(ed25519, rogue_kp);
    auto rogue_cmd = rogue_signer.create_signed(
        command::CommandType::SHUTDOWN, {},
        command::PropagationMode::BROADCAST);

    auto rogue_result = propagator.propagate(rogue_cmd);
    NG_LOG_INFO("  Authenticated: {} (expected: NO)", rogue_result.authenticated ? "YES" : "NO");
    if (!rogue_result.auth_error.empty()) {
        NG_LOG_INFO("  Rejection reason: {}", rogue_result.auth_error);
    }

    // Demo 5: Master SHUTDOWN - disable all neurons
    NG_LOG_INFO("--- Demo 5: Master BROADCAST SHUTDOWN ---");
    auto shutdown_cmd = signer.create_signed(
        command::CommandType::SHUTDOWN, {},
        command::PropagationMode::BROADCAST);
    auto shutdown_result = propagator.propagate(shutdown_cmd);
    NG_LOG_INFO("  Shut down {} neurons", shutdown_result.neurons_succeeded);

    std::cout << "\n--- After Command Demo ---\n";
    print_stats(net.stats());

    // ============================================================
    // Phase 3: Persistence Demo (Neo4j)
    // ============================================================
    std::cout << "\n========================================\n";
    std::cout << "    PERSISTENCE DEMO (Neo4j + Sync)\n";
    std::cout << "========================================\n\n";

    // Re-enable neurons for persistence demo (shutdown disabled them)
    for (const auto& [nid, _] : net.neurons()) {
        auto* n = net.get_neuron(nid);
        if (n) n->set_state(core::NeuronState::IDLE);
    }

    // Configure Neo4j connection
    persistence::Neo4jConfig neo4j_config;
    neo4j_config.host = "localhost";
    neo4j_config.port = 7474;
    neo4j_config.database = "neo4j";
    neo4j_config.username = "neo4j";
    neo4j_config.password = "neurograph";

    NG_LOG_INFO("Neo4j endpoint: {}", neo4j_config.query_endpoint());

    // Create client and check connectivity
    persistence::Neo4jClient client(neo4j_config);
    NG_LOG_INFO("Checking Neo4j connectivity...");

    bool neo4j_available = false;
    try {
        neo4j_available = client.health_check();
        if (neo4j_available) {
            NG_LOG_INFO("Neo4j is UP and responding.");
        } else {
            NG_LOG_WARN("Neo4j health check failed.");
        }
    } catch (const std::exception& e) {
        NG_LOG_WARN("Neo4j not available: {} (this is OK for demo)", e.what());
    }

    if (neo4j_available) {
        // Full persistence demo with live Neo4j
        persistence::Neo4jGraphRepository repo(client);

        NG_LOG_INFO("--- Demo P1: Schema Setup ---");
        auto schema_result = repo.ensure_schema();
        if (schema_result.ok()) {
            NG_LOG_INFO("  Database schema (constraints + indexes) created.");
        } else {
            NG_LOG_ERROR("  Schema setup failed: {}", schema_result.error().message);
        }

        NG_LOG_INFO("--- Demo P2: SyncManager - Initial Sync to DB ---");
        persistence::SyncManager sync(net, repo);
        sync.mark_all_dirty();

        auto sync_result = sync.sync_to_db();
        if (sync_result.ok()) {
            auto sync_stats = sync.stats();
            NG_LOG_INFO("  Synced to Neo4j: {} neurons, {} synapses saved",
                        sync_stats.neurons_synced, sync_stats.synapses_synced);
        } else {
            NG_LOG_ERROR("  Sync failed: {}", sync_result.error().message);
        }

        NG_LOG_INFO("--- Demo P3: Modify + Incremental Sync ---");
        // Create a new neuron and mark it dirty
        auto new_id = core::NeuronId::generate();
        auto new_neuron = core::Neuron(new_id, "persistence_demo_neuron");
        new_neuron.set_position(core::Vec3{1.0, 2.0, 3.0});
        new_neuron.set_threshold(0.5);
        net.add_neuron(std::move(new_neuron));
        sync.mark_neuron_dirty(new_id);

        // Connect it to an existing neuron
        auto existing_id = net.neurons().begin()->first;
        auto conn_result = net.connect(new_id, existing_id, 0.75);
        if (conn_result.ok()) {
            sync.mark_synapse_dirty(conn_result.value());
        }

        sync_result = sync.sync_to_db();
        if (sync_result.ok()) {
            NG_LOG_INFO("  Incremental sync: 1 new neuron + 1 new synapse persisted.");
        }

        NG_LOG_INFO("--- Demo P4: Periodic Background Sync ---");
        sync.start_periodic_sync(std::chrono::seconds(2));
        NG_LOG_INFO("  Background sync started (every 2s).");

        // Simulate some activity
        auto another_id = core::NeuronId::generate();
        net.add_neuron(core::Neuron(another_id, "bg_sync_neuron"));
        sync.mark_neuron_dirty(another_id);

        // Wait for one background sync cycle
        std::this_thread::sleep_for(std::chrono::seconds(3));
        auto bg_stats = sync.stats();
        NG_LOG_INFO("  After background cycle: {} total neurons synced, {} total synapses synced",
                    bg_stats.neurons_synced, bg_stats.synapses_synced);

        sync.stop_periodic_sync();
        NG_LOG_INFO("  Background sync stopped.");

        NG_LOG_INFO("--- Demo P5: Load from DB ---");
        auto load_result = sync.sync_from_db();
        if (load_result.ok()) {
            auto loaded_stats = net.stats();
            NG_LOG_INFO("  Loaded from Neo4j: {} neurons, {} synapses",
                        loaded_stats.neuron_count, loaded_stats.synapse_count);
        }

    } else {
        // Offline demo: show config info and Docker instructions
        NG_LOG_INFO("Neo4j is not running. Showing persistence configuration...\n");

        NG_LOG_INFO("--- Persistence Configuration ---");
        NG_LOG_INFO("  Query endpoint:     {}", neo4j_config.query_endpoint());
        NG_LOG_INFO("  TX commit endpoint: {}", neo4j_config.tx_commit_endpoint());
        NG_LOG_INFO("  Database: {}", neo4j_config.database);

        NG_LOG_INFO("--- Dirty Tracking Preview ---");
        NG_LOG_INFO("  Network has {} neurons and {} synapses that would be synced.",
                    net.stats().neuron_count, net.stats().synapse_count);

        NG_LOG_INFO("\n  To run with full persistence, start Neo4j:");
        NG_LOG_INFO("    docker run -d -p 7474:7474 -p 7687:7687 \\");
        NG_LOG_INFO("      -e NEO4J_AUTH=neo4j/neurograph neo4j:5");
        NG_LOG_INFO("  Then re-run this program.");
    }

    // ============================================================
    // Phase 4A: Module System Demo
    // ============================================================
    std::cout << "\n========================================\n";
    std::cout << "    MODULE SYSTEM DEMO\n";
    std::cout << "========================================\n\n";

    // Re-enable neurons for remaining demos
    for (const auto& [nid, _] : net.neurons()) {
        auto* n = net.get_neuron(nid);
        if (n) n->set_state(core::NeuronState::IDLE);
    }

    module::ModuleRegistry mod_registry;

    NG_LOG_INFO("--- Demo M1: In-process Module Registration ---");

    // Create a simple analyzer module inline (simulates a loaded plugin)
    class AnalyzerModule : public module::ModuleInterface {
        uint64_t total_ticks_ = 0;
        double last_avg_activation_ = 0.0;
    public:
        module::ModuleInfo info() const override {
            return {"activity_analyzer", "1.0.0", "Tracks network activity over time",
                    "NeuroGraph", module::ModuleCapability::ANALYZER |
                                  module::ModuleCapability::CUSTOM_COMMANDS};
        }
        util::Result<void> initialize(core::Network&) override {
            return util::Result<void>::success();
        }
        util::Result<void> shutdown() override {
            return util::Result<void>::success();
        }
        void on_tick(core::Network& network, uint64_t tick) override {
            total_ticks_++;
            auto stats = network.stats();
            last_avg_activation_ = stats.avg_activation;
        }
        util::Result<nlohmann::json> handle_command(
            const std::string& cmd, const nlohmann::json& payload) override {
            if (cmd == "report") {
                return nlohmann::json{
                    {"total_ticks", total_ticks_},
                    {"last_avg_activation", last_avg_activation_},
                };
            }
            return util::Error("Unknown command: " + cmd);
        }
        nlohmann::json status() const override {
            return {{"ticks", total_ticks_}, {"alive", true}};
        }
    };

    auto* analyzer = new AnalyzerModule();
    module::ModuleEntry analyzer_entry;
    analyzer_entry.info = analyzer->info();
    analyzer_entry.instance = analyzer;
    analyzer_entry.destroy = [](module::ModuleInterface* m) { delete m; };

    auto reg_result = mod_registry.register_module(std::move(analyzer_entry));
    NG_LOG_INFO("  Registered: {} ({})", analyzer->info().name,
                reg_result.ok() ? "OK" : "FAIL");

    NG_LOG_INFO("--- Demo M2: Module Initialization ---");
    auto init_result = mod_registry.initialize_all(net);
    NG_LOG_INFO("  Initialize all: {}", init_result.ok() ? "OK" : "FAIL");

    NG_LOG_INFO("--- Demo M3: Module Tick Dispatch ---");
    for (uint64_t tick = 0; tick < 10; ++tick) {
        mod_registry.tick_all(net, tick);
    }
    auto mod_status = analyzer->status();
    NG_LOG_INFO("  After 10 ticks: {} ticks processed, alive={}",
                mod_status["ticks"].get<uint64_t>(),
                mod_status["alive"].get<bool>());

    NG_LOG_INFO("--- Demo M4: Module Command Dispatch ---");
    auto cmd_result = mod_registry.dispatch_command("activity_analyzer", "report", {});
    if (cmd_result.ok()) {
        NG_LOG_INFO("  Report: total_ticks={}, last_avg_activation={:.4f}",
                    cmd_result.value()["total_ticks"].get<uint64_t>(),
                    cmd_result.value()["last_avg_activation"].get<double>());
    }

    NG_LOG_INFO("--- Demo M5: Module Loader (directory scan) ---");
    module::ModuleLoader loader(mod_registry);
    auto loaded = loader.load_directory("./modules");
    NG_LOG_INFO("  Scanned ./modules directory: {} plugins found", loaded.size());
    NG_LOG_INFO("  (Place .dll/.so files in ./modules to auto-load plugins)");

    NG_LOG_INFO("  Active modules:");
    for (const auto& minfo : mod_registry.all_info()) {
        NG_LOG_INFO("    - {} v{}: {}", minfo.name, minfo.version, minfo.description);
    }

    // ============================================================
    // Phase 4B: External Network Communication Demo
    // ============================================================
    std::cout << "\n========================================\n";
    std::cout << "    NETWORK COMMUNICATION DEMO\n";
    std::cout << "========================================\n\n";

    NG_LOG_INFO("--- Demo N1: Protocol Serialization ---");
    auto handshake = net::make_handshake("neurograph-main", "NeuroGraph Demo");
    auto wire_data = net::serialize_message(handshake);
    NG_LOG_INFO("  Handshake message: {} bytes on wire", wire_data.size());

    std::vector<std::pair<std::string, double>> signal_data;
    int sig_count = 0;
    for (const auto& [nid, neuron] : net.neurons()) {
        if (neuron.activation_level() > 0.0 && sig_count < 5) {
            signal_data.emplace_back(nid.value(), neuron.activation_level());
            sig_count++;
        }
    }
    auto sig_msg = net::make_signal("neurograph-main", signal_data);
    auto sig_wire = net::serialize_message(sig_msg);
    NG_LOG_INFO("  Signal message ({} neurons): {} bytes", signal_data.size(), sig_wire.size());

    NG_LOG_INFO("--- Demo N2: Server + Client Loopback ---");
    constexpr uint16_t DEMO_PORT = 9876;

    net::NetworkServer server("server-net", "DemoServer");
    auto server_start = server.start(DEMO_PORT);

    if (server_start.ok()) {
        NG_LOG_INFO("  Server listening on port {}", DEMO_PORT);

        // Track messages received by server
        int server_msgs = 0;
        server.on_message([&](const std::string& peer, const net::Message& msg) {
            server_msgs++;
            NG_LOG_INFO("  Server received {} from {}",
                        net::message_type_to_string(msg.type), peer);
        });

        server.on_connect([](const std::string& peer) {
            NG_LOG_INFO("  Server: peer connected: {}", peer);
        });

        // Connect client to server
        net::NetworkClient demo_client("client-net", "DemoClient");
        auto conn = demo_client.connect("127.0.0.1", DEMO_PORT);

        if (conn.ok()) {
            NG_LOG_INFO("  Client connected to server.");

            // Wait for handshake exchange
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            // Send signals from client to server
            auto client_sig = net::make_signal("client-net", signal_data);
            demo_client.send(client_sig);
            NG_LOG_INFO("  Client sent signal with {} neuron activations", signal_data.size());

            // Send a heartbeat
            auto hb = net::make_heartbeat("client-net");
            demo_client.send(hb);

            // Wait for server to process
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            NG_LOG_INFO("  Server processed {} messages", server_msgs);

            // Graceful disconnect
            demo_client.disconnect();
            NG_LOG_INFO("  Client disconnected.");
        } else {
            NG_LOG_WARN("  Client connection failed: {}", conn.error().message);
        }

        // Show connected peers
        auto peers = server.connected_peers();
        NG_LOG_INFO("  Active peers after disconnect: {}", peers.size());

        server.stop();
        NG_LOG_INFO("  Server stopped.");
    } else {
        NG_LOG_WARN("  Server failed to start: {} (port {} may be in use)",
                    server_start.error().message, DEMO_PORT);
        NG_LOG_INFO("  Skipping loopback demo.");
    }

    NG_LOG_INFO("--- Demo N3: NetworkAdapter (concept) ---");
    NG_LOG_INFO("  The NetworkAdapter bridges remote signals <-> local neurons:");
    NG_LOG_INFO("  - map_input():  remote neuron -> local neuron (receive signals)");
    NG_LOG_INFO("  - map_output(): local neuron -> remote neuron (send signals)");
    NG_LOG_INFO("  - flush_outputs(): scan output neurons and send activations");
    NG_LOG_INFO("  - Incoming COMMAND messages are routed through CommandPropagator");

    // Shutdown modules
    mod_registry.shutdown_all();
    NG_LOG_INFO("  All modules shut down.");

    // ============================================================
    // Phase 5A: Neural Image System Demo
    // ============================================================
    std::cout << "\n========================================\n";
    std::cout << "    NEURAL IMAGE SYSTEM DEMO\n";
    std::cout << "========================================\n\n";

    // Re-enable neurons for snapshot
    for (const auto& [nid, _] : net.neurons()) {
        auto* n = net.get_neuron(nid);
        if (n) n->set_state(core::NeuronState::IDLE);
    }

    NG_LOG_INFO("--- Demo S1: Save Neural Image (Binary) ---");
    snapshot::SnapshotMetadata meta;
    meta.name = "demo_network_v1";
    meta.description = "Demo network after self-organization and command processing";
    meta.version = "1.0.0";
    meta.author = "NeuroGraph Demo";
    meta.task_type = snapshot::TaskType::CLASSIFICATION;
    meta.task_name = "demo_classification";
    meta.tags = {"demo", "self-organized", "gpu-trained"};
    meta.training.learning_rate = 0.01;
    meta.training.epochs_completed = 10;
    meta.training.self_org_iterations = 10;
    meta.training.topology_strategy = "hebbian";
    meta.metrics.accuracy = 0.85;
    meta.metrics.loss = 0.35;

    auto save_result_s = snapshot::ModelSnapshot::save(
        net, meta, "./models/demo_network_v1.ngmodel",
        snapshot::SnapshotFormat::BINARY);
    if (save_result_s.ok()) {
        NG_LOG_INFO("  Saved binary neural image: ./models/demo_network_v1.ngmodel");
    } else {
        NG_LOG_WARN("  Save failed: {}", save_result_s.error().message);
    }

    NG_LOG_INFO("--- Demo S2: Save Neural Image (JSON) ---");
    meta.name = "demo_network_v1_json";
    auto json_save = snapshot::ModelSnapshot::save(
        net, meta, "./models/demo_network_v1.ngmodel.json",
        snapshot::SnapshotFormat::JSON);
    if (json_save.ok()) {
        NG_LOG_INFO("  Saved JSON neural image: ./models/demo_network_v1.ngmodel.json");
    } else {
        NG_LOG_WARN("  Save failed: {}", json_save.error().message);
    }

    NG_LOG_INFO("--- Demo S3: Read Metadata (without loading network) ---");
    auto meta_result = snapshot::ModelSnapshot::read_metadata(
        "./models/demo_network_v1.ngmodel");
    if (meta_result.ok()) {
        auto& m = meta_result.value();
        NG_LOG_INFO("  Name: {}", m.name);
        NG_LOG_INFO("  Task: {} ({})", m.task_name, task_type_str(m.task_type));
        NG_LOG_INFO("  Neurons: {}, Synapses: {}", m.neuron_count, m.synapse_count);
        NG_LOG_INFO("  Accuracy: {:.1f}%, Epochs: {}",
                    m.metrics.accuracy * 100.0, m.training.epochs_completed);
        std::string tags_str;
        for (size_t i = 0; i < m.tags.size(); ++i) {
            if (i > 0) tags_str += ", ";
            tags_str += m.tags[i];
        }
        NG_LOG_INFO("  Tags: {}", tags_str);
    }

    NG_LOG_INFO("--- Demo S4: Load Neural Image from Disk ---");
    auto load_result_s = snapshot::ModelSnapshot::load(
        "./models/demo_network_v1.ngmodel");
    if (load_result_s.ok()) {
        auto& image = load_result_s.value();
        NG_LOG_INFO("  Loaded: {} ({} neurons, {} synapses)",
                    image.metadata.name,
                    image.network.neuron_count(),
                    image.network.synapse_count());
        NG_LOG_INFO("  Network is ready to use — all weights and topology restored.");
    }

    NG_LOG_INFO("--- Demo S5: Snapshot Registry ---");

    // Save a second model with different accuracy
    snapshot::SnapshotMetadata meta2;
    meta2.name = "demo_network_v2";
    meta2.description = "Improved version with higher accuracy";
    meta2.task_type = snapshot::TaskType::CLASSIFICATION;
    meta2.task_name = "demo_classification";
    meta2.tags = {"demo", "improved"};
    meta2.training.epochs_completed = 50;
    meta2.metrics.accuracy = 0.93;

    snapshot::SnapshotRegistry registry("./models");
    registry.save(net, meta2);

    auto scan_result = registry.scan();
    if (scan_result.ok()) {
        NG_LOG_INFO("  Registry: {} neural images found", scan_result.value());
    }

    NG_LOG_INFO("  Catalog:");
    registry.print_catalog();

    auto best = registry.find_best(snapshot::TaskType::CLASSIFICATION);
    if (best.ok()) {
        NG_LOG_INFO("  Best classifier: {} (accuracy: {:.1f}%)",
                    best.value()->metadata.name,
                    best.value()->metadata.metrics.accuracy * 100.0);
    }

    auto by_tag = registry.find_by_tag("demo");
    NG_LOG_INFO("  Models tagged 'demo': {}", by_tag.size());

    NG_LOG_INFO("  Neural images can be loaded like functions:");
    NG_LOG_INFO("    auto image = registry.load(\"demo_network_v2\");");
    NG_LOG_INFO("    // image.network is ready — feed inputs, read outputs");

    // ============================================================
    // Phase 5B: Live Workbench (WebSocket)
    // ============================================================
    std::cout << "\n========================================\n";
    std::cout << "    NEUROGRAPH WORKBENCH (WebSocket)\n";
    std::cout << "========================================\n\n";

    // Re-enable neurons for visualization
    for (const auto& [nid, _] : net.neurons()) {
        auto* n = net.get_neuron(nid);
        if (n) n->set_state(core::NeuronState::IDLE);
    }

    constexpr uint16_t VIS_PORT = 8765;
    visualizer::VisualizerBridge vis_bridge(net, VIS_PORT);
    vis_bridge.set_registry(&registry);

    auto vis_start = vis_bridge.start();
    if (vis_start.ok()) {
        NG_LOG_INFO("Workbench server started on ws://localhost:{}", VIS_PORT);
        NG_LOG_INFO("Open neurograph_workbench.html in your browser to connect.");
        NG_LOG_INFO("");
        NG_LOG_INFO("  Features available from the workbench:");
        NG_LOG_INFO("    - Create new networks (specify neurons, connections)");
        NG_LOG_INFO("    - Load existing neural images from the registry");
        NG_LOG_INFO("    - Train with live self-organization (step or continuous)");
        NG_LOG_INFO("    - Tune parameters (learning rate, thresholds)");
        NG_LOG_INFO("    - Save networks with custom name, tags, description");
        NG_LOG_INFO("    - Delete neural images from the registry");
        NG_LOG_INFO("    - Inspect individual neurons and their history");
        NG_LOG_INFO("");
        std::cout << "\n--- Current Network State ---\n";
        print_stats(net.stats());

        std::cout << "\n[Press Enter to stop the server and exit...]\n" << std::flush;

        // Wait for the server to be running — block this thread
        // Use a simple polling loop so it works in both interactive and piped modes
        while (vis_bridge.is_connected() || true) {
            // Check if user pressed Enter (non-blocking on Windows)
            #ifdef _WIN32
            if (_kbhit()) {
                _getch();
                break;
            }
            #else
            // On POSIX, just use cin
            if (std::cin.peek() != EOF) {
                std::cin.get();
                break;
            }
            #endif
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        vis_bridge.stop();
        NG_LOG_INFO("Workbench server stopped.");
    } else {
        NG_LOG_WARN("Failed to start workbench: {}", vis_start.error().message);

        std::cout << "\n--- Final Network State ---\n";
        print_stats(net.stats());
    }

    NG_LOG_INFO("=== NeuroGraph Demo Complete ===");
    return 0;
}
