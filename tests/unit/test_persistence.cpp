#include <gtest/gtest.h>
#include "neurograph/persistence/neo4j_config.hpp"
#include "neurograph/persistence/neo4j_client.hpp"
#include "neurograph/persistence/neo4j_repository.hpp"
#include "neurograph/persistence/sync_manager.hpp"
#include "neurograph/core/network.hpp"

using namespace neurograph;

// ============================================================
// Neo4jConfig Tests
// ============================================================

TEST(Neo4jConfigTest, DefaultValues) {
    persistence::Neo4jConfig config;
    EXPECT_EQ(config.host, "localhost");
    EXPECT_EQ(config.port, 7474);
    EXPECT_EQ(config.database, "neo4j");
    EXPECT_EQ(config.username, "neo4j");
    EXPECT_FALSE(config.use_tls);
}

TEST(Neo4jConfigTest, BaseUrl) {
    persistence::Neo4jConfig config;
    EXPECT_EQ(config.base_url(), "http://localhost:7474");

    config.use_tls = true;
    config.port = 7473;
    EXPECT_EQ(config.base_url(), "https://localhost:7473");
}

TEST(Neo4jConfigTest, QueryEndpoint) {
    persistence::Neo4jConfig config;
    EXPECT_EQ(config.query_endpoint(), "http://localhost:7474/db/neo4j/query/v2");

    config.database = "mydb";
    EXPECT_EQ(config.query_endpoint(), "http://localhost:7474/db/mydb/query/v2");
}

TEST(Neo4jConfigTest, TxCommitEndpoint) {
    persistence::Neo4jConfig config;
    EXPECT_EQ(config.tx_commit_endpoint(), "http://localhost:7474/db/neo4j/tx/commit");
}

// ============================================================
// In-Memory Mock Repository for SyncManager Testing
// ============================================================

/// A simple in-memory repository that implements GraphRepository
/// for testing SyncManager without a real Neo4j instance.
class InMemoryRepository : public persistence::GraphRepository {
public:
    // Neuron storage
    std::unordered_map<std::string, nlohmann::json> neurons;
    std::unordered_map<std::string, nlohmann::json> synapses;
    bool connected = true;

    util::Result<void> save_neuron(const core::Neuron& neuron) override {
        neurons[neuron.id().value()] = neuron.to_json();
        return util::Result<void>::success();
    }

    util::Result<core::Neuron> load_neuron(const core::NeuronId& id) override {
        auto it = neurons.find(id.value());
        if (it == neurons.end()) return util::Error("Not found");
        return core::Neuron::from_json(it->second);
    }

    util::Result<void> delete_neuron(const core::NeuronId& id) override {
        neurons.erase(id.value());
        return util::Result<void>::success();
    }

    util::Result<void> save_neurons_batch(
        const std::vector<const core::Neuron*>& batch) override {
        for (const auto* n : batch) {
            neurons[n->id().value()] = n->to_json();
        }
        return util::Result<void>::success();
    }

    util::Result<void> save_synapse(const core::Synapse& synapse) override {
        synapses[synapse.id().value()] = synapse.to_json();
        return util::Result<void>::success();
    }

    util::Result<core::Synapse> load_synapse(const core::SynapseId& id) override {
        auto it = synapses.find(id.value());
        if (it == synapses.end()) return util::Error("Not found");
        return core::Synapse::from_json(it->second);
    }

    util::Result<void> delete_synapse(const core::SynapseId& id) override {
        synapses.erase(id.value());
        return util::Result<void>::success();
    }

    util::Result<void> save_synapses_batch(
        const std::vector<const core::Synapse*>& batch) override {
        for (const auto* s : batch) {
            synapses[s->id().value()] = s->to_json();
        }
        return util::Result<void>::success();
    }

    util::Result<void> save_network(const core::Network& network) override {
        neurons.clear();
        synapses.clear();
        for (const auto& [id, n] : network.neurons()) {
            neurons[id.value()] = n.to_json();
        }
        for (const auto& [id, s] : network.synapses()) {
            synapses[id.value()] = s.to_json();
        }
        return util::Result<void>::success();
    }

    util::Result<core::Network> load_network() override {
        core::Network net;
        for (const auto& [id, j] : neurons) {
            net.add_neuron(core::Neuron::from_json(j));
        }
        for (const auto& [id, j] : synapses) {
            net.add_synapse(core::Synapse::from_json(j));
        }
        return net;
    }

    util::Result<void> clear_all() override {
        neurons.clear();
        synapses.clear();
        return util::Result<void>::success();
    }

    util::Result<void> ensure_schema() override {
        return util::Result<void>::success();
    }

    bool is_connected() const override { return connected; }
};

// ============================================================
// SyncManager Tests
// ============================================================

class SyncManagerTest : public ::testing::Test {
protected:
    core::Network net;
    InMemoryRepository repo;
    core::NeuronId id_a, id_b;

    void SetUp() override {
        core::Neuron a(core::NeuronId::generate(), "alpha");
        core::Neuron b(core::NeuronId::generate(), "beta");
        id_a = a.id();
        id_b = b.id();
        net.add_neuron(std::move(a));
        net.add_neuron(std::move(b));
        net.connect(id_a, id_b, 0.7);
    }
};

TEST_F(SyncManagerTest, MarkDirtyAndSync) {
    persistence::SyncManager sync(net, repo);

    sync.mark_neuron_dirty(id_a);
    sync.mark_neuron_dirty(id_b);
    EXPECT_EQ(sync.pending_count(), 2);

    auto result = sync.sync_to_db();
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(sync.pending_count(), 0);

    // Verify neurons were saved to repo
    EXPECT_EQ(repo.neurons.size(), 2);
    EXPECT_TRUE(repo.neurons.contains(id_a.value()));
    EXPECT_TRUE(repo.neurons.contains(id_b.value()));
}

TEST_F(SyncManagerTest, MarkAllDirty) {
    persistence::SyncManager sync(net, repo);

    sync.mark_all_dirty();
    // 2 neurons + 1 synapse = 3 pending
    EXPECT_EQ(sync.pending_count(), 3);

    auto result = sync.sync_to_db();
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(sync.pending_count(), 0);
    EXPECT_EQ(repo.neurons.size(), 2);
    EXPECT_EQ(repo.synapses.size(), 1);
}

TEST_F(SyncManagerTest, DeleteTracking) {
    persistence::SyncManager sync(net, repo);

    // First save everything
    sync.mark_all_dirty();
    sync.sync_to_db();
    EXPECT_EQ(repo.neurons.size(), 2);

    // Delete a neuron from in-memory network
    auto synapse_ids = net.get_neuron(id_b)->incoming_synapses();
    for (const auto& sid : synapse_ids) {
        sync.mark_synapse_deleted(sid);
    }
    sync.mark_neuron_deleted(id_b);
    net.remove_neuron(id_b);

    EXPECT_GE(sync.pending_count(), 1); // at least the deleted neuron

    auto result = sync.sync_to_db();
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(repo.neurons.size(), 1); // only alpha remains
}

TEST_F(SyncManagerTest, SyncFromDb) {
    // Pre-populate repo with different data
    core::Neuron c(core::NeuronId::generate(), "gamma");
    repo.neurons[c.id().value()] = c.to_json();

    persistence::SyncManager sync(net, repo);

    // Current network has 2 neurons
    EXPECT_EQ(net.neuron_count(), 2);

    auto result = sync.sync_from_db();
    EXPECT_TRUE(result.ok());

    // After sync_from_db, network should match repo (1 neuron: gamma)
    EXPECT_EQ(net.neuron_count(), 1);
    EXPECT_EQ(sync.pending_count(), 0);
}

TEST_F(SyncManagerTest, SyncStats) {
    persistence::SyncManager sync(net, repo);

    sync.mark_all_dirty();
    sync.sync_to_db();

    auto stats = sync.stats();
    EXPECT_EQ(stats.total_syncs, 1);
    EXPECT_EQ(stats.neurons_synced, 2);
    EXPECT_EQ(stats.synapses_synced, 1);
    EXPECT_EQ(stats.errors, 0);
}

TEST_F(SyncManagerTest, NoPendingNoSync) {
    persistence::SyncManager sync(net, repo);

    // No dirty entities — sync should be a no-op
    EXPECT_EQ(sync.pending_count(), 0);
    auto result = sync.sync_to_db();
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(repo.neurons.size(), 0); // nothing was written
}

// ============================================================
// Neo4j Client Tests (connectivity-dependent, skip if no server)
// ============================================================

TEST(Neo4jClientTest, HealthCheckFailsWhenNoServer) {
    persistence::Neo4jConfig config;
    config.host = "localhost";
    config.port = 17474; // unlikely to have a server here
    config.timeout_seconds = 2;

    persistence::Neo4jClient client(config);
    EXPECT_FALSE(client.health_check());
}
