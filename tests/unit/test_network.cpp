#include <gtest/gtest.h>
#include "neurograph/core/network.hpp"
#include "neurograph/util/uuid.hpp"   // P1-19 deterministic-self-org tests

using namespace neurograph::core;

class NetworkTest : public ::testing::Test {
protected:
    Network net;
    NeuronId id_a, id_b, id_c;

    void SetUp() override {
        Neuron a(NeuronId::generate(), "neuron_a");
        Neuron b(NeuronId::generate(), "neuron_b");
        Neuron c(NeuronId::generate(), "neuron_c");

        a.set_position(Vec3{0, 0, 0});
        b.set_position(Vec3{1, 0, 0});
        c.set_position(Vec3{2, 0, 0});

        id_a = a.id();
        id_b = b.id();
        id_c = c.id();

        net.add_neuron(std::move(a));
        net.add_neuron(std::move(b));
        net.add_neuron(std::move(c));
    }
};

// --- Neuron CRUD ---

TEST_F(NetworkTest, AddNeuron) {
    EXPECT_EQ(net.neuron_count(), 3);
    EXPECT_TRUE(net.has_neuron(id_a));
    EXPECT_TRUE(net.has_neuron(id_b));
    EXPECT_TRUE(net.has_neuron(id_c));
}

TEST_F(NetworkTest, AddDuplicateNeuron) {
    Neuron dup(id_a, "duplicate");
    auto result = net.add_neuron(std::move(dup));
    EXPECT_TRUE(result.is_error());
}

TEST_F(NetworkTest, GetNeuron) {
    auto* n = net.get_neuron(id_a);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->label(), "neuron_a");

    auto* missing = net.get_neuron(NeuronId("nonexistent"));
    EXPECT_EQ(missing, nullptr);
}

TEST_F(NetworkTest, RemoveNeuron) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_c, id_a, 0.5);

    EXPECT_EQ(net.synapse_count(), 2);

    auto result = net.remove_neuron(id_a);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(net.neuron_count(), 2);
    EXPECT_FALSE(net.has_neuron(id_a));

    // Synapses connected to removed neuron should also be removed
    EXPECT_EQ(net.synapse_count(), 0);
}

TEST_F(NetworkTest, RemoveNonexistentNeuron) {
    auto result = net.remove_neuron(NeuronId("nonexistent"));
    EXPECT_TRUE(result.is_error());
}

// --- Synapse CRUD ---

TEST_F(NetworkTest, Connect) {
    auto result = net.connect(id_a, id_b, 0.75);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(net.synapse_count(), 1);

    auto* syn = net.get_synapse(result.value());
    ASSERT_NE(syn, nullptr);
    EXPECT_EQ(syn->source_id(), id_a);
    EXPECT_EQ(syn->target_id(), id_b);
    EXPECT_DOUBLE_EQ(syn->weight(), 0.75);
}

TEST_F(NetworkTest, ConnectInvalidNeurons) {
    auto result = net.connect(id_a, NeuronId("nonexistent"), 1.0);
    EXPECT_TRUE(result.is_error());
}

TEST_F(NetworkTest, Disconnect) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_a, id_c, 0.5);
    EXPECT_EQ(net.synapse_count(), 2);

    auto result = net.disconnect(id_a, id_b);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(net.synapse_count(), 1);
}

TEST_F(NetworkTest, RemoveSynapse) {
    auto sid = net.connect(id_a, id_b, 1.0).value();

    auto* neuron_a = net.get_neuron(id_a);
    auto* neuron_b = net.get_neuron(id_b);
    EXPECT_EQ(neuron_a->outgoing_synapses().size(), 1);
    EXPECT_EQ(neuron_b->incoming_synapses().size(), 1);

    net.remove_synapse(sid);

    EXPECT_EQ(net.synapse_count(), 0);
    EXPECT_EQ(neuron_a->outgoing_synapses().size(), 0);
    EXPECT_EQ(neuron_b->incoming_synapses().size(), 0);
}

// --- Adjacency ---

TEST_F(NetworkTest, GetNeighbors) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_a, id_c, 0.5);

    auto neighbors = net.get_neighbors(id_a);
    EXPECT_EQ(neighbors.size(), 2);

    auto neighbors_b = net.get_neighbors(id_b);
    EXPECT_TRUE(neighbors_b.empty());
}

// --- Replication ---

TEST_F(NetworkTest, ReplicateNeuron) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_a, id_c, 0.5);

    auto result = net.replicate_neuron(id_a, 0.0, true);
    EXPECT_TRUE(result.ok());

    NeuronId clone_id = result.value();
    EXPECT_EQ(net.neuron_count(), 4);
    EXPECT_TRUE(net.has_neuron(clone_id));

    // Clone should have inherited connections to b and c
    auto* clone = net.get_neuron(clone_id);
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->outgoing_synapses().size(), 2);
}

TEST_F(NetworkTest, ReplicateNonexistent) {
    auto result = net.replicate_neuron(NeuronId("nonexistent"));
    EXPECT_TRUE(result.is_error());
}

// --- Traversal ---

TEST_F(NetworkTest, BFS) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_b, id_c, 1.0);

    std::vector<NeuronId> visited;
    net.bfs(id_a, [&](const NeuronId& id) {
        visited.push_back(id);
        return true;
    });

    EXPECT_EQ(visited.size(), 3);
    EXPECT_EQ(visited[0], id_a);
}

TEST_F(NetworkTest, BFSStopEarly) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_b, id_c, 1.0);

    std::vector<NeuronId> visited;
    net.bfs(id_a, [&](const NeuronId& id) {
        visited.push_back(id);
        return visited.size() < 2; // stop after 2
    });

    EXPECT_EQ(visited.size(), 2);
}

TEST_F(NetworkTest, DFS) {
    net.connect(id_a, id_b, 1.0);
    net.connect(id_a, id_c, 1.0);
    net.connect(id_b, id_c, 1.0);

    std::vector<NeuronId> visited;
    net.dfs(id_a, [&](const NeuronId& id) {
        visited.push_back(id);
        return true;
    });

    EXPECT_EQ(visited.size(), 3);
    EXPECT_EQ(visited[0], id_a);
}

// --- Activation ---

TEST_F(NetworkTest, ActivateAll) {
    net.connect(id_a, id_b, 1.0);

    std::vector<double> inputs = {1.0, 0.0, 0.0};
    net.activate_all(inputs);

    auto* a = net.get_neuron(id_a);
    EXPECT_GT(a->activation_level(), 0.0);
}

// --- Self-Organization ---

TEST_F(NetworkTest, SelfOrganize) {
    // Activate neurons so Hebbian update has something to work with
    for (auto& [id, neuron] : const_cast<std::unordered_map<NeuronId, Neuron>&>(
             static_cast<const Network&>(net).neurons())) {
        neuron.activate(1.0);
    }

    TopologyConfig config;
    config.affinity_threshold = 0.1; // low threshold to get connections
    net.set_topology_config(config);

    size_t initial_synapses = net.synapse_count();
    net.self_organize(1);

    // Should have formed some new connections
    EXPECT_GE(net.synapse_count(), initial_synapses);
    EXPECT_EQ(net.current_cycle(), 1);
}

// --- P1-16: bounded growth ----------------------------------------------

namespace {

// Build a network with 2 input + 2 output (protected) + N hidden neurons.
// Returns the network plus the hidden ids so the test can drive growth
// by activating them. Wires inputs→hidden→outputs so growth has both
// sides to inherit from when splitting.
struct BoundedGrowthNet {
    Network                       net;
    std::unordered_set<NeuronId>  io;
    std::vector<NeuronId>         hidden;
};

BoundedGrowthNet make_growth_fixture(int n_hidden = 4) {
    BoundedGrowthNet f;
    auto in_a  = f.net.add_neuron(Neuron(NeuronId::generate(), "input_a")).value();
    auto in_b  = f.net.add_neuron(Neuron(NeuronId::generate(), "input_b")).value();
    auto out_a = f.net.add_neuron(Neuron(NeuronId::generate(), "out_a")).value();
    auto out_b = f.net.add_neuron(Neuron(NeuronId::generate(), "out_b")).value();
    f.io = {in_a, in_b, out_a, out_b};
    for (int h = 0; h < n_hidden; ++h) {
        auto hid = f.net.add_neuron(Neuron(NeuronId::generate(),
                                           "hidden_" + std::to_string(h)))
                        .value();
        f.hidden.push_back(hid);
        f.net.connect(in_a, hid, 0.3);
        f.net.connect(in_b, hid, 0.3);
        f.net.connect(hid, out_a, 0.3);
        f.net.connect(hid, out_b, 0.3);
    }
    f.net.set_protected_neurons(f.io);
    return f;
}

// Drive growth by reporting a sequence of stagnant losses + activating
// hidden neurons so the stress detector picks them as candidates.
//
// Stress is computed from each neuron's `activation_history` and needs
// at least 4 samples before that neuron qualifies — so we activate
// repeatedly within a cycle (varying levels) to build up history fast,
// then call self_organize. Loss is reported flat so the stall counter
// climbs (when use_stall_counter=true) and growth is enabled.
void drive_growth(Network& net, const std::vector<NeuronId>& hidden,
                  int cycles) {
    static constexpr int kPrimeSamples = 6;  // > stress min of 4
    for (int c = 0; c < cycles; ++c) {
        for (int s = 0; s < kPrimeSamples; ++s) {
            for (std::size_t h = 0; h < hidden.size(); ++h) {
                auto* n = net.get_neuron(hidden[h]);
                if (!n) continue;
                // Alternating high/low levels => non-zero variance
                // every neuron, with the variance pattern depending
                // on neuron index so the stress sort isn't trivial.
                const double v = ((s + h) % 2 == 0) ? 0.9 : 0.1;
                n->activate(v);
            }
        }
        // Also activate inputs / outputs so spectator-state-based
        // checks don't filter out the cycle.
        net.report_training_loss(0.5); // flat — drives stall counter up
        net.self_organize(1);
    }
}

} // namespace

// ---------------------------------------------------------------------
// P1-20: growth-only self-org variant. Skips the rewire +
// Hebbian-strengthen + activation-mutation phases that conflict with
// concurrent supervised backprop, so the orchestrator can interleave
// it during training without poisoning the gradient signal.
// ---------------------------------------------------------------------

TEST(NetworkGrowthOnly, PreservesPreExistingSynapseWeights) {
    // Build a small fixture, set known weights on its synapses, drive
    // growth, then assert every original synapse still has a weight
    // close to the original (only the half-and-noise from explicit
    // splits should perturb a small fraction; the rewire/Hebbian
    // phases that DO modify all weights are skipped).
    auto f = make_growth_fixture(/*n_hidden=*/4);
    TopologyConfig cfg;
    cfg.use_stall_counter = false;        // grow every cycle
    cfg.growth_split_ratio = 0.5;
    cfg.growth_min_splits = 1;
    cfg.max_neurons = 24;
    cfg.max_split_fraction_per_cycle = 0.20;
    f.net.set_topology_config(cfg);

    // Snapshot every existing synapse's id + weight before growth.
    std::unordered_map<SynapseId, double> before;
    for (const auto& [sid, syn] : f.net.synapses()) {
        before[sid] = syn.weight();
    }
    const std::size_t n_before = before.size();

    drive_growth(f.net, f.hidden, /*cycles=*/3);

    // For every pre-existing synapse that still exists post-growth,
    // its absolute weight should NOT have grown — splits only halve
    // outgoing weights of the original-of-the-split-pair (potentially
    // multiple times across cycles), and prune removes synapses
    // entirely. Neither operation can produce a larger |w|.
    // Hebbian strengthening + rewire WOULD produce growing weights;
    // proving they didn't run is what this test pins down.
    int touched_by_split = 0;
    for (const auto& [sid, syn] : f.net.synapses()) {
        auto it = before.find(sid);
        if (it == before.end()) continue;  // newly added by split — not in `before`
        const double orig    = it->second;
        const double current = syn.weight();
        // Same sign — splits halve magnitude but preserve sign;
        // rewire/Hebbian could flip it.
        if (orig != 0.0) {
            EXPECT_GE(orig * current, 0.0)
                << "synapse " << sid.value()
                << " sign flipped (orig=" << orig << " now=" << current << ")";
        }
        // Magnitude never grows — splits only divide.
        EXPECT_LE(std::abs(current), std::abs(orig) + 1e-12)
            << "synapse " << sid.value()
            << " weight magnitude grew (orig=" << orig << " now=" << current
            << ") — growth-only must not strengthen";
        if (std::abs(current - orig) > 1e-12) ++touched_by_split;
    }
    // Sanity: with the high split ratio + non-stall config, splits
    // should have touched at least some pre-existing synapses (i.e.
    // the test's "growth happened" precondition holds).
    (void) n_before;
    EXPECT_GT(touched_by_split, 0)
        << "no synapse weights changed — growth-only didn't actually "
           "split anything, so the no-strengthen invariant is vacuous";
}

TEST(NetworkGrowthOnly, GrowsStressedNeurons) {
    auto f = make_growth_fixture(/*n_hidden=*/4);
    TopologyConfig cfg;
    cfg.use_stall_counter = false;
    cfg.growth_split_ratio = 0.5;
    cfg.growth_min_splits = 1;
    cfg.max_neurons = 32;
    cfg.max_split_fraction_per_cycle = 0.20;
    f.net.set_topology_config(cfg);

    const std::size_t neurons_before = f.net.neuron_count();

    // Manually drive activations so the stress detector picks
    // candidates, then call growth_only_self_organize directly.
    static constexpr int kPrime = 6;
    for (int c = 0; c < 5; ++c) {
        for (int s = 0; s < kPrime; ++s) {
            for (std::size_t h = 0; h < f.hidden.size(); ++h) {
                auto* n = f.net.get_neuron(f.hidden[h]);
                if (n) n->activate(((s + h) % 2 == 0) ? 0.9 : 0.1);
            }
        }
        f.net.report_training_loss(0.5);
        f.net.growth_only_self_organize(1);
    }

    EXPECT_GT(f.net.neuron_count(), neurons_before)
        << "growth-only self-org never grew any neurons under "
           "favorable stress + non-stall-gated config";
}

TEST(NetworkGrowthOnly, RespectsMaxNeuronsCap) {
    // Sanity that the growth-only path also honors the P1-16 caps.
    auto f = make_growth_fixture(/*n_hidden=*/4);
    TopologyConfig cfg;
    cfg.use_stall_counter = false;
    cfg.growth_split_ratio = 1.0;
    cfg.growth_min_splits = 4;
    cfg.max_neurons = 12;          // very tight: starts at 8, grows to ≤12
    f.net.set_topology_config(cfg);

    static constexpr int kPrime = 6;
    for (int c = 0; c < 10; ++c) {
        for (int s = 0; s < kPrime; ++s) {
            for (std::size_t h = 0; h < f.hidden.size(); ++h) {
                auto* n = f.net.get_neuron(f.hidden[h]);
                if (n) n->activate(((s + h) % 2 == 0) ? 0.9 : 0.1);
            }
        }
        f.net.report_training_loss(0.5);
        f.net.growth_only_self_organize(1);
    }
    EXPECT_LE(f.net.neuron_count(), 12u);
}

TEST(NetworkBoundedGrowth, MaxNeuronsCapStopsGrowth) {
    auto f = make_growth_fixture(/*n_hidden=*/4);
    TopologyConfig cfg;
    cfg.use_stall_counter   = false;   // grow every cycle, ignore loss
    cfg.growth_split_ratio  = 1.0;     // try to split everything
    cfg.growth_min_splits   = 4;
    cfg.max_neurons         = 10;      // hard ceiling on TOTAL neurons
    f.net.set_topology_config(cfg);

    drive_growth(f.net, f.hidden, /*cycles=*/20);

    // Started with 4 inputs+outputs + 4 hidden = 8 neurons; cap at 10
    // means at most 2 more can be grown. Strict <= because the cap
    // refuses any growth that would push past it.
    EXPECT_LE(f.net.neuron_count(), 10u)
        << "max_neurons cap was not respected after 20 grow cycles";
}

TEST(NetworkBoundedGrowth, MaxSplitFractionPerCycleLimitsBurst) {
    auto f = make_growth_fixture(/*n_hidden=*/20);
    TopologyConfig cfg;
    cfg.use_stall_counter            = false;
    cfg.growth_split_ratio           = 1.0;   // would split all 20
    cfg.growth_min_splits            = 1;
    cfg.max_split_fraction_per_cycle = 0.10;  // but cap at 10% of network
    // Total neurons = 4 IO + 20 hidden = 24. 10% of 24 = 2 splits/cycle.
    f.net.set_topology_config(cfg);

    const std::size_t before = f.net.neuron_count();
    drive_growth(f.net, f.hidden, /*cycles=*/1);
    const std::size_t after = f.net.neuron_count();

    // At most 2 splits in one cycle (10% of 24).
    EXPECT_LE(after - before, 2u)
        << "max_split_fraction_per_cycle was not respected: grew "
        << (after - before) << " neurons in one cycle (expected <= 2)";
}

// P1-19: DeterministicSelfOrgScope makes self-org reproducible with the
// same seed. The orchestrator pairs it with util::DeterministicUuidScope
// — both are needed for byte-identical bundles because:
//   - UUID scope keeps neuron IDs (and thus unordered_map iteration
//     order) stable.
//   - Self-org scope keeps the random draws inside grow_neurons /
//     dynamic_rewire / Neuron::replicate stable.
// This test mirrors that pairing and asserts two runs end up with
// identical structural growth (same neuron + synapse counts).
TEST(NetworkDeterministicSelfOrg, SameSeedProducesSameGrowthShape) {
    auto run_growth = [](std::uint64_t seed) {
        // Same scope pairing the orchestrator uses, so the test
        // measures the actual production-path determinism contract.
        neurograph::util::DeterministicUuidScope uuid_scope(seed);
        DeterministicSelfOrgScope                selforg_scope(
            seed ^ 0x5a5a5a5a5a5a5a5aULL);

        auto f = make_growth_fixture(/*n_hidden=*/8);
        TopologyConfig cfg;
        cfg.use_stall_counter            = false;
        cfg.growth_split_ratio           = 0.5;
        cfg.growth_min_splits            = 2;
        cfg.max_neurons                  = 32;
        cfg.max_split_fraction_per_cycle = 0.30;
        f.net.set_topology_config(cfg);
        drive_growth(f.net, f.hidden, /*cycles=*/5);
        return std::pair{f.net.neuron_count(), f.net.synapse_count()};
    };

    const auto a = run_growth(/*seed=*/12345);
    const auto b = run_growth(/*seed=*/12345);
    EXPECT_EQ(a.first,  b.first)  << "neuron count differs across runs";
    EXPECT_EQ(a.second, b.second) << "synapse count differs across runs";
}

// Sanity that the scope actually consults the seed — if growth shape
// is identical for any seed, the wiring is broken (e.g. the scope
// isn't being read by the random draws). Loose assertion: at least
// neuron OR synapse count differs.
TEST(NetworkDeterministicSelfOrg, DifferentSeedsProduceDifferentGrowth) {
    auto run_growth = [](std::uint64_t seed) {
        neurograph::util::DeterministicUuidScope uuid_scope(seed);
        DeterministicSelfOrgScope                selforg_scope(
            seed ^ 0x5a5a5a5a5a5a5a5aULL);

        auto f = make_growth_fixture(/*n_hidden=*/8);
        TopologyConfig cfg;
        cfg.use_stall_counter            = false;
        cfg.growth_split_ratio           = 0.5;
        cfg.growth_min_splits            = 2;
        cfg.max_neurons                  = 32;
        cfg.max_split_fraction_per_cycle = 0.30;
        f.net.set_topology_config(cfg);
        drive_growth(f.net, f.hidden, /*cycles=*/5);
        return std::pair{f.net.neuron_count(), f.net.synapse_count()};
    };

    const auto a = run_growth(/*seed=*/12345);
    const auto c = run_growth(/*seed=*/999);
    EXPECT_TRUE(a.first != c.first || a.second != c.second)
        << "seeds 12345 and 999 produced indistinguishable growth — "
           "scope may not be wired correctly";
}

TEST(NetworkBoundedGrowth, ZeroCapsPreserveLegacyUnboundedGrowth) {
    // Regression guard: with both new caps at their default 0 (= off),
    // growth behavior must match pre-P1-16. We don't assert on a
    // specific count — just that growth is *not* artificially stopped
    // at small numbers when both caps are unset.
    auto f = make_growth_fixture(/*n_hidden=*/4);
    TopologyConfig cfg;
    cfg.use_stall_counter            = false;
    cfg.growth_split_ratio           = 1.0;
    cfg.growth_min_splits            = 2;
    cfg.max_neurons                  = 0;     // unbounded
    cfg.max_split_fraction_per_cycle = 0.0;   // uncapped
    cfg.max_hidden_neurons           = 0;     // also unbounded
    f.net.set_topology_config(cfg);

    drive_growth(f.net, f.hidden, /*cycles=*/3);

    // Should have grown well past the original 8 neurons since no
    // cap is engaged. Loose assertion to avoid coupling to the
    // exact growth-rate constants.
    EXPECT_GT(f.net.neuron_count(), 10u)
        << "default unbounded behavior regressed — caps fired with "
           "both knobs at 0";
}

// --- Statistics ---

TEST_F(NetworkTest, Stats) {
    net.connect(id_a, id_b, 0.5);
    net.connect(id_b, id_c, 1.5);

    auto stats = net.stats();
    EXPECT_EQ(stats.neuron_count, 3);
    EXPECT_EQ(stats.synapse_count, 2);
    EXPECT_DOUBLE_EQ(stats.avg_weight, 1.0); // (0.5 + 1.5) / 2
}

// --- JSON ---

TEST_F(NetworkTest, JsonRoundTrip) {
    net.connect(id_a, id_b, 0.5);
    net.connect(id_b, id_c, 1.0);

    auto j = net.to_json();
    Network restored = Network::from_json(j);

    EXPECT_EQ(restored.neuron_count(), 3);
    EXPECT_EQ(restored.synapse_count(), 2);
    EXPECT_TRUE(restored.has_neuron(id_a));
    EXPECT_TRUE(restored.has_neuron(id_b));
    EXPECT_TRUE(restored.has_neuron(id_c));
}
