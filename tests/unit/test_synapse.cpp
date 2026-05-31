#include <gtest/gtest.h>
#include "neurograph/core/synapse.hpp"

using namespace neurograph::core;

TEST(SynapseTest, ConstructAndIdentity) {
    auto sid = SynapseId::generate();
    auto src = NeuronId::generate();
    auto tgt = NeuronId::generate();

    Synapse s(sid, src, tgt, 0.75, SynapseType::EXCITATORY);

    EXPECT_EQ(s.id(), sid);
    EXPECT_EQ(s.source_id(), src);
    EXPECT_EQ(s.target_id(), tgt);
    EXPECT_DOUBLE_EQ(s.weight(), 0.75);
    EXPECT_EQ(s.type(), SynapseType::EXCITATORY);
    EXPECT_DOUBLE_EQ(s.plasticity(), 1.0);
    EXPECT_EQ(s.delay_ms(), 0u);
}

TEST(SynapseTest, PropagateExcitatory) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              0.5, SynapseType::EXCITATORY);

    // Excitatory: signal * weight * (+1)
    EXPECT_DOUBLE_EQ(s.propagate(1.0), 0.5);
    EXPECT_DOUBLE_EQ(s.propagate(2.0), 1.0);
    EXPECT_DOUBLE_EQ(s.propagate(0.0), 0.0);
    EXPECT_DOUBLE_EQ(s.propagate(-1.0), -0.5);
}

TEST(SynapseTest, PropagateInhibitory) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              0.5, SynapseType::INHIBITORY);

    // Inhibitory: signal * weight * (-1)
    EXPECT_DOUBLE_EQ(s.propagate(1.0), -0.5);
    EXPECT_DOUBLE_EQ(s.propagate(2.0), -1.0);
}

TEST(SynapseTest, Strengthen) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              0.5, SynapseType::EXCITATORY);

    s.strengthen(0.1); // delta * plasticity(1.0) = 0.1
    EXPECT_DOUBLE_EQ(s.weight(), 0.6);

    s.strengthen(0.2);
    EXPECT_DOUBLE_EQ(s.weight(), 0.8);
}

TEST(SynapseTest, StrengthenWithPlasticity) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              0.5, SynapseType::EXCITATORY);

    s.set_plasticity(0.5);
    s.strengthen(0.2); // 0.2 * 0.5 = 0.1 added
    EXPECT_DOUBLE_EQ(s.weight(), 0.6);
}

TEST(SynapseTest, Weaken) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              0.5, SynapseType::EXCITATORY);

    s.weaken(0.1);
    EXPECT_DOUBLE_EQ(s.weight(), 0.4);

    // Should not go below 0 for positive weights
    s.weaken(10.0);
    EXPECT_DOUBLE_EQ(s.weight(), 0.0);
}

TEST(SynapseTest, SetProperties) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              1.0, SynapseType::EXCITATORY);

    s.set_weight(0.3);
    EXPECT_DOUBLE_EQ(s.weight(), 0.3);

    s.set_plasticity(2.5);
    EXPECT_DOUBLE_EQ(s.plasticity(), 2.5);

    // Plasticity should be clamped to [0, 10]
    s.set_plasticity(15.0);
    EXPECT_DOUBLE_EQ(s.plasticity(), 10.0);
    s.set_plasticity(-1.0);
    EXPECT_DOUBLE_EQ(s.plasticity(), 0.0);

    s.set_delay(100);
    EXPECT_EQ(s.delay_ms(), 100u);

    s.set_type(SynapseType::INHIBITORY);
    EXPECT_EQ(s.type(), SynapseType::INHIBITORY);
}

TEST(SynapseTest, Metadata) {
    Synapse s(SynapseId::generate(), NeuronId::generate(), NeuronId::generate(),
              1.0, SynapseType::EXCITATORY);

    EXPECT_TRUE(s.metadata().is_object());
    EXPECT_TRUE(s.metadata().empty());

    s.set_metadata({{"last_activated_cycle", 42}, {"tag", "important"}});
    EXPECT_EQ(s.metadata()["last_activated_cycle"], 42);
    EXPECT_EQ(s.metadata()["tag"], "important");
}

TEST(SynapseTest, JsonRoundTrip) {
    auto sid = SynapseId::generate();
    auto src = NeuronId::generate();
    auto tgt = NeuronId::generate();

    Synapse original(sid, src, tgt, 0.42, SynapseType::INHIBITORY);
    original.set_plasticity(1.5);
    original.set_delay(50);
    original.set_metadata({{"layer", 2}});

    auto j = original.to_json();
    Synapse restored = Synapse::from_json(j);

    EXPECT_EQ(restored.id(), sid);
    EXPECT_EQ(restored.source_id(), src);
    EXPECT_EQ(restored.target_id(), tgt);
    EXPECT_DOUBLE_EQ(restored.weight(), 0.42);
    EXPECT_EQ(restored.type(), SynapseType::INHIBITORY);
    EXPECT_DOUBLE_EQ(restored.plasticity(), 1.5);
    EXPECT_EQ(restored.delay_ms(), 50u);
    EXPECT_EQ(restored.metadata()["layer"], 2);
}

TEST(SynapseTest, TypeStrings) {
    EXPECT_EQ(synapse_type_str(SynapseType::EXCITATORY), "excitatory");
    EXPECT_EQ(synapse_type_str(SynapseType::INHIBITORY), "inhibitory");

    EXPECT_EQ(parse_synapse_type("excitatory"), SynapseType::EXCITATORY);
    EXPECT_EQ(parse_synapse_type("inhibitory"), SynapseType::INHIBITORY);
    EXPECT_EQ(parse_synapse_type("unknown"), SynapseType::EXCITATORY); // default
}
