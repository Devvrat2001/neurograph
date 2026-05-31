#include <gtest/gtest.h>
#include "neurograph/core/neuron.hpp"

using namespace neurograph::core;

TEST(NeuronTest, ConstructAndIdentity) {
    auto id = NeuronId::generate();
    Neuron n(id, "test_neuron");

    EXPECT_EQ(n.id(), id);
    EXPECT_EQ(n.label(), "test_neuron");
    EXPECT_EQ(n.state(), NeuronState::IDLE);
    EXPECT_DOUBLE_EQ(n.activation_level(), 0.0);
    EXPECT_DOUBLE_EQ(n.threshold(), 0.5);
}

TEST(NeuronTest, ActivateSigmoid) {
    Neuron n(NeuronId::generate(), "sigmoid_neuron");
    n.set_activation_type(ActivationType::SIGMOID);
    n.set_threshold(0.5);

    // Sigmoid(0) = 0.5 -> exactly at threshold -> ACTIVE
    double result = n.activate(0.0);
    EXPECT_NEAR(result, 0.5, 1e-6);
    EXPECT_EQ(n.state(), NeuronState::ACTIVE);

    // Sigmoid(-5) ~ 0.0067 -> below threshold -> IDLE
    result = n.activate(-5.0);
    EXPECT_LT(result, 0.01);
    EXPECT_EQ(n.state(), NeuronState::IDLE);

    // Sigmoid(5) ~ 0.9933 -> above threshold -> ACTIVE
    result = n.activate(5.0);
    EXPECT_GT(result, 0.99);
    EXPECT_EQ(n.state(), NeuronState::ACTIVE);
}

TEST(NeuronTest, ActivateReLU) {
    Neuron n(NeuronId::generate(), "relu_neuron");
    n.set_activation_type(ActivationType::RELU);
    n.set_threshold(0.5);

    EXPECT_DOUBLE_EQ(n.activate(-1.0), 0.0);
    EXPECT_EQ(n.state(), NeuronState::IDLE);

    EXPECT_DOUBLE_EQ(n.activate(2.0), 2.0);
    EXPECT_EQ(n.state(), NeuronState::ACTIVE);
}

TEST(NeuronTest, ActivateDisabledNeuron) {
    Neuron n(NeuronId::generate(), "disabled");
    n.set_state(NeuronState::DISABLED);

    double result = n.activate(10.0);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(NeuronTest, ActivationHistory) {
    Neuron n(NeuronId::generate(), "history_test");
    n.set_activation_type(ActivationType::SIGMOID);

    for (int i = 0; i < 10; ++i) {
        n.activate(static_cast<double>(i) - 5.0);
    }

    EXPECT_EQ(n.activation_history().size(), 10);

    // Fill beyond history size
    for (size_t i = 0; i < Neuron::ACTIVATION_HISTORY_SIZE + 10; ++i) {
        n.activate(0.5);
    }
    EXPECT_EQ(n.activation_history().size(), Neuron::ACTIVATION_HISTORY_SIZE);
}

TEST(NeuronTest, Replicate) {
    Neuron original(NeuronId::generate(), "original");
    original.set_threshold(0.7);
    original.set_position(Vec3{1.0, 2.0, 3.0});
    original.set_activation_type(ActivationType::TANH);
    original.set_metadata({{"type", "sensory"}});

    Neuron clone = original.replicate(0.0); // no mutation

    EXPECT_NE(clone.id(), original.id());
    EXPECT_EQ(clone.label(), "original_clone");
    EXPECT_DOUBLE_EQ(clone.threshold(), 0.7);
    EXPECT_EQ(clone.activation_type(), ActivationType::TANH);
    EXPECT_EQ(clone.state(), NeuronState::IDLE);
    EXPECT_DOUBLE_EQ(clone.activation_level(), 0.0);
    EXPECT_TRUE(clone.outgoing_synapses().empty());
    EXPECT_TRUE(clone.incoming_synapses().empty());
    EXPECT_TRUE(clone.position().has_value());
    EXPECT_EQ(clone.metadata()["type"], "sensory");
}

TEST(NeuronTest, ReplicateWithMutation) {
    Neuron original(NeuronId::generate(), "original");
    original.set_threshold(0.5);

    // High mutation rate to ensure threshold changes
    bool threshold_changed = false;
    for (int i = 0; i < 100; ++i) {
        Neuron clone = original.replicate(0.5);
        if (std::abs(clone.threshold() - 0.5) > 0.01) {
            threshold_changed = true;
            break;
        }
    }
    EXPECT_TRUE(threshold_changed);
}

TEST(NeuronTest, SynapseTracking) {
    Neuron n(NeuronId::generate(), "tracked");
    auto s1 = SynapseId::generate();
    auto s2 = SynapseId::generate();

    n.add_outgoing(s1);
    n.add_outgoing(s2);
    n.add_incoming(s1);

    EXPECT_EQ(n.outgoing_synapses().size(), 2);
    EXPECT_EQ(n.incoming_synapses().size(), 1);

    n.remove_outgoing(s1);
    EXPECT_EQ(n.outgoing_synapses().size(), 1);
    EXPECT_EQ(n.outgoing_synapses()[0], s2);
}

TEST(NeuronTest, ConnectionAffinity) {
    Neuron a(NeuronId::generate(), "type_a");
    Neuron b(NeuronId::generate(), "type_b");

    a.set_position(Vec3{0, 0, 0});
    b.set_position(Vec3{1, 0, 0});

    double score = a.evaluate_connection_affinity(b);
    EXPECT_GE(score, 0.0);
    EXPECT_LE(score, 1.0);
}

TEST(NeuronTest, ConnectionAffinityFarApart) {
    Neuron a(NeuronId::generate(), "near");
    Neuron b(NeuronId::generate(), "near");
    Neuron c(NeuronId::generate(), "far");

    a.set_position(Vec3{0, 0, 0});
    b.set_position(Vec3{0.5, 0, 0});
    c.set_position(Vec3{9, 9, 9});

    double score_near = a.evaluate_connection_affinity(b);
    double score_far = a.evaluate_connection_affinity(c);

    // Closer neurons should have higher spatial affinity
    EXPECT_GT(score_near, score_far);
}

TEST(NeuronTest, JsonRoundTrip) {
    Neuron original(NeuronId::generate(), "json_test");
    original.set_threshold(0.42);
    original.set_activation_type(ActivationType::TANH);
    original.set_position(Vec3{1.5, -2.5, 3.0});
    original.set_metadata({{"layer", 3}});
    original.add_outgoing(SynapseId::generate());

    auto j = original.to_json();
    Neuron restored = Neuron::from_json(j);

    EXPECT_EQ(restored.id(), original.id());
    EXPECT_EQ(restored.label(), original.label());
    EXPECT_DOUBLE_EQ(restored.threshold(), 0.42);
    EXPECT_EQ(restored.activation_type(), ActivationType::TANH);
    EXPECT_TRUE(restored.position().has_value());
    EXPECT_NEAR(restored.position()->x, 1.5, 1e-9);
    EXPECT_EQ(restored.metadata()["layer"], 3);
    EXPECT_EQ(restored.outgoing_synapses().size(), 1);
}

TEST(NeuronTest, StateStrings) {
    EXPECT_EQ(neuron_state_str(NeuronState::IDLE), "idle");
    EXPECT_EQ(neuron_state_str(NeuronState::ACTIVE), "active");
    EXPECT_EQ(neuron_state_str(NeuronState::DISABLED), "disabled");

    EXPECT_EQ(parse_neuron_state("active"), NeuronState::ACTIVE);
    EXPECT_EQ(parse_neuron_state("refractory"), NeuronState::REFRACTORY);
    EXPECT_EQ(parse_neuron_state("unknown_value"), NeuronState::IDLE);
}
