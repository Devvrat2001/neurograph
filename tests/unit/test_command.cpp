#include <gtest/gtest.h>
#include "neurograph/crypto/ed25519_provider.hpp"
#include "neurograph/crypto/key_manager.hpp"
#include "neurograph/command/command.hpp"
#include "neurograph/command/command_signer.hpp"
#include "neurograph/command/command_verifier.hpp"
#include "neurograph/command/command_executor.hpp"
#include "neurograph/command/command_propagator.hpp"
#include "neurograph/core/network.hpp"

using namespace neurograph;

// ============================================================
// Ed25519Provider Tests
// ============================================================

class CryptoTest : public ::testing::Test {
protected:
    crypto::Ed25519Provider provider;
};

TEST_F(CryptoTest, GenerateKeyPair) {
    auto kp = provider.generate_keypair();
    // Keys should be non-zero
    bool all_zero = true;
    for (auto b : kp.public_key) { if (b != 0) { all_zero = false; break; } }
    EXPECT_FALSE(all_zero);
}

TEST_F(CryptoTest, SignAndVerify) {
    auto kp = provider.generate_keypair();
    std::vector<uint8_t> message = {1, 2, 3, 4, 5, 6, 7, 8};

    auto sig = provider.sign(message, kp.secret_key);
    EXPECT_TRUE(provider.verify(message, sig, kp.public_key));
}

TEST_F(CryptoTest, VerifyFailsWithWrongKey) {
    auto kp1 = provider.generate_keypair();
    auto kp2 = provider.generate_keypair();
    std::vector<uint8_t> message = {10, 20, 30};

    auto sig = provider.sign(message, kp1.secret_key);
    // Verification with wrong public key should fail
    EXPECT_FALSE(provider.verify(message, sig, kp2.public_key));
}

TEST_F(CryptoTest, VerifyFailsWithTamperedMessage) {
    auto kp = provider.generate_keypair();
    std::vector<uint8_t> message = {1, 2, 3};

    auto sig = provider.sign(message, kp.secret_key);

    // Tamper with message
    std::vector<uint8_t> tampered = {1, 2, 4};
    EXPECT_FALSE(provider.verify(tampered, sig, kp.public_key));
}

TEST_F(CryptoTest, DeterministicKeyFromSeed) {
    std::array<uint8_t, crypto::ED25519_SEED_SIZE> seed{};
    seed.fill(42);

    auto kp1 = provider.keypair_from_seed(seed);
    auto kp2 = provider.keypair_from_seed(seed);

    EXPECT_EQ(kp1.public_key, kp2.public_key);
    EXPECT_EQ(kp1.secret_key, kp2.secret_key);
}

// ============================================================
// KeyManager Tests
// ============================================================

TEST_F(CryptoTest, KeyManagerAuthorization) {
    crypto::KeyManager km(provider);

    auto kp = km.generate_master_key();
    EXPECT_TRUE(km.is_authorized_master(kp.public_key));
    EXPECT_TRUE(km.master_keypair().has_value());

    // Another key should not be authorized
    auto other = provider.generate_keypair();
    EXPECT_FALSE(km.is_authorized_master(other.public_key));

    // Authorize it
    km.authorize_master(other.public_key);
    EXPECT_TRUE(km.is_authorized_master(other.public_key));
    EXPECT_EQ(km.authorized_masters().size(), 2);

    // Revoke
    km.revoke_master(other.public_key);
    EXPECT_FALSE(km.is_authorized_master(other.public_key));
}

// ============================================================
// Command Tests
// ============================================================

class CommandTest : public ::testing::Test {
protected:
    crypto::Ed25519Provider provider;
    crypto::KeyManager key_manager{provider};
    crypto::KeyPair master_kp;

    void SetUp() override {
        master_kp = key_manager.generate_master_key();
    }
};

TEST_F(CommandTest, CreateCommand) {
    command::Command cmd(command::CommandType::ACTIVATE,
                         {{"input", 1.5}},
                         command::PropagationMode::BFS);

    EXPECT_EQ(cmd.type(), command::CommandType::ACTIVATE);
    EXPECT_EQ(cmd.propagation_mode(), command::PropagationMode::BFS);
    EXPECT_FALSE(cmd.is_signed());
    EXPECT_FALSE(cmd.is_expired());
    EXPECT_EQ(cmd.ttl(), 256u);
    EXPECT_DOUBLE_EQ(cmd.payload()["input"].get<double>(), 1.5);
}

TEST_F(CommandTest, SignAndVerify) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);

    auto cmd = signer.create_signed(
        command::CommandType::ACTIVATE,
        {{"input", 2.0}},
        command::PropagationMode::BROADCAST);

    EXPECT_TRUE(cmd.is_signed());

    auto result = verifier.verify(cmd);
    EXPECT_TRUE(result.valid);
}

TEST_F(CommandTest, VerifyFailsUnsigned) {
    command::CommandVerifier verifier(provider, key_manager);

    command::Command cmd(command::CommandType::ACTIVATE, {});
    auto result = verifier.verify(cmd);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, "Command is not signed");
}

TEST_F(CommandTest, VerifyFailsUnauthorized) {
    // Sign with a key that's not authorized
    auto rogue_kp = provider.generate_keypair();
    command::CommandSigner rogue_signer(provider, rogue_kp);
    command::CommandVerifier verifier(provider, key_manager);

    auto cmd = rogue_signer.create_signed(command::CommandType::SHUTDOWN, {});
    auto result = verifier.verify(cmd);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, "Issuer is not an authorized master");
}

TEST_F(CommandTest, VerifyFailsTampered) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);

    auto cmd = signer.create_signed(command::CommandType::ACTIVATE,
                                     {{"input", 1.0}});

    // Tamper with payload after signing
    cmd.payload()["input"] = 999.0;

    auto result = verifier.verify(cmd);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, "Invalid signature");
}

TEST_F(CommandTest, TargetedCommand) {
    command::CommandSigner signer(provider, master_kp);

    auto target = core::NeuronId::generate();
    auto cmd = signer.create_signed_targeted(
        command::CommandType::READ_NEURON, {}, target);

    EXPECT_EQ(cmd.propagation_mode(), command::PropagationMode::TARGETED);
    EXPECT_TRUE(cmd.target_neuron().has_value());
    EXPECT_EQ(cmd.target_neuron()->value(), target.value());
}

TEST_F(CommandTest, CommandJsonRoundTrip) {
    command::Command cmd(command::CommandType::REPLICATE,
                         {{"mutation_rate", 0.05}},
                         command::PropagationMode::DFS);
    cmd.set_ttl(100);

    auto j = cmd.to_json();
    EXPECT_EQ(j["type"], static_cast<int>(command::CommandType::REPLICATE));
    EXPECT_EQ(j["type_name"], "REPLICATE");
    EXPECT_EQ(j["ttl"], 100);
}

// ============================================================
// CommandExecutor Tests
// ============================================================

class ExecutorTest : public ::testing::Test {
protected:
    core::Network net;
    core::NeuronId id_a, id_b;

    void SetUp() override {
        core::Neuron a(core::NeuronId::generate(), "alpha");
        core::Neuron b(core::NeuronId::generate(), "beta");
        a.set_position(core::Vec3{0, 0, 0});
        b.set_position(core::Vec3{1, 0, 0});

        id_a = a.id();
        id_b = b.id();

        net.add_neuron(std::move(a));
        net.add_neuron(std::move(b));
        net.connect(id_a, id_b, 0.5);
    }
};

TEST_F(ExecutorTest, CreateNeuron) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::CREATE_NEURON,
                         {{"label", "gamma"}, {"threshold", 0.3}});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(net.neuron_count(), 3);
}

TEST_F(ExecutorTest, ReadNeuron) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::READ_NEURON, {});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.data["label"], "alpha");
}

TEST_F(ExecutorTest, UpdateNeuron) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::UPDATE_NEURON,
                         {{"label", "alpha_v2"}, {"threshold", 0.8}});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);

    auto* n = net.get_neuron(id_a);
    EXPECT_EQ(n->label(), "alpha_v2");
    EXPECT_DOUBLE_EQ(n->threshold(), 0.8);
}

TEST_F(ExecutorTest, DeleteNeuron) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::DELETE_NEURON, {});

    auto result = exec.execute(cmd, id_b);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(net.neuron_count(), 1);
    EXPECT_EQ(net.synapse_count(), 0); // connected synapse should be gone
}

TEST_F(ExecutorTest, Replicate) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::REPLICATE,
                         {{"mutation_rate", 0.0}, {"inherit_connections", true}});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(net.neuron_count(), 3);
    EXPECT_TRUE(result.data.contains("clone_id"));
}

TEST_F(ExecutorTest, Connect) {
    // Remove existing connection first, then reconnect
    net.disconnect(id_a, id_b);
    EXPECT_EQ(net.synapse_count(), 0);

    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::CONNECT,
                         {{"target_id", id_b.value()}, {"weight", 0.9}});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(net.synapse_count(), 1);
}

TEST_F(ExecutorTest, Activate) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::ACTIVATE, {{"input", 5.0}});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.data["output"].get<double>(), 0.0);
}

TEST_F(ExecutorTest, Shutdown) {
    command::CommandExecutor exec(net);
    command::Command cmd(command::CommandType::SHUTDOWN, {});

    auto result = exec.execute(cmd, id_a);
    EXPECT_TRUE(result.success);

    auto* n = net.get_neuron(id_a);
    EXPECT_EQ(n->state(), core::NeuronState::DISABLED);
}

// ============================================================
// CommandPropagator Tests
// ============================================================

class PropagatorTest : public ::testing::Test {
protected:
    crypto::Ed25519Provider provider;
    crypto::KeyManager key_manager{provider};
    crypto::KeyPair master_kp;
    core::Network net;
    core::NeuronId id_a, id_b, id_c;

    void SetUp() override {
        master_kp = key_manager.generate_master_key();

        core::Neuron a(core::NeuronId::generate(), "a");
        core::Neuron b(core::NeuronId::generate(), "b");
        core::Neuron c(core::NeuronId::generate(), "c");

        id_a = a.id(); id_b = b.id(); id_c = c.id();

        net.add_neuron(std::move(a));
        net.add_neuron(std::move(b));
        net.add_neuron(std::move(c));

        net.connect(id_a, id_b, 1.0);
        net.connect(id_b, id_c, 1.0);
    }
};

TEST_F(PropagatorTest, BFSPropagation) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = signer.create_signed(
        command::CommandType::ACTIVATE,
        {{"input", 1.0}},
        command::PropagationMode::BFS);

    auto result = propagator.propagate(cmd, id_a);

    EXPECT_TRUE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 3); // a -> b -> c
    EXPECT_EQ(result.neurons_succeeded, 3);
    EXPECT_EQ(result.neurons_failed, 0);
}

TEST_F(PropagatorTest, DFSPropagation) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = signer.create_signed(
        command::CommandType::ACTIVATE,
        {{"input", 0.5}},
        command::PropagationMode::DFS);

    auto result = propagator.propagate(cmd, id_a);

    EXPECT_TRUE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 3);
    EXPECT_EQ(result.neurons_succeeded, 3);
}

TEST_F(PropagatorTest, TargetedPropagation) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = signer.create_signed_targeted(
        command::CommandType::READ_NEURON, {}, id_b);

    auto result = propagator.propagate(cmd);

    EXPECT_TRUE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 1);
    EXPECT_EQ(result.neurons_succeeded, 1);
    EXPECT_EQ(result.results[0].second.data["label"], "b");
}

TEST_F(PropagatorTest, BroadcastPropagation) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = signer.create_signed(
        command::CommandType::ACTIVATE,
        {{"input", 2.0}},
        command::PropagationMode::BROADCAST);

    auto result = propagator.propagate(cmd);

    EXPECT_TRUE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 3);
    EXPECT_EQ(result.neurons_succeeded, 3);
}

TEST_F(PropagatorTest, RejectsUnsignedCommand) {
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    command::Command cmd(command::CommandType::SHUTDOWN, {});
    auto result = propagator.propagate(cmd, id_a);

    EXPECT_FALSE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 0);
}

TEST_F(PropagatorTest, RejectsUnauthorizedCommand) {
    auto rogue_kp = provider.generate_keypair();
    command::CommandSigner rogue_signer(provider, rogue_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = rogue_signer.create_signed(command::CommandType::SHUTDOWN, {});
    auto result = propagator.propagate(cmd, id_a);

    EXPECT_FALSE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 0);
}

TEST_F(PropagatorTest, TTLLimitsTraversal) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = signer.create_signed(
        command::CommandType::ACTIVATE,
        {{"input", 1.0}},
        command::PropagationMode::BFS);
    cmd.set_ttl(1); // Only 1 hop

    // Need to re-sign after modifying TTL since it's part of the signed data
    // Actually TTL is set before signing in create_signed, but we modified it after.
    // For this test, let's create with the right TTL from the start.
    command::Command cmd2(command::CommandType::ACTIVATE,
                          {{"input", 1.0}},
                          command::PropagationMode::BFS);
    cmd2.set_ttl(1);
    signer.sign(cmd2);

    auto result = propagator.propagate(cmd2, id_a);

    EXPECT_TRUE(result.authenticated);
    EXPECT_EQ(result.neurons_visited, 1); // only 1 due to TTL
}

TEST_F(PropagatorTest, ShutdownPropagatesThroughNetwork) {
    command::CommandSigner signer(provider, master_kp);
    command::CommandVerifier verifier(provider, key_manager);
    command::CommandExecutor executor(net);
    command::CommandPropagator propagator(net, verifier, executor);

    auto cmd = signer.create_signed(
        command::CommandType::SHUTDOWN, {},
        command::PropagationMode::BROADCAST);

    auto result = propagator.propagate(cmd);

    EXPECT_TRUE(result.authenticated);
    EXPECT_EQ(result.neurons_succeeded, 3);

    // All neurons should now be disabled
    EXPECT_EQ(net.get_neuron(id_a)->state(), core::NeuronState::DISABLED);
    EXPECT_EQ(net.get_neuron(id_b)->state(), core::NeuronState::DISABLED);
    EXPECT_EQ(net.get_neuron(id_c)->state(), core::NeuronState::DISABLED);
}
