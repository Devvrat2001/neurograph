#include "neurograph/command/command_signer.hpp"

namespace neurograph::command {

CommandSigner::CommandSigner(const crypto::Ed25519Provider& provider,
                             crypto::KeyPair keypair)
    : provider_(provider), keypair_(std::move(keypair)) {}

void CommandSigner::sign(Command& cmd) const {
    auto message = cmd.serialize_for_signing();
    auto sig = provider_.sign(message, keypair_.secret_key);
    cmd.set_signature(sig, keypair_.public_key);
}

Command CommandSigner::create_signed(CommandType type, nlohmann::json payload,
                                      PropagationMode mode) const {
    Command cmd(type, std::move(payload), mode);
    sign(cmd);
    return cmd;
}

Command CommandSigner::create_signed_targeted(CommandType type, nlohmann::json payload,
                                               core::NeuronId target) const {
    Command cmd(type, std::move(payload), PropagationMode::TARGETED);
    cmd.set_target_neuron(std::move(target));
    sign(cmd);
    return cmd;
}

} // namespace neurograph::command
