#pragma once

#include "neurograph/command/command.hpp"
#include "neurograph/crypto/ed25519_provider.hpp"

namespace neurograph::command {

/// Signs commands with an Ed25519 secret key.
/// Only the master user should have access to the signing key.
class CommandSigner {
public:
    CommandSigner(const crypto::Ed25519Provider& provider,
                  crypto::KeyPair keypair);

    /// Sign a command in-place. Sets the signature and issuer public key.
    void sign(Command& cmd) const;

    /// Create a new command and sign it in one step.
    Command create_signed(CommandType type, nlohmann::json payload,
                          PropagationMode mode = PropagationMode::BFS) const;

    /// Create a signed command targeted at a specific neuron.
    Command create_signed_targeted(CommandType type, nlohmann::json payload,
                                    core::NeuronId target) const;

    const crypto::PublicKey& public_key() const noexcept {
        return keypair_.public_key;
    }

private:
    const crypto::Ed25519Provider& provider_;
    crypto::KeyPair keypair_;
};

} // namespace neurograph::command
