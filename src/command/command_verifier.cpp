#include "neurograph/command/command_verifier.hpp"

namespace neurograph::command {

CommandVerifier::CommandVerifier(const crypto::Ed25519Provider& provider,
                                 const crypto::KeyManager& key_manager)
    : provider_(provider), key_manager_(key_manager) {}

VerificationResult CommandVerifier::verify(const Command& cmd) const {
    // 1. Check if command is signed
    if (!cmd.is_signed()) {
        return VerificationResult::failure("Command is not signed");
    }

    // 2. Check if command has expired
    if (cmd.is_expired()) {
        return VerificationResult::failure("Command has expired");
    }

    // 3. Verify the cryptographic signature
    if (!verify_signature(cmd)) {
        return VerificationResult::failure("Invalid signature");
    }

    // 4. Check if the issuer is an authorized master
    if (!key_manager_.is_authorized_master(cmd.issuer_public_key())) {
        return VerificationResult::failure("Issuer is not an authorized master");
    }

    return VerificationResult::success();
}

bool CommandVerifier::verify_signature(const Command& cmd) const {
    if (!cmd.is_signed()) return false;

    auto message = cmd.serialize_for_signing();
    return provider_.verify(message, cmd.signature(), cmd.issuer_public_key());
}

} // namespace neurograph::command
