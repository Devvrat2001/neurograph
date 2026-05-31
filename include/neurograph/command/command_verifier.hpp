#pragma once

#include "neurograph/command/command.hpp"
#include "neurograph/crypto/ed25519_provider.hpp"
#include "neurograph/crypto/key_manager.hpp"

namespace neurograph::command {

/// Verification result with reason for failure.
struct VerificationResult {
    bool valid = false;
    std::string reason;

    static VerificationResult success() { return {true, ""}; }
    static VerificationResult failure(std::string reason) {
        return {false, std::move(reason)};
    }

    explicit operator bool() const noexcept { return valid; }
};

/// Verifies command signatures and authorization.
/// Checks: (1) command is signed, (2) signature is valid, (3) issuer is authorized master.
class CommandVerifier {
public:
    CommandVerifier(const crypto::Ed25519Provider& provider,
                    const crypto::KeyManager& key_manager);

    /// Full verification: signature + authorization + expiry check.
    VerificationResult verify(const Command& cmd) const;

    /// Signature-only check (no authorization check).
    bool verify_signature(const Command& cmd) const;

private:
    const crypto::Ed25519Provider& provider_;
    const crypto::KeyManager& key_manager_;
};

} // namespace neurograph::command
