#include "neurograph/crypto/ed25519_provider.hpp"

#include <sodium.h>
#include <stdexcept>

namespace neurograph::crypto {

Ed25519Provider::Ed25519Provider() {
    if (sodium_init() < 0) {
        throw std::runtime_error("Failed to initialize libsodium");
    }
}

KeyPair Ed25519Provider::generate_keypair() const {
    KeyPair kp;
    crypto_sign_keypair(kp.public_key.data(), kp.secret_key.data());
    return kp;
}

KeyPair Ed25519Provider::keypair_from_seed(
    const std::array<uint8_t, ED25519_SEED_SIZE>& seed) const
{
    KeyPair kp;
    crypto_sign_seed_keypair(kp.public_key.data(), kp.secret_key.data(), seed.data());
    return kp;
}

Signature Ed25519Provider::sign(const std::vector<uint8_t>& message,
                                 const SecretKey& secret_key) const
{
    Signature sig;
    unsigned long long sig_len = 0;
    crypto_sign_detached(sig.data(), &sig_len,
                         message.data(), message.size(),
                         secret_key.data());
    return sig;
}

bool Ed25519Provider::verify(const std::vector<uint8_t>& message,
                              const Signature& signature,
                              const PublicKey& public_key) const
{
    return crypto_sign_verify_detached(
        signature.data(),
        message.data(), message.size(),
        public_key.data()) == 0;
}

} // namespace neurograph::crypto
