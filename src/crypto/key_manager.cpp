#include "neurograph/crypto/key_manager.hpp"

#include <fstream>
#include <algorithm>

namespace neurograph::crypto {

KeyManager::KeyManager(const Ed25519Provider& provider)
    : provider_(provider) {}

KeyPair KeyManager::generate_master_key() {
    auto kp = provider_.generate_keypair();
    master_keypair_ = kp;
    authorize_master(kp.public_key);
    return kp;
}

util::Result<KeyPair> KeyManager::load_keypair(
    const std::string& public_key_path,
    const std::string& secret_key_path) const
{
    KeyPair kp;

    // Read public key
    std::ifstream pub_file(public_key_path, std::ios::binary);
    if (!pub_file) {
        return util::Error("Failed to open public key file: " + public_key_path);
    }
    pub_file.read(reinterpret_cast<char*>(kp.public_key.data()),
                  ED25519_PUBLIC_KEY_SIZE);
    if (pub_file.gcount() != static_cast<std::streamsize>(ED25519_PUBLIC_KEY_SIZE)) {
        return util::Error("Invalid public key file size");
    }

    // Read secret key
    std::ifstream sec_file(secret_key_path, std::ios::binary);
    if (!sec_file) {
        return util::Error("Failed to open secret key file: " + secret_key_path);
    }
    sec_file.read(reinterpret_cast<char*>(kp.secret_key.data()),
                  ED25519_SECRET_KEY_SIZE);
    if (sec_file.gcount() != static_cast<std::streamsize>(ED25519_SECRET_KEY_SIZE)) {
        return util::Error("Invalid secret key file size");
    }

    return kp;
}

util::Result<void> KeyManager::save_keypair(
    const KeyPair& kp,
    const std::string& public_key_path,
    const std::string& secret_key_path) const
{
    std::ofstream pub_file(public_key_path, std::ios::binary);
    if (!pub_file) {
        return util::Error("Failed to create public key file: " + public_key_path);
    }
    pub_file.write(reinterpret_cast<const char*>(kp.public_key.data()),
                   ED25519_PUBLIC_KEY_SIZE);

    std::ofstream sec_file(secret_key_path, std::ios::binary);
    if (!sec_file) {
        return util::Error("Failed to create secret key file: " + secret_key_path);
    }
    sec_file.write(reinterpret_cast<const char*>(kp.secret_key.data()),
                   ED25519_SECRET_KEY_SIZE);

    return util::Result<void>::success();
}

void KeyManager::authorize_master(const PublicKey& public_key) {
    if (!is_authorized_master(public_key)) {
        authorized_masters_.push_back(public_key);
    }
}

void KeyManager::revoke_master(const PublicKey& public_key) {
    authorized_masters_.erase(
        std::remove(authorized_masters_.begin(), authorized_masters_.end(), public_key),
        authorized_masters_.end());
}

bool KeyManager::is_authorized_master(const PublicKey& public_key) const {
    return std::find(authorized_masters_.begin(), authorized_masters_.end(),
                     public_key) != authorized_masters_.end();
}

void KeyManager::set_master_keypair(KeyPair kp) {
    authorize_master(kp.public_key);
    master_keypair_ = std::move(kp);
}

} // namespace neurograph::crypto
