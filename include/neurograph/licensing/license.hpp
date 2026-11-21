#pragma once

// P1-11 — License key system (cryptographic core).
//
// Offline-verifiable per-seat licenses. The user pastes a key string;
// the running app verifies an Ed25519 signature against an embedded
// public key, decodes the payload (tier, expiry, holder, features), and
// gates functionality accordingly. No network call is required for
// verification — air-gapped customers must work.
//
// What's in this header:
//   - License + LicensePayload value types and Tier enum.
//   - issue_key() / verify_key_with() — pure crypto operations driven
//     entirely by the caller-supplied keypair, used by the ops keygen
//     tool and by the test suite.
//   - verify_key() — wraps verify_key_with() using the public key
//     embedded in the build (a placeholder until ops mints the real
//     production keypair).
//   - has_feature() / is_active() — runtime gating helpers.
//
// Persistence + trial layer (lower in this header):
//   - activate() / current_license() / persist_license() / load_persisted_license()
//     — round-trip a verified key through %APPDATA%\NeuroGraph\license.dat,
//     DPAPI-encrypted on Windows.
//   - TrialState + load_or_init_trial() / trial_active() — 30-day free
//     trial with anti-rollback via a high-watermark wall-clock anchor.
//   - is_revoked() — best-effort online revocation check (returns false
//     on any network error so air-gapped customers stay unblocked).
//
// Key format: RFC 4648 base32 of `payload || ed25519_sig`, grouped in
// chunks of 4 chars separated by dashes for readability. Example:
//
//   AEAQ-CAJB-EAQC-AJB...  (~140 chars total — long but always offline-verifiable)
//
// Documented as a long key from day one rather than promising a 20-char
// format we'd have to walk back when the signature size requirements
// surface. JetBrains-style long keys are accepted in the SMB market.

#include "neurograph/util/result.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace neurograph::licensing {

enum class Tier : std::uint8_t {
    Trial        = 0,
    Personal     = 1,
    Professional = 2,
    Enterprise   = 3,
};

/// Caller-facing license object — what `verify_key` returns. The `key`
/// field stores the original input string for re-display in UIs.
struct License {
    std::string                            key;
    std::string                            holder;
    Tier                                   tier = Tier::Trial;
    std::chrono::system_clock::time_point  expires;
    std::vector<std::string>               features;   // e.g. "ONNX_EXPORT"
};

/// What the ops keygen tool serializes + signs to mint a key. Caller
/// constructs this, then passes it to `issue_key`.
struct LicensePayload {
    Tier                                   tier = Tier::Personal;
    std::chrono::system_clock::time_point  expires;
    std::string                            holder;
    std::vector<std::string>               features;
};

/// Ed25519 keypair, sized per libsodium's constants.
struct Keypair {
    std::vector<unsigned char> public_key;   // crypto_sign_PUBLICKEYBYTES
    std::vector<unsigned char> secret_key;   // crypto_sign_SECRETKEYBYTES
};

// ---------------------------------------------------------------------------
// Crypto primitives
// ---------------------------------------------------------------------------

/// Generate a fresh Ed25519 keypair via libsodium's CSPRNG. Tests and the
/// ops keygen tool use this; production keys live in an HSM and never
/// touch this code path.
Keypair generate_keypair();

/// Mint a license key by serializing `p` and signing it with `secret_key`.
/// Returns the formatted (dashed base32) key string.
util::Result<std::string>
issue_key(const LicensePayload& p,
          const std::vector<unsigned char>& secret_key);

/// Verify a key against the production public key embedded in the binary.
/// Equivalent to `verify_key_with(key, embedded_public_key())`.
util::Result<License> verify_key(const std::string& key);

/// Verify a key against a caller-supplied public key. Returns the parsed
/// License on success. On failure, the error message names the specific
/// failure mode (bad format, bad signature, expired, unknown tier) so the
/// UI can show a useful message.
util::Result<License>
verify_key_with(const std::string& key,
                const std::vector<unsigned char>& public_key);

/// The public key the production binary verifies against. In Debug builds
/// (and when no production key has been linked in) this returns a
/// placeholder zero-filled key — useful so the rest of the app can call
/// verify_key() during development without crashing, but `verify_key`
/// will always fail until a real keypair is patched in.
const std::vector<unsigned char>& embedded_public_key();

// ---------------------------------------------------------------------------
// Runtime gating
// ---------------------------------------------------------------------------

/// True iff the license has not yet expired (compares `expires` to
/// `system_clock::now()`).
bool is_active(const License& l);

/// True iff `l.features` contains `feature` (case-sensitive).
bool has_feature(const License& l, std::string_view feature);

// ---------------------------------------------------------------------------
// Persistence layer (P1-11 activation + trial)
// ---------------------------------------------------------------------------
//
// The state path defaults to a per-user location:
//   Windows: %APPDATA%\NeuroGraph\license.dat   (and trial.dat)
//   POSIX:   $XDG_CONFIG_HOME/neurograph/license.dat (or ~/.config/...)
//
// Tests pass an explicit path so they don't trample the developer's real
// AppData state.

/// Default location for the activated-license envelope.
std::filesystem::path default_state_path();

/// Default location for the trial-state envelope.
std::filesystem::path default_trial_path();

/// Encrypt + write a verified key to disk. On Windows this calls
/// `CryptProtectData` (CurrentUser scope) so the file is unreadable
/// from other Windows accounts and from the same account on a
/// different machine. On non-Windows we obfuscate with a build-time
/// pepper -- not real security, but Phase 1 only ships on Windows so
/// the non-Windows path exists for unit tests on POSIX CI.
util::Result<void>
persist_license(const License& l,
                const std::filesystem::path& state_path = default_state_path());

/// Decrypt + re-verify the persisted blob. The original key string is
/// stored inside the envelope and re-checked against
/// `embedded_public_key()`, so an attacker who somehow forges the
/// DPAPI ciphertext still has to forge an Ed25519 signature.
util::Result<License>
load_persisted_license(const std::filesystem::path& state_path = default_state_path());

/// Test-friendly variant: re-verify against an explicit public key
/// instead of the embedded one. Used by the unit tests so they can
/// round-trip persistence with a freshly-generated keypair without
/// depending on the production embedded key being a real (non-
/// placeholder) value.
util::Result<License>
load_persisted_license_with(const std::filesystem::path& state_path,
                            const std::vector<unsigned char>& public_key);

/// Verify, refuse-if-expired, persist, return. The user-facing entry
/// point invoked from the workbench license dialog.
util::Result<License>
activate(const std::string& key,
         const std::filesystem::path& state_path = default_state_path());

/// Convenience wrapper: load + return only on success. Re-verifies on
/// every call; a tampered file returns nullopt rather than throwing.
std::optional<License>
current_license(const std::filesystem::path& state_path = default_state_path());

/// Erase any persisted license (best-effort). Used by the "Sign out"
/// button in the license dialog and by tests for clean reset.
void clear_persisted_license(
    const std::filesystem::path& state_path = default_state_path());

// ---------------------------------------------------------------------------
// Trial state machine (anti-rollback)
// ---------------------------------------------------------------------------

/// Trial bookkeeping persisted between launches. `started_at` is the
/// first time the app ever ran on this machine; `high_watermark` is
/// the latest wall-clock the app has observed and is monotonically
/// non-decreasing across launches even if the user backdates the
/// system clock.
struct TrialState {
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point high_watermark;
};

/// Default trial length -- a calendar month. Centralised so tests and
/// the dialog UI can reference the same constant.
inline constexpr std::chrono::hours kDefaultTrialLength{24 * 30};

/// Load + advance the trial file. On first run this initializes both
/// timestamps to `now` and writes the file. On subsequent runs it
/// bumps `high_watermark` to max(prev, now) and rewrites the file.
util::Result<TrialState>
load_or_init_trial(const std::filesystem::path& trial_path = default_trial_path());

/// Test-friendly variant that takes an explicit "now" instead of
/// reading the system clock. Used by the unit tests to simulate clock
/// rollback / time-travel without touching the real wall clock.
util::Result<TrialState>
load_or_init_trial_at(const std::filesystem::path& trial_path,
                      std::chrono::system_clock::time_point now);

/// True iff the trial has not yet elapsed. Computed against the
/// high-watermark, so backdating the system clock cannot extend the
/// trial.
bool trial_active(const TrialState& t,
                  std::chrono::hours length = kDefaultTrialLength);

/// Whole days remaining in the trial, clamped to [0, length_in_days].
int trial_days_remaining(const TrialState& t,
                         std::chrono::hours length = kDefaultTrialLength);

/// Synthesize a License view for an active trial -- the workbench
/// renders trial mode through the same License code path as a real
/// activated key. Tier=Trial, no premium features, expires at
/// started_at + length.
License make_trial_license(const TrialState& t,
                           std::chrono::hours length = kDefaultTrialLength);

// ---------------------------------------------------------------------------
// Online revocation (best-effort)
// ---------------------------------------------------------------------------

/// Hash the key, POST to `endpoint`, parse `{"revoked": true|false}`.
/// Returns Ok(false) on any network error or non-200 -- offline and
/// air-gapped users must NEVER be locked out, so an unreachable
/// licensing service is a soft signal only. Returns Ok(true) only
/// when the server explicitly says revoked. Caller is expected to
/// treat the soft signal as "unknown -- allow".
util::Result<bool>
is_revoked(const std::string& key,
           const std::string& endpoint = "https://licensing.neurograph.ai/v1/check",
           std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

/// True when this build carries a real vendor public key. With the
/// zero-filled placeholder no key can ever verify, so the application
/// must not lock anyone out: it runs unlicensed with every feature on.
bool licensing_configured();

} // namespace neurograph::licensing
