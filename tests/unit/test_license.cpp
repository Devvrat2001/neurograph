// Unit tests for neurograph::licensing (P1-11 cryptographic core).
//
// Coverage:
//   - Keypair generation produces Ed25519-sized buffers.
//   - Issue → verify round-trip preserves every payload field.
//   - Tampered key bytes (anywhere in payload or signature) fail verify.
//   - Wrong public key fails verify (sanity check that we're actually
//     checking the signature, not just decoding).
//   - Malformed inputs are rejected without throwing.
//   - is_active correctly reports expiry, has_feature is case-sensitive.
//   - Production verify_key fails with the placeholder embedded key
//     (so we don't accidentally ship a "default-allow" build).

#include "neurograph/licensing/license.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

using neurograph::licensing::embedded_public_key;
using neurograph::licensing::generate_keypair;
using neurograph::licensing::has_feature;
using neurograph::licensing::is_active;
using neurograph::licensing::issue_key;
using neurograph::licensing::License;
using neurograph::licensing::LicensePayload;
using neurograph::licensing::Tier;
using neurograph::licensing::verify_key;
using neurograph::licensing::verify_key_with;

namespace {

LicensePayload make_payload(std::chrono::system_clock::time_point exp =
                                std::chrono::system_clock::now() +
                                std::chrono::hours(24 * 365),
                            Tier t = Tier::Professional) {
    LicensePayload p;
    p.tier     = t;
    p.expires  = exp;
    p.holder   = "Alice Example";
    p.features = {"ONNX_EXPORT", "MULTI_GPU"};
    return p;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Keypair shapes
// ---------------------------------------------------------------------------

TEST(License, GenerateKeypairProducesEd25519SizedBuffers) {
    auto kp = generate_keypair();
    EXPECT_EQ(kp.public_key.size(), 32u);   // crypto_sign_PUBLICKEYBYTES
    EXPECT_EQ(kp.secret_key.size(), 64u);   // crypto_sign_SECRETKEYBYTES
    // Two consecutive calls should produce different keys with very high
    // probability — quick smoke test that the CSPRNG is wired.
    auto kp2 = generate_keypair();
    EXPECT_NE(kp.public_key, kp2.public_key);
}

// ---------------------------------------------------------------------------
// 2. Round-trip preserves payload
// ---------------------------------------------------------------------------

TEST(License, IssueVerifyRoundTripPreservesAllFields) {
    auto kp = generate_keypair();
    auto p  = make_payload();

    auto key_r = issue_key(p, kp.secret_key);
    ASSERT_TRUE(key_r.ok()) << key_r.error().message;
    const std::string& key = key_r.value();
    ASSERT_FALSE(key.empty());

    auto lic_r = verify_key_with(key, kp.public_key);
    ASSERT_TRUE(lic_r.ok()) << lic_r.error().message;
    const License& lic = lic_r.value();

    EXPECT_EQ(lic.holder, p.holder);
    EXPECT_EQ(lic.tier,   p.tier);
    EXPECT_EQ(lic.features, p.features);

    // Expires should round-trip to the second (we serialize as int64
    // seconds-since-epoch, so any sub-second precision is dropped).
    using namespace std::chrono;
    const auto delta = duration_cast<seconds>(lic.expires - p.expires);
    EXPECT_LE(std::abs(delta.count()), 1);
    EXPECT_EQ(lic.key, key);  // re-displayed input string preserved
}

TEST(License, RoundTripsTrialTier) {
    auto kp = generate_keypair();
    auto p  = make_payload(std::chrono::system_clock::now() +
                            std::chrono::hours(24 * 30), Tier::Trial);
    auto key = issue_key(p, kp.secret_key).value();
    auto lic = verify_key_with(key, kp.public_key).value();
    EXPECT_EQ(lic.tier, Tier::Trial);
}

TEST(License, RoundTripsEnterpriseTierWithManyFeatures) {
    auto kp = generate_keypair();
    auto p  = make_payload();
    p.tier  = Tier::Enterprise;
    p.features = {"ONNX_EXPORT", "MULTI_GPU", "DISTRIBUTED",
                  "AUDIT_LOG", "CUSTOM_KERNELS"};
    auto key = issue_key(p, kp.secret_key).value();
    auto lic = verify_key_with(key, kp.public_key).value();
    EXPECT_EQ(lic.tier, Tier::Enterprise);
    EXPECT_EQ(lic.features.size(), 5u);
}

// ---------------------------------------------------------------------------
// 3. Tampering — every byte position should be detected
// ---------------------------------------------------------------------------

TEST(License, FlippedPayloadByteFailsVerification) {
    auto kp  = generate_keypair();
    auto key = issue_key(make_payload(), kp.secret_key).value();

    // Flip a printable char in the middle of the key. Skip dashes.
    auto tampered = key;
    for (char& c : tampered) {
        if (c != '-') {
            c = (c == 'A') ? 'B' : 'A';
            break;
        }
    }
    ASSERT_NE(tampered, key);
    auto r = verify_key_with(tampered, kp.public_key);
    EXPECT_FALSE(r.ok());
}

TEST(License, FlippedSignatureByteFailsVerification) {
    auto kp  = generate_keypair();
    auto key = issue_key(make_payload(), kp.secret_key).value();

    // The signature lives at the tail — flip the last non-dash char.
    auto tampered = key;
    for (auto it = tampered.rbegin(); it != tampered.rend(); ++it) {
        if (*it != '-') {
            *it = (*it == 'Z') ? 'A' : static_cast<char>(*it + 1);
            break;
        }
    }
    ASSERT_NE(tampered, key);
    auto r = verify_key_with(tampered, kp.public_key);
    EXPECT_FALSE(r.ok());
}

// ---------------------------------------------------------------------------
// 4. Wrong public key fails
// ---------------------------------------------------------------------------

TEST(License, WrongPublicKeyFailsVerification) {
    auto kp_a = generate_keypair();
    auto kp_b = generate_keypair();
    auto key  = issue_key(make_payload(), kp_a.secret_key).value();

    auto r = verify_key_with(key, kp_b.public_key);
    EXPECT_FALSE(r.ok());
}

// ---------------------------------------------------------------------------
// 5. Embedded production key is the placeholder until ops mints one
// ---------------------------------------------------------------------------
//
// Belt-and-braces: a valid key signed with a *fresh* keypair must NOT
// verify against the embedded (placeholder) production key. If this
// test ever passes, we have an "all-zeros key trusts everything"
// vulnerability shipping in the binary.

TEST(License, EmbeddedKeyDoesNotAcceptArbitrarySignedKeys) {
    auto kp  = generate_keypair();
    auto key = issue_key(make_payload(), kp.secret_key).value();
    auto r   = verify_key(key);   // uses embedded_public_key() internally
    EXPECT_FALSE(r.ok());
}

TEST(License, EmbeddedPublicKeyHasCorrectSize) {
    EXPECT_EQ(embedded_public_key().size(), 32u);
}

// ---------------------------------------------------------------------------
// 6. Malformed key inputs
// ---------------------------------------------------------------------------

TEST(License, EmptyKeyRejected) {
    auto kp = generate_keypair();
    auto r  = verify_key_with("", kp.public_key);
    EXPECT_FALSE(r.ok());
}

TEST(License, NonBase32CharsRejected) {
    auto kp = generate_keypair();
    auto r  = verify_key_with("0189-!!!!", kp.public_key);   // 0/1/8/9/! aren't base32
    EXPECT_FALSE(r.ok());
}

TEST(License, ShortButValidBase32Rejected) {
    auto kp = generate_keypair();
    auto r  = verify_key_with("ABCD-EFGH", kp.public_key);   // far too short for sig
    EXPECT_FALSE(r.ok());
}

// ---------------------------------------------------------------------------
// 7. Expiry semantics
// ---------------------------------------------------------------------------

TEST(License, IsActiveTrueForFutureExpiry) {
    License l;
    l.expires = std::chrono::system_clock::now() + std::chrono::hours(1);
    EXPECT_TRUE(is_active(l));
}

TEST(License, IsActiveFalseForPastExpiry) {
    License l;
    l.expires = std::chrono::system_clock::now() - std::chrono::hours(1);
    EXPECT_FALSE(is_active(l));
}

TEST(License, ExpiredKeyRoundTripsButReportsInactive) {
    auto kp = generate_keypair();
    auto p  = make_payload(std::chrono::system_clock::now() -
                            std::chrono::hours(24 * 30));
    auto key = issue_key(p, kp.secret_key).value();
    auto lic = verify_key_with(key, kp.public_key).value();
    // verify_key only checks the signature; expiry gating is the
    // caller's job via is_active(). This separation lets the UI show
    // a useful "your key expired" message instead of a generic
    // "verification failed".
    EXPECT_FALSE(is_active(lic));
}

// ---------------------------------------------------------------------------
// 8. Feature gating
// ---------------------------------------------------------------------------

TEST(License, HasFeatureCaseSensitive) {
    License l;
    l.features = {"ONNX_EXPORT"};
    EXPECT_TRUE(has_feature(l, "ONNX_EXPORT"));
    EXPECT_FALSE(has_feature(l, "onnx_export"));   // case-sensitive on purpose
    EXPECT_FALSE(has_feature(l, "MULTI_GPU"));
}

TEST(License, FeatureListPreservedAcrossRoundTrip) {
    auto kp = generate_keypair();
    auto p  = make_payload();
    p.features = {"A", "B", "CCC", "DDDD"};
    auto key = issue_key(p, kp.secret_key).value();
    auto lic = verify_key_with(key, kp.public_key).value();
    EXPECT_EQ(lic.features, p.features);
}

// ---------------------------------------------------------------------------
// 9. UTF-8 holder names survive
// ---------------------------------------------------------------------------

TEST(License, UnicodeHolderNamePreserved) {
    auto kp = generate_keypair();
    auto p  = make_payload();
    p.holder = "山田太郎";   // multi-byte UTF-8
    auto key = issue_key(p, kp.secret_key).value();
    auto lic = verify_key_with(key, kp.public_key).value();
    EXPECT_EQ(lic.holder, p.holder);
}

// ---------------------------------------------------------------------------
// 10. Whitespace and dashes are tolerated in the key string
// ---------------------------------------------------------------------------
//
// Users will paste keys with stray newlines / tabs from emails. The
// decoder skips dashes already; we also accept spaces.

TEST(License, KeyWithExtraWhitespaceStillVerifies) {
    auto kp  = generate_keypair();
    auto key = issue_key(make_payload(), kp.secret_key).value();

    // Insert spaces around dashes.
    std::string spaced;
    for (char c : key) {
        if (c == '-') spaced += " - ";
        else          spaced.push_back(c);
    }
    auto r = verify_key_with(spaced, kp.public_key);
    EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}
