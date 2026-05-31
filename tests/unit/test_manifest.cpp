// Unit tests for neurograph::reporting::Manifest (P1-13).
//
// Coverage:
//   - SHA-256 known-answer test against the standard "abc" / empty
//     vectors so a regression in libsodium wiring is caught immediately.
//   - File hashing matches the byte-buffer hash for the same content.
//   - save → load round-trip preserves every field including embedded JSON.
//   - verify_against returns true on an untouched bundle, false on
//     tampered/missing artifacts.
//   - Empty hash fields are treated as "not tracked" and skipped.
//   - ISO 8601 timestamp format matches the spec.
//   - Schema version mismatch is rejected.

#include "neurograph/reporting/manifest.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <string>

using neurograph::reporting::current_iso8601_utc;
using neurograph::reporting::load_manifest;
using neurograph::reporting::Manifest;
using neurograph::reporting::populate_host_info;
using neurograph::reporting::save_manifest;
using neurograph::reporting::sha256_of_bytes;
using neurograph::reporting::sha256_of_file;
using neurograph::reporting::verify_against;

namespace {

// Deterministic temp directory per test. We can't share `tempdir`
// across tests because parallel runs would clobber each other.
struct TempDir {
    std::filesystem::path path;
    explicit TempDir(const std::string& tag) {
        std::random_device rd;
        const auto stamp = std::to_string(rd()) + "_" + tag;
        path = std::filesystem::temp_directory_path() /
               ("neurograph_manifest_" + stamp);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void write_file(const std::filesystem::path& p, const std::string& content) {
    std::ofstream f(p, std::ios::binary);
    ASSERT_TRUE(f.is_open()) << "cannot open " << p;
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    ASSERT_TRUE(f.good());
}

// Produce a fully-populated manifest pointing at the supplied bundle
// dir. SHAs are real SHA-256s of the on-disk files.
Manifest make_manifest_for_bundle(const std::filesystem::path& dir) {
    Manifest m;
    m.neurograph_version = "0.1.0";
    m.build_git_sha      = "abc1234";
    m.dataset_sha256     = sha256_of_file(dir / "dataset.csv");
    m.pipeline_sha256    = sha256_of_file(dir / "pipeline.ngpipe");
    m.model_sha256       = sha256_of_file(dir / "model.ngm");
    m.rng_seed           = 42;
    m.os                 = "TestOS 1.0";
    m.cpu                = "FakeCPU";
    m.gpu                = "FakeGPU";
    m.timestamp_iso8601  = "2026-04-18T12:00:00Z";
    m.pipeline_config    = nlohmann::json{{"kind", "pipeline"}, {"v", 1}};
    m.training_config    = nlohmann::json{{"kind", "training"}, {"epochs", 50}};
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. SHA-256 known-answer test
// ---------------------------------------------------------------------------
//
// FIPS 180-4 / RFC 4634 standard test vectors: "abc" → ba7816..., and
// the empty string → e3b0c4... If either fails, libsodium isn't linked
// or the byte-order is wrong.

TEST(Manifest, Sha256MatchesKnownVectors) {
    EXPECT_EQ(sha256_of_bytes("abc", 3),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_of_bytes("", 0),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

// ---------------------------------------------------------------------------
// 2. File hash matches byte-buffer hash
// ---------------------------------------------------------------------------

TEST(Manifest, Sha256OfFileMatchesByteBuffer) {
    TempDir t("file_vs_bytes");
    const std::string body = "hello, manifest world\n";
    write_file(t.path / "x.bin", body);
    EXPECT_EQ(sha256_of_file(t.path / "x.bin"),
              sha256_of_bytes(body.data(), body.size()));
}

// ---------------------------------------------------------------------------
// 3. Streaming hash works for a file larger than one chunk
// ---------------------------------------------------------------------------
//
// kChunkBytes = 64 KiB inside the .cpp. We push past 200 KiB so the
// loop runs at least 3 times.

TEST(Manifest, Sha256OfLargeFileMatchesByteBuffer) {
    TempDir t("large");
    std::string body(200 * 1024 + 17, 'A');
    for (std::size_t i = 0; i < body.size(); ++i) {
        body[i] = static_cast<char>('A' + (i % 26));
    }
    write_file(t.path / "big.bin", body);
    EXPECT_EQ(sha256_of_file(t.path / "big.bin"),
              sha256_of_bytes(body.data(), body.size()));
}

// ---------------------------------------------------------------------------
// 4. save → load round-trip preserves all fields
// ---------------------------------------------------------------------------

TEST(Manifest, SaveLoadRoundTripPreservesFields) {
    TempDir t("roundtrip");
    write_file(t.path / "dataset.csv",     "x,y\n1,2\n");
    write_file(t.path / "pipeline.ngpipe", R"({"v":1})");
    write_file(t.path / "model.ngm",       "binary-blob");

    const auto m = make_manifest_for_bundle(t.path);
    save_manifest(m, t.path / "manifest.json");

    const auto loaded = load_manifest(t.path / "manifest.json");
    EXPECT_EQ(loaded.neurograph_version, m.neurograph_version);
    EXPECT_EQ(loaded.build_git_sha,      m.build_git_sha);
    EXPECT_EQ(loaded.dataset_sha256,     m.dataset_sha256);
    EXPECT_EQ(loaded.pipeline_sha256,    m.pipeline_sha256);
    EXPECT_EQ(loaded.model_sha256,       m.model_sha256);
    EXPECT_EQ(loaded.rng_seed,           m.rng_seed);
    EXPECT_EQ(loaded.os,                 m.os);
    EXPECT_EQ(loaded.cpu,                m.cpu);
    EXPECT_EQ(loaded.gpu,                m.gpu);
    EXPECT_EQ(loaded.timestamp_iso8601,  m.timestamp_iso8601);
    EXPECT_EQ(loaded.pipeline_config,    m.pipeline_config);
    EXPECT_EQ(loaded.training_config,    m.training_config);
}

// ---------------------------------------------------------------------------
// 5. verify_against on an untampered bundle
// ---------------------------------------------------------------------------

TEST(Manifest, VerifyAgainstReturnsTrueOnUntamperedBundle) {
    TempDir t("verify_ok");
    write_file(t.path / "dataset.csv",     "a,b\n1,2\n3,4\n");
    write_file(t.path / "pipeline.ngpipe", R"({"v":1,"cols":[]})");
    write_file(t.path / "model.ngm",       "weights-bytes");

    const auto m = make_manifest_for_bundle(t.path);
    EXPECT_TRUE(verify_against(m, t.path));
}

// ---------------------------------------------------------------------------
// 6. verify_against fails on tampered file
// ---------------------------------------------------------------------------

TEST(Manifest, VerifyAgainstReturnsFalseWhenModelTampered) {
    TempDir t("verify_tamper");
    write_file(t.path / "dataset.csv",     "a,b\n1,2\n");
    write_file(t.path / "pipeline.ngpipe", "{}");
    write_file(t.path / "model.ngm",       "original");

    const auto m = make_manifest_for_bundle(t.path);
    write_file(t.path / "model.ngm", "tampered");
    EXPECT_FALSE(verify_against(m, t.path));
}

// ---------------------------------------------------------------------------
// 7. verify_against fails when a tracked file is missing
// ---------------------------------------------------------------------------

TEST(Manifest, VerifyAgainstReturnsFalseWhenFileMissing) {
    TempDir t("verify_missing");
    write_file(t.path / "dataset.csv",     "a\n1\n");
    write_file(t.path / "pipeline.ngpipe", "{}");
    write_file(t.path / "model.ngm",       "weights");

    const auto m = make_manifest_for_bundle(t.path);
    std::filesystem::remove(t.path / "model.ngm");
    EXPECT_FALSE(verify_against(m, t.path));
}

// ---------------------------------------------------------------------------
// 8. Empty hash fields are skipped
// ---------------------------------------------------------------------------

TEST(Manifest, EmptyHashFieldsAreNotTrackedAndSkipped) {
    TempDir t("empty_fields");
    // Bundle only contains pipeline + model — no dataset shipped.
    write_file(t.path / "pipeline.ngpipe", "{}");
    write_file(t.path / "model.ngm",       "weights");

    Manifest m;
    m.dataset_sha256.clear();   // not tracked
    m.pipeline_sha256 = sha256_of_file(t.path / "pipeline.ngpipe");
    m.model_sha256    = sha256_of_file(t.path / "model.ngm");

    EXPECT_TRUE(verify_against(m, t.path));
}

// ---------------------------------------------------------------------------
// 9. ISO 8601 UTC timestamp format
// ---------------------------------------------------------------------------

TEST(Manifest, CurrentIso8601IsStrictUtcFormat) {
    const auto ts = current_iso8601_utc();
    // YYYY-MM-DDTHH:MM:SSZ — 20 chars exactly.
    ASSERT_EQ(ts.size(), 20u) << "got '" << ts << "'";
    static const std::regex pat(
        R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$)");
    EXPECT_TRUE(std::regex_match(ts, pat)) << "got '" << ts << "'";
}

// ---------------------------------------------------------------------------
// 10. Schema version mismatch is rejected
// ---------------------------------------------------------------------------

TEST(Manifest, LoadRejectsUnknownSchemaVersion) {
    TempDir t("schema");
    nlohmann::json j;
    j["schema_version"]    = 999;
    j["neurograph_version"] = "x";
    write_file(t.path / "manifest.json", j.dump(2));
    EXPECT_THROW((void)load_manifest(t.path / "manifest.json"),
                 std::runtime_error);
}

TEST(Manifest, LoadRejectsMissingSchemaVersion) {
    TempDir t("no_schema");
    write_file(t.path / "manifest.json", "{}");
    EXPECT_THROW((void)load_manifest(t.path / "manifest.json"),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// 11. populate_host_info doesn't throw and fills os field on supported plats
// ---------------------------------------------------------------------------

TEST(Manifest, PopulateHostInfoNeverThrows) {
    Manifest m;
    EXPECT_NO_THROW(populate_host_info(m));
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
    EXPECT_FALSE(m.os.empty()) << "expected non-empty OS string on this platform";
#endif
}

// ---------------------------------------------------------------------------
// 12. Manifest file size stays under the spec's 8 KB target
// ---------------------------------------------------------------------------
//
// Acceptance: "File size < 8 KB typical." We check with a realistic
// fully-populated manifest plus a non-trivial config.

TEST(Manifest, TypicalManifestStaysUnder8KiB) {
    TempDir t("size");
    write_file(t.path / "dataset.csv",     "a,b\n1,2\n");
    write_file(t.path / "pipeline.ngpipe", "{}");
    write_file(t.path / "model.ngm",       "weights");

    auto m = make_manifest_for_bundle(t.path);
    // Larger configs: 30 columns × 5 fields each.
    nlohmann::json cols = nlohmann::json::array();
    for (int i = 0; i < 30; ++i) {
        cols.push_back({
            {"name",   "col_" + std::to_string(i)},
            {"impute", "mean"},
            {"scale",  "standard"},
            {"encode", "onehot"},
            {"drop",   false},
        });
    }
    m.pipeline_config = {{"columns", cols}, {"target", "y"}};
    m.training_config = {{"epochs", 100}, {"optimizer", "adam"},
                         {"learning_rate", 1e-3}};

    save_manifest(m, t.path / "manifest.json");
    const auto sz = std::filesystem::file_size(t.path / "manifest.json");
    EXPECT_LT(sz, 8u * 1024u) << "manifest.json grew to " << sz << " bytes";
}
