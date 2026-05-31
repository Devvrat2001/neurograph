#pragma once

// P1-13 — Reproducibility Manifest.
//
// Every exported model bundle ships with a `manifest.json` that pins
// down exactly which inputs, configurations, and binaries produced the
// model. SOC2/ISO27001 audits ask "prove this exact model is the one
// you trained on day X with input Y" — `verify_against` is the answer.
//
// Captured fields:
//   - SHA-256 of dataset, fitted pipeline, and trained model files
//   - Pipeline + training configurations (full JSON, not just summaries)
//   - NeuroGraph version, build SHA, RNG seed
//   - Host info (OS / CPU / GPU strings) to flag platform-specific results
//   - ISO 8601 UTC timestamp
//
// All hashing uses libsodium's `crypto_hash_sha256` so we don't add a
// new transitive dependency.

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace neurograph::reporting {

struct Manifest {
    std::string   neurograph_version;
    std::string   build_git_sha;

    // Lower-case hex SHA-256s. Empty string means "not tracked" — that
    // artifact is skipped during verify_against (e.g. dataset isn't
    // shipped with the bundle for privacy reasons but still hashed).
    std::string   dataset_sha256;
    std::string   pipeline_sha256;
    std::string   model_sha256;

    std::uint64_t rng_seed = 0;

    std::string   os;   // e.g. "Windows 10.0", "Linux 5.15.0"
    std::string   cpu;  // best-effort brand string; may be empty
    std::string   gpu;  // best-effort; may be empty when no detection path

    /// "YYYY-MM-DDTHH:MM:SSZ" — strictly UTC, no timezone offset.
    std::string   timestamp_iso8601;

    // Full embedded configurations. Stored as JSON so the manifest is
    // self-describing and resilient to schema additions in the source
    // structs. The orchestrator fills these from existing serialization.
    nlohmann::json pipeline_config;
    nlohmann::json training_config;
};

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/// Serialize to JSON and write to disk (UTF-8, LF newlines, 2-space
/// indent). Throws `std::runtime_error` on filesystem error.
void save_manifest(const Manifest& m, const std::filesystem::path& path);

/// Load and parse. Throws `std::runtime_error` if the file is missing,
/// not valid JSON, or the schema version is unrecognized.
Manifest load_manifest(const std::filesystem::path& path);

/// Re-hash the artifacts in `bundle_dir` and compare against the
/// manifest. Conventions:
///
///   - Looks for `dataset.csv`, `pipeline.ngpipe`, `model.ngm`.
///   - For each artifact, the corresponding `*_sha256` field is checked
///     only when non-empty. Empty = not tracked, skipped.
///   - When a hash field is non-empty and the file is missing, returns
///     false (the manifest claims an artifact that isn't there).
///   - Returns true iff every tracked artifact present matches.
///
/// I/O failures (read errors, permission denied) are treated as
/// verification failures rather than thrown — the caller cares about
/// "did this pass yes/no", not the underlying errno.
bool verify_against(const Manifest& m, const std::filesystem::path& bundle_dir);

// ---------------------------------------------------------------------------
// Helpers — exposed so callers can build a Manifest without re-implementing
// SHA-256 or the host-info dance.
// ---------------------------------------------------------------------------

/// Lower-case hex SHA-256 of a file's contents. Streams the file in
/// chunks so even multi-GB datasets are safe. Throws `std::runtime_error`
/// on read failure.
std::string sha256_of_file(const std::filesystem::path& path);

/// Lower-case hex SHA-256 of an in-memory byte buffer. Convenience
/// wrapper around libsodium's `crypto_hash_sha256`.
std::string sha256_of_bytes(const void* data, std::size_t n_bytes);

/// Best-effort population of `os`, `cpu`, `gpu` fields. Never throws.
/// Whatever can't be detected on the current platform stays empty.
void populate_host_info(Manifest& m);

/// "now" as a strictly-UTC ISO 8601 string ("YYYY-MM-DDTHH:MM:SSZ").
std::string current_iso8601_utc();

} // namespace neurograph::reporting
