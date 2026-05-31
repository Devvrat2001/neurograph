#pragma once

#include "neurograph/snapshot/snapshot_metadata.hpp"
#include "neurograph/core/network.hpp"
#include "neurograph/util/result.hpp"

#include <string>
#include <vector>

namespace neurograph::snapshot {

/// Save format for neural images.
enum class SnapshotFormat : uint8_t {
    BINARY = 0,    // Compact binary: [magic][version][meta_len][meta_json][net_len][net_json]
    JSON   = 1     // Single JSON: {"metadata": {...}, "network": {...}}
};

/// A loaded neural image — metadata + network ready to use.
struct NeuralImage {
    SnapshotMetadata metadata;
    core::Network network;
};

/// Saves and loads neural images (.ngmodel files).
///
/// Binary format (.ngmodel):
///   [4 bytes]  magic number (0x4E47494D = "NGIM")
///   [2 bytes]  format version (1)
///   [4 bytes]  metadata JSON length (big-endian)
///   [N bytes]  metadata JSON (UTF-8)
///   [4 bytes]  network JSON length (big-endian)
///   [M bytes]  network JSON (UTF-8)
///
/// JSON format (.ngmodel.json):
///   Single JSON object with "metadata" and "network" keys.
///
class ModelSnapshot {
public:
    /// Save a network as a neural image.
    /// Automatically populates neuron_count, synapse_count, organization_cycle,
    /// avg_activation, avg_weight from the network's current state.
    static util::Result<void> save(
        const core::Network& network,
        const SnapshotMetadata& metadata,
        const std::string& path,
        SnapshotFormat format = SnapshotFormat::BINARY);

    /// Load a neural image from file.
    /// Auto-detects format from magic bytes vs JSON.
    static util::Result<NeuralImage> load(const std::string& path);

    /// Read only the metadata header without loading the full network.
    /// Useful for browsing/searching snapshots without the cost of
    /// deserializing the entire network.
    static util::Result<SnapshotMetadata> read_metadata(const std::string& path);

    /// Validate a snapshot file (check magic, version, JSON parse).
    static util::Result<void> validate(const std::string& path);

    /// Default file extension for binary format.
    static constexpr const char* EXTENSION = ".ngmodel";
    /// Default file extension for JSON format.
    static constexpr const char* JSON_EXTENSION = ".ngmodel.json";

private:
    static util::Result<void> save_binary(
        const nlohmann::json& meta_json,
        const nlohmann::json& net_json,
        const std::string& path);

    static util::Result<void> save_json(
        const nlohmann::json& meta_json,
        const nlohmann::json& net_json,
        const std::string& path);

    static util::Result<NeuralImage> load_binary(const std::string& path);
    static util::Result<NeuralImage> load_json(const std::string& path);

    static void write_u32_be(std::ostream& out, uint32_t val);
    static void write_u16_be(std::ostream& out, uint16_t val);
    static uint32_t read_u32_be(std::istream& in);
    static uint16_t read_u16_be(std::istream& in);
};

} // namespace neurograph::snapshot
