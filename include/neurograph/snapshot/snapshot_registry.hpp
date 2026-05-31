#pragma once

#include "neurograph/snapshot/snapshot_metadata.hpp"
#include "neurograph/snapshot/model_snapshot.hpp"
#include "neurograph/util/result.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace neurograph::snapshot {

/// Entry in the snapshot registry — metadata + file path.
struct RegistryEntry {
    SnapshotMetadata metadata;
    std::string file_path;
    SnapshotFormat format = SnapshotFormat::BINARY;
    size_t file_size_bytes = 0;
};

/// Manages a directory of neural images.
///
/// The registry scans a directory for .ngmodel and .ngmodel.json files,
/// reads their metadata headers, and provides search/filter/load operations.
///
/// Usage:
///   SnapshotRegistry registry("./models");
///   registry.scan();  // discover all snapshots
///
///   // Search by task
///   auto classifiers = registry.find_by_task(TaskType::CLASSIFICATION);
///
///   // Search by name
///   auto image = registry.load("mnist_classifier_v2");
///
///   // Save a new snapshot into the registry
///   registry.save(network, metadata);
///
class SnapshotRegistry {
public:
    /// Create a registry backed by a directory path.
    explicit SnapshotRegistry(const std::string& directory);

    /// Scan the directory for snapshot files and index their metadata.
    /// Returns the number of snapshots found.
    util::Result<size_t> scan();

    /// Save a network as a neural image into the registry directory.
    /// The file is named after metadata.name (e.g., "mnist_v2.ngmodel").
    util::Result<std::string> save(
        const core::Network& network,
        const SnapshotMetadata& metadata,
        SnapshotFormat format = SnapshotFormat::BINARY);

    /// Load a neural image by name.
    util::Result<NeuralImage> load(const std::string& name) const;

    /// Delete a snapshot by name (removes file and registry entry).
    util::Result<void> remove(const std::string& name);

    /// Check if a snapshot exists by name.
    bool has(const std::string& name) const;

    /// Get metadata for a specific snapshot.
    util::Result<SnapshotMetadata> get_metadata(const std::string& name) const;

    // --- Search / Filter ---

    /// Find all snapshots matching a task type.
    std::vector<const RegistryEntry*> find_by_task(TaskType type) const;

    /// Find all snapshots with a specific tag.
    std::vector<const RegistryEntry*> find_by_tag(const std::string& tag) const;

    /// Find all snapshots matching a custom predicate.
    std::vector<const RegistryEntry*> find_if(
        std::function<bool(const RegistryEntry&)> predicate) const;

    /// Find the best snapshot for a task (highest accuracy).
    util::Result<const RegistryEntry*> find_best(TaskType type) const;

    /// Get all registered entries.
    const std::unordered_map<std::string, RegistryEntry>& entries() const noexcept {
        return entries_;
    }

    /// Get the number of registered snapshots.
    size_t count() const noexcept { return entries_.size(); }

    /// Get the registry directory path.
    const std::string& directory() const noexcept { return directory_; }

    /// List all snapshot names.
    std::vector<std::string> names() const;

    /// Print a summary of all registered snapshots.
    void print_catalog() const;

private:
    std::string directory_;
    std::unordered_map<std::string, RegistryEntry> entries_;

    /// Build the file path for a snapshot name + format.
    std::string build_path(const std::string& name, SnapshotFormat format) const;
};

} // namespace neurograph::snapshot
