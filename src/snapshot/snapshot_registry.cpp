#include "neurograph/snapshot/snapshot_registry.hpp"
#include "neurograph/util/logger.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;

namespace neurograph::snapshot {

SnapshotRegistry::SnapshotRegistry(const std::string& directory)
    : directory_(directory) {}

util::Result<size_t> SnapshotRegistry::scan() {
    entries_.clear();

    if (!fs::exists(directory_)) {
        NG_LOG_WARN("Registry directory does not exist: {}", directory_);
        return static_cast<size_t>(0);
    }

    if (!fs::is_directory(directory_)) {
        return util::Error("Path is not a directory: " + directory_);
    }

    size_t found = 0;
    for (const auto& entry : fs::directory_iterator(directory_)) {
        if (!entry.is_regular_file()) continue;

        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();

        // Check for .ngmodel or .ngmodel.json
        bool is_binary = filename.size() > 8 &&
                         filename.substr(filename.size() - 8) == ".ngmodel" &&
                         filename.substr(filename.size() - 13) != ".ngmodel.json";
        bool is_json = filename.size() > 13 &&
                       filename.substr(filename.size() - 13) == ".ngmodel.json";

        if (!is_binary && !is_json) continue;

        auto meta_result = ModelSnapshot::read_metadata(path);
        if (!meta_result.ok()) {
            NG_LOG_WARN("Skipping invalid snapshot {}: {}", filename, meta_result.error().message);
            continue;
        }

        RegistryEntry reg_entry;
        reg_entry.metadata = meta_result.value();
        reg_entry.file_path = path;
        reg_entry.format = is_binary ? SnapshotFormat::BINARY : SnapshotFormat::JSON;
        reg_entry.file_size_bytes = entry.file_size();

        std::string name = reg_entry.metadata.name;
        if (name.empty()) {
            // Derive name from filename
            name = entry.path().stem().string();
            if (is_json) {
                // Remove extra .ngmodel from stem of .ngmodel.json
                if (name.size() > 8 && name.substr(name.size() - 8) == ".ngmodel") {
                    name = name.substr(0, name.size() - 8);
                }
            }
        }

        entries_[name] = std::move(reg_entry);
        found++;
    }

    NG_LOG_INFO("Registry scan: {} neural images found in {}", found, directory_);
    return found;
}

util::Result<std::string> SnapshotRegistry::save(
    const core::Network& network,
    const SnapshotMetadata& metadata,
    SnapshotFormat format)
{
    if (metadata.name.empty()) {
        return util::Error("Snapshot name cannot be empty");
    }

    // Ensure directory exists
    if (!fs::exists(directory_)) {
        fs::create_directories(directory_);
    }

    std::string path = build_path(metadata.name, format);
    auto result = ModelSnapshot::save(network, metadata, path, format);
    if (!result.ok()) {
        return util::Error(result.error().message);
    }

    // Add to registry
    RegistryEntry entry;
    entry.metadata = metadata;
    auto stats = network.stats();
    entry.metadata.neuron_count = stats.neuron_count;
    entry.metadata.synapse_count = stats.synapse_count;
    entry.metadata.organization_cycle = stats.organization_cycle;
    entry.file_path = path;
    entry.format = format;
    entry.file_size_bytes = fs::file_size(path);

    entries_[metadata.name] = std::move(entry);

    return path;
}

util::Result<NeuralImage> SnapshotRegistry::load(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        return util::Error("Snapshot not found: " + name);
    }

    return ModelSnapshot::load(it->second.file_path);
}

util::Result<void> SnapshotRegistry::remove(const std::string& name) {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        return util::Error("Snapshot not found: " + name);
    }

    std::string path = it->second.file_path;
    entries_.erase(it);

    std::error_code ec;
    fs::remove(path, ec);
    if (ec) {
        return util::Error("Failed to delete file: " + ec.message());
    }

    NG_LOG_INFO("Removed neural image: {}", name);
    return util::Result<void>::success();
}

bool SnapshotRegistry::has(const std::string& name) const {
    return entries_.find(name) != entries_.end();
}

util::Result<SnapshotMetadata> SnapshotRegistry::get_metadata(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        return util::Error("Snapshot not found: " + name);
    }
    return it->second.metadata;
}

std::vector<const RegistryEntry*> SnapshotRegistry::find_by_task(TaskType type) const {
    std::vector<const RegistryEntry*> results;
    for (const auto& [name, entry] : entries_) {
        if (entry.metadata.task_type == type) {
            results.push_back(&entry);
        }
    }
    // Sort by accuracy descending
    std::sort(results.begin(), results.end(),
              [](const RegistryEntry* a, const RegistryEntry* b) {
                  return a->metadata.metrics.accuracy > b->metadata.metrics.accuracy;
              });
    return results;
}

std::vector<const RegistryEntry*> SnapshotRegistry::find_by_tag(const std::string& tag) const {
    std::vector<const RegistryEntry*> results;
    for (const auto& [name, entry] : entries_) {
        for (const auto& t : entry.metadata.tags) {
            if (t == tag) {
                results.push_back(&entry);
                break;
            }
        }
    }
    return results;
}

std::vector<const RegistryEntry*> SnapshotRegistry::find_if(
    std::function<bool(const RegistryEntry&)> predicate) const
{
    std::vector<const RegistryEntry*> results;
    for (const auto& [name, entry] : entries_) {
        if (predicate(entry)) {
            results.push_back(&entry);
        }
    }
    return results;
}

util::Result<const RegistryEntry*> SnapshotRegistry::find_best(TaskType type) const {
    auto matches = find_by_task(type);
    if (matches.empty()) {
        return util::Error("No snapshots found for task type: " +
                          std::string(task_type_str(type)));
    }
    return matches.front();  // already sorted by accuracy descending
}

std::vector<std::string> SnapshotRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(entries_.size());
    for (const auto& [name, _] : entries_) {
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

void SnapshotRegistry::print_catalog() const {
    if (entries_.empty()) {
        std::cout << "  (no neural images in registry)\n";
        return;
    }

    std::cout << std::left
              << std::setw(25) << "Name"
              << std::setw(18) << "Task"
              << std::setw(10) << "Neurons"
              << std::setw(10) << "Synapses"
              << std::setw(10) << "Accuracy"
              << std::setw(10) << "Epochs"
              << "Size\n";
    std::cout << std::string(93, '-') << "\n";

    for (const auto& [name, entry] : entries_) {
        const auto& m = entry.metadata;
        std::cout << std::left
                  << std::setw(25) << name
                  << std::setw(18) << task_type_str(m.task_type)
                  << std::setw(10) << m.neuron_count
                  << std::setw(10) << m.synapse_count
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << (m.metrics.accuracy * 100.0) << "%"
                  << std::setw(8) << m.training.epochs_completed;

        // Human-readable file size
        if (entry.file_size_bytes < 1024) {
            std::cout << entry.file_size_bytes << " B";
        } else if (entry.file_size_bytes < 1024 * 1024) {
            std::cout << std::fixed << std::setprecision(1)
                      << (entry.file_size_bytes / 1024.0) << " KB";
        } else {
            std::cout << std::fixed << std::setprecision(1)
                      << (entry.file_size_bytes / (1024.0 * 1024.0)) << " MB";
        }
        std::cout << "\n";
    }
}

std::string SnapshotRegistry::build_path(
    const std::string& name, SnapshotFormat format) const
{
    std::string ext = (format == SnapshotFormat::BINARY)
        ? ModelSnapshot::EXTENSION
        : ModelSnapshot::JSON_EXTENSION;

    return (fs::path(directory_) / (name + ext)).string();
}

} // namespace neurograph::snapshot
