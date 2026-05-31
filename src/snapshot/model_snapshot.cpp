#include "neurograph/snapshot/model_snapshot.hpp"
#include "neurograph/util/logger.hpp"

#include <fstream>
#include <sstream>
#include <cstring>

namespace neurograph::snapshot {

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

util::Result<void> ModelSnapshot::save(
    const core::Network& network,
    const SnapshotMetadata& metadata,
    const std::string& path,
    SnapshotFormat format)
{
    // Build metadata with live network stats
    SnapshotMetadata meta = metadata;
    auto stats = network.stats();
    meta.neuron_count = stats.neuron_count;
    meta.synapse_count = stats.synapse_count;
    meta.organization_cycle = stats.organization_cycle;
    meta.metrics.avg_activation = stats.avg_activation;
    meta.metrics.avg_weight = stats.avg_weight;

    if (meta.created_timestamp == 0) {
        meta.stamp_now();
    }

    auto meta_json = meta.to_json();
    auto net_json = network.to_json();

    if (format == SnapshotFormat::BINARY) {
        return save_binary(meta_json, net_json, path);
    } else {
        return save_json(meta_json, net_json, path);
    }
}

util::Result<NeuralImage> ModelSnapshot::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return util::Error("Cannot open snapshot file: " + path);
    }

    // Peek at first 4 bytes to detect format
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.seekg(0);

    // Check if it starts with the binary magic (big-endian "NGIM")
    uint8_t magic_bytes[4];
    std::memcpy(magic_bytes, &magic, 4);
    uint32_t magic_be = (static_cast<uint32_t>(magic_bytes[0]) << 24) |
                        (static_cast<uint32_t>(magic_bytes[1]) << 16) |
                        (static_cast<uint32_t>(magic_bytes[2]) << 8)  |
                        (static_cast<uint32_t>(magic_bytes[3]));

    file.close();

    if (magic_be == SnapshotMetadata::MAGIC) {
        return load_binary(path);
    } else {
        return load_json(path);
    }
}

util::Result<SnapshotMetadata> ModelSnapshot::read_metadata(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return util::Error("Cannot open snapshot file: " + path);
    }

    // Peek at format
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.seekg(0);

    uint8_t magic_bytes[4];
    std::memcpy(magic_bytes, &magic, 4);
    uint32_t magic_be = (static_cast<uint32_t>(magic_bytes[0]) << 24) |
                        (static_cast<uint32_t>(magic_bytes[1]) << 16) |
                        (static_cast<uint32_t>(magic_bytes[2]) << 8)  |
                        (static_cast<uint32_t>(magic_bytes[3]));

    if (magic_be == SnapshotMetadata::MAGIC) {
        // Binary: skip magic + version, read meta length, read meta JSON
        file.seekg(0);
        auto m = read_u32_be(file); // magic
        (void)m;
        auto v = read_u16_be(file); // version
        (void)v;
        uint32_t meta_len = read_u32_be(file);

        std::string meta_str(meta_len, '\0');
        file.read(meta_str.data(), meta_len);
        if (!file.good()) {
            return util::Error("Truncated metadata in binary snapshot");
        }

        try {
            auto meta_json = nlohmann::json::parse(meta_str);
            return SnapshotMetadata::from_json(meta_json);
        } catch (const std::exception& e) {
            return util::Error(std::string("Failed to parse metadata JSON: ") + e.what());
        }
    } else {
        // JSON format
        file.close();
        std::ifstream jfile(path);
        if (!jfile.is_open()) {
            return util::Error("Cannot open snapshot file: " + path);
        }

        try {
            nlohmann::json j;
            jfile >> j;
            if (!j.contains("metadata")) {
                return util::Error("JSON snapshot missing 'metadata' key");
            }
            return SnapshotMetadata::from_json(j["metadata"]);
        } catch (const std::exception& e) {
            return util::Error(std::string("Failed to parse JSON snapshot: ") + e.what());
        }
    }
}

util::Result<void> ModelSnapshot::validate(const std::string& path) {
    auto meta_result = read_metadata(path);
    if (!meta_result.ok()) {
        return util::Error("Validation failed: " + meta_result.error().message);
    }

    // Try full load to validate network data
    auto load_result = load(path);
    if (!load_result.ok()) {
        return util::Error("Validation failed: " + load_result.error().message);
    }

    auto& image = load_result.value();
    if (image.metadata.neuron_count != image.network.neuron_count()) {
        return util::Error("Metadata neuron_count mismatch");
    }
    if (image.metadata.synapse_count != image.network.synapse_count()) {
        return util::Error("Metadata synapse_count mismatch");
    }

    return util::Result<void>::success();
}

// ---------------------------------------------------------------------------
// Binary format
// ---------------------------------------------------------------------------

util::Result<void> ModelSnapshot::save_binary(
    const nlohmann::json& meta_json,
    const nlohmann::json& net_json,
    const std::string& path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return util::Error("Cannot create snapshot file: " + path);
    }

    std::string meta_str = meta_json.dump();
    std::string net_str = net_json.dump();

    // Header
    write_u32_be(file, SnapshotMetadata::MAGIC);
    write_u16_be(file, SnapshotMetadata::FORMAT_VERSION);

    // Metadata block
    write_u32_be(file, static_cast<uint32_t>(meta_str.size()));
    file.write(meta_str.data(), meta_str.size());

    // Network block
    write_u32_be(file, static_cast<uint32_t>(net_str.size()));
    file.write(net_str.data(), net_str.size());

    if (!file.good()) {
        return util::Error("Write error while saving snapshot");
    }

    NG_LOG_INFO("Saved neural image: {} ({} neurons, {} synapses, {} bytes)",
                path, meta_json["neuron_count"].get<size_t>(),
                meta_json["synapse_count"].get<size_t>(),
                static_cast<size_t>(file.tellp()));

    return util::Result<void>::success();
}

util::Result<NeuralImage> ModelSnapshot::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return util::Error("Cannot open snapshot file: " + path);
    }

    // Read and verify magic
    uint32_t magic = read_u32_be(file);
    if (magic != SnapshotMetadata::MAGIC) {
        return util::Error("Invalid magic number — not a NeuroGraph image");
    }

    // Read and check version
    uint16_t version = read_u16_be(file);
    if (version > SnapshotMetadata::FORMAT_VERSION) {
        return util::Error("Unsupported format version: " + std::to_string(version));
    }

    // Read metadata
    uint32_t meta_len = read_u32_be(file);
    std::string meta_str(meta_len, '\0');
    file.read(meta_str.data(), meta_len);
    if (!file.good()) {
        return util::Error("Truncated metadata in snapshot");
    }

    // Read network
    uint32_t net_len = read_u32_be(file);
    std::string net_str(net_len, '\0');
    file.read(net_str.data(), net_len);
    if (!file.good()) {
        return util::Error("Truncated network data in snapshot");
    }

    try {
        NeuralImage image;
        image.metadata = SnapshotMetadata::from_json(nlohmann::json::parse(meta_str));
        image.network = core::Network::from_json(nlohmann::json::parse(net_str));

        NG_LOG_INFO("Loaded neural image: {} ({} neurons, {} synapses)",
                    image.metadata.name, image.network.neuron_count(),
                    image.network.synapse_count());

        return image;
    } catch (const std::exception& e) {
        return util::Error(std::string("Failed to deserialize snapshot: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// JSON format
// ---------------------------------------------------------------------------

util::Result<void> ModelSnapshot::save_json(
    const nlohmann::json& meta_json,
    const nlohmann::json& net_json,
    const std::string& path)
{
    std::ofstream file(path);
    if (!file.is_open()) {
        return util::Error("Cannot create snapshot file: " + path);
    }

    nlohmann::json doc = {
        {"metadata", meta_json},
        {"network", net_json}
    };

    file << doc.dump(2);

    if (!file.good()) {
        return util::Error("Write error while saving JSON snapshot");
    }

    NG_LOG_INFO("Saved neural image (JSON): {} ({} neurons, {} synapses)",
                path, meta_json["neuron_count"].get<size_t>(),
                meta_json["synapse_count"].get<size_t>());

    return util::Result<void>::success();
}

util::Result<NeuralImage> ModelSnapshot::load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return util::Error("Cannot open snapshot file: " + path);
    }

    try {
        nlohmann::json doc;
        file >> doc;

        if (!doc.contains("metadata") || !doc.contains("network")) {
            return util::Error("JSON snapshot missing 'metadata' or 'network' key");
        }

        NeuralImage image;
        image.metadata = SnapshotMetadata::from_json(doc["metadata"]);
        image.network = core::Network::from_json(doc["network"]);

        NG_LOG_INFO("Loaded neural image (JSON): {} ({} neurons, {} synapses)",
                    image.metadata.name, image.network.neuron_count(),
                    image.network.synapse_count());

        return image;
    } catch (const std::exception& e) {
        return util::Error(std::string("Failed to parse JSON snapshot: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// Binary helpers (big-endian)
// ---------------------------------------------------------------------------

void ModelSnapshot::write_u32_be(std::ostream& out, uint32_t val) {
    uint8_t bytes[4] = {
        static_cast<uint8_t>((val >> 24) & 0xFF),
        static_cast<uint8_t>((val >> 16) & 0xFF),
        static_cast<uint8_t>((val >> 8)  & 0xFF),
        static_cast<uint8_t>((val)       & 0xFF)
    };
    out.write(reinterpret_cast<const char*>(bytes), 4);
}

void ModelSnapshot::write_u16_be(std::ostream& out, uint16_t val) {
    uint8_t bytes[2] = {
        static_cast<uint8_t>((val >> 8) & 0xFF),
        static_cast<uint8_t>((val)      & 0xFF)
    };
    out.write(reinterpret_cast<const char*>(bytes), 2);
}

uint32_t ModelSnapshot::read_u32_be(std::istream& in) {
    uint8_t bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

uint16_t ModelSnapshot::read_u16_be(std::istream& in) {
    uint8_t bytes[2];
    in.read(reinterpret_cast<char*>(bytes), 2);
    return (static_cast<uint16_t>(bytes[0]) << 8) |
           (static_cast<uint16_t>(bytes[1]));
}

} // namespace neurograph::snapshot
