#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include <vector>
#include <cstdint>

namespace neurograph::snapshot {

/// Describes the task a neural image was trained for.
enum class TaskType : uint8_t {
    CLASSIFICATION = 0,
    REGRESSION     = 1,
    CLUSTERING     = 2,
    ASSOCIATION    = 3,   // associative memory / pattern recall
    ANOMALY        = 4,   // anomaly detection
    CUSTOM         = 255
};

inline std::string_view task_type_str(TaskType t) {
    switch (t) {
        case TaskType::CLASSIFICATION: return "classification";
        case TaskType::REGRESSION:     return "regression";
        case TaskType::CLUSTERING:     return "clustering";
        case TaskType::ASSOCIATION:    return "association";
        case TaskType::ANOMALY:        return "anomaly";
        case TaskType::CUSTOM:         return "custom";
        default:                       return "unknown";
    }
}

inline TaskType parse_task_type(const std::string& s) {
    if (s == "classification") return TaskType::CLASSIFICATION;
    if (s == "regression")     return TaskType::REGRESSION;
    if (s == "clustering")     return TaskType::CLUSTERING;
    if (s == "association")    return TaskType::ASSOCIATION;
    if (s == "anomaly")        return TaskType::ANOMALY;
    return TaskType::CUSTOM;
}

/// Training configuration captured at snapshot time.
struct TrainingConfig {
    double learning_rate = 0.01;
    int epochs_completed = 0;
    int self_org_iterations = 0;
    double pruning_threshold = 0.05;
    double affinity_threshold = 0.3;
    std::string topology_strategy = "hebbian";
    nlohmann::json extra;  // any additional training params

    nlohmann::json to_json() const {
        auto j = nlohmann::json{
            {"learning_rate", learning_rate},
            {"epochs_completed", epochs_completed},
            {"self_org_iterations", self_org_iterations},
            {"pruning_threshold", pruning_threshold},
            {"affinity_threshold", affinity_threshold},
            {"topology_strategy", topology_strategy}
        };
        if (!extra.empty()) j["extra"] = extra;
        return j;
    }

    static TrainingConfig from_json(const nlohmann::json& j) {
        TrainingConfig c;
        if (j.contains("learning_rate"))       c.learning_rate = j["learning_rate"];
        if (j.contains("epochs_completed"))    c.epochs_completed = j["epochs_completed"];
        if (j.contains("self_org_iterations")) c.self_org_iterations = j["self_org_iterations"];
        if (j.contains("pruning_threshold"))   c.pruning_threshold = j["pruning_threshold"];
        if (j.contains("affinity_threshold"))  c.affinity_threshold = j["affinity_threshold"];
        if (j.contains("topology_strategy"))   c.topology_strategy = j["topology_strategy"];
        if (j.contains("extra"))               c.extra = j["extra"];
        return c;
    }
};

/// Performance metrics captured at snapshot time.
struct PerformanceMetrics {
    double accuracy = 0.0;         // classification accuracy [0, 1]
    double loss = 0.0;             // training loss
    double avg_activation = 0.0;
    double avg_weight = 0.0;
    nlohmann::json extra;          // task-specific metrics (precision, recall, etc.)

    nlohmann::json to_json() const {
        auto j = nlohmann::json{
            {"accuracy", accuracy},
            {"loss", loss},
            {"avg_activation", avg_activation},
            {"avg_weight", avg_weight}
        };
        if (!extra.empty()) j["extra"] = extra;
        return j;
    }

    static PerformanceMetrics from_json(const nlohmann::json& j) {
        PerformanceMetrics m;
        if (j.contains("accuracy"))       m.accuracy = j["accuracy"];
        if (j.contains("loss"))           m.loss = j["loss"];
        if (j.contains("avg_activation")) m.avg_activation = j["avg_activation"];
        if (j.contains("avg_weight"))     m.avg_weight = j["avg_weight"];
        if (j.contains("extra"))          m.extra = j["extra"];
        return m;
    }
};

/// Input/output neuron group mapping stored in the snapshot.
struct IOMapping {
    std::vector<std::string> input_neuron_ids;
    std::vector<std::string> output_neuron_ids;
    int input_features = 0;
    int output_classes = 0;
    std::vector<std::string> class_labels;  // optional: ["cat", "dog", "bird"]

    nlohmann::json to_json() const {
        return {
            {"input_neuron_ids", input_neuron_ids},
            {"output_neuron_ids", output_neuron_ids},
            {"input_features", input_features},
            {"output_classes", output_classes},
            {"class_labels", class_labels}
        };
    }

    static IOMapping from_json(const nlohmann::json& j) {
        IOMapping m;
        if (j.contains("input_neuron_ids"))  m.input_neuron_ids = j["input_neuron_ids"].get<std::vector<std::string>>();
        if (j.contains("output_neuron_ids")) m.output_neuron_ids = j["output_neuron_ids"].get<std::vector<std::string>>();
        if (j.contains("input_features"))    m.input_features = j["input_features"];
        if (j.contains("output_classes"))    m.output_classes = j["output_classes"];
        if (j.contains("class_labels"))      m.class_labels = j["class_labels"].get<std::vector<std::string>>();
        return m;
    }
};

/// Complete metadata header for a neural image.
struct SnapshotMetadata {
    // Identity
    std::string name;                // unique snapshot name (e.g., "mnist_classifier_v2")
    std::string description;         // human-readable description
    std::string version = "1.0.0";   // snapshot version
    std::string author;

    // Task
    TaskType task_type = TaskType::CUSTOM;
    std::string task_name;           // specific task (e.g., "digit_recognition")
    std::vector<std::string> tags;   // searchable tags ["vision", "mnist", "classifier"]

    // Network topology at snapshot time
    size_t neuron_count = 0;
    size_t synapse_count = 0;
    int organization_cycle = 0;

    // Training info
    TrainingConfig training;
    PerformanceMetrics metrics;
    IOMapping io_mapping;

    // Timestamps
    uint64_t created_timestamp = 0;  // unix epoch ms
    uint64_t training_duration_ms = 0;

    // Format info
    static constexpr uint32_t MAGIC = 0x4E47494D;  // "NGIM" — NeuroGraph Image
    static constexpr uint16_t FORMAT_VERSION = 1;

    void stamp_now() {
        created_timestamp = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );
    }

    nlohmann::json to_json() const {
        return {
            {"magic", MAGIC},
            {"format_version", FORMAT_VERSION},
            {"name", name},
            {"description", description},
            {"version", version},
            {"author", author},
            {"task_type", std::string(task_type_str(task_type))},
            {"task_name", task_name},
            {"tags", tags},
            {"neuron_count", neuron_count},
            {"synapse_count", synapse_count},
            {"organization_cycle", organization_cycle},
            {"training", training.to_json()},
            {"metrics", metrics.to_json()},
            {"io_mapping", io_mapping.to_json()},
            {"created_timestamp", created_timestamp},
            {"training_duration_ms", training_duration_ms}
        };
    }

    static SnapshotMetadata from_json(const nlohmann::json& j) {
        SnapshotMetadata m;
        if (j.contains("name"))               m.name = j["name"];
        if (j.contains("description"))        m.description = j["description"];
        if (j.contains("version"))            m.version = j["version"];
        if (j.contains("author"))             m.author = j["author"];
        if (j.contains("task_type"))          m.task_type = parse_task_type(j["task_type"]);
        if (j.contains("task_name"))          m.task_name = j["task_name"];
        if (j.contains("tags"))              m.tags = j["tags"].get<std::vector<std::string>>();
        if (j.contains("neuron_count"))       m.neuron_count = j["neuron_count"];
        if (j.contains("synapse_count"))      m.synapse_count = j["synapse_count"];
        if (j.contains("organization_cycle")) m.organization_cycle = j["organization_cycle"];
        if (j.contains("training"))           m.training = TrainingConfig::from_json(j["training"]);
        if (j.contains("metrics"))            m.metrics = PerformanceMetrics::from_json(j["metrics"]);
        if (j.contains("io_mapping"))         m.io_mapping = IOMapping::from_json(j["io_mapping"]);
        if (j.contains("created_timestamp"))  m.created_timestamp = j["created_timestamp"];
        if (j.contains("training_duration_ms")) m.training_duration_ms = j["training_duration_ms"];
        return m;
    }
};

} // namespace neurograph::snapshot
