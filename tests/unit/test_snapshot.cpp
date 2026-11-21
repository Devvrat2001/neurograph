#include <gtest/gtest.h>
#include "neurograph/snapshot/snapshot_metadata.hpp"
#include "neurograph/snapshot/model_snapshot.hpp"
#include "neurograph/snapshot/snapshot_registry.hpp"
#include "neurograph/core/network.hpp"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace neurograph;

// Helper: create a small test network
static core::Network make_test_network(int n = 10, int connections = 15) {
    core::Network net;
    std::vector<core::NeuronId> ids;
    for (int i = 0; i < n; ++i) {
        auto id = core::NeuronId::generate();
        auto neuron = core::Neuron(id, "test_neuron_" + std::to_string(i));
        neuron.set_position(core::Vec3{
            static_cast<double>(i), static_cast<double>(i * 0.5), 0.0});
        neuron.set_threshold(0.3 + 0.01 * i);
        neuron.set_activation_level(0.1 * i);
        net.add_neuron(std::move(neuron));
        ids.push_back(id);
    }
    int made = 0;
    for (int i = 0; i < n && made < connections; ++i) {
        for (int j = i + 1; j < n && made < connections; ++j) {
            net.connect(ids[i], ids[j], 0.1 + 0.05 * made);
            made++;
        }
    }
    return net;
}

// Helper: create test metadata
static snapshot::SnapshotMetadata make_test_metadata(const std::string& name = "test_model") {
    snapshot::SnapshotMetadata meta;
    meta.name = name;
    meta.description = "Test neural image";
    meta.version = "1.0.0";
    meta.author = "NeuroGraph Test";
    meta.task_type = snapshot::TaskType::CLASSIFICATION;
    meta.task_name = "digit_recognition";
    meta.tags = {"test", "classification", "digits"};
    meta.training.learning_rate = 0.01;
    meta.training.epochs_completed = 50;
    meta.training.topology_strategy = "hebbian";
    meta.metrics.accuracy = 0.87;
    meta.metrics.loss = 0.32;
    meta.io_mapping.input_features = 784;
    meta.io_mapping.output_classes = 10;
    meta.io_mapping.class_labels = {"0","1","2","3","4","5","6","7","8","9"};
    meta.stamp_now();
    return meta;
}

// Each test gets its own folder. CTest runs every test as a separate
// process, several at once, and with one folder for the whole fixture a
// test's TearDown deleted files that another test was still using.
class SnapshotTestFixture : public ::testing::Test {
protected:
    std::string test_dir;

    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()->current_test_info();
        const auto dir = fs::temp_directory_path() / "neurograph_test_snapshots"
                         / info.test_suite_name() / info.name();
        std::error_code ec;
        fs::remove_all(dir, ec);  // left over from an interrupted run
        fs::create_directories(dir);
        test_dir = dir.string();
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(test_dir, ec);
    }
};

// ===========================================================================
// Metadata Tests
// ===========================================================================

TEST(SnapshotMetadata, TaskTypeRoundtrip) {
    using namespace snapshot;
    EXPECT_EQ(parse_task_type("classification"), TaskType::CLASSIFICATION);
    EXPECT_EQ(parse_task_type("regression"), TaskType::REGRESSION);
    EXPECT_EQ(parse_task_type("clustering"), TaskType::CLUSTERING);
    EXPECT_EQ(parse_task_type("association"), TaskType::ASSOCIATION);
    EXPECT_EQ(parse_task_type("anomaly"), TaskType::ANOMALY);
    EXPECT_EQ(parse_task_type("custom"), TaskType::CUSTOM);
    EXPECT_EQ(parse_task_type("unknown_xyz"), TaskType::CUSTOM);

    EXPECT_EQ(task_type_str(TaskType::CLASSIFICATION), "classification");
    EXPECT_EQ(task_type_str(TaskType::REGRESSION), "regression");
}

TEST(SnapshotMetadata, JsonRoundtrip) {
    auto meta = make_test_metadata();
    auto j = meta.to_json();
    auto restored = snapshot::SnapshotMetadata::from_json(j);

    EXPECT_EQ(restored.name, "test_model");
    EXPECT_EQ(restored.description, "Test neural image");
    EXPECT_EQ(restored.task_type, snapshot::TaskType::CLASSIFICATION);
    EXPECT_EQ(restored.task_name, "digit_recognition");
    EXPECT_EQ(restored.tags.size(), 3);
    EXPECT_DOUBLE_EQ(restored.training.learning_rate, 0.01);
    EXPECT_EQ(restored.training.epochs_completed, 50);
    EXPECT_DOUBLE_EQ(restored.metrics.accuracy, 0.87);
    EXPECT_EQ(restored.io_mapping.input_features, 784);
    EXPECT_EQ(restored.io_mapping.output_classes, 10);
    EXPECT_EQ(restored.io_mapping.class_labels.size(), 10);
    EXPECT_GT(restored.created_timestamp, 0u);
}

TEST(SnapshotMetadata, TrainingConfigRoundtrip) {
    snapshot::TrainingConfig config;
    config.learning_rate = 0.005;
    config.epochs_completed = 100;
    config.self_org_iterations = 500;
    config.pruning_threshold = 0.02;
    config.extra = {{"batch_size", 32}, {"warmup", true}};

    auto j = config.to_json();
    auto restored = snapshot::TrainingConfig::from_json(j);

    EXPECT_DOUBLE_EQ(restored.learning_rate, 0.005);
    EXPECT_EQ(restored.epochs_completed, 100);
    EXPECT_EQ(restored.self_org_iterations, 500);
    EXPECT_DOUBLE_EQ(restored.pruning_threshold, 0.02);
    EXPECT_EQ(restored.extra["batch_size"], 32);
    EXPECT_EQ(restored.extra["warmup"], true);
}

TEST(SnapshotMetadata, IOMapping) {
    snapshot::IOMapping io;
    io.input_neuron_ids = {"id1", "id2", "id3"};
    io.output_neuron_ids = {"out1", "out2"};
    io.input_features = 3;
    io.output_classes = 2;
    io.class_labels = {"cat", "dog"};

    auto j = io.to_json();
    auto restored = snapshot::IOMapping::from_json(j);

    EXPECT_EQ(restored.input_neuron_ids.size(), 3);
    EXPECT_EQ(restored.output_neuron_ids.size(), 2);
    EXPECT_EQ(restored.input_features, 3);
    EXPECT_EQ(restored.class_labels[0], "cat");
    EXPECT_EQ(restored.class_labels[1], "dog");
}

TEST(SnapshotMetadata, MagicNumber) {
    EXPECT_EQ(snapshot::SnapshotMetadata::MAGIC, 0x4E47494D);
    EXPECT_EQ(snapshot::SnapshotMetadata::FORMAT_VERSION, 1);
}

// ===========================================================================
// ModelSnapshot Tests
// ===========================================================================

TEST_F(SnapshotTestFixture, SaveLoadBinary) {
    auto net = make_test_network();
    auto meta = make_test_metadata("binary_test");
    std::string path = test_dir + "/binary_test.ngmodel";

    auto save_result = snapshot::ModelSnapshot::save(
        net, meta, path, snapshot::SnapshotFormat::BINARY);
    ASSERT_TRUE(save_result.ok()) << save_result.error().message;
    EXPECT_TRUE(fs::exists(path));

    auto load_result = snapshot::ModelSnapshot::load(path);
    ASSERT_TRUE(load_result.ok()) << load_result.error().message;

    auto& image = load_result.value();
    EXPECT_EQ(image.metadata.name, "binary_test");
    EXPECT_EQ(image.network.neuron_count(), net.neuron_count());
    EXPECT_EQ(image.network.synapse_count(), net.synapse_count());
    EXPECT_DOUBLE_EQ(image.metadata.metrics.accuracy, 0.87);
}

TEST_F(SnapshotTestFixture, SaveLoadJson) {
    auto net = make_test_network();
    auto meta = make_test_metadata("json_test");
    std::string path = test_dir + "/json_test.ngmodel.json";

    auto save_result = snapshot::ModelSnapshot::save(
        net, meta, path, snapshot::SnapshotFormat::JSON);
    ASSERT_TRUE(save_result.ok()) << save_result.error().message;
    EXPECT_TRUE(fs::exists(path));

    auto load_result = snapshot::ModelSnapshot::load(path);
    ASSERT_TRUE(load_result.ok()) << load_result.error().message;

    auto& image = load_result.value();
    EXPECT_EQ(image.metadata.name, "json_test");
    EXPECT_EQ(image.network.neuron_count(), net.neuron_count());
    EXPECT_EQ(image.network.synapse_count(), net.synapse_count());
}

TEST_F(SnapshotTestFixture, ReadMetadataOnly) {
    auto net = make_test_network();
    auto meta = make_test_metadata("meta_only");
    std::string path = test_dir + "/meta_only.ngmodel";

    snapshot::ModelSnapshot::save(net, meta, path, snapshot::SnapshotFormat::BINARY);

    auto meta_result = snapshot::ModelSnapshot::read_metadata(path);
    ASSERT_TRUE(meta_result.ok());
    EXPECT_EQ(meta_result.value().name, "meta_only");
    EXPECT_EQ(meta_result.value().neuron_count, net.neuron_count());
    EXPECT_EQ(meta_result.value().task_type, snapshot::TaskType::CLASSIFICATION);
}

TEST_F(SnapshotTestFixture, ReadMetadataJson) {
    auto net = make_test_network();
    auto meta = make_test_metadata("meta_json");
    std::string path = test_dir + "/meta_json.ngmodel.json";

    snapshot::ModelSnapshot::save(net, meta, path, snapshot::SnapshotFormat::JSON);

    auto meta_result = snapshot::ModelSnapshot::read_metadata(path);
    ASSERT_TRUE(meta_result.ok());
    EXPECT_EQ(meta_result.value().name, "meta_json");
}

TEST_F(SnapshotTestFixture, Validate) {
    auto net = make_test_network();
    auto meta = make_test_metadata("validate_test");
    std::string path = test_dir + "/validate_test.ngmodel";

    snapshot::ModelSnapshot::save(net, meta, path, snapshot::SnapshotFormat::BINARY);

    auto val_result = snapshot::ModelSnapshot::validate(path);
    EXPECT_TRUE(val_result.ok());
}

TEST_F(SnapshotTestFixture, LoadNonexistentFile) {
    auto result = snapshot::ModelSnapshot::load(test_dir + "/nonexistent.ngmodel");
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.error().message.find("Cannot open") != std::string::npos);
}

TEST_F(SnapshotTestFixture, AutoDetectFormat) {
    auto net = make_test_network();
    auto meta = make_test_metadata("autodetect");

    // Save as binary
    std::string bin_path = test_dir + "/autodetect.ngmodel";
    snapshot::ModelSnapshot::save(net, meta, bin_path, snapshot::SnapshotFormat::BINARY);

    // Save as JSON
    meta.name = "autodetect_json";
    std::string json_path = test_dir + "/autodetect.ngmodel.json";
    snapshot::ModelSnapshot::save(net, meta, json_path, snapshot::SnapshotFormat::JSON);

    // Load both — format auto-detected
    auto bin_result = snapshot::ModelSnapshot::load(bin_path);
    ASSERT_TRUE(bin_result.ok());
    EXPECT_EQ(bin_result.value().metadata.name, "autodetect");

    auto json_result = snapshot::ModelSnapshot::load(json_path);
    ASSERT_TRUE(json_result.ok());
    EXPECT_EQ(json_result.value().metadata.name, "autodetect_json");
}

TEST_F(SnapshotTestFixture, NetworkIntegrity) {
    // Create network with known structure
    auto net = make_test_network(20, 30);
    auto original_stats = net.stats();

    auto meta = make_test_metadata("integrity");
    std::string path = test_dir + "/integrity.ngmodel";

    snapshot::ModelSnapshot::save(net, meta, path, snapshot::SnapshotFormat::BINARY);
    auto load_result = snapshot::ModelSnapshot::load(path);
    ASSERT_TRUE(load_result.ok());

    auto loaded_stats = load_result.value().network.stats();
    EXPECT_EQ(loaded_stats.neuron_count, original_stats.neuron_count);
    EXPECT_EQ(loaded_stats.synapse_count, original_stats.synapse_count);
    EXPECT_NEAR(loaded_stats.avg_weight, original_stats.avg_weight, 0.001);
}

// ===========================================================================
// SnapshotRegistry Tests
// ===========================================================================

TEST_F(SnapshotTestFixture, RegistrySaveAndScan) {
    snapshot::SnapshotRegistry registry(test_dir);

    auto net = make_test_network();
    auto meta = make_test_metadata("reg_test_1");
    auto save_result = registry.save(net, meta);
    ASSERT_TRUE(save_result.ok());

    meta.name = "reg_test_2";
    meta.task_type = snapshot::TaskType::REGRESSION;
    meta.metrics.accuracy = 0.92;
    registry.save(net, meta);

    EXPECT_EQ(registry.count(), 2);
    EXPECT_TRUE(registry.has("reg_test_1"));
    EXPECT_TRUE(registry.has("reg_test_2"));

    // Rescan from disk
    snapshot::SnapshotRegistry fresh_registry(test_dir);
    auto scan_result = fresh_registry.scan();
    ASSERT_TRUE(scan_result.ok());
    EXPECT_EQ(scan_result.value(), 2);
}

TEST_F(SnapshotTestFixture, RegistryLoad) {
    snapshot::SnapshotRegistry registry(test_dir);

    auto net = make_test_network();
    auto meta = make_test_metadata("loadable");
    registry.save(net, meta);

    auto load_result = registry.load("loadable");
    ASSERT_TRUE(load_result.ok());
    EXPECT_EQ(load_result.value().network.neuron_count(), net.neuron_count());
}

TEST_F(SnapshotTestFixture, RegistryRemove) {
    snapshot::SnapshotRegistry registry(test_dir);

    auto net = make_test_network();
    auto meta = make_test_metadata("removable");
    registry.save(net, meta);
    EXPECT_TRUE(registry.has("removable"));

    auto remove_result = registry.remove("removable");
    EXPECT_TRUE(remove_result.ok());
    EXPECT_FALSE(registry.has("removable"));
    EXPECT_EQ(registry.count(), 0);
}

TEST_F(SnapshotTestFixture, RegistryFindByTask) {
    snapshot::SnapshotRegistry registry(test_dir);
    auto net = make_test_network();

    auto m1 = make_test_metadata("classifier_1");
    m1.task_type = snapshot::TaskType::CLASSIFICATION;
    m1.metrics.accuracy = 0.80;
    registry.save(net, m1);

    auto m2 = make_test_metadata("classifier_2");
    m2.task_type = snapshot::TaskType::CLASSIFICATION;
    m2.metrics.accuracy = 0.95;
    registry.save(net, m2);

    auto m3 = make_test_metadata("regressor_1");
    m3.task_type = snapshot::TaskType::REGRESSION;
    m3.metrics.accuracy = 0.70;
    registry.save(net, m3);

    auto classifiers = registry.find_by_task(snapshot::TaskType::CLASSIFICATION);
    EXPECT_EQ(classifiers.size(), 2);
    // Should be sorted by accuracy descending
    EXPECT_DOUBLE_EQ(classifiers[0]->metadata.metrics.accuracy, 0.95);
    EXPECT_DOUBLE_EQ(classifiers[1]->metadata.metrics.accuracy, 0.80);

    auto regressors = registry.find_by_task(snapshot::TaskType::REGRESSION);
    EXPECT_EQ(regressors.size(), 1);
}

TEST_F(SnapshotTestFixture, RegistryFindByTag) {
    snapshot::SnapshotRegistry registry(test_dir);
    auto net = make_test_network();

    auto m1 = make_test_metadata("tagged_1");
    m1.tags = {"vision", "mnist"};
    registry.save(net, m1);

    auto m2 = make_test_metadata("tagged_2");
    m2.tags = {"nlp", "sentiment"};
    registry.save(net, m2);

    auto vision = registry.find_by_tag("vision");
    EXPECT_EQ(vision.size(), 1);
    EXPECT_EQ(vision[0]->metadata.name, "tagged_1");

    auto nlp = registry.find_by_tag("nlp");
    EXPECT_EQ(nlp.size(), 1);
    EXPECT_EQ(nlp[0]->metadata.name, "tagged_2");

    auto empty = registry.find_by_tag("nonexistent");
    EXPECT_TRUE(empty.empty());
}

TEST_F(SnapshotTestFixture, RegistryFindBest) {
    snapshot::SnapshotRegistry registry(test_dir);
    auto net = make_test_network();

    auto m1 = make_test_metadata("low_acc");
    m1.metrics.accuracy = 0.60;
    registry.save(net, m1);

    auto m2 = make_test_metadata("high_acc");
    m2.metrics.accuracy = 0.98;
    registry.save(net, m2);

    auto best = registry.find_best(snapshot::TaskType::CLASSIFICATION);
    ASSERT_TRUE(best.ok());
    EXPECT_EQ(best.value()->metadata.name, "high_acc");
    EXPECT_DOUBLE_EQ(best.value()->metadata.metrics.accuracy, 0.98);
}

TEST_F(SnapshotTestFixture, RegistryNames) {
    snapshot::SnapshotRegistry registry(test_dir);
    auto net = make_test_network();

    registry.save(net, make_test_metadata("zebra"));
    registry.save(net, make_test_metadata("alpha"));
    registry.save(net, make_test_metadata("beta"));

    auto names = registry.names();
    ASSERT_EQ(names.size(), 3);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "beta");
    EXPECT_EQ(names[2], "zebra");
}

TEST_F(SnapshotTestFixture, RegistryFindIf) {
    snapshot::SnapshotRegistry registry(test_dir);
    auto net = make_test_network();

    auto m1 = make_test_metadata("small");
    m1.training.epochs_completed = 10;
    registry.save(net, m1);

    auto m2 = make_test_metadata("large");
    m2.training.epochs_completed = 500;
    registry.save(net, m2);

    auto heavily_trained = registry.find_if([](const snapshot::RegistryEntry& e) {
        return e.metadata.training.epochs_completed > 100;
    });
    EXPECT_EQ(heavily_trained.size(), 1);
    EXPECT_EQ(heavily_trained[0]->metadata.name, "large");
}

TEST_F(SnapshotTestFixture, RegistryEmptyDirectory) {
    std::string empty_dir = test_dir + "/empty_sub";
    fs::create_directories(empty_dir);

    snapshot::SnapshotRegistry registry(empty_dir);
    auto scan_result = registry.scan();
    ASSERT_TRUE(scan_result.ok());
    EXPECT_EQ(scan_result.value(), 0);
    EXPECT_EQ(registry.count(), 0);
}

TEST_F(SnapshotTestFixture, RegistryNonexistentDirectory) {
    snapshot::SnapshotRegistry registry(test_dir + "/does_not_exist");
    auto scan_result = registry.scan();
    ASSERT_TRUE(scan_result.ok());
    EXPECT_EQ(scan_result.value(), 0);
}
