/// Integration test for all three phases of NeuroGraph improvements.
/// Tests: activation functions, LR scheduling, dropout, batch norm,
///        augmentation, embeddings, LR finder, metrics export,
///        adaptive topology, and image folder loader.

#include "neurograph/core/network.hpp"
#include "neurograph/core/activation.hpp"
#include "neurograph/core/topology.hpp"
#include "neurograph/training/training_engine.hpp"
#include "neurograph/training/data_loader.hpp"
#include "neurograph/training/augmentation.hpp"
#include "neurograph/training/embedding.hpp"
#include "neurograph/training/metrics_export.hpp"
#include "neurograph/training/optimizer.hpp"
#include "neurograph/training/loss.hpp"
#include "neurograph/training/backprop.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace neurograph;
using namespace neurograph::core;
using namespace neurograph::training;

// ============================================================
// PHASE 1 TESTS: Core Improvements
// ============================================================

TEST(Phase1, NewActivationFunctions) {
    // Test all new activation functions: forward + derivative
    auto test_activation = [](ActivationType type, const std::string& name) {
        auto fn = make_activation(type);
        ASSERT_NE(fn, nullptr) << "Failed to create: " << name;

        // Should not crash on various inputs
        double vals[] = {-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0};
        for (double v : vals) {
            double result = fn->compute(v);
            double deriv = fn->derivative(v);
            EXPECT_TRUE(std::isfinite(result)) << name << " forward NaN at " << v;
            EXPECT_TRUE(std::isfinite(deriv)) << name << " derivative NaN at " << v;
        }
    };

    test_activation(ActivationType::LEAKY_RELU, "LeakyReLU");
    test_activation(ActivationType::GELU, "GELU");
    test_activation(ActivationType::SWISH, "Swish");
    test_activation(ActivationType::ELU, "ELU");

    // Verify specific behaviors
    auto gelu = make_activation(ActivationType::GELU);
    EXPECT_NEAR(gelu->compute(0.0), 0.0, 1e-6);  // GELU(0) = 0

    auto leaky = make_activation(ActivationType::LEAKY_RELU);
    EXPECT_EQ(leaky->compute(1.0), 1.0);
    EXPECT_NEAR(leaky->compute(-1.0), -0.01, 1e-6);

    auto swish = make_activation(ActivationType::SWISH);
    EXPECT_NEAR(swish->compute(0.0), 0.0, 1e-6);  // Swish(0) = 0

    auto elu = make_activation(ActivationType::ELU);
    EXPECT_EQ(elu->compute(1.0), 1.0);
    EXPECT_LT(elu->compute(-1.0), 0.0);
}

TEST(Phase1, ActivationDeviceFunctions) {
    // Test CUDA-compatible (but host-callable) device functions
    EXPECT_NEAR(device_activate(0.0f, ActivationType::GELU), 0.0f, 1e-4f);
    EXPECT_NEAR(device_activate(1.0f, ActivationType::LEAKY_RELU), 1.0f, 1e-4f);
    EXPECT_NEAR(device_activate(-1.0f, ActivationType::LEAKY_RELU), -0.01f, 1e-4f);
    EXPECT_NEAR(device_activate(0.0f, ActivationType::SWISH), 0.0f, 1e-4f);
    EXPECT_EQ(device_activate(1.0f, ActivationType::ELU), 1.0f);

    // Derivatives
    EXPECT_TRUE(std::isfinite(device_activate_derivative(0.5f, ActivationType::GELU)));
    EXPECT_TRUE(std::isfinite(device_activate_derivative(0.5f, ActivationType::SWISH)));
    EXPECT_TRUE(std::isfinite(device_activate_derivative(-0.5f, ActivationType::ELU)));
}

TEST(Phase1, ActivationStringParsing) {
    EXPECT_EQ(parse_activation_type("gelu"), ActivationType::GELU);
    EXPECT_EQ(parse_activation_type("swish"), ActivationType::SWISH);
    EXPECT_EQ(parse_activation_type("silu"), ActivationType::SWISH);
    EXPECT_EQ(parse_activation_type("leaky_relu"), ActivationType::LEAKY_RELU);
    EXPECT_EQ(parse_activation_type("elu"), ActivationType::ELU);
    // P1-17: LINEAR with both names.
    EXPECT_EQ(parse_activation_type("linear"),   ActivationType::LINEAR);
    EXPECT_EQ(parse_activation_type("identity"), ActivationType::LINEAR);

    EXPECT_EQ(activation_type_str(ActivationType::GELU), "gelu");
    EXPECT_EQ(activation_type_str(ActivationType::SWISH), "swish");
    EXPECT_EQ(activation_type_str(ActivationType::LINEAR), "linear");
}

// P1-17: LINEAR activation. Identity forward, derivative always 1.
// The whole point is gradient flow without saturation — verify both
// the host and device paths and the factory all agree.
TEST(Phase1, LinearActivationIsIdentity) {
    auto fn = make_activation(ActivationType::LINEAR);
    ASSERT_NE(fn, nullptr);
    EXPECT_EQ(fn->name(), "linear");
    EXPECT_EQ(fn->type(), ActivationType::LINEAR);

    const double sample[] = {-1e6, -3.14, -0.001, 0.0, 0.001, 1.0, 42.0, 1e6};
    for (double x : sample) {
        EXPECT_DOUBLE_EQ(fn->compute(x), x)
            << "LINEAR forward not identity at x=" << x;
        EXPECT_DOUBLE_EQ(fn->derivative(x), 1.0)
            << "LINEAR derivative not 1 at x=" << x;
    }

    // Device path same contract — gradient flow through CUDA kernel
    // depends on this matching the CPU path bit-for-bit (within float
    // precision, since device_activate uses float not double).
    for (float x : {-7.5f, 0.0f, 0.123f, 99.0f}) {
        EXPECT_FLOAT_EQ(device_activate(x, ActivationType::LINEAR), x);
        EXPECT_FLOAT_EQ(device_activate_derivative(x, ActivationType::LINEAR), 1.0f);
    }
}

TEST(Phase1, LRSchedules) {
    // Constant schedule
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01;
    cfg.schedule_name = "constant";
    cfg.warmup_epochs = 0;
    auto sched = make_schedule(cfg);
    ASSERT_NE(sched, nullptr);
    EXPECT_NEAR(sched->compute_lr(0.01, 0, 100), 0.01, 1e-8);
    EXPECT_NEAR(sched->compute_lr(0.01, 50, 100), 0.01, 1e-8);

    // Cosine schedule
    cfg.schedule_name = "cosine";
    cfg.min_lr = 0.0;
    sched = make_schedule(cfg);
    ASSERT_NE(sched, nullptr);
    double lr_start = sched->compute_lr(0.01, 0, 100);
    double lr_mid = sched->compute_lr(0.01, 50, 100);
    double lr_end = sched->compute_lr(0.01, 99, 100);
    EXPECT_NEAR(lr_start, 0.01, 1e-6);  // full LR at start
    EXPECT_LT(lr_mid, 0.01);             // decayed at midpoint
    EXPECT_GT(lr_mid, 0.0);
    EXPECT_NEAR(lr_end, 0.0, 1e-4);     // near zero at end

    // Exponential schedule
    cfg.schedule_name = "exponential";
    cfg.decay_factor = 0.9;
    sched = make_schedule(cfg);
    ASSERT_NE(sched, nullptr);
    double lr_e0 = sched->compute_lr(0.01, 0, 100);
    double lr_e1 = sched->compute_lr(0.01, 1, 100);
    double lr_e10 = sched->compute_lr(0.01, 10, 100);
    EXPECT_NEAR(lr_e0, 0.01, 1e-8);
    EXPECT_NEAR(lr_e1, 0.009, 1e-6);     // 0.01 * 0.9^1
    EXPECT_NEAR(lr_e10, 0.01 * std::pow(0.9, 10), 1e-6);
}

TEST(Phase1, WarmupSchedule) {
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01;
    cfg.schedule_name = "cosine";
    cfg.warmup_epochs = 10;
    cfg.min_lr = 0.0;
    auto sched = make_schedule(cfg);
    ASSERT_NE(sched, nullptr);

    // During warmup: should ramp linearly from initial_lr/warmup to initial_lr
    // Formula: initial_lr * (epoch + 1) / warmup_epochs
    double lr_0 = sched->compute_lr(0.01, 0, 100);
    double lr_5 = sched->compute_lr(0.01, 5, 100);
    double lr_10 = sched->compute_lr(0.01, 10, 100);
    EXPECT_NEAR(lr_0, 0.001, 1e-6);        // epoch 0: 0.01 * 1/10
    EXPECT_NEAR(lr_5, 0.006, 1e-3);        // epoch 5: 0.01 * 6/10
    EXPECT_GT(lr_10, 0.008);                // near full LR after warmup
}

TEST(Phase1, DropoutInBackprop) {
    Network net;
    // Create a simple 2->2->1 network
    auto in1 = NeuronId::generate(), in2 = NeuronId::generate();
    auto h1 = NeuronId::generate(), h2 = NeuronId::generate();
    auto out1 = NeuronId::generate();

    net.add_neuron(Neuron(in1, "input_0"));
    net.add_neuron(Neuron(in2, "input_1"));
    net.add_neuron(Neuron(h1, "hidden_0"));
    net.add_neuron(Neuron(h2, "hidden_1"));
    net.add_neuron(Neuron(out1, "output_0"));

    net.connect(in1, h1, 0.5); net.connect(in1, h2, 0.3);
    net.connect(in2, h1, 0.4); net.connect(in2, h2, 0.6);
    net.connect(h1, out1, 0.7); net.connect(h2, out1, 0.8);

    IOManager io(net);
    io.ensure_io_neurons(0, 0); // IDs already exist
    // Manually set up IO
    // Use the backprop with dropout
    Backprop bp(net, io);
    bp.set_dropout_rate(0.5);
    bp.set_training_mode(true);

    // Run forward multiple times; with 50% dropout, outputs should vary
    io.ensure_input_neurons(2);
    io.ensure_output_neurons(1);
    // Can't easily test stochastic behavior deterministically,
    // but verify it doesn't crash
    EXPECT_NO_THROW(bp.rebuild_topo_order());
}

TEST(Phase1, BatchNormInBackprop) {
    Network net;
    auto in1 = NeuronId::generate();
    auto h1 = NeuronId::generate();
    auto out1 = NeuronId::generate();

    net.add_neuron(Neuron(in1, "input_0"));
    net.add_neuron(Neuron(h1, "hidden_0"));
    net.add_neuron(Neuron(out1, "output_0"));
    net.connect(in1, h1, 0.5);
    net.connect(h1, out1, 0.7);

    IOManager io(net);
    io.ensure_input_neurons(0);
    io.ensure_output_neurons(0);

    Backprop bp(net, io);
    bp.set_batch_norm(true);
    bp.set_training_mode(true);
    EXPECT_NO_THROW(bp.rebuild_topo_order());
}

TEST(Phase1, OptimizerSetLR) {
    OptimizerConfig cfg;
    cfg.learning_rate = 0.001;
    cfg.schedule_name = "constant";
    auto opt = make_optimizer("adam", cfg);
    ASSERT_NE(opt, nullptr);

    opt->set_learning_rate(0.05);
    // Verify it doesn't crash
    EXPECT_NO_THROW(opt->update_lr(0, 100));
}

TEST(Phase1, AdaptiveTopologyConfig) {
    TopologyConfig config;
    // Verify new defaults
    EXPECT_TRUE(config.enable_neuron_pruning);
    EXPECT_TRUE(config.enable_neuron_growth);
    EXPECT_TRUE(config.enable_dynamic_rewiring);
    EXPECT_TRUE(config.use_stall_counter);   // new: stall counter on by default
    EXPECT_NEAR(config.growth_split_ratio, 0.3, 1e-6);
    EXPECT_EQ(config.growth_min_splits, 2);
    EXPECT_EQ(config.growth_patience, 10);
}

TEST(Phase1, SplitPreservesOutputConnections) {
    // Regression: when a hidden neuron is split, the clone should inherit
    // connections to the OUTPUT layer too, not just the input layer.
    // Previously the clone received only half the outgoing synapses, which
    // stranded clones (0 outputs) when the parent had few downstream links.
    Network net;
    TopologyConfig cfg;
    cfg.use_stall_counter = false;
    cfg.enable_neuron_pruning = false;
    cfg.enable_dynamic_rewiring = false;
    cfg.growth_min_splits = 1;
    cfg.growth_split_ratio = 1.0;  // split every stressed neuron
    net.set_topology_config(cfg);

    auto i1 = NeuronId::generate();
    auto h1 = NeuronId::generate();
    auto o1 = NeuronId::generate();
    net.add_neuron(Neuron(i1, "input_0"));
    net.add_neuron(Neuron(h1, "hidden_0"));
    net.add_neuron(Neuron(o1, "output_0"));
    net.set_protected_neurons({i1, o1});  // h1 is the only splittable

    net.connect(i1, h1, 0.5);
    net.connect(h1, o1, 0.7);

    // Force h1 to look stressed: push high-variance activation history
    // via activate() (the public API that populates activation_history_).
    auto* hidden = net.get_neuron(h1);
    ASSERT_NE(hidden, nullptr);
    for (int i = 0; i < 12; ++i) {
        hidden->activate(i % 2 == 0 ? 3.0 : -3.0);  // saturates sigmoid either way
    }

    net.self_organize();

    // Find any clone of h1 (hidden neurons other than h1)
    std::vector<NeuronId> clones;
    for (auto& [nid, neuron] : net.neurons()) {
        if (nid == i1 || nid == o1 || nid == h1) continue;
        clones.push_back(nid);
    }
    ASSERT_FALSE(clones.empty()) << "Expected at least one clone after split";

    // Every clone must have a path to o1 (i.e., at least one outgoing synapse
    // targeting the output layer).
    for (auto& cid : clones) {
        auto* clone = net.get_neuron(cid);
        ASSERT_NE(clone, nullptr);
        bool reaches_output = false;
        for (auto& sid : clone->outgoing_synapses()) {
            auto* syn = net.get_synapse(sid);
            if (syn && syn->target_id() == o1) { reaches_output = true; break; }
        }
        EXPECT_TRUE(reaches_output)
            << "Clone " << cid.value() << " is not connected to output layer";
    }
}

TEST(Phase1, StallCounterCanBeDisabled) {
    // When use_stall_counter=false, grow_neurons runs even without any
    // reported loss. Seeding hidden neurons from a 2-input/2-output network
    // should succeed on the first self-org cycle.
    Network net;
    TopologyConfig cfg;
    cfg.use_stall_counter = false;
    cfg.enable_neuron_pruning = false;  // don't immediately prune what we seed
    cfg.enable_dynamic_rewiring = false;
    net.set_topology_config(cfg);

    auto i1 = NeuronId::generate(), i2 = NeuronId::generate();
    auto o1 = NeuronId::generate(), o2 = NeuronId::generate();
    net.add_neuron(Neuron(i1, "input_0"));
    net.add_neuron(Neuron(i2, "input_1"));
    net.add_neuron(Neuron(o1, "output_0"));
    net.add_neuron(Neuron(o2, "output_1"));
    net.set_protected_neurons({i1, i2, o1, o2});

    // No loss ever reported — with stall counter this would block growth.
    size_t before = net.neuron_count();
    net.self_organize();  // should still seed hidden neurons
    EXPECT_GT(net.neuron_count(), before)
        << "grow_neurons should run when use_stall_counter=false "
        << "even with no loss reported";
}

TEST(Phase1, ProtectedNeurons) {
    Network net;
    auto n1 = NeuronId::generate();
    auto n2 = NeuronId::generate();
    net.add_neuron(Neuron(n1, "input_0"));
    net.add_neuron(Neuron(n2, "output_0"));

    std::unordered_set<NeuronId> protected_set = {n1, n2};
    net.set_protected_neurons(protected_set);
    EXPECT_EQ(net.protected_neurons().size(), 2);
    EXPECT_TRUE(net.protected_neurons().count(n1));
}

TEST(Phase1, LossReporting) {
    Network net;
    net.report_training_loss(2.0);
    net.report_training_loss(1.5);
    net.report_training_loss(1.0);
    EXPECT_NEAR(net.adaptive_stats().best_loss, 1.0, 0.1);
}

// ============================================================
// PHASE 2 TESTS: New Data Types
// ============================================================

TEST(Phase2, AugmentorConfig) {
    AugmentationConfig cfg;
    EXPECT_FALSE(cfg.enabled);
    EXPECT_TRUE(cfg.random_rotation);
    EXPECT_TRUE(cfg.random_shift);
    EXPECT_TRUE(cfg.random_noise);
    EXPECT_FALSE(cfg.horizontal_flip);
    EXPECT_EQ(cfg.image_width, 28);
    EXPECT_EQ(cfg.image_height, 28);
}

TEST(Phase2, AugmentorNoise) {
    AugmentationConfig cfg;
    cfg.enabled = true;
    cfg.random_rotation = false;
    cfg.random_shift = false;
    cfg.horizontal_flip = false;
    cfg.random_noise = true;
    cfg.noise_stddev = 0.1;
    cfg.image_width = 4;
    cfg.image_height = 4;

    Augmentor aug(cfg);
    Sample s;
    s.features.resize(16, 0.5);
    s.target = std::vector<double>{1.0, 0.0};
    s.label = "0";

    Sample augmented = aug.augment(s);
    EXPECT_EQ(augmented.features.size(), 16u);
    EXPECT_EQ(augmented.target, s.target);
    EXPECT_EQ(augmented.label, s.label);

    // At least some features should have changed due to noise
    bool any_different = false;
    for (size_t i = 0; i < 16; ++i) {
        if (std::abs(augmented.features[i] - 0.5) > 1e-10) {
            any_different = true;
            break;
        }
        // All values should be clamped to [0, 1]
        EXPECT_GE(augmented.features[i], 0.0);
        EXPECT_LE(augmented.features[i], 1.0);
    }
    EXPECT_TRUE(any_different) << "Noise augmentation should change at least one pixel";
}

TEST(Phase2, AugmentorRotation) {
    AugmentationConfig cfg;
    cfg.enabled = true;
    cfg.random_rotation = true;
    cfg.random_shift = false;
    cfg.horizontal_flip = false;
    cfg.random_noise = false;
    cfg.image_width = 4;
    cfg.image_height = 4;

    Augmentor aug(cfg);
    Sample s;
    s.features.resize(16, 0.0);
    s.features[5] = 1.0;  // set one pixel
    s.target = std::vector<double>{1.0};
    s.label = "0";

    // Rotation should redistribute the pixel
    Sample augmented = aug.augment(s);
    EXPECT_EQ(augmented.features.size(), 16u);
}

TEST(Phase2, AugmentorShift) {
    AugmentationConfig cfg;
    cfg.enabled = true;
    cfg.random_rotation = false;
    cfg.random_shift = true;
    cfg.horizontal_flip = false;
    cfg.random_noise = false;
    cfg.image_width = 4;
    cfg.image_height = 4;

    Augmentor aug(cfg);
    Sample s;
    s.features = {
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0
    };
    s.target = std::vector<double>{1.0};
    s.label = "0";

    Sample augmented = aug.augment(s);
    EXPECT_EQ(augmented.features.size(), 16u);
    // Sum should be preserved (shift just moves pixels)
    // (approximately, since pixels shifted out are lost)
}

TEST(Phase2, EmbeddingTableCreation) {
    EmbeddingTable emb(100, 16);
    EXPECT_EQ(emb.vocab_size(), 100);
    EXPECT_EQ(emb.embed_dim(), 16);

    auto vec = emb.lookup(0);
    EXPECT_EQ(vec.size(), 16u);

    // Values should be Xavier-initialized (not all zero)
    bool any_nonzero = false;
    for (double v : vec) {
        if (std::abs(v) > 1e-10) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(Phase2, EmbeddingLookupBounds) {
    EmbeddingTable emb(10, 8);

    // Valid lookups
    auto v0 = emb.lookup(0);
    auto v9 = emb.lookup(9);
    EXPECT_EQ(v0.size(), 8u);
    EXPECT_EQ(v9.size(), 8u);

    // Out of bounds should return zeros
    auto vbad = emb.lookup(100);
    EXPECT_EQ(vbad.size(), 8u);
    for (double v : vbad) EXPECT_EQ(v, 0.0);

    auto vneg = emb.lookup(-1);
    EXPECT_EQ(vneg.size(), 8u);
}

TEST(Phase2, EmbeddingUpdate) {
    EmbeddingTable emb(10, 4);
    auto before = emb.lookup(3);

    std::vector<double> grad = {1.0, 0.5, -0.5, -1.0};
    emb.update(3, grad, 0.1);

    auto after = emb.lookup(3);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(after[i], before[i] - 0.1 * grad[i], 1e-10);
    }
}

TEST(Phase2, EmbeddingConfig) {
    EmbeddingConfig cfg;
    EXPECT_FALSE(cfg.enabled);
    EXPECT_EQ(cfg.embed_dim, 16);
    EXPECT_TRUE(cfg.categorical_columns.empty());
}

TEST(Phase2, DataLoaderImageResize) {
    // Test the resize function
    std::vector<double> src = {
        0.0, 1.0,
        1.0, 0.0
    };
    auto resized = DataLoader::resize_image(src, 2, 2, 4, 4);
    EXPECT_EQ(resized.size(), 16u);
    // Top-left quadrant should be ~0.0 (nearest neighbor from src[0,0])
    EXPECT_NEAR(resized[0], 0.0, 1e-6);
    // Top-right quadrant should be ~1.0 (nearest neighbor from src[0,1])
    EXPECT_NEAR(resized[3], 1.0, 1e-6);
}

TEST(Phase2, MNISTLoaderExists) {
    // Verify MNIST files exist from earlier download
    std::string img_path = "C:/Users/maind/OneDrive/Documents/project/neurograph/datasets/mnist/train-images-idx3-ubyte";
    std::string lbl_path = "C:/Users/maind/OneDrive/Documents/project/neurograph/datasets/mnist/train-labels-idx1-ubyte";

    if (std::filesystem::exists(img_path) && std::filesystem::exists(lbl_path)) {
        auto result = DataLoader::load_mnist_idx(img_path, lbl_path, 100);
        ASSERT_TRUE(result.ok()) << "MNIST load failed";
        auto& data = result.value();
        EXPECT_EQ(data.size(), 100u);
        EXPECT_EQ(data[0].features.size(), 784u);
        EXPECT_EQ(data[0].target.size(), 10u);

        // Pixel values should be normalized [0,1]
        for (double v : data[0].features) {
            EXPECT_GE(v, 0.0);
            EXPECT_LE(v, 1.0);
        }
    } else {
        GTEST_SKIP() << "MNIST files not found, skipping";
    }
}

// ============================================================
// PHASE 3 TESTS: Production Quality
// ============================================================

TEST(Phase3, LRFinderSmallNetwork) {
    // Create a tiny network and dataset for LR finder
    Network net;
    TrainingEngine engine(net);

    // Load iris dataset
    std::string iris_path = "C:/Users/maind/OneDrive/Documents/project/neurograph/datasets/iris.csv";
    if (!std::filesystem::exists(iris_path)) {
        GTEST_SKIP() << "Iris dataset not found";
    }

    auto result = engine.load_dataset(iris_path, 1, true);
    ASSERT_TRUE(result.ok()) << result.error().message;

    auto lr_result = engine.find_learning_rate(1e-6, 0.1, 30);
    EXPECT_FALSE(lr_result.learning_rates.empty());
    EXPECT_FALSE(lr_result.losses.empty());
    EXPECT_EQ(lr_result.learning_rates.size(), lr_result.losses.size());
    EXPECT_GT(lr_result.suggested_lr, 0.0);
    EXPECT_LT(lr_result.suggested_lr, 1.0);

    // Verify network weights are restored (not corrupted)
    EXPECT_GT(net.synapse_count(), 0u);
}

TEST(Phase3, MetricsExportCSV) {
    std::vector<EpochResult> history;
    for (int i = 0; i < 5; ++i) {
        EpochResult r;
        r.epoch = i;
        r.train_loss = 2.0 - i * 0.3;
        r.train_accuracy = 0.5 + i * 0.1;
        r.val_loss = 2.1 - i * 0.25;
        r.val_accuracy = 0.48 + i * 0.1;
        r.epoch_duration_ms = 100.0 + i;
        history.push_back(r);
    }

    std::string path = "C:/Users/maind/OneDrive/Documents/project/neurograph/build/test_metrics.csv";
    EXPECT_NO_THROW(MetricsExporter::export_csv(path, history));
    EXPECT_TRUE(std::filesystem::exists(path));

    // Read and verify contents
    std::ifstream f(path);
    std::string header;
    std::getline(f, header);
    EXPECT_NE(header.find("epoch"), std::string::npos);
    EXPECT_NE(header.find("train_loss"), std::string::npos);
    EXPECT_NE(header.find("val_accuracy"), std::string::npos);

    // Verify 5 data rows exist
    int line_count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) line_count++;
    }
    EXPECT_EQ(line_count, 5);

    f.close();
    std::filesystem::remove(path);
}

TEST(Phase3, MetricsExportJSON) {
    std::vector<EpochResult> history;
    EpochResult r;
    r.epoch = 0; r.train_loss = 1.5; r.train_accuracy = 0.8;
    r.val_loss = 1.6; r.val_accuracy = 0.78; r.epoch_duration_ms = 50;
    history.push_back(r);

    std::string path = "C:/Users/maind/OneDrive/Documents/project/neurograph/build/test_metrics.json";
    EXPECT_NO_THROW(MetricsExporter::export_json(path, history, "test_model", "test_data"));
    EXPECT_TRUE(std::filesystem::exists(path));

    // Read and verify it's valid JSON-ish
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("test_model"), std::string::npos);
    EXPECT_NE(content.find("test_data"), std::string::npos);
    EXPECT_NE(content.find("best_val_accuracy"), std::string::npos);
    EXPECT_NE(content.find("history"), std::string::npos);

    f.close();
    std::filesystem::remove(path);
}

TEST(Phase3, MetricsTimestampedName) {
    auto name = MetricsExporter::timestamped_name();
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name.find("metrics_"), std::string::npos);
}

// ============================================================
// INTEGRATION: End-to-End Training with All Features
// ============================================================

TEST(Integration, IrisTrainingWithAllFeatures) {
    std::string iris_path = "C:/Users/maind/OneDrive/Documents/project/neurograph/datasets/iris.csv";
    if (!std::filesystem::exists(iris_path)) {
        GTEST_SKIP() << "Iris dataset not found";
    }

    Network net;
    TrainingEngine engine(net);

    auto load_result = engine.load_dataset(iris_path, 1, true);
    ASSERT_TRUE(load_result.ok()) << load_result.error().message;

    SupervisedConfig config;
    config.epochs = 20;
    config.batch_size = 16;
    config.optimizer_name = "adam";
    config.optimizer_config.learning_rate = 0.01;
    config.loss_type = LossType::CROSS_ENTROPY;

    // Phase 1 features
    config.schedule_name = "cosine";
    config.warmup_epochs = 3;
    config.dropout_rate = 0.1;
    config.batch_norm = true;

    // Phase 2 features (augmentation off for tabular data)
    config.augmentation.enabled = false;

    // Self-organization
    config.interleave_self_org = true;
    config.self_org_interval = 5;

    // Early stopping
    config.patience = 10;

    std::vector<EpochResult> results;
    auto train_result = engine.train(config, [&](const EpochResult& r) {
        results.push_back(r);
    });
    ASSERT_TRUE(train_result.ok()) << train_result.error().message;

    EXPECT_FALSE(results.empty());
    EXPECT_GT(results.back().train_accuracy, 0.3);  // should learn something

    // Verify network has neurons and synapses
    EXPECT_GT(net.neuron_count(), 0u);
    EXPECT_GT(net.synapse_count(), 0u);

    // Test inference
    auto prediction = engine.predict({5.1, 3.5, 1.4, 0.2});
    EXPECT_EQ(prediction.size(), 3u);  // 3 Iris classes

    int predicted_class = engine.predict_class({5.1, 3.5, 1.4, 0.2});
    EXPECT_GE(predicted_class, 0);
    EXPECT_LT(predicted_class, 3);
}

TEST(Integration, MNISTSmallTraining) {
    std::string img_path = "C:/Users/maind/OneDrive/Documents/project/neurograph/datasets/mnist/train-images-idx3-ubyte";
    std::string lbl_path = "C:/Users/maind/OneDrive/Documents/project/neurograph/datasets/mnist/train-labels-idx1-ubyte";

    if (!std::filesystem::exists(img_path) || !std::filesystem::exists(lbl_path)) {
        GTEST_SKIP() << "MNIST files not found";
    }

    Network net;
    TrainingEngine engine(net);

    // Load just 200 samples for speed
    auto load_result = engine.load_mnist(img_path, lbl_path, 200);
    ASSERT_TRUE(load_result.ok()) << load_result.error().message;
    EXPECT_EQ(engine.num_features(), 784);
    EXPECT_EQ(engine.num_outputs(), 10);

    SupervisedConfig config;
    config.epochs = 5;
    config.batch_size = 32;
    config.optimizer_name = "adam";
    config.optimizer_config.learning_rate = 0.001;
    config.loss_type = LossType::CROSS_ENTROPY;
    config.schedule_name = "cosine";
    config.warmup_epochs = 1;
    config.dropout_rate = 0.1;
    config.batch_norm = true;

    // Enable augmentation for images
    config.augmentation.enabled = true;
    config.augmentation.image_width = 28;
    config.augmentation.image_height = 28;

    std::vector<EpochResult> results;
    auto train_result = engine.train(config, [&](const EpochResult& r) {
        results.push_back(r);
    });
    ASSERT_TRUE(train_result.ok()) << train_result.error().message;
    EXPECT_EQ(results.size(), 5u);

    // Loss should decrease over epochs
    EXPECT_LT(results.back().train_loss, results.front().train_loss);
}
