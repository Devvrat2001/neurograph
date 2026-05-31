// Unit tests for TrainingEngine::load_prepared_dataset (P1-18).
//
// Focus: the hidden-layer auto-allocation behavior added in P1-18.
// Smoke-level coverage of the three branches (auto, explicit > 0,
// explicit == 0 for legacy direct-wiring), plus a convergence smoke
// to confirm the Glorot-initialized network with ReLU hidden actually
// learns a linearly separable problem (which it should beat trivially,
// proving the wiring is sane).

#include "neurograph/core/network.hpp"
#include "neurograph/training/training_engine.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using neurograph::core::Network;
using neurograph::training::Sample;
using neurograph::training::SupervisedConfig;
using neurograph::training::TrainingEngine;
using neurograph::training::LossType;

namespace {

// Build a tiny linearly-separable 2-class dataset: 100 samples,
// 3 features, label = sign(x1 + x2 - x3). Returns (train, val) split.
struct ToyDataset {
    std::vector<Sample> train;
    std::vector<Sample> val;
    int num_features = 3;
    int num_classes  = 2;
};

ToyDataset make_toy_classification(std::uint64_t seed = 42) {
    ToyDataset out;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    auto gen_one = [&]() {
        Sample s;
        s.features.resize(3);
        for (int d = 0; d < 3; ++d) s.features[d] = nd(rng);
        const double sgn = s.features[0] + s.features[1] - s.features[2];
        const int cls = (sgn > 0.0) ? 1 : 0;
        // One-hot target.
        s.target = std::vector<double>(2, 0.0);
        s.target[cls] = 1.0;
        s.label = (cls == 1) ? "pos" : "neg";
        return s;
    };

    for (int i = 0; i < 80; ++i) out.train.push_back(gen_one());
    for (int i = 0; i < 20; ++i) out.val.push_back(gen_one());
    return out;
}

} // namespace

// --------------------------------------------------------------------
// 1. Default kAutoHiddenSize picks within [8, 64], formula = 2 * features
// --------------------------------------------------------------------
TEST(TrainingEngine, LoadPreparedAutoSizesHiddenLayer) {
    Network net;
    TrainingEngine engine(net);
    auto ds = make_toy_classification();

    auto r = engine.load_prepared_dataset(
        ds.train, ds.val, ds.num_features, ds.num_classes, {"neg", "pos"},
        /*seed=*/42);   // hidden_layer_size omitted → auto
    ASSERT_TRUE(r.ok()) << r.error().message;

    // 3 features → auto = clamp(6, 8, 64) = 8 hidden neurons.
    // Total = 3 input + 8 hidden + 2 output = 13.
    EXPECT_EQ(net.neuron_count(), 13u);

    // Synapse count = (3*8) + (8*2) = 40 — proves both layers wired.
    EXPECT_EQ(net.synapse_count(), 40u);
}

// --------------------------------------------------------------------
// 2. Explicit hidden_layer_size > 0 honored
// --------------------------------------------------------------------
TEST(TrainingEngine, LoadPreparedExplicitHiddenSize) {
    Network net;
    TrainingEngine engine(net);
    auto ds = make_toy_classification();

    auto r = engine.load_prepared_dataset(
        ds.train, ds.val, ds.num_features, ds.num_classes, {"neg", "pos"},
        /*seed=*/42, /*hidden_layer_size=*/16);
    ASSERT_TRUE(r.ok()) << r.error().message;

    EXPECT_EQ(net.neuron_count(), 3u + 16u + 2u);
    EXPECT_EQ(net.synapse_count(), 3u * 16u + 16u * 2u);
}

// --------------------------------------------------------------------
// 3. hidden_layer_size = 0 reproduces legacy direct-wiring behavior
// --------------------------------------------------------------------
TEST(TrainingEngine, LoadPreparedZeroPreservesDirectWiring) {
    Network net;
    TrainingEngine engine(net);
    auto ds = make_toy_classification();

    auto r = engine.load_prepared_dataset(
        ds.train, ds.val, ds.num_features, ds.num_classes, {"neg", "pos"},
        /*seed=*/42, /*hidden_layer_size=*/0);
    ASSERT_TRUE(r.ok()) << r.error().message;

    // Just input + output, no hidden.
    EXPECT_EQ(net.neuron_count(), 3u + 2u);
    EXPECT_EQ(net.synapse_count(), 3u * 2u);
}

// --------------------------------------------------------------------
// 4. Auto sizing bounds — large feature count caps at 64
// --------------------------------------------------------------------
TEST(TrainingEngine, LoadPreparedAutoCapsAtUpperBound) {
    Network net;
    TrainingEngine engine(net);

    // Synthesize 100 features so 2 * 100 = 200 would exceed the cap.
    std::vector<Sample> train, val;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto gen_one = [&]() {
        Sample s;
        s.features.resize(100);
        for (int d = 0; d < 100; ++d) s.features[d] = nd(rng);
        s.target = {0.5};
        return s;
    };
    for (int i = 0; i < 60; ++i) train.push_back(gen_one());
    for (int i = 0; i < 15; ++i) val.push_back(gen_one());

    auto r = engine.load_prepared_dataset(train, val, 100, 1, {}, 42);
    ASSERT_TRUE(r.ok()) << r.error().message;

    // 100 features + 64 hidden (capped) + 1 output = 165.
    EXPECT_EQ(net.neuron_count(), 165u);
}

// --------------------------------------------------------------------
// 5. Auto sizing floor — single-feature input still gets 8 hidden
// --------------------------------------------------------------------
TEST(TrainingEngine, LoadPreparedAutoRespectsFloor) {
    Network net;
    TrainingEngine engine(net);

    std::vector<Sample> train, val;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto gen_one = [&]() {
        Sample s;
        s.features = {nd(rng)};
        s.target = {0.5};
        return s;
    };
    for (int i = 0; i < 60; ++i) train.push_back(gen_one());
    for (int i = 0; i < 15; ++i) val.push_back(gen_one());

    auto r = engine.load_prepared_dataset(train, val, 1, 1, {}, 42);
    ASSERT_TRUE(r.ok()) << r.error().message;

    // 1 feature → 2 * 1 = 2, clamped up to 8.
    EXPECT_EQ(net.neuron_count(), 1u + 8u + 1u);
}

// --------------------------------------------------------------------
// 6. End-to-end: hidden-layer network actually learns separable data.
//
// This is the convergence smoke that proves the new wiring isn't
// just topologically correct but functionally usable. We assert
// the train accuracy beats trivial baseline (0.5) by a clear
// margin after a modest number of epochs, with cross-entropy + Adam.
// --------------------------------------------------------------------
TEST(TrainingEngine, HiddenLayerNetworkLearnsSeparableData) {
    Network net;
    TrainingEngine engine(net);
    auto ds = make_toy_classification();

    // Default auto-hidden is 8 neurons for 3 features.
    auto load_r = engine.load_prepared_dataset(
        ds.train, ds.val, ds.num_features, ds.num_classes, {"neg", "pos"},
        /*seed=*/42);
    ASSERT_TRUE(load_r.ok()) << load_r.error().message;

    SupervisedConfig cfg;
    cfg.epochs                            = 30;
    cfg.batch_size                        = 16;
    cfg.optimizer_name                    = "adam";
    cfg.optimizer_config.learning_rate    = 0.01;
    cfg.loss_type                         = LossType::MSE;
    cfg.warmup_epochs                     = 3;
    cfg.dropout_rate                      = 0.0;
    cfg.batch_norm                        = false;
    cfg.interleave_self_org               = false;  // P1-16 pending
    cfg.patience                          = 30;     // run all epochs

    auto train_r = engine.train(cfg);
    ASSERT_TRUE(train_r.ok()) << train_r.error().message;

    // Final-epoch train accuracy should be clearly above chance.
    // Loose threshold (0.7) so the test isn't sensitive to small
    // tuning shifts but still fails if wiring regresses to chance.
    ASSERT_FALSE(engine.history().empty());
    EXPECT_GT(engine.history().back().train_accuracy, 0.7)
        << "Hidden-layer network failed to learn linearly-separable data";
}
