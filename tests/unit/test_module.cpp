#include <gtest/gtest.h>
#include "neurograph/module/module_interface.hpp"
#include "neurograph/module/module_registry.hpp"
#include "neurograph/core/network.hpp"

using namespace neurograph;

// ============================================================
// Test module implementation (in-process, no dynamic loading)
// ============================================================

class TestModule : public module::ModuleInterface {
public:
    int init_count = 0;
    int shutdown_count = 0;
    int tick_count = 0;
    uint64_t last_tick = 0;

    module::ModuleInfo info() const override {
        return {
            "test_module",
            "1.0.0",
            "A test module for unit testing",
            "NeuroGraph",
            module::ModuleCapability::CUSTOM_COMMANDS |
            module::ModuleCapability::NEURON_PROCESSOR
        };
    }

    util::Result<void> initialize(core::Network& network) override {
        init_count++;
        return util::Result<void>::success();
    }

    util::Result<void> shutdown() override {
        shutdown_count++;
        return util::Result<void>::success();
    }

    void on_tick(core::Network& network, uint64_t tick) override {
        tick_count++;
        last_tick = tick;
    }

    util::Result<nlohmann::json> handle_command(
        const std::string& command_name,
        const nlohmann::json& payload) override {
        if (command_name == "echo") {
            return nlohmann::json{{"echo", payload}};
        }
        return util::Error("Unknown command: " + command_name);
    }

    nlohmann::json status() const override {
        return {{"ticks", tick_count}, {"initialized", init_count > 0}};
    }
};

class FailingModule : public module::ModuleInterface {
public:
    module::ModuleInfo info() const override {
        return {"failing_module", "0.1.0", "Always fails", "", module::ModuleCapability::NONE};
    }

    util::Result<void> initialize(core::Network&) override {
        return util::Error("Initialization failed");
    }

    util::Result<void> shutdown() override {
        return util::Result<void>::success();
    }

    void on_tick(core::Network&, uint64_t) override {}
};

// ============================================================
// ModuleCapability Tests
// ============================================================

TEST(ModuleCapability, BitwiseOr) {
    auto caps = module::ModuleCapability::CUSTOM_COMMANDS |
                module::ModuleCapability::DATA_SOURCE;
    EXPECT_TRUE(module::has_capability(caps, module::ModuleCapability::CUSTOM_COMMANDS));
    EXPECT_TRUE(module::has_capability(caps, module::ModuleCapability::DATA_SOURCE));
    EXPECT_FALSE(module::has_capability(caps, module::ModuleCapability::ANALYZER));
}

TEST(ModuleCapability, None) {
    EXPECT_FALSE(module::has_capability(module::ModuleCapability::NONE,
                                        module::ModuleCapability::CUSTOM_COMMANDS));
}

// ============================================================
// ModuleInfo Tests
// ============================================================

TEST(ModuleInfo, TestModuleInfo) {
    TestModule mod;
    auto info = mod.info();
    EXPECT_EQ(info.name, "test_module");
    EXPECT_EQ(info.version, "1.0.0");
    EXPECT_TRUE(module::has_capability(info.capabilities,
                                       module::ModuleCapability::CUSTOM_COMMANDS));
    EXPECT_TRUE(module::has_capability(info.capabilities,
                                       module::ModuleCapability::NEURON_PROCESSOR));
}

// ============================================================
// ModuleRegistry Tests
// ============================================================

TEST(ModuleRegistry, RegisterAndLookup) {
    module::ModuleRegistry registry;
    auto mod = new TestModule();

    module::ModuleEntry entry;
    entry.info = mod->info();
    entry.instance = mod;
    entry.destroy = [](module::ModuleInterface* m) { delete m; };

    auto result = registry.register_module(std::move(entry));
    ASSERT_TRUE(result.ok());

    EXPECT_TRUE(registry.has_module("test_module"));
    EXPECT_EQ(registry.size(), 1);
    EXPECT_NE(registry.get_module("test_module"), nullptr);
    EXPECT_EQ(registry.get_module("nonexistent"), nullptr);
}

TEST(ModuleRegistry, DuplicateRegistration) {
    module::ModuleRegistry registry;
    auto mod1 = new TestModule();
    auto mod2 = new TestModule();

    module::ModuleEntry entry1;
    entry1.info = mod1->info();
    entry1.instance = mod1;
    entry1.destroy = [](module::ModuleInterface* m) { delete m; };

    module::ModuleEntry entry2;
    entry2.info = mod2->info();
    entry2.instance = mod2;
    entry2.destroy = [](module::ModuleInterface* m) { delete m; };

    ASSERT_TRUE(registry.register_module(std::move(entry1)).ok());
    EXPECT_TRUE(registry.register_module(std::move(entry2)).is_error());
}

TEST(ModuleRegistry, InitializeAll) {
    module::ModuleRegistry registry;
    auto mod = new TestModule();

    module::ModuleEntry entry;
    entry.info = mod->info();
    entry.instance = mod;
    entry.destroy = [](module::ModuleInterface* m) { delete m; };

    registry.register_module(std::move(entry));

    core::Network net;
    auto result = registry.initialize_all(net);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(mod->init_count, 1);
}

TEST(ModuleRegistry, InitializeFailure) {
    module::ModuleRegistry registry;
    auto mod = new FailingModule();

    module::ModuleEntry entry;
    entry.info = mod->info();
    entry.instance = mod;
    entry.destroy = [](module::ModuleInterface* m) { delete m; };

    registry.register_module(std::move(entry));

    core::Network net;
    auto result = registry.initialize_all(net);
    EXPECT_TRUE(result.is_error());
}

TEST(ModuleRegistry, TickAll) {
    module::ModuleRegistry registry;
    auto mod = new TestModule();

    module::ModuleEntry entry;
    entry.info = mod->info();
    entry.instance = mod;
    entry.destroy = [](module::ModuleInterface* m) { delete m; };

    registry.register_module(std::move(entry));

    core::Network net;
    registry.initialize_all(net);
    registry.tick_all(net, 42);

    EXPECT_EQ(mod->tick_count, 1);
    EXPECT_EQ(mod->last_tick, 42);
}

TEST(ModuleRegistry, DispatchCommand) {
    module::ModuleRegistry registry;
    auto mod = new TestModule();

    module::ModuleEntry entry;
    entry.info = mod->info();
    entry.instance = mod;
    entry.destroy = [](module::ModuleInterface* m) { delete m; };

    registry.register_module(std::move(entry));

    core::Network net;
    registry.initialize_all(net);

    auto result = registry.dispatch_command("test_module", "echo",
                                            {{"message", "hello"}});
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value()["echo"]["message"], "hello");
}

TEST(ModuleRegistry, DispatchCommandUnknownModule) {
    module::ModuleRegistry registry;
    auto result = registry.dispatch_command("nonexistent", "echo", {});
    EXPECT_TRUE(result.is_error());
}

TEST(ModuleRegistry, UnregisterModule) {
    module::ModuleRegistry registry;
    auto mod = new TestModule();

    module::ModuleEntry entry;
    entry.info = mod->info();
    entry.instance = mod;
    entry.destroy = [](module::ModuleInterface* m) { delete m; };

    registry.register_module(std::move(entry));

    core::Network net;
    registry.initialize_all(net);

    auto result = registry.unregister_module("test_module");
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(registry.has_module("test_module"));
    EXPECT_EQ(registry.size(), 0);
}

TEST(ModuleRegistry, ModuleNames) {
    module::ModuleRegistry registry;

    auto mod1 = new TestModule();
    module::ModuleEntry entry1;
    entry1.info = mod1->info();
    entry1.instance = mod1;
    entry1.destroy = [](module::ModuleInterface* m) { delete m; };
    registry.register_module(std::move(entry1));

    auto names = registry.module_names();
    EXPECT_EQ(names.size(), 1);
    EXPECT_EQ(names[0], "test_module");
}

TEST(ModuleRegistry, ShutdownAll) {
    auto mod = new TestModule();

    {
        module::ModuleRegistry registry;
        module::ModuleEntry entry;
        entry.info = mod->info();
        entry.instance = mod;
        // Don't auto-delete so we can check shutdown count
        entry.destroy = [](module::ModuleInterface*) {};

        registry.register_module(std::move(entry));

        core::Network net;
        registry.initialize_all(net);
        registry.shutdown_all();

        EXPECT_EQ(mod->shutdown_count, 1);
    }

    delete mod;
}

TEST(ModuleRegistry, ModuleStatus) {
    TestModule mod;
    auto status = mod.status();
    EXPECT_EQ(status["ticks"], 0);
    EXPECT_EQ(status["initialized"], false);
}
