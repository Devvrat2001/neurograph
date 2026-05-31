#include "neurograph/persistence/sync_manager.hpp"
#include "neurograph/util/logger.hpp"

namespace neurograph::persistence {

SyncManager::SyncManager(core::Network& network, GraphRepository& repo)
    : network_(network), repo_(repo) {}

SyncManager::~SyncManager() {
    stop_periodic_sync();
}

// ============================================================
// Dirty Tracking
// ============================================================

void SyncManager::mark_neuron_dirty(const core::NeuronId& id) {
    std::lock_guard lock(mutex_);
    dirty_neurons_.insert(id.value());
}

void SyncManager::mark_synapse_dirty(const core::SynapseId& id) {
    std::lock_guard lock(mutex_);
    dirty_synapses_.insert(id.value());
}

void SyncManager::mark_neuron_deleted(const core::NeuronId& id) {
    std::lock_guard lock(mutex_);
    deleted_neurons_.insert(id.value());
    dirty_neurons_.erase(id.value()); // no need to save a deleted neuron
}

void SyncManager::mark_synapse_deleted(const core::SynapseId& id) {
    std::lock_guard lock(mutex_);
    deleted_synapses_.insert(id.value());
    dirty_synapses_.erase(id.value());
}

void SyncManager::mark_all_dirty() {
    std::lock_guard lock(mutex_);
    for (const auto& [id, neuron] : network_.neurons()) {
        dirty_neurons_.insert(id.value());
    }
    for (const auto& [id, synapse] : network_.synapses()) {
        dirty_synapses_.insert(id.value());
    }
}

size_t SyncManager::pending_count() const {
    std::lock_guard lock(mutex_);
    return dirty_neurons_.size() + dirty_synapses_.size() +
           deleted_neurons_.size() + deleted_synapses_.size();
}

// ============================================================
// Sync Operations
// ============================================================

util::Result<void> SyncManager::sync_to_db() {
    std::unordered_set<std::string> neurons_to_save;
    std::unordered_set<std::string> synapses_to_save;
    std::unordered_set<std::string> neurons_to_delete;
    std::unordered_set<std::string> synapses_to_delete;

    // Snapshot the dirty sets under lock
    {
        std::lock_guard lock(mutex_);
        neurons_to_save = dirty_neurons_;
        synapses_to_save = dirty_synapses_;
        neurons_to_delete = deleted_neurons_;
        synapses_to_delete = deleted_synapses_;
    }

    bool has_errors = false;

    // 1. Delete removed synapses
    for (const auto& sid : synapses_to_delete) {
        auto result = repo_.delete_synapse(core::SynapseId(sid));
        if (result.is_error()) {
            NG_LOG_WARN("Failed to delete synapse {}: {}", sid, result.error().message);
            has_errors = true;
        } else {
            std::lock_guard lock(mutex_);
            deleted_synapses_.erase(sid);
            stats_.synapses_deleted++;
        }
    }

    // 2. Delete removed neurons
    for (const auto& nid : neurons_to_delete) {
        auto result = repo_.delete_neuron(core::NeuronId(nid));
        if (result.is_error()) {
            NG_LOG_WARN("Failed to delete neuron {}: {}", nid, result.error().message);
            has_errors = true;
        } else {
            std::lock_guard lock(mutex_);
            deleted_neurons_.erase(nid);
            stats_.neurons_deleted++;
        }
    }

    // 3. Save dirty neurons in batch
    if (!neurons_to_save.empty()) {
        std::vector<const core::Neuron*> batch;
        batch.reserve(neurons_to_save.size());
        for (const auto& nid : neurons_to_save) {
            const auto* neuron = network_.get_neuron(core::NeuronId(nid));
            if (neuron) batch.push_back(neuron);
        }

        auto result = repo_.save_neurons_batch(batch);
        if (result.is_error()) {
            NG_LOG_ERROR("Batch neuron save failed: {}", result.error().message);
            has_errors = true;
        } else {
            std::lock_guard lock(mutex_);
            for (const auto& nid : neurons_to_save) {
                dirty_neurons_.erase(nid);
            }
            stats_.neurons_synced += batch.size();
        }
    }

    // 4. Save dirty synapses in batch
    if (!synapses_to_save.empty()) {
        std::vector<const core::Synapse*> batch;
        batch.reserve(synapses_to_save.size());
        for (const auto& sid : synapses_to_save) {
            const auto* synapse = network_.get_synapse(core::SynapseId(sid));
            if (synapse) batch.push_back(synapse);
        }

        auto result = repo_.save_synapses_batch(batch);
        if (result.is_error()) {
            NG_LOG_ERROR("Batch synapse save failed: {}", result.error().message);
            has_errors = true;
        } else {
            std::lock_guard lock(mutex_);
            for (const auto& sid : synapses_to_save) {
                dirty_synapses_.erase(sid);
            }
            stats_.synapses_synced += batch.size();
        }
    }

    stats_.total_syncs++;
    if (has_errors) stats_.errors++;

    NG_LOG_INFO("Sync to DB complete: {} neurons, {} synapses saved; {} neurons, {} synapses deleted",
                neurons_to_save.size(), synapses_to_save.size(),
                neurons_to_delete.size(), synapses_to_delete.size());

    return has_errors ? util::Result<void>(util::Error("Some sync operations failed"))
                      : util::Result<void>::success();
}

util::Result<void> SyncManager::sync_from_db() {
    auto result = repo_.load_network();
    if (result.is_error()) {
        return util::Error("Failed to load network from DB: " + result.error().message);
    }

    // Replace the in-memory network
    // Note: This is destructive - clears all in-memory state
    auto loaded = std::move(result).value();

    // Clear current network by removing all neurons (which cascades to synapses)
    std::vector<core::NeuronId> current_ids;
    for (const auto& [id, _] : network_.neurons()) {
        current_ids.push_back(id);
    }
    for (const auto& id : current_ids) {
        network_.remove_neuron(id);
    }

    // Add loaded neurons and synapses
    // We need the JSON round-trip since Network doesn't have a bulk replace
    auto j = loaded.to_json();
    for (const auto& nj : j["neurons"]) {
        network_.add_neuron(core::Neuron::from_json(nj));
    }
    for (const auto& sj : j["synapses"]) {
        network_.add_synapse(core::Synapse::from_json(sj));
    }

    // Clear dirty sets
    {
        std::lock_guard lock(mutex_);
        dirty_neurons_.clear();
        dirty_synapses_.clear();
        deleted_neurons_.clear();
        deleted_synapses_.clear();
    }

    NG_LOG_INFO("Synced from DB: {} neurons, {} synapses loaded",
                network_.neuron_count(), network_.synapse_count());
    return util::Result<void>::success();
}

// ============================================================
// Periodic Sync
// ============================================================

void SyncManager::start_periodic_sync(std::chrono::milliseconds interval) {
    stop_periodic_sync();

    sync_active_.store(true);
    sync_thread_ = std::thread([this, interval]() {
        NG_LOG_INFO("Periodic sync started (interval: {}ms)", interval.count());
        while (sync_active_.load()) {
            std::this_thread::sleep_for(interval);
            if (!sync_active_.load()) break;

            if (pending_count() > 0) {
                auto result = sync_to_db();
                if (result.is_error()) {
                    NG_LOG_WARN("Periodic sync failed: {}", result.error().message);
                }
            }
        }
        NG_LOG_INFO("Periodic sync stopped");
    });
}

void SyncManager::stop_periodic_sync() {
    sync_active_.store(false);
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }
}

SyncManager::SyncStats SyncManager::stats() const {
    return stats_;
}

} // namespace neurograph::persistence
