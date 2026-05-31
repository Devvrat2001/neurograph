#pragma once

#include "neurograph/persistence/graph_repository.hpp"
#include "neurograph/core/network.hpp"
#include "neurograph/util/result.hpp"
#include <unordered_set>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

namespace neurograph::persistence {

/// Synchronizes an in-memory Network with a GraphRepository.
/// Tracks dirty (modified) neurons and synapses, and flushes them
/// to the database in batch operations.
///
/// Supports both manual sync and periodic background sync.
class SyncManager {
public:
    SyncManager(core::Network& network, GraphRepository& repo);
    ~SyncManager();

    // Non-copyable
    SyncManager(const SyncManager&) = delete;
    SyncManager& operator=(const SyncManager&) = delete;

    // --- Dirty tracking ---

    /// Mark a neuron as modified (needs sync to DB).
    void mark_neuron_dirty(const core::NeuronId& id);

    /// Mark a synapse as modified.
    void mark_synapse_dirty(const core::SynapseId& id);

    /// Mark a neuron as deleted (needs removal from DB).
    void mark_neuron_deleted(const core::NeuronId& id);

    /// Mark a synapse as deleted.
    void mark_synapse_deleted(const core::SynapseId& id);

    /// Mark the entire network as dirty (full resync needed).
    void mark_all_dirty();

    /// Get count of pending changes.
    size_t pending_count() const;

    // --- Sync operations ---

    /// Flush all dirty entities to the database.
    util::Result<void> sync_to_db();

    /// Load the full network from the database, replacing in-memory state.
    util::Result<void> sync_from_db();

    // --- Periodic sync ---

    /// Start periodic background sync at the given interval.
    void start_periodic_sync(std::chrono::milliseconds interval);

    /// Stop the periodic sync thread.
    void stop_periodic_sync();

    /// Check if periodic sync is running.
    bool is_periodic_sync_active() const noexcept { return sync_active_.load(); }

    // --- Stats ---
    struct SyncStats {
        size_t total_syncs = 0;
        size_t neurons_synced = 0;
        size_t synapses_synced = 0;
        size_t neurons_deleted = 0;
        size_t synapses_deleted = 0;
        size_t errors = 0;
    };

    SyncStats stats() const;

private:
    core::Network& network_;
    GraphRepository& repo_;

    mutable std::mutex mutex_;
    std::unordered_set<std::string> dirty_neurons_;
    std::unordered_set<std::string> dirty_synapses_;
    std::unordered_set<std::string> deleted_neurons_;
    std::unordered_set<std::string> deleted_synapses_;

    // Periodic sync
    std::atomic<bool> sync_active_{false};
    std::thread sync_thread_;

    // Stats
    SyncStats stats_;
};

} // namespace neurograph::persistence
