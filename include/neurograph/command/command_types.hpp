#pragma once

#include <string_view>
#include <cstdint>

namespace neurograph::command {

enum class CommandType : uint8_t {
    // Neuron CRUD
    CREATE_NEURON   = 0,
    READ_NEURON     = 1,
    UPDATE_NEURON   = 2,
    DELETE_NEURON   = 3,

    // Synapse CRUD
    CREATE_SYNAPSE  = 10,
    READ_SYNAPSE    = 11,
    UPDATE_SYNAPSE  = 12,
    DELETE_SYNAPSE  = 13,

    // Network operations
    REPLICATE       = 20,
    CONNECT         = 21,
    DISCONNECT      = 22,

    // Activation & control
    ACTIVATE        = 30,
    SHUTDOWN        = 31,
    SELF_ORGANIZE   = 32,

    // System
    SYNC_TO_DB      = 40,
    LOAD_MODULE     = 41,
    UNLOAD_MODULE   = 42,

    // Custom (for module-defined commands)
    CUSTOM          = 255
};

enum class PropagationMode : uint8_t {
    BFS,           // Breadth-first through entire network
    DFS,           // Depth-first through entire network
    TARGETED,      // Single neuron only (no propagation)
    BROADCAST      // All neurons simultaneously (parallel)
};

inline std::string_view command_type_str(CommandType t) {
    switch (t) {
        case CommandType::CREATE_NEURON:  return "CREATE_NEURON";
        case CommandType::READ_NEURON:    return "READ_NEURON";
        case CommandType::UPDATE_NEURON:  return "UPDATE_NEURON";
        case CommandType::DELETE_NEURON:  return "DELETE_NEURON";
        case CommandType::CREATE_SYNAPSE: return "CREATE_SYNAPSE";
        case CommandType::READ_SYNAPSE:   return "READ_SYNAPSE";
        case CommandType::UPDATE_SYNAPSE: return "UPDATE_SYNAPSE";
        case CommandType::DELETE_SYNAPSE: return "DELETE_SYNAPSE";
        case CommandType::REPLICATE:      return "REPLICATE";
        case CommandType::CONNECT:        return "CONNECT";
        case CommandType::DISCONNECT:     return "DISCONNECT";
        case CommandType::ACTIVATE:       return "ACTIVATE";
        case CommandType::SHUTDOWN:       return "SHUTDOWN";
        case CommandType::SELF_ORGANIZE:  return "SELF_ORGANIZE";
        case CommandType::SYNC_TO_DB:     return "SYNC_TO_DB";
        case CommandType::LOAD_MODULE:    return "LOAD_MODULE";
        case CommandType::UNLOAD_MODULE:  return "UNLOAD_MODULE";
        case CommandType::CUSTOM:         return "CUSTOM";
        default:                          return "UNKNOWN";
    }
}

inline std::string_view propagation_mode_str(PropagationMode m) {
    switch (m) {
        case PropagationMode::BFS:       return "BFS";
        case PropagationMode::DFS:       return "DFS";
        case PropagationMode::TARGETED:  return "TARGETED";
        case PropagationMode::BROADCAST: return "BROADCAST";
        default:                         return "UNKNOWN";
    }
}

} // namespace neurograph::command
