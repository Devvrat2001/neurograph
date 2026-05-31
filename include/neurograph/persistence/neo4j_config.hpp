#pragma once

#include <string>
#include <cstdint>

namespace neurograph::persistence {

/// Configuration for connecting to a Neo4j instance.
struct Neo4jConfig {
    std::string host = "localhost";
    uint16_t port = 7474;
    std::string database = "neo4j";
    std::string username = "neo4j";
    std::string password = "neo4j";
    bool use_tls = false;
    int timeout_seconds = 30;

    /// Returns the base URL for the Neo4j HTTP API.
    /// e.g. "http://localhost:7474"
    std::string base_url() const {
        std::string scheme = use_tls ? "https" : "http";
        return scheme + "://" + host + ":" + std::to_string(port);
    }

    /// Returns the Query API v2 endpoint.
    /// e.g. "http://localhost:7474/db/neo4j/query/v2"
    std::string query_endpoint() const {
        return base_url() + "/db/" + database + "/query/v2";
    }

    /// Returns the transaction commit endpoint (alternative).
    /// e.g. "http://localhost:7474/db/neo4j/tx/commit"
    std::string tx_commit_endpoint() const {
        return base_url() + "/db/" + database + "/tx/commit";
    }
};

} // namespace neurograph::persistence
