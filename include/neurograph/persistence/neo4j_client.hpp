#pragma once

#include "neurograph/persistence/neo4j_config.hpp"
#include "neurograph/util/result.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace neurograph::persistence {

/// Low-level HTTP client for executing Cypher queries against Neo4j.
/// Uses libcurl to communicate via the Neo4j Query API v2.
class Neo4jClient {
public:
    explicit Neo4jClient(Neo4jConfig config);
    ~Neo4jClient();

    // Non-copyable, movable
    Neo4jClient(const Neo4jClient&) = delete;
    Neo4jClient& operator=(const Neo4jClient&) = delete;
    Neo4jClient(Neo4jClient&& other) noexcept;
    Neo4jClient& operator=(Neo4jClient&& other) noexcept;

    /// Execute a single Cypher query with parameters.
    /// Returns the parsed JSON response from Neo4j.
    util::Result<nlohmann::json> execute_cypher(
        const std::string& query,
        const nlohmann::json& params = nlohmann::json::object()) const;

    /// Execute multiple Cypher statements in a single transaction.
    /// Uses the /tx/commit endpoint for atomic batch execution.
    util::Result<nlohmann::json> execute_batch(
        const std::vector<std::pair<std::string, nlohmann::json>>& statements) const;

    /// Health check: verifies connectivity to Neo4j.
    bool health_check() const;

    const Neo4jConfig& config() const noexcept { return config_; }

private:
    Neo4jConfig config_;
    void* curl_handle_ = nullptr; // CURL* - opaque to avoid leaking curl headers

    void init_curl();
    void cleanup_curl();

    /// Perform an HTTP POST request and return the response body.
    util::Result<std::string> http_post(const std::string& url,
                                         const std::string& body) const;

    /// Build the Authorization header value (Basic auth).
    std::string auth_header() const;
};

} // namespace neurograph::persistence
