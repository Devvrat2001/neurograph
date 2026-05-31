#include "neurograph/persistence/neo4j_client.hpp"
#include "neurograph/util/logger.hpp"

#include <curl/curl.h>
#include <sstream>

namespace neurograph::persistence {

// libcurl write callback: appends received data to a std::string.
static size_t write_callback(char* ptr, size_t size, size_t nmemb,
                              void* userdata) {
    auto* response = static_cast<std::string*>(userdata);
    size_t total = size * nmemb;
    response->append(ptr, total);
    return total;
}

Neo4jClient::Neo4jClient(Neo4jConfig config)
    : config_(std::move(config))
{
    init_curl();
}

Neo4jClient::~Neo4jClient() {
    cleanup_curl();
}

Neo4jClient::Neo4jClient(Neo4jClient&& other) noexcept
    : config_(std::move(other.config_))
    , curl_handle_(other.curl_handle_)
{
    other.curl_handle_ = nullptr;
}

Neo4jClient& Neo4jClient::operator=(Neo4jClient&& other) noexcept {
    if (this != &other) {
        cleanup_curl();
        config_ = std::move(other.config_);
        curl_handle_ = other.curl_handle_;
        other.curl_handle_ = nullptr;
    }
    return *this;
}

void Neo4jClient::init_curl() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        NG_LOG_ERROR("Failed to initialize libcurl");
    }
}

void Neo4jClient::cleanup_curl() {
    if (curl_handle_) {
        curl_easy_cleanup(static_cast<CURL*>(curl_handle_));
        curl_handle_ = nullptr;
    }
}

std::string Neo4jClient::auth_header() const {
    // Base64 encode "username:password"
    std::string credentials = config_.username + ":" + config_.password;
    // Use curl's base64 encoding
    CURL* curl = static_cast<CURL*>(curl_handle_);
    char* encoded = nullptr;
    size_t encoded_len = 0;

    // Manual base64 encoding (simple implementation)
    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string result;
    int i = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    const char* bytes = credentials.c_str();
    size_t len = credentials.size();

    size_t pos = 0;
    while (pos < len) {
        int bytes_read = 0;
        for (i = 0; i < 3 && pos < len; ++i, ++pos) {
            char_array_3[i] = static_cast<unsigned char>(bytes[pos]);
            bytes_read++;
        }
        for (; i < 3; ++i) char_array_3[i] = 0;

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) | ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) | ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (i = 0; i < bytes_read + 1; ++i) {
            result += base64_chars[char_array_4[i]];
        }
        while (bytes_read++ < 3) result += '=';
    }

    return "Basic " + result;
}

util::Result<std::string> Neo4jClient::http_post(const std::string& url,
                                                   const std::string& body) const {
    if (!curl_handle_) {
        return util::Error("CURL not initialized");
    }

    CURL* curl = static_cast<CURL*>(curl_handle_);
    std::string response_body;

    // Reset curl handle for reuse
    curl_easy_reset(curl);

    // Set URL
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    // Set POST data
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));

    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    std::string auth = auth_header();
    headers = curl_slist_append(headers, ("Authorization: " + auth).c_str());

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Set write callback
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

    // Timeout
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(config_.timeout_seconds));

    // TLS verification (disable for local dev if not using TLS)
    if (!config_.use_tls) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    }

    // Perform request
    CURLcode res = curl_easy_perform(curl);

    // Cleanup headers
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        return util::Error("HTTP request failed: " + std::string(curl_easy_strerror(res)));
    }

    // Check HTTP status code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code < 200 || http_code >= 300) {
        return util::Error("Neo4j HTTP error " + std::to_string(http_code) +
                          ": " + response_body);
    }

    return response_body;
}

util::Result<nlohmann::json> Neo4jClient::execute_cypher(
    const std::string& query,
    const nlohmann::json& params) const
{
    // Build Query API v2 request body
    nlohmann::json request_body = {
        {"statement", query},
        {"parameters", params}
    };

    std::string url = config_.query_endpoint();
    std::string body = request_body.dump();

    NG_LOG_TRACE("Neo4j query: {}", query);

    auto result = http_post(url, body);
    if (result.is_error()) {
        NG_LOG_ERROR("Neo4j query failed: {}", result.error().message);
        return util::Error(result.error().message);
    }

    try {
        auto response = nlohmann::json::parse(result.value());

        // Check for Neo4j errors in the response
        if (response.contains("errors") && !response["errors"].empty()) {
            std::string error_msg;
            for (const auto& err : response["errors"]) {
                error_msg += err.value("message", "Unknown error") + "; ";
            }
            return util::Error("Neo4j error: " + error_msg);
        }

        return response;
    } catch (const nlohmann::json::parse_error& e) {
        return util::Error("Failed to parse Neo4j response: " + std::string(e.what()));
    }
}

util::Result<nlohmann::json> Neo4jClient::execute_batch(
    const std::vector<std::pair<std::string, nlohmann::json>>& statements) const
{
    // Build transaction commit request
    nlohmann::json stmts = nlohmann::json::array();
    for (const auto& [query, params] : statements) {
        stmts.push_back({
            {"statement", query},
            {"parameters", params}
        });
    }

    nlohmann::json request_body = {{"statements", stmts}};

    std::string url = config_.tx_commit_endpoint();
    std::string body = request_body.dump();

    NG_LOG_TRACE("Neo4j batch: {} statements", statements.size());

    auto result = http_post(url, body);
    if (result.is_error()) {
        return util::Error(result.error().message);
    }

    try {
        auto response = nlohmann::json::parse(result.value());

        if (response.contains("errors") && !response["errors"].empty()) {
            std::string error_msg;
            for (const auto& err : response["errors"]) {
                error_msg += err.value("message", "Unknown error") + "; ";
            }
            return util::Error("Neo4j batch error: " + error_msg);
        }

        return response;
    } catch (const nlohmann::json::parse_error& e) {
        return util::Error("Failed to parse Neo4j response: " + std::string(e.what()));
    }
}

bool Neo4jClient::health_check() const {
    if (!curl_handle_) return false;

    CURL* curl = static_cast<CURL*>(curl_handle_);
    std::string response_body;

    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, config_.base_url().c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    if (!config_.use_tls) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    }

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        NG_LOG_WARN("Neo4j health check failed: {}", curl_easy_strerror(res));
        return false;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    return http_code == 200;
}

} // namespace neurograph::persistence
