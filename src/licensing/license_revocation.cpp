// P1-11 -- Online revocation check (best-effort, soft-fail).
//
// Hashes the key (so we never wire the raw secret over the network),
// POSTs `{"key_hash": "..."}` to the licensing service, and parses
// `{"revoked": true|false}` out of the response. Any error -- DNS,
// timeout, non-200, malformed JSON -- collapses to Ok(false). Air-
// gapped customers MUST stay unblocked, so "we couldn't talk to the
// server" is never grounds to refuse a launch.
//
// Caller is expected to treat Ok(false) as "we have no negative
// signal" rather than "we have positive proof of validity". The
// strong validity signal still comes from the embedded-public-key
// signature check inside verify_key().

#include "neurograph/licensing/license.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sodium.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace neurograph::licensing {

namespace {

void ensure_sodium() {
    static const bool ok = []() {
        return sodium_init() >= 0;
    }();
    if (!ok) throw std::runtime_error("revocation: sodium_init failed");
}

// SHA-256 of the key string, hex-encoded. The server only ever sees
// the hash so logs / dumps don't accidentally leak working keys.
std::string sha256_hex(const std::string& s) {
    ensure_sodium();
    unsigned char digest[crypto_hash_sha256_BYTES];
    crypto_hash_sha256(digest,
                       reinterpret_cast<const unsigned char*>(s.data()),
                       s.size());
    std::ostringstream o;
    o << std::hex;
    for (unsigned char b : digest) {
        o.width(2);
        o.fill('0');
        o << static_cast<int>(b);
    }
    return o.str();
}

std::size_t curl_write_cb(char* ptr, std::size_t size, std::size_t nmemb,
                          void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

} // namespace

util::Result<bool>
is_revoked(const std::string& key,
           const std::string& endpoint,
           std::chrono::milliseconds timeout) {
    if (key.empty()) {
        return util::Error("revocation: empty key");
    }

    // CURL global init is normally process-wide, but doing it here per
    // call is cheap (idempotent reference-counted init under the hood)
    // and avoids order-of-init headaches with the workbench's static
    // initializers.
    CURL* h = curl_easy_init();
    if (!h) {
        // This is "we can't even allocate a CURL handle" -- treat as
        // not-revoked rather than blocking the user. Logged for ops.
        return false;
    }

    nlohmann::json body = { {"key_hash", sha256_hex(key)} };
    const std::string body_str = body.dump();

    std::string response;
    curl_easy_setopt(h, CURLOPT_URL,            endpoint.c_str());
    curl_easy_setopt(h, CURLOPT_POST,           1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS,     body_str.c_str());
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE,  static_cast<long>(body_str.size()));
    curl_easy_setopt(h, CURLOPT_TIMEOUT_MS,     static_cast<long>(timeout.count()));
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(std::min<long>(2000, static_cast<long>(timeout.count()))));
    curl_easy_setopt(h, CURLOPT_NOSIGNAL,       1L);   // safe in worker thread
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION,  curl_write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA,      &response);
    curl_easy_setopt(h, CURLOPT_USERAGENT,      "neurograph-license/1.0");

    curl_slist* hdrs = nullptr;
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);

    const CURLcode rc = curl_easy_perform(h);
    long http_code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(hdrs);
    curl_easy_cleanup(h);

    if (rc != CURLE_OK || http_code != 200) {
        // Soft-fail: server unreachable / non-200 -> assume not revoked.
        return false;
    }

    try {
        auto j = nlohmann::json::parse(response);
        if (j.contains("revoked") && j["revoked"].is_boolean()) {
            return j["revoked"].get<bool>();
        }
    } catch (const std::exception&) {
        // Malformed body. Soft-fail.
        return false;
    }
    return false;
}

} // namespace neurograph::licensing
