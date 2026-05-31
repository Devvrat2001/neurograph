// P1-11 -- License keygen CLI (ops-only).
//
// This binary is what an ops engineer runs to mint a customer key.
// The Ed25519 secret key lives only on the ops machine (or HSM); the
// running NeuroGraph workbench never sees it. The same secret is what
// `embedded_public_key()` on the customer side validates against.
//
// Subcommands:
//   genkey  --out-priv <path> --out-pub <path>
//       Generate a fresh keypair. Run once per product line; the
//       resulting public key gets baked into the workbench binary
//       (replacing the placeholder in license.cpp).
//
//   issue   --secret-key <path> --tier <t> --expires <YYYY-MM-DD>
//           --holder "Name" [--features F1,F2,...]
//       Mint a license key signed by the secret-key file. Prints the
//       grouped base32 key string to stdout.
//
//   verify  --public-key <path> <key-string>
//       Round-trip a key against an explicit public key. Sanity check
//       for ops -- "did I mint this with the right private key?".
//
// Deliberately a flat argv parser, no third-party CLI library: this
// is a release-time tool that has to build on whatever ops box is
// handy, so we keep its dep list down to just neurograph_lib.

#include "neurograph/licensing/license.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace neurograph::licensing;

namespace {

void print_usage() {
    std::cerr <<
        "usage: neurograph_keygen <subcommand> [args]\n"
        "\n"
        "subcommands:\n"
        "  genkey  --out-priv <path> --out-pub <path>\n"
        "          Generate a fresh Ed25519 keypair.\n"
        "\n"
        "  issue   --secret-key <path> --tier <trial|personal|professional|enterprise>\n"
        "          --expires <YYYY-MM-DD> --holder <name>\n"
        "          [--features F1,F2,...]\n"
        "          Mint and print a license key.\n"
        "\n"
        "  verify  --public-key <path> <key-string>\n"
        "          Verify and print the parsed License.\n";
}

bool read_file(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "error: cannot open " << path << "\n";
        return false;
    }
    f.seekg(0, std::ios::end);
    auto n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.assign(static_cast<std::size_t>(n), 0);
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return f.good() || f.eof();
}

bool write_file(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::cerr << "error: cannot write " << path << "\n";
        return false;
    }
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    }
    return f.good();
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

bool parse_tier(const std::string& s, Tier& out) {
    if (s == "trial")        { out = Tier::Trial; return true; }
    if (s == "personal")     { out = Tier::Personal; return true; }
    if (s == "professional") { out = Tier::Professional; return true; }
    if (s == "enterprise")   { out = Tier::Enterprise; return true; }
    return false;
}

const char* tier_name(Tier t) {
    switch (t) {
        case Tier::Trial:        return "trial";
        case Tier::Personal:     return "personal";
        case Tier::Professional: return "professional";
        case Tier::Enterprise:   return "enterprise";
    }
    return "?";
}

// Parse "YYYY-MM-DD" as a UTC date at 23:59:59 -- expiry is end-of-day
// in the customer's favour. Hand-rolled (no <chrono>::parse on MSVC
// 2022 yet, no Howard Hinnant date dep yet).
bool parse_date_eod_utc(const std::string& s,
                        std::chrono::system_clock::time_point& out) {
    int y = 0, m = 0, d = 0;
    char dash1 = 0, dash2 = 0;
    std::istringstream is(s);
    is >> y >> dash1 >> m >> dash2 >> d;
    if (!is || dash1 != '-' || dash2 != '-') return false;
    if (m < 1 || m > 12 || d < 1 || d > 31)  return false;

    std::tm tm{};
    tm.tm_year = y - 1900;
    tm.tm_mon  = m - 1;
    tm.tm_mday = d;
    tm.tm_hour = 23;
    tm.tm_min  = 59;
    tm.tm_sec  = 59;

#ifdef _WIN32
    const std::time_t t = _mkgmtime(&tm);
#else
    const std::time_t t = timegm(&tm);
#endif
    if (t == -1) return false;
    out = std::chrono::system_clock::from_time_t(t);
    return true;
}

std::string flag(int argc, char** argv, const std::string& name,
                 const std::string& fallback = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) return argv[i + 1];
    }
    return fallback;
}

int cmd_genkey(int argc, char** argv) {
    const std::string priv = flag(argc, argv, "--out-priv");
    const std::string pub  = flag(argc, argv, "--out-pub");
    if (priv.empty() || pub.empty()) {
        std::cerr << "genkey: --out-priv and --out-pub are required\n";
        return 2;
    }
    auto kp = generate_keypair();
    if (!write_file(priv, kp.secret_key)) return 1;
    if (!write_file(pub,  kp.public_key)) return 1;
    std::cout << "wrote secret key (" << kp.secret_key.size() << " B) to " << priv << "\n";
    std::cout << "wrote public key (" << kp.public_key.size() << " B) to " << pub  << "\n";
    return 0;
}

int cmd_issue(int argc, char** argv) {
    const std::string secret_path = flag(argc, argv, "--secret-key");
    const std::string tier_str    = flag(argc, argv, "--tier");
    const std::string expires_str = flag(argc, argv, "--expires");
    const std::string holder      = flag(argc, argv, "--holder");
    const std::string features    = flag(argc, argv, "--features");
    if (secret_path.empty() || tier_str.empty() || expires_str.empty() || holder.empty()) {
        std::cerr << "issue: --secret-key, --tier, --expires, --holder are required\n";
        return 2;
    }

    LicensePayload p;
    if (!parse_tier(tier_str, p.tier)) {
        std::cerr << "issue: unknown tier '" << tier_str << "'\n";
        return 2;
    }
    if (!parse_date_eod_utc(expires_str, p.expires)) {
        std::cerr << "issue: --expires must be YYYY-MM-DD\n";
        return 2;
    }
    p.holder   = holder;
    p.features = split_csv(features);

    std::vector<unsigned char> sk;
    if (!read_file(secret_path, sk)) return 1;

    auto key = issue_key(p, sk);
    if (key.is_error()) {
        std::cerr << "issue: " << key.error().message << "\n";
        return 1;
    }
    std::cout << key.value() << "\n";
    return 0;
}

int cmd_verify(int argc, char** argv) {
    const std::string pub_path = flag(argc, argv, "--public-key");
    if (pub_path.empty()) {
        std::cerr << "verify: --public-key is required\n";
        return 2;
    }
    // Last positional argument is the key string.
    std::string key;
    for (int i = argc - 1; i > 0; --i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0 && a != pub_path) {
            key = a;
            break;
        }
    }
    if (key.empty()) {
        std::cerr << "verify: missing key string positional argument\n";
        return 2;
    }
    std::vector<unsigned char> pk;
    if (!read_file(pub_path, pk)) return 1;

    auto r = verify_key_with(key, pk);
    if (r.is_error()) {
        std::cerr << "verify: " << r.error().message << "\n";
        return 1;
    }
    const License& l = r.value();
    auto exp_t = std::chrono::system_clock::to_time_t(l.expires);
    std::cout << "holder:   " << l.holder    << "\n";
    std::cout << "tier:     " << tier_name(l.tier) << "\n";
    std::cout << "expires:  " << std::put_time(std::gmtime(&exp_t), "%Y-%m-%dT%H:%M:%SZ") << "\n";
    std::cout << "active:   " << (is_active(l) ? "yes" : "no (expired)") << "\n";
    std::cout << "features:";
    for (const auto& f : l.features) std::cout << " " << f;
    std::cout << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 2; }
    const std::string sub = argv[1];
    if (sub == "genkey") return cmd_genkey(argc, argv);
    if (sub == "issue")  return cmd_issue(argc, argv);
    if (sub == "verify") return cmd_verify(argc, argv);
    if (sub == "--help" || sub == "-h" || sub == "help") {
        print_usage();
        return 0;
    }
    std::cerr << "unknown subcommand: " << sub << "\n\n";
    print_usage();
    return 2;
}
