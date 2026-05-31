// Binary-size budget for predict.exe (P1-07 AC4).
//
// Spec target from docs/phase1-tabular-mvp.md:
//   ".exe under 20 MB (strip debug info in Release)"
//
// This is not a "performance" test in the traditional sense — it's a
// shipping-budget gate. It lives in tests/perf/ alongside the wall-time
// gates because both are about end-user-facing budgets that don't fit
// the unit-test "asserts on values" mold. They share the same env-flag
// override convention so a single CI knob can disable / loosen the
// whole suite.
//
// Config awareness: the 20 MB target only applies to Release builds —
// Debug binaries with full PDB info are easily 5-10× larger and that's
// fine. We detect the build config via the standard NDEBUG macro
// (defined by CMake in Release / RelWithDebInfo / MinSizeRel). When
// NDEBUG is undefined, the test reports the size for trending but
// skips the gate.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#ifndef NEUROGRAPH_PREDICT_EXE
#  error "NEUROGRAPH_PREDICT_EXE must be defined by the test build"
#endif

namespace fs = std::filesystem;

namespace {

long env_long_or(const char* name, long fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    try {
        return std::stol(v);
    } catch (...) {
        return fallback;
    }
}

bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}

constexpr bool is_release_build() {
#ifdef NDEBUG
    return true;
#else
    return false;
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// Predict.exe size gate.
//
// Default cap: 20 MiB (20 * 1024 * 1024 = 20'971'520 bytes), straight
// from the spec. Override via NEUROGRAPH_PERF_PREDICT_SIZE_MAX_MB.
// Skip via NEUROGRAPH_PERF_SKIP_GATE=1.
// ---------------------------------------------------------------------------

TEST(PerfBinarySize, PredictExeFitsInBudget) {
    const fs::path exe = NEUROGRAPH_PREDICT_EXE;
    ASSERT_TRUE(fs::exists(exe))
        << "predict.exe not found at " << exe.string()
        << " — build dependency wired correctly?";

    std::error_code ec;
    const auto bytes = fs::file_size(exe, ec);
    ASSERT_FALSE(ec) << "fs::file_size failed: " << ec.message();

    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    const long   max_mib = env_long_or("NEUROGRAPH_PERF_PREDICT_SIZE_MAX_MB", 20);
    const bool   skip    = env_truthy("NEUROGRAPH_PERF_SKIP_GATE");
    const char*  config  = is_release_build() ? "Release" : "Debug";

    std::cout << "[perf] predict_exe_size config=" << config
              << " bytes=" << bytes
              << " (" << mib << " MiB)"
              << " gate=" << max_mib << " MiB"
              << ((!is_release_build()) ? " (Debug — gate skipped)" :
                  (skip ? " (gate skipped)" : "")) << '\n';
    ::testing::Test::RecordProperty("predict_exe_size_bytes",
                                    static_cast<int>(bytes));

    // Debug builds are intentionally fat — gate only applies to Release.
    if (!is_release_build()) {
        GTEST_SKIP() << "Skipping size gate: not a Release build "
                     << "(NDEBUG undefined). Measured size still recorded "
                     << "via RecordProperty for trending.";
    }
    if (skip) {
        GTEST_SKIP() << "NEUROGRAPH_PERF_SKIP_GATE=1 — recording size only.";
    }

    EXPECT_LT(mib, static_cast<double>(max_mib))
        << "predict.exe size budget exceeded: " << mib
        << " MiB > " << max_mib << " MiB cap. "
        << "Override via NEUROGRAPH_PERF_PREDICT_SIZE_MAX_MB=<n> "
        << "or set NEUROGRAPH_PERF_SKIP_GATE=1.";
}
