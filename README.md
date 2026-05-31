# NeuroGraph

A C++/CUDA self-organizing neural network with a desktop wizard for
end-to-end "drop in a CSV → get a trained, exported model" workflows.

> **Status:** Phase 1 (Tabular MVP) feature-complete. 432 unit/integration
> tests + 12 perf tests passing. Pre-release; not yet shipping.

---

## What it does

1. **Wizard** (`neurograph_workbench.exe`) — Load CSV → Data-quality scan
   → Train → Export. 4-step ImGui flow for non-ML users.
2. **Standalone inference** (`neurograph_predict.exe`) — Run a trained
   bundle against new CSV/JSON rows from the command line.
3. **C-ABI library** (`neurograph_rt.dll`) — C89-clean ABI for embedding
   inference in C / C++ / Rust / Go / C# / Python (`ctypes`) / Java (JNI).
4. **Adaptive topology** — `core::Network` grows + prunes neurons during
   training (`growth_only_self_organize` runs alongside backprop without
   poisoning the gradient signal).

Detailed roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md).
Phase 1 engineering spec: [`docs/phase1-tabular-mvp.md`](docs/phase1-tabular-mvp.md).

---

## Build (Windows / MSVC 2022)

Prerequisites:
- Visual Studio 2022 Build Tools (MSVC 14.4+)
- CUDA Toolkit 13.2
- CMake 3.25+
- vcpkg

```powershell
# One-time: install dependencies via vcpkg manifest mode
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg-toolchain.cmake

# Build everything
cmake --build build --config Release

# Run the wizard
build\Release\neurograph_workbench.exe

# Run all tests
build\tests\Release\neurograph_tests.exe
```

---

## Targets

| Binary | Role |
|---|---|
| `neurograph_workbench.exe` | ImGui desktop wizard (Load → Quality → Train → Export). |
| `neurograph_predict.exe` | Standalone inference CLI. |
| `neurograph_rt.dll` | C-ABI shared library for embedding inference. |
| `neurograph_keygen.exe` | Ops-only Ed25519 license keygen. |
| `neurograph_tests` | 432 unit + integration tests. |
| `neurograph_perf` | 12 chrono-timed perf gates. |
| `neurograph` | Legacy demo program (walks all subsystems). |

---

## Optional build flags

| Flag | Default | Purpose |
|---|---|---|
| `NEUROGRAPH_BUILD_TESTS=ON` | ON | Build unit/integration/perf test targets. |
| `NEUROGRAPH_ENABLE_COVERAGE=ON` | OFF | Add `neurograph_coverage` target via OpenCppCoverage. |
| `NEUROGRAPH_ENABLE_ASAN=ON` | OFF | Build with MSVC AddressSanitizer (auto-copies runtime DLL). |

---

## License

To be determined. All rights reserved until a `LICENSE` file lands.
