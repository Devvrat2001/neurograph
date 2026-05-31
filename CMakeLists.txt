cmake_minimum_required(VERSION 3.25)
project(neurograph VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architecture: Ampere (RTX 3050/3060/3070/3080)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# --- Options ---
option(NEUROGRAPH_BUILD_TESTS "Build unit and integration tests" ON)
option(NEUROGRAPH_ENABLE_COVERAGE
       "Add coverage targets via OpenCppCoverage (Windows/MSVC, Debug build)"
       OFF)
option(NEUROGRAPH_ENABLE_ASAN
       "Build with AddressSanitizer (MSVC /fsanitize=address or GCC/Clang -fsanitize=address)"
       OFF)

# --- CMake module path ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# --- AddressSanitizer wiring ---
#
# Closes the Windows leg of the P1-08 "memory leak free" AC by exposing
# MSVC's built-in ASan (VS 2022 17.0+) as a single-flag opt-in. Same flag
# also works on GCC/Clang for the Linux CI lane.
#
# Usage:
#   cmake -B build -DNEUROGRAPH_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build --config Debug --target neurograph_tests
#   ctest --test-dir build --output-on-failure
#
# Notes:
#   - MSVC ASan requires /MTd or /MT runtime selection consistent across
#     all linked libs. Our vcpkg ports default to /MD which works with
#     ASan as long as we don't mix configs — keep the build clean.
#   - ASan adds ~2-5x runtime overhead and ~3x memory; tests that rely
#     on tight perf gates (tests/perf/*) should be skipped when ASan is
#     on (they're in a separate target, so this is automatic).
#   - On MSVC, /Zi (debug info) is needed for symbolicated stacks, which
#     CMake adds for Debug builds by default.
if(NEUROGRAPH_ENABLE_ASAN)
    if(MSVC)
        # MSVC: scope to CXX/C — nvcc doesn't understand /fsanitize=address
        # as a raw flag and the CUDA kernels don't have host-allocated
        # memory worth instrumenting anyway. CMake's
        # `$<COMPILE_LANGUAGE:...>` generator expression filters per TU.
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>
            $<$<COMPILE_LANGUAGE:C>:/fsanitize=address>
        )
        # /MTd + /fsanitize=address can deadlock at startup; pin to the
        # DLL CRT which is what vcpkg's deps already use. Same per-language
        # scoping rationale.
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/MD$<$<CONFIG:Debug>:d>>
            $<$<COMPILE_LANGUAGE:C>:/MD$<$<CONFIG:Debug>:d>>
        )
        # MSVC ASan injects "annotate_vector"/"annotate_string" sentinels
        # into ASan-built objects. CUDA TUs (skipped above) and any
        # third-party static libs built without ASan don't have those
        # sentinels — the linker then rejects the mix with LNK2038
        # "mismatch detected for 'annotate_vector'". The documented
        # fix is to globally disable the annotations so every TU sees
        # the same value (0). Trades off a small amount of ASan
        # coverage in std::vector/std::string operations for working
        # mixed-CUDA builds. Apply globally (not just CXX) so the
        # /D propagates into every consumer.
        add_compile_definitions(
            _DISABLE_VECTOR_ANNOTATION
            _DISABLE_STRING_ANNOTATION
        )
    else()
        # GCC/Clang on Linux/macOS: separate compile and link flags.
        # Same per-language scoping for nvcc compatibility.
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
            $<$<COMPILE_LANGUAGE:C>:-fsanitize=address>
            $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer>
        )
        add_link_options(-fsanitize=address)
    endif()
    message(STATUS "AddressSanitizer enabled — expect 2-5x runtime overhead. "
                   "CUDA TUs are excluded (nvcc doesn't grok the flag).")

    # Locate the ASan runtime DLL once at configure time so post-build
    # steps can copy it next to each exe. Without this, ASan-built
    # binaries crash at startup with STATUS_DLL_NOT_FOUND because the
    # runtime DLL only sits in the MSVC compiler's bin directory, not on
    # PATH for normal users. We propagate the variable so per-target
    # CMakeLists (e.g. tests/CMakeLists.txt) can attach the same step.
    if(MSVC)
        find_file(NEUROGRAPH_ASAN_DLL
            NAMES clang_rt.asan_dynamic-x86_64.dll
            PATHS
                "$ENV{VCToolsInstallDir}/bin/Hostx64/x64"
                "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
                "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
                "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
                "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
            DOC "MSVC AddressSanitizer runtime DLL")
        if(NEUROGRAPH_ASAN_DLL)
            message(STATUS "ASan runtime DLL: ${NEUROGRAPH_ASAN_DLL}")
        else()
            message(WARNING
                "ASan enabled but clang_rt.asan_dynamic-x86_64.dll not found. "
                "ASan-built exes will fail to start unless the DLL is on PATH "
                "or copied next to them manually. Set -DNEUROGRAPH_ASAN_DLL=<path>.")
        endif()
    endif()
endif()

# Helper used by exe targets: when ASan is on (and the DLL was found),
# copy the runtime next to the exe after build so it's loadable. No-op
# when ASan is off — keeps exe target CMakeLists clean of conditionals.
function(neurograph_attach_asan_runtime target)
    if(NEUROGRAPH_ENABLE_ASAN AND MSVC AND NEUROGRAPH_ASAN_DLL)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${NEUROGRAPH_ASAN_DLL}"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "ASan: copy runtime DLL next to ${target}")
    endif()
endfunction()

# --- Dependencies via vcpkg ---
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(unofficial-sodium CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

# --- CUDA Kernels Library ---
add_library(neurograph_cuda STATIC
    cuda/kernels/activate_neurons.cu
    cuda/kernels/propagate_signals.cu
    cuda/kernels/hebbian_update.cu
    cuda/kernels/normalize_weights.cu
    cuda/kernels/compute_affinity.cu
)

target_include_directories(neurograph_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cuda
)

set_target_properties(neurograph_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# --- Main Library ---
add_library(neurograph_lib STATIC
    src/util/uuid.cpp
    src/util/logger.cpp
    src/core/activation.cpp
    src/core/synapse.cpp
    src/core/neuron.cpp
    src/core/topology.cpp
    src/core/network.cpp
    src/crypto/ed25519_provider.cpp
    src/crypto/key_manager.cpp
    src/command/command.cpp
    src/command/command_signer.cpp
    src/command/command_verifier.cpp
    src/command/command_executor.cpp
    src/command/command_propagator.cpp
    src/persistence/neo4j_client.cpp
    src/persistence/neo4j_repository.cpp
    src/persistence/sync_manager.cpp
    src/module/module_registry.cpp
    src/module/module_loader.cpp
    src/net/protocol.cpp
    src/net/network_server.cpp
    src/net/network_client.cpp
    src/net/network_adapter.cpp
    src/snapshot/model_snapshot.cpp
    src/snapshot/snapshot_registry.cpp
    src/visualizer/ws_server.cpp
    src/visualizer/visualizer_bridge.cpp
    src/training/data_loader.cpp
    src/training/io_manager.cpp
    src/training/backprop.cpp
    src/training/training_engine.cpp
    src/eval/regression.cpp
    src/eval/classification.cpp
    src/eval/baselines.cpp
    src/eval/learning_curve.cpp
    src/reporting/model_card.cpp
    src/reporting/manifest.cpp
    src/licensing/license.cpp
    # P1-11 — persistence (DPAPI on Windows, pepper-XOR fallback on POSIX)
    # plus the trial state machine, and a soft-fail HTTP revocation check.
    src/licensing/license_store.cpp
    src/licensing/license_revocation.cpp
    src/orchestration/auto_pipeline.cpp
    src/runtime/bundle.cpp
    src/workbench/wizard/wizard_state.cpp
    src/workbench/wizard/csv_preview.cpp
    src/dataqc/scanner.cpp
    src/preprocess/transforms.cpp
    src/preprocess/pipeline.cpp
    src/gpu/device_network.cu
)

target_include_directories(neurograph_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cuda
)

# Prevent Windows min/max macro conflicts with std::min/std::max
if(WIN32)
    target_compile_definitions(neurograph_lib PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(neurograph_lib PUBLIC
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    unofficial-sodium::sodium
    CURL::libcurl
    neurograph_cuda
)

# Winsock2 for networking on Windows
if(WIN32)
    target_link_libraries(neurograph_lib PUBLIC ws2_32)
    # P1-11 — DPAPI (CryptProtectData/CryptUnprotectData) lives in
    # crypt32.dll. SHGetKnownFolderPath is in Shell32.lib (already
    # linked transitively via the C runtime, but be explicit).
    target_link_libraries(neurograph_lib PUBLIC crypt32 Shell32)
endif()

set_target_properties(neurograph_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- Executable ---
add_executable(neurograph src/main.cpp src/cuda_link_stub.cu)
target_link_libraries(neurograph PRIVATE neurograph_lib)
# Use CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES (set automatically by CMake CUDA language support)
target_include_directories(neurograph PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(neurograph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
neurograph_attach_asan_runtime(neurograph)

# --- Predict CLI (P1-07) ---
add_executable(neurograph_predict
    src/runtime/predict_main.cpp
    src/cuda_link_stub.cu
)
target_link_libraries(neurograph_predict PRIVATE neurograph_lib)
target_include_directories(neurograph_predict PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(neurograph_predict PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
neurograph_attach_asan_runtime(neurograph_predict)

# --- C ABI shared library (P1-08) ---
#
# `neurograph_rt` is the DLL customers link against from C / C++ / Rust /
# Go / etc. The header uses NEUROGRAPH_RT_BUILDING to switch __declspec
# between dllexport (here) and dllimport (consumers). The CUDA link stub
# is pulled in so the device-symbol resolver finds it even though the C
# ABI itself has no CUDA — `neurograph_lib` links against `neurograph_cuda`
# and the shared library inherits that dependency.
add_library(neurograph_rt SHARED
    src/capi/neurograph_rt.cpp
    src/cuda_link_stub.cu
)
target_link_libraries(neurograph_rt PRIVATE neurograph_lib)
target_include_directories(neurograph_rt PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(neurograph_rt PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_compile_definitions(neurograph_rt PRIVATE NEUROGRAPH_RT_BUILDING=1)
set_target_properties(neurograph_rt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# --- C ABI example program (P1-08) ---
#
# Built as a real C target. If anything in neurograph_rt.h is C-
# incompatible (stray //, bool, etc.), this target fails to compile and
# the build breaks — which is cheaper than a dedicated static-check CI
# step and catches regressions immediately.
add_executable(neurograph_capi_example examples/capi/example.c)
target_link_libraries(neurograph_capi_example PRIVATE neurograph_rt)
target_include_directories(neurograph_capi_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
neurograph_attach_asan_runtime(neurograph_capi_example)

# --- License keygen CLI (P1-11, ops-only) ---
#
# Mints customer keys and verifies them against an explicit public key.
# This binary is for ops -- the Ed25519 secret key never lives on a
# customer machine. Built from the one .cpp; everything it needs is
# already in neurograph_lib (issue_key / verify_key_with /
# generate_keypair).
add_executable(neurograph_keygen src/licensing/license_keygen_main.cpp)
target_link_libraries(neurograph_keygen PRIVATE neurograph_lib)
neurograph_attach_asan_runtime(neurograph_keygen)

# --- Workbench (ImGui + DirectX 11) ---
find_package(imgui CONFIG REQUIRED)

add_executable(neurograph_workbench WIN32
    src/workbench/workbench_app.cpp
    # P1-09 wizard UI. State + csv_preview translation units already
    # live in neurograph_lib so the unit tests can link them without
    # pulling ImGui in; only the render layer is workbench-only.
    src/workbench/wizard/wizard_ui.cpp
    src/workbench/wizard/step_load.cpp
    src/workbench/wizard/step_quality.cpp
    src/workbench/wizard/step_train.cpp
    src/workbench/wizard/step_results.cpp
    # P1-11 license dialog -- ImGui-bound, so workbench-only. The
    # crypto/persistence/trial logic lives in neurograph_lib.
    src/workbench/license_dialog.cpp
    src/cuda_link_stub.cu
)
target_link_libraries(neurograph_workbench PRIVATE
    neurograph_lib
    imgui::imgui
    d3d11
    dxgi
)
target_include_directories(neurograph_workbench PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    # The wizard step .cpp files include "wizard_ui.hpp" with no
    # relative qualifier; add the wizard directory so that resolves.
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard
)
set_target_properties(neurograph_workbench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
neurograph_attach_asan_runtime(neurograph_workbench)

# --- Tests ---
if(NEUROGRAPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)

    # Coverage helper. Off by default — flip with
    # `cmake -DNEUROGRAPH_ENABLE_COVERAGE=ON` and build the
    # `neurograph_coverage` target in Debug. See cmake/OpenCppCoverage.cmake
    # for usage details and platform requirements (MSVC + PDBs).
    if(NEUROGRAPH_ENABLE_COVERAGE)
        include(OpenCppCoverage)
        add_coverage_run(neurograph_coverage
            TARGET      neurograph_tests
            SOURCES_DIR "${CMAKE_SOURCE_DIR}/src"
                        "${CMAKE_SOURCE_DIR}/include"
            OUTPUT_DIR  "${CMAKE_BINARY_DIR}/coverage"
        )
    endif()
endif()
