find_package(GTest CONFIG REQUIRED)

add_executable(neurograph_tests
    unit/test_neuron.cpp
    unit/test_synapse.cpp
    unit/test_network.cpp
    unit/test_command.cpp
    unit/test_persistence.cpp
    unit/test_module.cpp
    unit/test_network_protocol.cpp
    unit/test_snapshot.cpp
    unit/test_regression_metrics.cpp
    unit/test_classification_metrics.cpp
    # P1-18: hidden-layer auto-allocation in load_prepared_dataset.
    # Exercises auto-sizing bounds (clamp(2 * features, 8, 64)),
    # explicit-size honoring, legacy zero-hidden direct wiring, and
    # the convergence smoke that the Glorot+ReLU network actually
    # learns linearly-separable data.
    unit/test_training_engine.cpp
    # P1-04 sklearn-parity. Validates evaluate_classification against
    # tests/fixtures/eval/sklearn_golden.json (regenerable via
    # tools/gen_sklearn_golden.py). Path is baked in at compile time
    # via NEUROGRAPH_SKLEARN_GOLDEN_PATH below.
    unit/test_sklearn_parity.cpp
    unit/test_dataqc_scanner.cpp
    unit/test_preprocess_pipeline.cpp
    unit/test_baselines.cpp
    unit/test_learning_curve.cpp
    unit/test_model_card.cpp
    unit/test_manifest.cpp
    unit/test_license.cpp
    # P1-11 persistence + trial coverage. Lives in a separate TU from
    # the crypto-only test_license.cpp so the file boundaries match
    # the implementation split (license.cpp vs license_store.cpp).
    unit/test_license_store.cpp
    unit/test_runtime_bundle.cpp
    unit/test_capi.cpp
    unit/test_wizard_state.cpp
    unit/test_models.cpp
    # The C ABI translation unit is compiled *into* the test exe with
    # NEUROGRAPH_RT_STATIC=1 so we don't have to chase the neurograph_rt
    # DLL around at test time. The same source, built separately into
    # the neurograph_rt SHARED target, exposes the exported symbols for
    # real customers.
    ${CMAKE_SOURCE_DIR}/src/capi/neurograph_rt.cpp
    integration/test_all_phases.cpp
    integration/test_auto_pipeline.cpp
    integration/test_predict_binary.cpp
    ${NEUROGRAPH_CUDA_LINK_STUB}
)

target_link_libraries(neurograph_tests PRIVATE
    neurograph_lib
    GTest::gtest_main
)

# Predict-binary test shells out to the built `neurograph_predict` exe,
# so the test must be rebuilt whenever the exe changes and needs the
# resolved path baked in at compile time.
add_dependencies(neurograph_tests neurograph_predict)
target_compile_definitions(neurograph_tests PRIVATE
    NEUROGRAPH_PREDICT_EXE="$<TARGET_FILE:neurograph_predict>"
    # Switches NG_EXPORT off so the C ABI symbols compile as plain
    # statically-linked functions in the test binary.
    NEUROGRAPH_RT_STATIC=1
    # Absolute path to the sklearn golden fixture. Compile-time bake
    # so the test runs from any CWD without env vars.
    NEUROGRAPH_SKLEARN_GOLDEN_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/eval/sklearn_golden.json"
    # Sample datasets and a scratch location, without hard-coding a
    # machine's own folders.
    NEUROGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    NEUROGRAPH_BINARY_DIR="${CMAKE_BINARY_DIR}"
)

set_target_properties(neurograph_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Runtime DLLs must be next to the exe before test discovery runs it
# (post-build steps execute in the order they are declared).
neurograph_attach_asan_runtime(neurograph_tests)
neurograph_copy_runtime_dlls(neurograph_tests)

include(GoogleTest)
gtest_discover_tests(neurograph_tests)

# ---------------------------------------------------------------------------
# neurograph_perf — separate executable for chrono-timed perf gates.
#
# Lives in its own target so devs can run unit tests without paying for
# the multi-second perf benchmarks, and CI can skip/run perf on a
# different cadence (e.g. nightly only). Tests are labeled "perf" via
# gtest_discover_tests so `ctest -L perf` selects only this target.
#
# Build dep on neurograph_predict matches the unit-test target — the
# predict-binary perf test shells out to the built exe.
# ---------------------------------------------------------------------------
add_executable(neurograph_perf
    perf/test_perf_classification_metrics.cpp
    perf/test_perf_predict_binary.cpp
    perf/test_perf_auto_pipeline.cpp
    perf/test_perf_binary_size.cpp
    # P1-09 AC2: headless ImGui frame-time bench. Compiled with the
    # wizard render TUs since they're not in neurograph_lib (the lib
    # avoids the ImGui dependency for unit-test isolation; render
    # files only ship inside neurograph_workbench.exe + this perf
    # target).
    perf/test_perf_wizard_render.cpp
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard/wizard_ui.cpp
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard/step_load.cpp
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard/step_quality.cpp
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard/step_train.cpp
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard/step_results.cpp
    ${NEUROGRAPH_CUDA_LINK_STUB}
)

target_link_libraries(neurograph_perf PRIVATE
    neurograph_lib
    GTest::gtest_main
    imgui::imgui   # P1-09 AC2: wizard render perf test needs ImGui
)

# Wizard render headers live in src/workbench/wizard/. The perf test
# includes "wizard_ui.hpp" with no relative qualifier (matches what
# the workbench exe's CMake does).
target_include_directories(neurograph_perf PRIVATE
    ${CMAKE_SOURCE_DIR}/src/workbench/wizard
)

add_dependencies(neurograph_perf neurograph_predict)
target_compile_definitions(neurograph_perf PRIVATE
    NEUROGRAPH_PREDICT_EXE="$<TARGET_FILE:neurograph_predict>"
)

set_target_properties(neurograph_perf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

neurograph_attach_asan_runtime(neurograph_perf)
neurograph_copy_runtime_dlls(neurograph_perf)

gtest_discover_tests(neurograph_perf
    PROPERTIES LABELS "perf"
)
