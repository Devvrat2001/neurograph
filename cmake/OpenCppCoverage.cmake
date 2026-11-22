# cmake/OpenCppCoverage.cmake
#
# Optional Windows-only coverage helper. Wraps OpenCppCoverage so the
# DoD's "≥ 70% on new code" gate can actually be measured rather than
# just claimed.
#
# Why OpenCppCoverage and not gcov/llvm-cov: this project ships against
# MSVC + Visual Studio's PDB format. OpenCppCoverage is the only
# coverage tool that reads PDBs natively without instrumentation
# overhead at compile time — meaning we don't have to maintain a
# parallel "coverage build" with custom flags. It instruments at
# runtime via DbgHelp and PDBs we already produce in Debug.
#
# Usage from a CMakeLists.txt:
#
#   set(NEUROGRAPH_ENABLE_COVERAGE ON CACHE BOOL "Enable coverage helpers")
#   if(NEUROGRAPH_ENABLE_COVERAGE)
#     include(OpenCppCoverage)
#     add_coverage_run(neurograph_coverage
#       TARGET      neurograph_tests
#       SOURCES_DIR ${CMAKE_SOURCE_DIR}/src
#                   ${CMAKE_SOURCE_DIR}/include
#       OUTPUT_DIR  ${CMAKE_BINARY_DIR}/coverage
#     )
#   endif()
#
# Then run:  cmake --build build --target neurograph_coverage --config Debug
# Reports land under: build/coverage/{html/index.html, cobertura.xml}.
#
# Notes:
#   - Coverage requires Debug PDBs, so build the test target with
#     --config Debug. Release strips line info and OpenCppCoverage
#     will report 0% lines covered.
#   - When OpenCppCoverage isn't installed, the helper still defines
#     the target — running it just prints an install hint and exits
#     non-zero so CI fails loudly rather than silently passing.

# --- Discovery ----------------------------------------------------------

# Standard install path produced by the OpenCppCoverage MSI. We also
# check PATH via find_program in case the user has it elsewhere.
find_program(OPENCPPCOVERAGE_EXE
    NAMES OpenCppCoverage
    PATHS
        "$ENV{ProgramFiles}/OpenCppCoverage"
        "$ENV{ProgramFiles\(x86\)}/OpenCppCoverage"
    DOC "Path to OpenCppCoverage.exe (Windows MSVC code coverage tool)"
)

if(OPENCPPCOVERAGE_EXE)
    message(STATUS "Found OpenCppCoverage: ${OPENCPPCOVERAGE_EXE}")
else()
    message(STATUS "OpenCppCoverage not found. Coverage targets will "
                   "exist but emit an install hint when invoked. Install "
                   "from https://github.com/OpenCppCoverage/OpenCppCoverage/releases")
endif()

# --- Public function ----------------------------------------------------

# add_coverage_run(<name>
#   TARGET      <test_executable_target>
#   SOURCES_DIR <dir>...   # one or more "--sources" filters
#   OUTPUT_DIR  <dir>      # report output base
#   [EXCLUDE_DIR <dir>...] # extra "--excluded_sources" filters
# )
#
# Creates a custom target `<name>` that, when built, runs the test
# executable under OpenCppCoverage and emits HTML + Cobertura reports.
function(add_coverage_run name)
    set(options)
    set(oneValueArgs TARGET OUTPUT_DIR)
    set(multiValueArgs SOURCES_DIR EXCLUDE_DIR)
    cmake_parse_arguments(COV "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(NOT COV_TARGET)
        message(FATAL_ERROR "add_coverage_run(${name}): TARGET is required")
    endif()
    if(NOT COV_SOURCES_DIR)
        message(FATAL_ERROR "add_coverage_run(${name}): SOURCES_DIR is required")
    endif()
    if(NOT COV_OUTPUT_DIR)
        set(COV_OUTPUT_DIR "${CMAKE_BINARY_DIR}/coverage")
    endif()

    # Build the OpenCppCoverage argument list. --sources filters which
    # source files count toward the percentage; everything else
    # (vcpkg, system headers, GoogleTest) is treated as ignored cover
    # rather than missing cover, which is exactly what we want — we're
    # gating on *our* code's coverage, not the world's.
    #
    # The filters are matched against the paths stored in the PDBs,
    # which use backslashes, and OpenCppCoverage refuses a filter that
    # contains '/' or a "." or ".." part. CMake paths use '/', so pass
    # each one in native form.
    set(_src_args "")
    foreach(_dir IN LISTS COV_SOURCES_DIR)
        cmake_path(NATIVE_PATH _dir NORMALIZE _native_dir)
        list(APPEND _src_args --sources "${_native_dir}")
    endforeach()
    foreach(_dir IN LISTS COV_EXCLUDE_DIR)
        cmake_path(NATIVE_PATH _dir NORMALIZE _native_dir)
        list(APPEND _src_args --excluded_sources "${_native_dir}")
    endforeach()

    if(OPENCPPCOVERAGE_EXE)
        # Real run. Two outputs:
        #   1. html/  — human-readable, auto-opened by devs.
        #   2. cobertura.xml — Jenkins/Azure DevOps/GitHub Actions
        #                      parse this directly to publish coverage
        #                      summaries on PRs.
        add_custom_target(${name}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${COV_OUTPUT_DIR}"
            COMMAND ${OPENCPPCOVERAGE_EXE}
                    --quiet
                    --export_type=html:${COV_OUTPUT_DIR}/html
                    --export_type=cobertura:${COV_OUTPUT_DIR}/cobertura.xml
                    --working_dir "${CMAKE_BINARY_DIR}"
                    ${_src_args}
                    --
                    $<TARGET_FILE:${COV_TARGET}>
            COMMENT "Running ${COV_TARGET} under OpenCppCoverage → ${COV_OUTPUT_DIR}"
            DEPENDS ${COV_TARGET}
            VERBATIM
            USES_TERMINAL  # let test stdout stream so progress is visible
        )
    else()
        # Stub target. Prints install instructions and exits non-zero
        # so a CI step that includes this in its build graph fails
        # loudly rather than recording zero coverage.
        add_custom_target(${name}
            COMMAND ${CMAKE_COMMAND} -E echo
                    "ERROR: OpenCppCoverage not installed — coverage target ${name} cannot run."
            COMMAND ${CMAKE_COMMAND} -E echo
                    "Install from: https://github.com/OpenCppCoverage/OpenCppCoverage/releases"
            COMMAND ${CMAKE_COMMAND} -E false
            COMMENT "Coverage stub: OpenCppCoverage not found"
        )
    endif()
endfunction()
