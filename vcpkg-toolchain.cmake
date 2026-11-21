# Minimal vcpkg toolchain pointing at pre-installed packages.
#
# Use this when the packages already sit in build/vcpkg_installed and you
# do not want (or have) the full vcpkg checkout on the machine. It works
# from any build directory:
#
#   cmake -S . -B build-nocuda -DCMAKE_TOOLCHAIN_FILE=vcpkg-toolchain.cmake
#
# With a real vcpkg checkout, pass its scripts/buildsystems/vcpkg.cmake
# instead and manifest mode installs everything from vcpkg.json.

set(VCPKG_INSTALLED_DIR "${CMAKE_CURRENT_LIST_DIR}/build/vcpkg_installed"
    CACHE PATH "Root of the pre-installed vcpkg packages")
set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg triplet")

# Some vcpkg package configs (libsodium's, for one) read the toolchain's
# internal spelling of the same directory.
set(_VCPKG_INSTALLED_DIR "${VCPKG_INSTALLED_DIR}")

# Add the installed packages to the prefix path
list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share")

# Packages ship Release and Debug binaries only.
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL     Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

# Include directories
include_directories("${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")

# Library directories
link_directories("${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")

# Tell the main CMakeLists that it must copy runtime DLLs next to the
# executables itself (the real vcpkg toolchain does this automatically).
set(NEUROGRAPH_COPY_VCPKG_DLLS ON)
