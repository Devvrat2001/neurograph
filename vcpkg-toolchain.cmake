# Minimal vcpkg toolchain pointing to pre-installed packages
set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/build/vcpkg_installed")
set(VCPKG_TARGET_TRIPLET "x64-windows")

# Add the installed packages to the prefix path
list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/x64-windows")
list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/x64-windows/share")

# Include directories
include_directories("${VCPKG_INSTALLED_DIR}/x64-windows/include")

# Library directories
link_directories("${VCPKG_INSTALLED_DIR}/x64-windows/lib")
