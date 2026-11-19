#pragma once

// The product version, from `project(... VERSION x.y.z)` in the top-level
// CMakeLists.txt (passed in as a compile definition).

#ifndef NEUROGRAPH_VERSION_STRING
#define NEUROGRAPH_VERSION_STRING "0.0.0-dev"
#endif

namespace neurograph {

inline constexpr const char* kVersion = NEUROGRAPH_VERSION_STRING;

} // namespace neurograph
