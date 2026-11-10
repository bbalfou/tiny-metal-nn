#pragma once
#include <string>

namespace tcnn {
namespace detail {
// Return a build-embedded NN shader, resolving quoted includes against the
// embedded asset tree. Keys are relative to kernels/, including subdirectories.
// SDK includes remain for the Metal compiler. No source-tree files are needed.
const std::string& shader_source(const std::string& filename);

} // namespace detail
} // namespace tcnn
