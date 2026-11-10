#include <tiny-metal-nn/detail/shaders.h>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include "nn_shaders_generated.h"

namespace tcnn {
namespace detail {
namespace {
// Look up an immutable shader by its source filename; a typo fails before dispatch.
const std::string& shader_source_unexpanded(const std::string& filename) {
    const auto found = embedded_nn_shaders().find(filename);
    if (found == embedded_nn_shaders().end())
        throw std::invalid_argument("Unknown NN shader asset: " + filename);
    return found->second;
}

// Resolve only embedded assets. Installation and execution never read arbitrary
// host paths or depend on the working directory of the consuming application.
std::string include_key(const std::string& parent, const std::string& name) {
    const std::filesystem::path requested{name};
    if (requested.is_absolute())
        throw std::invalid_argument("NN shader include must be relative: " + name);
    const auto relative = (std::filesystem::path(parent).parent_path() / requested)
                              .lexically_normal()
                              .generic_string();
    const auto& assets = embedded_nn_shaders();
    if (assets.count(relative))
        return relative;
    const auto root_relative = requested.lexically_normal().generic_string();
    if (assets.count(root_relative))
        return root_relative;
    throw std::invalid_argument("Unknown NN shader include '" + name + "' in " + parent);
}

// Expand local includes before passing MSL to metalrt. The Metal compiler still
// evaluates conditional guards/macros and every SDK include itself.
std::string expand_shader(const std::string& filename, std::set<std::string>& active) {
    if (active.size() >= 32 || !active.insert(filename).second)
        throw std::invalid_argument("Cyclic or excessively nested NN shader include: " + filename);
    std::istringstream input{shader_source_unexpanded(filename)};
    std::ostringstream output;
    output << "#line 1 \"" << filename << "\"\n";
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        auto cursor = line.find_first_not_of(" \t");
        if (cursor != std::string::npos && line[cursor] == '#') {
            cursor = line.find_first_not_of(" \t", cursor + 1);
            if (cursor != std::string::npos && line.compare(cursor, 7, "include") == 0) {
                cursor = line.find_first_not_of(" \t", cursor + 7);
                if (cursor != std::string::npos && line[cursor] == '"') {
                    const auto end = line.find('"', cursor + 1);
                    if (end == std::string::npos)
                        throw std::invalid_argument("Unclosed NN shader include in " + filename);
                    const auto key =
                        include_key(filename, line.substr(cursor + 1, end - cursor - 1));
                    output << expand_shader(key, active);
                    output << "#line " << line_number + 1 << " \"" << filename << "\"\n";
                    continue;
                }
            }
        }
        output << line << '\n';
        if (output.tellp() > 16 * 1024 * 1024)
            throw std::invalid_argument("Expanded NN shader exceeds its size limit: " + filename);
    }
    active.erase(filename);
    return output.str();
}
} // namespace

// map nodes keep returned references stable across concurrent cache insertions.
const std::string& shader_source(const std::string& filename) {
    static std::mutex mutex;
    static std::map<std::string, std::string> expanded;
    std::lock_guard<std::mutex> lock{mutex};
    const auto found = expanded.find(filename);
    if (found != expanded.end())
        return found->second;
    std::set<std::string> active;
    return expanded.emplace(filename, expand_shader(filename, active)).first->second;
}

} // namespace detail
} // namespace tcnn
