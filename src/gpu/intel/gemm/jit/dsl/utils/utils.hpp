#pragma once

#include <iostream>
#include <string>

#include "gemmstone/config.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

namespace dsl {

inline bool stream_try_match(std::istream &in, const std::string &s) {
    in >> std::ws;
    auto pos = in.tellg();
    bool ok = true;
    for (auto &c : s) {
        if (in.get() != c || in.fail()) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        in.clear();
        in.seekg(pos);
    }
    return ok;
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
