#pragma once

#include <iostream>
#include <string>

#include "gemmstone/config.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

// TODO: Remove after upstream work finished
using ngen::utils::rounddown_pow2;
using ngen::utils::roundup_pow2;

namespace dsl {

const std::string &to_string(ngen::ProductFamily family);
const std::string &to_string(ngen::HW hw);
std::string to_string(const ngen::Product &product);

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
