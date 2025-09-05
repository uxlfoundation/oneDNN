/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <string>

#include "gpu/intel/jit/ir/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

std::string to_string(tensor_kind_t tensor) {
    switch (tensor) {
#define CASE(name) \
    case tensor_kind_t::name: return #name
        CASE(src);
        CASE(wei);
        CASE(dst);
        CASE(a);
        CASE(b);
        CASE(c);
#undef CASE
        default: gpu_error_not_expected();
    }
    return {};
}

pvar_t::name_t::name_t(const std::string &s) {
    gpu_assert(!s.empty() && s.length() <= max_len);
    s.copy(data_, s.length());
}

std::string pvar_t::name_t::str() const {
    return std::string(data_);
}

size_t pvar_t::name_t::get_hash() const {
    static_assert(max_len % sizeof(uint64_t) == 0,
            "max_len must be aligned to 64 bits");
    size_t h = 0;
    for (size_t i = 0; i < max_len; i += sizeof(uint64_t)) {
        uint64_t u64 = 0;
        std::memcpy(&u64, &data_[i], sizeof(u64));
        h = hash_combine(h, u64);
    }
    return h;
}

namespace pvars {
pvar_t g("g");
pvar_t ic("ic");
pvar_t id("id");
pvar_t ih("ih");
pvar_t iw("iw");
pvar_t kd("kd");
pvar_t kh("kh");
pvar_t kw("kw");
pvar_t mb("mb");
pvar_t oc("oc");
pvar_t od("od");
pvar_t oh("oh");
pvar_t ow("ow");
pvar_t sd("sd");
pvar_t sh("sh");
pvar_t sw("sw");
pvar_t dd("dd");
pvar_t dh("dh");
pvar_t dw("dw");
pvar_t pd("pd");
pvar_t ph("ph");
pvar_t pw("pw");
pvar_t b("b");
pvar_t m("m");
pvar_t n("n");
pvar_t k("k");
} // namespace pvars

bool is_spatial(const pvar_t &pvar, char prefix) {
    auto s = pvar.str();
    if (s.size() != 2) return false;
    char c0 = s[0];
    char c1 = s[1];
    return (c0 == prefix) && utils::one_of(c1, 'd', 'h', 'w');
}
bool is_input_spatial(const pvar_t &pvar) {
    return is_spatial(pvar, 'i');
}
bool is_output_spatial(const pvar_t &pvar) {
    return is_spatial(pvar, 'o');
}
bool is_kernel_spatial(const pvar_t &pvar) {
    return is_spatial(pvar, 'k');
}
bool is_dilation(const pvar_t &pvar) {
    return is_spatial(pvar, 'd');
}
bool is_padding(const pvar_t &pvar) {
    return is_spatial(pvar, 'p');
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
