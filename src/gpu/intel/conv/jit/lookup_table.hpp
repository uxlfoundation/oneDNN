/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_INTEL_CONV_JIT_LOOKUP_TABLE_HPP
#define GPU_INTEL_CONV_JIT_LOOKUP_TABLE_HPP

#include <iostream>
#include <unordered_map>

#include "gpu/intel/conv/jit/key.hpp"
#include "gpu/intel/jit/ir/blocking.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

using blocking_params_t = intel::jit::blocking_params_t;

class lookup_table_t {
public:
    struct entry_t {
        key_t key;
        blocking_params_t params;

        void stringify(std::ostream &out) const;
        void parse(std::istream &in);
    };

    lookup_table_t() = default;
    lookup_table_t(const char **entries);

    void set(const key_t &key, const blocking_params_t &params);
    void merge(const lookup_table_t &other);
    blocking_params_t find(const key_t &key) const;
    bool is_empty() const { return data_.empty(); }
    void stringify(std::ostream &out) const;
    void parse(std::istream &in);

private:
    std::unordered_map<std::string, std::vector<entry_t>> data_;
};

const lookup_table_t &const_lookup_table();
lookup_table_t &lookup_table();

} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
