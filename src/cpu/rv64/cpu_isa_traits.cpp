/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct isa_info_t {
    isa_info_t(cpu_isa_t aisa) : isa(aisa) {};
    cpu_isa_t isa;
};

static isa_info_t get_isa_info_t(void) {
    if (mayiuse(zvfh)) return isa_info_t(zvfh);
    if (mayiuse(v)) return isa_info_t(v);
    return isa_info_t(isa_undef);
}

cpu_isa_t get_max_cpu_isa() {
    return get_isa_info_t().isa;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
