/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_COMMON_ZEN_UNPACK_FALLBACK_HPP
#define CPU_X64_ZEN64_COMMON_ZEN_UNPACK_FALLBACK_HPP

#include <memory>
#include <string>

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {

// Runs a matmul with zen_packed weights that the Zen kernel cannot execute
// (for example, because of its attributes). No other implementation reads
// the opaque layout, so the weights are unpacked on every execution into a
// plain scratchpad buffer, and the whole problem, with the original
// attributes, runs through the first implementation that accepts it on plain
// weights.
//
// The pd owns an instance after init() / init_scratchpad(); the primitive
// keeps a copy and calls init_primitives() once.
struct unpack_fallback_t {
    // The descriptors of pd must be resolved (no format_any).
    status_t init(const engine_t *engine, const matmul_pd_t &pd);
    status_t init_primitives(engine_t *engine);
    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;

    bool enabled() const { return matmul_pd_ != nullptr; }
    // "zen:unpack+" followed by the name of the implementation that runs.
    const char *name() const { return name_.c_str(); }

    status_t execute(const exec_ctx_t &ctx) const;

private:
    memory_desc_t plain_wei_md_ {};
    std::shared_ptr<primitive_desc_t> unpack_pd_;
    std::shared_ptr<primitive_desc_t> matmul_pd_;
    std::shared_ptr<primitive_t> unpack_;
    std::shared_ptr<primitive_t> matmul_;
    std::string name_;
};

} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
