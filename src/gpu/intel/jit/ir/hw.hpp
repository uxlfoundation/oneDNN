/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_HW_HPP
#define GPU_INTEL_JIT_IR_HW_HPP

#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/ir/include/hw.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

inline hw_t make_ir_hw(const impl::engine_t *engine) {
    using namespace compute;
    auto intel_engine = utils::downcast<const engine_t *>(engine);

    auto *device_info = intel_engine->device_info();
    auto product = get_ngen_product(*device_info);
    int eu_count = device_info->eu_count();
    int max_wg_size = static_cast<int>(
            device_info->max_wg_size(/*large_grf_mode=*/false));
    size_t l3_cache_size = device_info->l3_cache_size();
    hw::attr_t attr = hw::attr_t::none;
    if (intel_engine->mayiuse_large_grf_mode()) attr |= hw::attr_t::large_grf;
    if (device_info->mayiuse_systolic()) attr |= hw_t::attr_t::systolic;
    if (device_info->mayiuse_float_atomic_add(data_type::f64))
        attr |= hw_t::attr_t::atomic_fp64;

    return hw_t(product, eu_count, max_wg_size, l3_cache_size, attr);
}

inline bool prefer_large_grf(
        const hw_t &hw, const gpu_primitive_attr_t *gpu_attr) {
    if (!gpu_attr || !hw.large_grf_support()) return false;
    return gpu_attr->threads_per_eu() * 2 == hw.threads_per_eu();
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
