/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_HOST_SCALARS_HPP
#define GPU_INTEL_GEMM_HOST_SCALARS_HPP

#include "common/host_scalar_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

// Host side scalars support for scales implemented via Alpha argument

template <typename ScalarType>
status_t update_alpha_by_scale_value(float &alpha,
        const host_scalar_memory_storage_t *scale_storage, bool is_dst) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    if (is_dst) {
        alpha /= static_cast<float>(value);
    } else {
        alpha *= static_cast<float>(value);
    }
    return status::success;
}

static status_t maybe_convert_scales_to_alpha(
        const memory_storage_t &scale_storage, float &alpha,
        const bool is_dst = false) {
#define SCALAR_DT_DISPATCH(sdt, vdt) \
    case sdt: { \
        CHECK(update_alpha_by_scale_value<vdt>( \
                alpha, scalar_storage, is_dst)); \
        break; \
    }

    using namespace data_type;
    auto scalar_storage = utils::downcast<const host_scalar_memory_storage_t *>(
            &scale_storage);
    switch ((int)scalar_storage->data_type()) {
        SCALAR_DT_DISPATCH(f32, float)
        SCALAR_DT_DISPATCH(f16, float16_t)
        SCALAR_DT_DISPATCH(bf16, bfloat16_t)
        SCALAR_DT_DISPATCH(s32, int32_t)
        SCALAR_DT_DISPATCH(s8, int8_t)
        SCALAR_DT_DISPATCH(u8, uint8_t)
        default:
            assert(!"Support for requested data type is missing for "
                    "host-side scalars");
    }
    return status::success;
#undef SCALAR_DT_DISPATCH
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
