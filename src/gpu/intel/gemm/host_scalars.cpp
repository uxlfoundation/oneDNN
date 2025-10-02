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

#include "gpu/intel/gemm/host_scalars.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

status_t maybe_get_scale_as_float(
        const memory_storage_t &scale_storage, float &scalar_value) {
#define SCALAR_DT_DISPATCH(sdt, vdt) \
    case sdt: { \
        CHECK(get_scalar_value_as_float<vdt>(scalar_value, scalar_storage)); \
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
