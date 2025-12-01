/*******************************************************************************
* Copyright 2020 Intel Corporation
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

// @@@@@@@@@@@@ unify ??????????????????
// Get value of host side scalar from storage and convert to float

template <typename ScalarType>
status_t get_scalar_value_as_float(float &scalar_value,
        const host_scalar_memory_storage_t *scale_storage) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    scalar_value = static_cast<float>(value);
    return status::success;
}

status_t maybe_get_scale_as_float(
        const memory_storage_t &scale_storage, float &scalar_value);

template <typename ScalarType>
status_t get_scalar_value_as_int(int &scalar_value,
        const host_scalar_memory_storage_t *scale_storage) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    scalar_value = static_cast<int>(value);
    return status::success;
}

status_t maybe_get_zp_as_int(
        const memory_storage_t &scale_storage, int &scalar_value);

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
