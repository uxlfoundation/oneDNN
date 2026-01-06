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

// Get value of host side scalar from storage

template <typename ScalarType, typename ResultType>
status_t get_scalar_value(ResultType &scalar_value,
        const host_scalar_memory_storage_t *scale_storage) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    scalar_value = static_cast<ResultType>(value);
    return status::success;
}

inline status_t maybe_get_host_scalar_value(
        const memory_storage_t &mem_storage, float &scalar_value) {
    using namespace data_type;
    const host_scalar_memory_storage_t *scalar_storage
            = utils::downcast<const host_scalar_memory_storage_t *>(
                    &mem_storage);
    status_t status = status::success;

    switch ((int)scalar_storage->data_type()) {
        case f32:
            status = get_scalar_value<float, float>(
                    scalar_value, scalar_storage);
            break;
        case f16:
            status = get_scalar_value<float16_t, float>(
                    scalar_value, scalar_storage);
            break;
        case bf16:
            status = get_scalar_value<bfloat16_t, float>(
                    scalar_value, scalar_storage);
            break;
        case s32:
            status = get_scalar_value<int32_t, float>(
                    scalar_value, scalar_storage);
            break;
        case s8:
            status = get_scalar_value<int8_t, float>(
                    scalar_value, scalar_storage);
            break;
        case u8:
            status = get_scalar_value<uint8_t, float>(
                    scalar_value, scalar_storage);
            break;
        default:
            assert(!"Support for requested data type is missing for host-side "
                    "scalars");
            status = status::invalid_arguments;
    }
    return status;
}

inline status_t maybe_get_host_scalar_value(
        const memory_storage_t &mem_storage, int &scalar_value) {
    using namespace data_type;
    const host_scalar_memory_storage_t *scalar_storage
            = utils::downcast<const host_scalar_memory_storage_t *>(
                    &mem_storage);
    status_t status = status::success;

    switch ((int)scalar_storage->data_type()) {
        case s32:
            status = get_scalar_value<int32_t, int>(
                    scalar_value, scalar_storage);
            break;
        case s8:
            status = get_scalar_value<int8_t, int>(
                    scalar_value, scalar_storage);
            break;
        case u8:
            status = get_scalar_value<uint8_t, int>(
                    scalar_value, scalar_storage);
            break;
        default:
            assert(!"Support for requested data type is missing for host-side "
                    "scalars");
            status = status::invalid_arguments;
    }
    return status;
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
