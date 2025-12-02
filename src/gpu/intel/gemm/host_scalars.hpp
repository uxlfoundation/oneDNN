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

#if 0
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

#else

#if 0
template <typename ScalarType, typename ResultType>
status_t get_scalar_value_as(ResultType &scalar_value,
        const host_scalar_memory_storage_t *scale_storage) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    scalar_value = static_cast<ResultType>(value);
    return status::success;
}

template<typename T>
status_t maybe_get_scalar_value(
        const memory_storage_t &scale_storage, T &scalar_value) {

#define SCALAR_DT_DISPATCH(sdt, vdt) \
    case sdt: { \
        CHECK(get_scalar_value_as<vdt, T>(scalar_value, scalar_storage)); \
        break; \
    }

    using namespace data_type;
    auto scalar_storage = utils::downcast<const host_scalar_memory_storage_t *>(
            &scale_storage);

    switch ((int)scalar_storage->data_type()) {
        if constexpr (std::is_same_v<T, float>) {
            SCALAR_DT_DISPATCH(f32, float)
            SCALAR_DT_DISPATCH(f16, float16_t)
            SCALAR_DT_DISPATCH(bf16, bfloat16_t)
        }
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
#endif

#if 0
template <typename ScalarType, typename ResultType>
status_t get_scalar_value_as(ResultType &scalar_value,
        const host_scalar_memory_storage_t *scale_storage) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    scalar_value = static_cast<ResultType>(value);
    return status::success;
}

template<typename T>
status_t maybe_get_scalar_value(
        const memory_storage_t &scale_storage, T &scalar_value) {

    using namespace data_type;
    auto scalar_storage = utils::downcast<const host_scalar_memory_storage_t *>(
            &scale_storage);

    status_t status = status::success;

    switch ((int)scalar_storage->data_type()) {
        case f32:
            status = get_scalar_value_as<float, T>(scalar_value, scalar_storage);
            break;
        case f16:
            if constexpr (std::is_same_v<T, float>) {
                status = get_scalar_value_as<float16_t, T>(scalar_value, scalar_storage);
            } else {
                assert(!"f16 not supported for non-float types");
            }
            break;
        case bf16:
            if constexpr (std::is_same_v<T, float>) {
                status = get_scalar_value_as<bfloat16_t, T>(scalar_value, scalar_storage);
            } else {
                assert(!"bf16 not supported for non-float types");
            }
            break;
        case s32:
            status = get_scalar_value_as<int32_t, T>(scalar_value, scalar_storage);
            break;
        case s8:
            status = get_scalar_value_as<int8_t, T>(scalar_value, scalar_storage);
            break;
        case u8:
            status = get_scalar_value_as<uint8_t, T>(scalar_value, scalar_storage);
            break;
        default:
            assert(!"Support for requested data type is missing for host-side scalars");
            status = status::invalid_arguments;
    }

    //CHECK(status);
    return status;
}
#endif

#if 1
template <typename ScalarType, typename ResultType>
status_t get_scalar_value_as(ResultType &scalar_value,
        const host_scalar_memory_storage_t *scale_storage) {
    ScalarType value = 0;
    status_t status = scale_storage->get_scalar_value(&value, sizeof(value));
    assert(status == status::success);
    if (status != status::success) return status;

    scalar_value = static_cast<ResultType>(value);
    return status::success;
}

// ?????????? ??? float
inline status_t maybe_get_scalar_value(
        const memory_storage_t &scale_storage, float &scalar_value) {

    using namespace data_type;
    const host_scalar_memory_storage_t *scalar_storage =
        utils::downcast<const host_scalar_memory_storage_t *>(&scale_storage);

    status_t status = status::success;

    switch ((int)scalar_storage->data_type()) {
        case f32:
            status = get_scalar_value_as<float, float>(scalar_value, scalar_storage);
            break;
        case f16:
            status = get_scalar_value_as<float16_t, float>(scalar_value, scalar_storage);
            break;
        case bf16:
            status = get_scalar_value_as<bfloat16_t, float>(scalar_value, scalar_storage);
            break;
        case s32:
            status = get_scalar_value_as<int32_t, float>(scalar_value, scalar_storage);
            break;
        case s8:
            status = get_scalar_value_as<int8_t, float>(scalar_value, scalar_storage);
            break;
        case u8:
            status = get_scalar_value_as<uint8_t, float>(scalar_value, scalar_storage);
            break;
        default:
            assert(!"Support for requested data type is missing for host-side scalars");
            status = status::invalid_arguments;
    }

    CHECK(status);
    return status;
}

// ?????????? ??? int
inline status_t maybe_get_scalar_value(
        const memory_storage_t &scale_storage, int &scalar_value) {

    using namespace data_type;
    const host_scalar_memory_storage_t *scalar_storage =
        utils::downcast<const host_scalar_memory_storage_t *>(&scale_storage);

    status_t status = status::success;

    switch ((int)scalar_storage->data_type()) {
        case s32:
            status = get_scalar_value_as<int32_t, int>(scalar_value, scalar_storage);
            break;
        case s8:
            status = get_scalar_value_as<int8_t, int>(scalar_value, scalar_storage);
            break;
        case u8:
            status = get_scalar_value_as<uint8_t, int>(scalar_value, scalar_storage);
            break;
        default:
            assert(!"Support for requested data type is missing for host-side scalars");
            status = status::invalid_arguments;
    }

    CHECK(status);
    return status;
}
#endif

#endif


} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
