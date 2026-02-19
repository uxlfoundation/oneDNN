/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef DNNL_TEST_COMMON_ZE_HPP
#define DNNL_TEST_COMMON_ZE_HPP

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_ze.hpp"
#include "gtest/gtest.h"

// Define a separate macro, that does not clash with ZE_CHECK from the library.
#ifdef DNNL_ENABLE_MEM_DEBUG

namespace mem_debug_utils {
// Copy-pasted from src/xpu/ze/utils.hpp::to_string() to avoid including
// .cpp file or exposing the symbol.
inline dnnl_status_t convert_to_dnnl(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return dnnl_success;
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return dnnl_out_of_memory;
        case ZE_RESULT_NOT_READY:
        case ZE_RESULT_ERROR_DEVICE_LOST:
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
        case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
        case ZE_RESULT_ERROR_NOT_AVAILABLE:
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
        case ZE_RESULT_ERROR_UNINITIALIZED:
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        case ZE_RESULT_ERROR_INVALID_SIZE:
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        case ZE_RESULT_ERROR_INVALID_ENUMERATION:
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
        case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
        case ZE_RESULT_ERROR_UNKNOWN:
        case ZE_RESULT_FORCE_UINT32:
        default: return dnnl_runtime_error;
    }
}

} // namespace mem_debug_utils

#define TEST_ZE_CHECK(x) \
    do { \
        dnnl_status_t s = mem_debug_utils::convert_to_dnnl(x); \
        dnnl::error::wrap_c_api(s, dnnl_status2str(s)); \
    } while (0)
#else
#define TEST_ZE_CHECK(x) \
    do { \
        int s = int(x); \
        EXPECT_EQ(s, ZE_RESULT_SUCCESS) << "Level Zero error: " << s; \
    } while (0)
#endif

// If there's one device found, return `true`, otherwise, `false`.
inline bool find_ze_device(ze_driver_handle_t *adriver = nullptr,
        ze_device_handle_t *adevice = nullptr,
        ze_context_handle_t *acontext = nullptr) {
    uint32_t driver_count = 1;
    ze_init_driver_type_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
    desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
    ze_driver_handle_t local_driver = nullptr;
    TEST_ZE_CHECK(zeInitDrivers(&driver_count, &local_driver, &desc));
    if (adriver) *adriver = local_driver;

    uint32_t device_count = 1;
    ze_device_handle_t local_device = nullptr;
    TEST_ZE_CHECK(zeDeviceGet(local_driver, &device_count, &local_device));
    if (adevice) *adevice = local_device;

    if (acontext) {
        ze_context_desc_t context_desc = {};
        context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        ze_context_handle_t local_context = nullptr;
        TEST_ZE_CHECK(
                zeContextCreate(local_driver, &context_desc, &local_context));
        *acontext = local_context;
    }

    return (bool)local_device;
}

// Returns a command list handle which user must destroy at the end.
inline ze_command_list_handle_t create_command_list(ze_context_handle_t ze_ctx,
        ze_device_handle_t ze_dev, bool in_order = true) {
    ze_command_queue_desc_t command_queue_desc {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.ordinal = 0;
    command_queue_desc.index = 0;
    // Note: when there's no in_order flag, it means out_of_order.
    command_queue_desc.flags = in_order ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t interop_ze_list = nullptr;
    TEST_ZE_CHECK(zeCommandListCreateImmediate(
            ze_ctx, ze_dev, &command_queue_desc, &interop_ze_list));
    return interop_ze_list;
}

#endif
