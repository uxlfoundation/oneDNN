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

#ifndef XPU_ZE_UTILS_HPP
#define XPU_ZE_UTILS_HPP

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include "level_zero/ze_api.h"

#include "xpu/utils.hpp"

#include <mutex>

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

static inline std::string to_string(ze_result_t r) {
#define ZE_STATUS_CASE(status) \
    case status: return #status
    switch (r) {
        ZE_STATUS_CASE(ZE_RESULT_SUCCESS);
        ZE_STATUS_CASE(ZE_RESULT_NOT_READY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_LOST);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_LINK_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_NOT_AVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNINITIALIZED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ARGUMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNKNOWN);
        ZE_STATUS_CASE(ZE_RESULT_FORCE_UINT32);
        default: return std::to_string((int)r);
    }
#undef ZE_STATUS_CASE
}

#define MAYBE_REPORT_ZE_ERROR(str) \
    do { \
        VERROR(primitive, level_zero, "errcode %s", str); \
    } while (false)

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::string err_str_ = xpu::ze::to_string(res_); \
            MAYBE_REPORT_ZE_ERROR(err_str_.c_str()); \
            return status::runtime_error; \
        } \
    } while (false)

#define ZE_CHECK_V(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::string err_str_ = xpu::ze::to_string(res_); \
            MAYBE_REPORT_ZE_ERROR(err_str_.c_str()); \
            return; \
        } \
    } while (false)

ze_result_t ze_initialize();

#if defined(_WIN32)
#define ZE_LIB_NAME "ze_loader.dll"
#elif defined(__linux__)
#define ZE_LIB_NAME "libze_loader.so.1"
#endif

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)xpu::find_symbol(ZE_LIB_NAME, symbol);
}
#undef ZE_LIB_NAME

#define INDIRECT_ZE_CALL(f) \
    template <typename... Args> \
    ze_result_t f(Args &&...args) { \
        ze_result_t init_res_ = ze_initialize(); \
        if (init_res_ != ZE_RESULT_SUCCESS) return init_res_; \
        static auto f_ = find_ze_symbol<decltype(&::f)>(#f); \
        if (!f_) return ZE_RESULT_ERROR_UNINITIALIZED; \
        return (f_(std::forward<Args>(args)...)); \
    }
INDIRECT_ZE_CALL(zeDriverGet)
INDIRECT_ZE_CALL(zeDriverGetProperties)
INDIRECT_ZE_CALL(zeDriverGetExtensionFunctionAddress)
INDIRECT_ZE_CALL(zeDeviceGet)
INDIRECT_ZE_CALL(zeDeviceGetProperties)
INDIRECT_ZE_CALL(zeDeviceGetComputeProperties)
INDIRECT_ZE_CALL(zeDeviceGetModuleProperties)
INDIRECT_ZE_CALL(zeDeviceGetMemoryAccessProperties)
INDIRECT_ZE_CALL(zeDeviceGetCacheProperties)
INDIRECT_ZE_CALL(zeContextCreate)
INDIRECT_ZE_CALL(zeContextDestroy)
INDIRECT_ZE_CALL(zeCommandListCreateImmediate)
INDIRECT_ZE_CALL(zeCommandListDestroy)
INDIRECT_ZE_CALL(zeCommandListHostSynchronize)
INDIRECT_ZE_CALL(zeCommandListGetContextHandle)
INDIRECT_ZE_CALL(zeCommandListAppendBarrier)
INDIRECT_ZE_CALL(zeCommandListAppendMemoryCopy)
INDIRECT_ZE_CALL(zeCommandListAppendMemoryFill)
INDIRECT_ZE_CALL(zeEventPoolCreate)
INDIRECT_ZE_CALL(zeEventPoolDestroy)
INDIRECT_ZE_CALL(zeEventCreate)
INDIRECT_ZE_CALL(zeEventDestroy)
INDIRECT_ZE_CALL(zeEventHostSynchronize)
INDIRECT_ZE_CALL(zeEventQueryKernelTimestamp)
INDIRECT_ZE_CALL(zeEventQueryStatus)
INDIRECT_ZE_CALL(zeMemAllocShared)
INDIRECT_ZE_CALL(zeMemAllocDevice)
INDIRECT_ZE_CALL(zeMemAllocHost)
INDIRECT_ZE_CALL(zeMemFree)
INDIRECT_ZE_CALL(zeMemGetAllocProperties)
INDIRECT_ZE_CALL(zeModuleCreate)
INDIRECT_ZE_CALL(zeModuleDestroy)
INDIRECT_ZE_CALL(zeModuleBuildLogGetString)
INDIRECT_ZE_CALL(zeModuleBuildLogDestroy)
INDIRECT_ZE_CALL(zeModuleGetKernelNames)
INDIRECT_ZE_CALL(zeModuleGetNativeBinary)
INDIRECT_ZE_CALL(zeKernelCreate)
INDIRECT_ZE_CALL(zeKernelDestroy)
INDIRECT_ZE_CALL(zeKernelSetArgumentValue)
INDIRECT_ZE_CALL(zeKernelGetName)
INDIRECT_ZE_CALL(zeKernelGetBinaryExp)
INDIRECT_ZE_CALL(zeKernelSetGroupSize)
INDIRECT_ZE_CALL(zeKernelSuggestGroupSize)
INDIRECT_ZE_CALL(zeCommandListAppendLaunchKernel)
#undef INDIRECT_ZE_CALL

// Level Zero objects destroy traits
template <typename T>
struct destroy_traits;
// {
//     static void destroy(T t) {}
// };

template <>
struct destroy_traits<ze_command_list_handle_t> {
    static void destroy(ze_command_list_handle_t t) {
        ZE_CHECK_V(xpu::ze::zeCommandListHostSynchronize(t, UINT64_MAX));
        auto ze_res = xpu::ze::zeCommandListDestroy(t);
        if (ze_res != ZE_RESULT_SUCCESS
                && ze_res != ZE_RESULT_ERROR_UNINITIALIZED) {
            ZE_CHECK_V(ze_res);
        }
    }
};

template <>
struct destroy_traits<ze_context_handle_t> {
    static void destroy(ze_context_handle_t t) {
        auto ze_res = xpu::ze::zeContextDestroy(t);
        if (ze_res != ZE_RESULT_SUCCESS
                && ze_res != ZE_RESULT_ERROR_UNINITIALIZED) {
            ZE_CHECK_V(ze_res);
        }
    }
};

template <>
struct destroy_traits<ze_event_handle_t> {
    static void destroy(ze_event_handle_t t) {
        ZE_CHECK_V(xpu::ze::zeEventHostSynchronize(t, UINT64_MAX));
        ZE_CHECK_V(xpu::ze::zeEventDestroy(t));
    }
};

template <>
struct destroy_traits<ze_event_pool_handle_t> {
    static void destroy(ze_event_pool_handle_t t) {
        auto ze_res = xpu::ze::zeEventPoolDestroy(t);
        if (ze_res != ZE_RESULT_SUCCESS
                && ze_res != ZE_RESULT_ERROR_UNINITIALIZED) {
            ZE_CHECK_V(ze_res);
        }
    }
};

// Kernels are part of the global static kernel cache object. It gets destroyed
// at static objects destruction point of program destruction. This moment
// happens after `atexit` system call, when Level Zero library will be unloaded.
// The unloaded library means there are no Level Zero resources that can be used
// for proper kernels destruction leading to reporting errors from this
// wrapper. This error reports specific `ZE_RESULT_ERROR_UNINITIALIZED` error
// code. Don't print an error message for this and successful codes.
//
// Note: since engine object is a singleton in benchdnn, objects created during
// engine construction shouldn't report `ZE_RESULT_ERROR_UNINITIALIZED` error
// message, too. Such objects are a context, a list, an event pool and modules
// associated with kernels.
template <>
struct destroy_traits<ze_kernel_handle_t> {
    static void destroy(ze_kernel_handle_t t) {
        auto ze_res = xpu::ze::zeKernelDestroy(t);
        if (ze_res != ZE_RESULT_SUCCESS
                && ze_res != ZE_RESULT_ERROR_UNINITIALIZED) {
            ZE_CHECK_V(ze_res);
        }
    }
};

template <>
struct destroy_traits<ze_module_handle_t> {
    static void destroy(ze_module_handle_t t) {
        auto ze_res = xpu::ze::zeModuleDestroy(t);
        if (ze_res != ZE_RESULT_SUCCESS
                && ze_res != ZE_RESULT_ERROR_UNINITIALIZED) {
            ZE_CHECK_V(ze_res);
        }
    }
};

template <typename T>
class wrapper_t {
public:
    wrapper_t(T t = nullptr, bool is_owner = true)
        : t_(t), is_owner_(is_owner) {}
    wrapper_t(const wrapper_t<T> &) = delete;
    wrapper_t &operator=(const wrapper_t<T> &) = delete;
    ~wrapper_t() { do_destroy(); }

    operator T() const { return t_; }
    T get() const { return t_; }
    // `unwrap` interfaces return a reference to the underlying object allowing
    // create an empty wrapper, "unwrap" its content to the correcpondent call
    // and fill it without additional actions.
    T &unwrap() { return t_; }
    const T &unwrap() const { return t_; }

private:
    T t_;
    bool is_owner_;

    void do_destroy() {
        if (is_owner_ && t_) { destroy_traits<T>::destroy(t_); }
    }
};

xpu::device_uuid_t get_device_uuid(ze_device_handle_t device);
status_t get_device_index(size_t *index, ze_device_handle_t device);
std::string get_kernel_name(ze_kernel_handle_t kernel);
ze_memory_type_t get_pointer_type(ze_context_handle_t, const void *ptr);
status_t append_memory_copy(ze_command_list_handle_t list,
        std::mutex &list_mutex, void *dst, const void *src, size_t size,
        ze_event_handle_t out_event, uint32_t num_deps_events,
        ze_event_handle_t *deps_events);
status_t append_memory_fill(ze_command_list_handle_t list,
        std::mutex &list_mutex, void *dst, const void *pattern,
        size_t pattern_size, size_t size, ze_event_handle_t out_event,
        uint32_t num_deps_events, ze_event_handle_t *deps_events);

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_UTILS_HPP
