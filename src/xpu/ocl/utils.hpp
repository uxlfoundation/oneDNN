/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef XPU_OCL_UTILS_HPP
#define XPU_OCL_UTILS_HPP

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_config.h"

#include "common/c_types_map.hpp"
#include "common/cpp_compat.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

inline status_t convert_to_dnnl(cl_int cl_status) {
    switch (cl_status) {
        case CL_SUCCESS: return status::success;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY: return status::out_of_memory;
        case CL_DEVICE_NOT_FOUND:
        case CL_DEVICE_NOT_AVAILABLE:
        case CL_COMPILER_NOT_AVAILABLE:
        case CL_PROFILING_INFO_NOT_AVAILABLE:
        case CL_MEM_COPY_OVERLAP:
        case CL_IMAGE_FORMAT_MISMATCH:
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        case CL_BUILD_PROGRAM_FAILURE:
        case CL_MAP_FAILURE:
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        case CL_COMPILE_PROGRAM_FAILURE:
        case CL_LINKER_NOT_AVAILABLE:
        case CL_LINK_PROGRAM_FAILURE:
        case CL_DEVICE_PARTITION_FAILED:
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        case CL_INVALID_PLATFORM:
        case CL_INVALID_DEVICE: return status::runtime_error;
        case CL_INVALID_VALUE:
        case CL_INVALID_DEVICE_TYPE:
        case CL_INVALID_CONTEXT:
        case CL_INVALID_QUEUE_PROPERTIES:
        case CL_INVALID_COMMAND_QUEUE:
        case CL_INVALID_HOST_PTR:
        case CL_INVALID_MEM_OBJECT:
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        case CL_INVALID_IMAGE_SIZE:
        case CL_INVALID_SAMPLER:
        case CL_INVALID_BINARY:
        case CL_INVALID_BUILD_OPTIONS:
        case CL_INVALID_PROGRAM:
        case CL_INVALID_PROGRAM_EXECUTABLE:
        case CL_INVALID_KERNEL_NAME:
        case CL_INVALID_KERNEL_DEFINITION:
        case CL_INVALID_KERNEL:
        case CL_INVALID_ARG_INDEX:
        case CL_INVALID_ARG_VALUE:
        case CL_INVALID_ARG_SIZE:
        case CL_INVALID_KERNEL_ARGS:
        case CL_INVALID_WORK_DIMENSION:
        case CL_INVALID_WORK_GROUP_SIZE:
        case CL_INVALID_WORK_ITEM_SIZE:
        case CL_INVALID_GLOBAL_OFFSET:
        case CL_INVALID_EVENT_WAIT_LIST:
        case CL_INVALID_EVENT:
        case CL_INVALID_OPERATION:
        case CL_INVALID_GL_OBJECT:
        case CL_INVALID_BUFFER_SIZE:
        case CL_INVALID_MIP_LEVEL:
        case CL_INVALID_GLOBAL_WORK_SIZE: return status::invalid_arguments;

        default: return status::runtime_error;
    }
}

// Ordered by value as defined by opencl
inline const char *convert_cl_int_to_str(cl_int cl_status) {
#define CL_STATUS_CASE(status) \
    case status: return #status
    switch (cl_status) {
        CL_STATUS_CASE(CL_SUCCESS);
        CL_STATUS_CASE(CL_DEVICE_NOT_FOUND);
        CL_STATUS_CASE(CL_DEVICE_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_COMPILER_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CL_STATUS_CASE(CL_OUT_OF_RESOURCES);
        CL_STATUS_CASE(CL_OUT_OF_HOST_MEMORY);
        CL_STATUS_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_MEM_COPY_OVERLAP);
        CL_STATUS_CASE(CL_IMAGE_FORMAT_MISMATCH);
        CL_STATUS_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CL_STATUS_CASE(CL_BUILD_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_MAP_FAILURE);
        CL_STATUS_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        CL_STATUS_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        CL_STATUS_CASE(CL_COMPILE_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_LINKER_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_LINK_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_DEVICE_PARTITION_FAILED);
        CL_STATUS_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_INVALID_VALUE);
        CL_STATUS_CASE(CL_INVALID_DEVICE_TYPE);
        CL_STATUS_CASE(CL_INVALID_PLATFORM);
        CL_STATUS_CASE(CL_INVALID_DEVICE);
        CL_STATUS_CASE(CL_INVALID_CONTEXT);
        CL_STATUS_CASE(CL_INVALID_QUEUE_PROPERTIES);
        CL_STATUS_CASE(CL_INVALID_COMMAND_QUEUE);
        CL_STATUS_CASE(CL_INVALID_HOST_PTR);
        CL_STATUS_CASE(CL_INVALID_MEM_OBJECT);
        CL_STATUS_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CL_STATUS_CASE(CL_INVALID_IMAGE_SIZE);
        CL_STATUS_CASE(CL_INVALID_SAMPLER);
        CL_STATUS_CASE(CL_INVALID_BINARY);
        CL_STATUS_CASE(CL_INVALID_BUILD_OPTIONS);
        CL_STATUS_CASE(CL_INVALID_PROGRAM);
        CL_STATUS_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        CL_STATUS_CASE(CL_INVALID_KERNEL_NAME);
        CL_STATUS_CASE(CL_INVALID_KERNEL_DEFINITION);
        CL_STATUS_CASE(CL_INVALID_KERNEL);
        CL_STATUS_CASE(CL_INVALID_ARG_INDEX);
        CL_STATUS_CASE(CL_INVALID_ARG_VALUE);
        CL_STATUS_CASE(CL_INVALID_ARG_SIZE);
        CL_STATUS_CASE(CL_INVALID_KERNEL_ARGS);
        CL_STATUS_CASE(CL_INVALID_WORK_DIMENSION);
        CL_STATUS_CASE(CL_INVALID_WORK_GROUP_SIZE);
        CL_STATUS_CASE(CL_INVALID_WORK_ITEM_SIZE);
        CL_STATUS_CASE(CL_INVALID_GLOBAL_OFFSET);
        CL_STATUS_CASE(CL_INVALID_EVENT_WAIT_LIST);
        CL_STATUS_CASE(CL_INVALID_EVENT);
        CL_STATUS_CASE(CL_INVALID_OPERATION);
        CL_STATUS_CASE(CL_INVALID_GL_OBJECT);
        CL_STATUS_CASE(CL_INVALID_BUFFER_SIZE);
        CL_STATUS_CASE(CL_INVALID_MIP_LEVEL);
        CL_STATUS_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
#undef CL_STATUS_CASE
        default: return "unknown macro name";
    }
}

#define MAYBE_REPORT_ERROR(msg) \
    do { \
        VERROR(primitive, gpu, msg); \
    } while (0)

#define MAYBE_REPORT_OCL_ERROR(s) \
    do { \
        VERROR(primitive, ocl, "errcode %d,%s,%s:%d", int(s), \
                dnnl::impl::xpu::ocl::convert_cl_int_to_str(s), __FILENAME__, \
                __LINE__); \
    } while (0)

#define OCL_CHECK_V(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { \
            MAYBE_REPORT_OCL_ERROR(s); \
            return; \
        } \
    } while (0)

#define OCL_CHECK(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { \
            MAYBE_REPORT_OCL_ERROR(s); \
            return dnnl::impl::xpu::ocl::convert_to_dnnl(s); \
        } \
    } while (0)

#define UNUSED_OCL_RESULT(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { MAYBE_REPORT_OCL_ERROR(s); } \
        assert(s == CL_SUCCESS); \
        MAYBE_UNUSED(s); \
    } while (false)

// OpenCL objects reference counting traits
template <typename T>
struct ref_traits;
//{
//    static void retain(T t) {}
//    static void release(T t) {}
//};

template <>
struct ref_traits<cl_context> {
    static void retain(cl_context t) { UNUSED_OCL_RESULT(clRetainContext(t)); }
    static void release(cl_context t) {
        UNUSED_OCL_RESULT(clReleaseContext(t));
    }
};

template <>
struct ref_traits<cl_command_queue> {
    static void retain(cl_command_queue t) {
        UNUSED_OCL_RESULT(clRetainCommandQueue(t));
    }
    static void release(cl_command_queue t) {
        UNUSED_OCL_RESULT(clReleaseCommandQueue(t));
    }
};

template <>
struct ref_traits<cl_program> {
    static void retain(cl_program t) { UNUSED_OCL_RESULT(clRetainProgram(t)); }
    static void release(cl_program t) {
        UNUSED_OCL_RESULT(clReleaseProgram(t));
    }
};

template <>
struct ref_traits<cl_kernel> {
    static void retain(cl_kernel t) { UNUSED_OCL_RESULT(clRetainKernel(t)); }
    static void release(cl_kernel t) { UNUSED_OCL_RESULT(clReleaseKernel(t)); }
};

template <>
struct ref_traits<cl_mem> {
    static void retain(cl_mem t) { UNUSED_OCL_RESULT(clRetainMemObject(t)); }
    static void release(cl_mem t) { UNUSED_OCL_RESULT(clReleaseMemObject(t)); }
};

template <>
struct ref_traits<cl_sampler> {
    static void retain(cl_sampler t) { UNUSED_OCL_RESULT(clRetainSampler(t)); }
    static void release(cl_sampler t) {
        UNUSED_OCL_RESULT(clReleaseSampler(t));
    }
};

template <>
struct ref_traits<cl_event> {
    static void retain(cl_event t) { UNUSED_OCL_RESULT(clRetainEvent(t)); }
    static void release(cl_event t) { UNUSED_OCL_RESULT(clReleaseEvent(t)); }
};

template <>
struct ref_traits<cl_device_id> {
    static void retain(cl_device_id t) { UNUSED_OCL_RESULT(clRetainDevice(t)); }
    static void release(cl_device_id t) {
        UNUSED_OCL_RESULT(clReleaseDevice(t));
    }
};

// Generic class providing RAII support for OpenCL objects
template <typename T>
struct wrapper_t {
    wrapper_t(T t = nullptr, bool retain = false) : t_(t) {
        if (retain) { do_retain(); }
    }

    wrapper_t(const wrapper_t &other) : t_(other.t_) { do_retain(); }

    wrapper_t(wrapper_t &&other) noexcept : wrapper_t() { swap(*this, other); }

    wrapper_t &operator=(wrapper_t other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(wrapper_t &a, wrapper_t &b) noexcept {
        using std::swap;
        swap(a.t_, b.t_);
    }

    ~wrapper_t() { do_release(); }

    operator T() const { return t_; }
    T get() const { return t_; }
    T &unwrap() { return t_; }
    const T &unwrap() const { return t_; }

    T release() {
        T t = t_;
        t_ = nullptr;
        return t;
    }

private:
    T t_;

    void do_retain() {
        if (t_) { ref_traits<T>::retain(t_); }
    }

    void do_release() {
        if (t_) { ref_traits<T>::release(t_); }
    }
};

// Constructs an OpenCL wrapper object (providing RAII support)
template <typename T>
wrapper_t<T> make_wrapper(T t, bool retain = false) {
    return wrapper_t<T>(t, retain);
}

cl_platform_id get_platform(cl_device_id device);
cl_platform_id get_platform(engine_t *engine);

template <typename F>
struct ext_func_t {
    ext_func_t(const char *ext_func_name, const char *vendor_name = "Intel")
        : ext_func_ptrs_(vendor_platforms(vendor_name).size()) {
        for (size_t i = 0; i < vendor_platforms(vendor_name).size(); ++i) {
            auto p = vendor_platforms(vendor_name)[i];
            auto it = ext_func_ptrs_.insert(
                    {p, load_ext_func(p, ext_func_name)});
            assert(it.second);
            MAYBE_UNUSED(it);
        }
    }

    template <typename... Args>
    typename cpp_compat::invoke_result<F, Args...>::type operator()(
            engine_t *engine, Args... args) const {
        auto f = get_func(engine);
        return f(args...);
    }

    F get_func(engine_t *engine) const {
        return get_func(get_platform(engine));
    }

    F get_func(cl_platform_id platform) const {
        return ext_func_ptrs_.at(platform);
    }

private:
    std::unordered_map<cl_platform_id, F> ext_func_ptrs_;

    static F load_ext_func(cl_platform_id platform, const char *ext_func_name) {
        return reinterpret_cast<F>(clGetExtensionFunctionAddressForPlatform(
                platform, ext_func_name));
    }

    static const std::vector<cl_platform_id> &vendor_platforms(
            const char *vendor_name) {
        static auto vendor_platforms = get_vendor_platforms(vendor_name);
        return vendor_platforms;
    }

    static std::vector<cl_platform_id> get_vendor_platforms(
            const char *vendor_name) {
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> vendor_platforms;
        char platform_vendor_name[128] = {};
        for (cl_platform_id p : platforms) {
            err = clGetPlatformInfo(p, CL_PLATFORM_VENDOR,
                    sizeof(platform_vendor_name), platform_vendor_name,
                    nullptr);
            if (err != CL_SUCCESS) continue;
            if (std::string(platform_vendor_name).find(vendor_name)
                    != std::string::npos)
                vendor_platforms.push_back(p);
        }

        // OpenCL can return a list of platforms that contains duplicates.
        std::sort(vendor_platforms.begin(), vendor_platforms.end());
        vendor_platforms.erase(
                std::unique(vendor_platforms.begin(), vendor_platforms.end()),
                vendor_platforms.end());
        return vendor_platforms;
    }
};

std::string get_kernel_name(cl_kernel kernel);

status_t get_devices(std::vector<cl_device_id> *devices,
        cl_device_type device_type, cl_uint vendor_id = 0x8086);

status_t get_devices(std::vector<cl_device_id> *devices,
        std::vector<wrapper_t<cl_device_id>> *sub_devices,
        cl_device_type device_type);

status_t get_device_index(size_t *index, cl_device_id device);

cl_platform_id get_platform(cl_device_id device);
cl_platform_id get_platform(engine_t *engine);

status_t create_program(ocl::wrapper_t<cl_program> &ocl_program,
        cl_device_id dev, cl_context ctx, const xpu::binary_t &binary);

#ifndef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
status_t get_device_uuid(xpu::device_uuid_t &uuid, cl_device_id ocl_dev);
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

// Check for three conditions:
// 1. Device and context are compatible, i.e. the device belongs to
//    the context devices.
// 2. Device type matches the passed engine kind
// 3. Device/context platfrom is an Intel platform
status_t check_device(engine_kind_t eng_kind, cl_device_id dev, cl_context ctx);

status_t clone_kernel(cl_kernel kernel, cl_kernel *cloned_kernel);

#ifdef DNNL_ENABLE_MEM_DEBUG
cl_mem DNNL_WEAK clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret);
#else
cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret);
#endif

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
