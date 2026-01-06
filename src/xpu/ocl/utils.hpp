/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#define CL_MEM_FLAGS_INTEL 0x10001
#define CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL (1 << 23)
#endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t convert_to_dnnl(cl_int cl_status);
const char *convert_cl_int_to_str(cl_int cl_status);

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

#if defined(_WIN32)
#define OCL_LIB_NAME "OpenCL.dll"
#elif defined(__linux__)
#define OCL_LIB_NAME "libOpenCL.so.1"
#endif

template <typename F>
F find_ocl_symbol(const char *symbol) {
    return (F)xpu::find_symbol(OCL_LIB_NAME, symbol);
}
#undef OCL_LIB_NAME

enum { CL_SYMBOL_NOT_FOUND = -128 };

// In case the OCL symbol is not found:
// - if the return value of OCL function is cl_int, return CL_SYMBOL_NOT_FOUND
// - if the return value of OCL function is a pointer, return nullptr
template <typename T>
typename std::enable_if<std::is_same<T, cl_int>::value, T>::type
no_ocl_symbol_error() {
    return CL_SYMBOL_NOT_FOUND;
}
template <typename T>
typename std::enable_if<std::is_pointer<T>::value, T>::type
no_ocl_symbol_error() {
    return nullptr;
}

#define INDIRECT_OCL_CALL(result_type, f) \
    template <typename... Args> \
    result_type f(Args &&...args) { \
        static auto f_ = find_ocl_symbol<decltype(&::f)>(#f); \
        if (f_) return f_(std::forward<Args>(args)...); \
        return no_ocl_symbol_error<result_type>(); \
    }

INDIRECT_OCL_CALL(cl_int, clBuildProgram)
INDIRECT_OCL_CALL(cl_mem, clCreateBuffer)
INDIRECT_OCL_CALL(cl_context, clCreateContext)
INDIRECT_OCL_CALL(cl_kernel, clCreateKernel)
INDIRECT_OCL_CALL(cl_program, clCreateProgramWithBinary)
INDIRECT_OCL_CALL(cl_program, clCreateProgramWithSource)
INDIRECT_OCL_CALL(cl_mem, clCreateSubBuffer)
INDIRECT_OCL_CALL(cl_int, clCreateSubDevices)
INDIRECT_OCL_CALL(cl_int, clEnqueueCopyBuffer)
INDIRECT_OCL_CALL(cl_int, clEnqueueFillBuffer)
INDIRECT_OCL_CALL(void *, clEnqueueMapBuffer)
INDIRECT_OCL_CALL(cl_int, clEnqueueMarkerWithWaitList)
INDIRECT_OCL_CALL(cl_int, clEnqueueNDRangeKernel)
INDIRECT_OCL_CALL(cl_int, clEnqueueReadBuffer)
INDIRECT_OCL_CALL(cl_int, clEnqueueUnmapMemObject)
INDIRECT_OCL_CALL(cl_int, clEnqueueWriteBuffer)
INDIRECT_OCL_CALL(cl_int, clFinish)
INDIRECT_OCL_CALL(cl_int, clGetCommandQueueInfo)
INDIRECT_OCL_CALL(cl_int, clGetContextInfo)
INDIRECT_OCL_CALL(cl_int, clGetDeviceIDs)
INDIRECT_OCL_CALL(cl_int, clGetDeviceInfo)
INDIRECT_OCL_CALL(cl_int, clGetEventProfilingInfo)
INDIRECT_OCL_CALL(void *, clGetExtensionFunctionAddressForPlatform)
INDIRECT_OCL_CALL(cl_int, clGetKernelArgInfo)
INDIRECT_OCL_CALL(cl_int, clGetKernelInfo)
INDIRECT_OCL_CALL(cl_int, clGetMemObjectInfo)
INDIRECT_OCL_CALL(cl_int, clGetPlatformIDs)
INDIRECT_OCL_CALL(cl_int, clGetPlatformInfo)
INDIRECT_OCL_CALL(cl_int, clGetProgramBuildInfo)
INDIRECT_OCL_CALL(cl_int, clGetProgramInfo)
INDIRECT_OCL_CALL(cl_int, clReleaseCommandQueue)
INDIRECT_OCL_CALL(cl_int, clReleaseContext)
INDIRECT_OCL_CALL(cl_int, clReleaseDevice)
INDIRECT_OCL_CALL(cl_int, clReleaseEvent)
INDIRECT_OCL_CALL(cl_int, clReleaseKernel)
INDIRECT_OCL_CALL(cl_int, clReleaseMemObject)
INDIRECT_OCL_CALL(cl_int, clReleaseProgram)
INDIRECT_OCL_CALL(cl_int, clReleaseSampler)
INDIRECT_OCL_CALL(cl_int, clRetainCommandQueue)
INDIRECT_OCL_CALL(cl_int, clRetainContext)
INDIRECT_OCL_CALL(cl_int, clRetainDevice)
INDIRECT_OCL_CALL(cl_int, clRetainEvent)
INDIRECT_OCL_CALL(cl_int, clRetainKernel)
INDIRECT_OCL_CALL(cl_int, clRetainMemObject)
INDIRECT_OCL_CALL(cl_int, clRetainProgram)
INDIRECT_OCL_CALL(cl_int, clRetainSampler)
INDIRECT_OCL_CALL(cl_int, clSetKernelArg)
INDIRECT_OCL_CALL(cl_int, clWaitForEvents)
#ifdef CL_VERSION_2_0
INDIRECT_OCL_CALL(cl_command_queue, clCreateCommandQueueWithProperties)
#else
INDIRECT_OCL_CALL(cl_command_queue, clCreateCommandQueue)
#endif
#ifdef CL_VERSION_2_1
INDIRECT_OCL_CALL(cl_kernel, clCloneKernel)
#endif

#undef INDIRECT_OCL_CALL

// OpenCL objects reference counting traits
template <typename T>
struct ref_traits;
//{
//    static void retain(T t) {}
//    static void release(T t) {}
//};

template <>
struct ref_traits<cl_context> {
    static void retain(cl_context t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainContext(t));
    }
    static void release(cl_context t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseContext(t));
    }
};

template <>
struct ref_traits<cl_command_queue> {
    static void retain(cl_command_queue t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainCommandQueue(t));
    }
    static void release(cl_command_queue t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseCommandQueue(t));
    }
};

template <>
struct ref_traits<cl_program> {
    static void retain(cl_program t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainProgram(t));
    }
    static void release(cl_program t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseProgram(t));
    }
};

template <>
struct ref_traits<cl_kernel> {
    static void retain(cl_kernel t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainKernel(t));
    }
    static void release(cl_kernel t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseKernel(t));
    }
};

template <>
struct ref_traits<cl_mem> {
    static void retain(cl_mem t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainMemObject(t));
    }
    static void release(cl_mem t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseMemObject(t));
    }
};

template <>
struct ref_traits<cl_sampler> {
    static void retain(cl_sampler t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainSampler(t));
    }
    static void release(cl_sampler t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseSampler(t));
    }
};

template <>
struct ref_traits<cl_event> {
    static void retain(cl_event t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainEvent(t));
    }
    static void release(cl_event t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseEvent(t));
    }
};

template <>
struct ref_traits<cl_device_id> {
    static void retain(cl_device_id t) {
        UNUSED_OCL_RESULT(xpu::ocl::clRetainDevice(t));
    }
    static void release(cl_device_id t) {
        UNUSED_OCL_RESULT(xpu::ocl::clReleaseDevice(t));
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
        return reinterpret_cast<F>(
                xpu::ocl::clGetExtensionFunctionAddressForPlatform(
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
        cl_int err = xpu::ocl::clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> platforms(num_platforms);
        err = xpu::ocl::clGetPlatformIDs(
                num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> vendor_platforms;
        char platform_vendor_name[128] = {};
        for (cl_platform_id p : platforms) {
            err = xpu::ocl::clGetPlatformInfo(p, CL_PLATFORM_VENDOR,
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
