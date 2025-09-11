/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef NGEN_SYCL_HPP
#define NGEN_SYCL_HPP

#include "ngen_config_internal.hpp"
#include "ngen_opencl.hpp"
#include "ngen_level_zero.hpp"
#include "ngen_interface.hpp"

#include <sycl/sycl.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>


namespace NGEN_NAMESPACE {


// Exceptions.
class unsupported_sycl_device : public std::runtime_error {
public:
    unsupported_sycl_device() : std::runtime_error("Unsupported SYCL device.") {}
};

// SYCL program generator class.
template <HW hw>
class SYCLCodeGenerator : public ELFCodeGenerator<hw>
{
public:
    using ELFCodeGenerator<hw>::getBinary;

    explicit SYCLCodeGenerator(Product product_, DebugConfig debugConfig = {})  : ELFCodeGenerator<hw>(product_, debugConfig) {}
    explicit SYCLCodeGenerator(int stepping_ = 0, DebugConfig debugConfig = {}) : ELFCodeGenerator<hw>(stepping_, debugConfig) {}
    explicit SYCLCodeGenerator(DebugConfig debugConfig) : ELFCodeGenerator<hw>(0, debugConfig) {}

    inline sycl::kernel getKernel(const sycl::context &context, const sycl::device &device);
    bool binaryIsZebin() { return true; }

    static inline HW detectHW(const sycl::context &context, const sycl::device &device);
    static inline Product detectHWInfo(const sycl::context &context, const sycl::device &device);

    // Queue-based convenience APIs.
    sycl::kernel getKernel(sycl::queue &queue) {
        return getKernel(queue.get_context(), queue.get_device());
    }
    static HW detectHW(sycl::queue &queue) {
        return detectHW(queue.get_context(), queue.get_device());
    }
    static Product detectHWInfo(sycl::queue &queue) {
        return detectHWInfo(queue.get_context(), queue.get_device());
    }
};

#define NGEN_FORWARD_SYCL(hw) NGEN_FORWARD_ELF(hw)

template <HW hw>
sycl::kernel SYCLCodeGenerator<hw>::getKernel(const sycl::context &context, const sycl::device &device)
{
    using namespace sycl;
    using super = ELFCodeGenerator<hw>;

    auto kernelName = super::interface_.getExternalName().c_str();
    auto binary = super::getBinary();

    const auto *binaryPtr = binary.data();
    size_t binarySize = binary.size();

    std::optional<kernel> outKernel;

    switch (device.get_backend()) {
        case backend::opencl: {
            auto contextCL = get_native<backend::opencl>(context);
            auto deviceCL = get_native<backend::opencl>(device);

            cl_int status = CL_SUCCESS;
            auto programCL = dynamic::clCreateProgramWithBinary(contextCL, 1, &deviceCL, &binarySize, &binaryPtr, nullptr, &status);

            detail::handleCL(status);
            if (programCL == nullptr)
                detail::handleCL(CL_OUT_OF_HOST_MEMORY);    /* a tried and true "default" error */

            detail::handleCL(dynamic::clBuildProgram(programCL, 1, &deviceCL, "-cl-std=CL2.0", nullptr, nullptr));

            auto kernelCL = dynamic::clCreateKernel(programCL, kernelName, &status);
            detail::handleCL(status);

            outKernel = make_kernel<backend::opencl>(kernelCL, context);

            detail::handleCL(dynamic::clReleaseKernel(kernelCL));
            detail::handleCL(dynamic::clReleaseProgram(programCL));
            detail::handleCL(dynamic::clReleaseContext(contextCL));
            break;
        }
        case backend::ext_oneapi_level_zero: {
            auto contextL0 = get_native<backend::ext_oneapi_level_zero>(context);
            auto deviceL0 = get_native<backend::ext_oneapi_level_zero>(device);

            ze_module_desc_t moduleDesc = {
                ZE_STRUCTURE_TYPE_MODULE_DESC,
                nullptr,
                ZE_MODULE_FORMAT_NATIVE,
                binarySize,
                binaryPtr,
                "",
                nullptr
            };

            ze_module_handle_t moduleL0;
            detail::handleL0(dynamic::zeModuleCreate(contextL0, deviceL0, &moduleDesc, &moduleL0, nullptr));

            ze_kernel_handle_t kernelL0;
            ze_kernel_desc_t kernelDesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernelName};
            detail::handleL0(dynamic::zeKernelCreate(moduleL0, &kernelDesc, &kernelL0));

            auto bundle = make_kernel_bundle<backend::ext_oneapi_level_zero, bundle_state::executable>({moduleL0}, context);
            outKernel = make_kernel<backend::ext_oneapi_level_zero>({bundle, kernelL0}, context);
            break;
        }
        default: throw unsupported_sycl_device();
    }

    return outKernel.value();
}

template <HW hw>
HW SYCLCodeGenerator<hw>::detectHW(const sycl::context &context, const sycl::device &device)
{
    return getCore(detectHWInfo(context, device).family);
}

template <HW hw>
Product SYCLCodeGenerator<hw>::detectHWInfo(const sycl::context &context, const sycl::device &device)
{
    using namespace sycl;
    switch (device.get_backend()) {
        case backend::opencl: {
            auto contextCL = get_native<backend::opencl>(context);
            auto deviceCL = get_native<backend::opencl>(device);
            auto ret = OpenCLCodeGenerator<hw>::detectHWInfo(contextCL, deviceCL);
            detail::handleCL(dynamic::clReleaseContext(contextCL));
            return ret;
        }
        case backend::ext_oneapi_level_zero:
            return LevelZeroCodeGenerator<hw>::detectHWInfo(get_native<backend::ext_oneapi_level_zero>(context),
                                                            get_native<backend::ext_oneapi_level_zero>(device));
        default: throw unsupported_sycl_device();
    }
    return Product{};
}

} /* namespace NGEN_NAMESPACE */

#endif
