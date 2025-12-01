/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#ifndef GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP
#define GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP

#include "gpu/intel/jit/ir/include/kernel.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include <sycl/sycl.hpp>
#define WITH_SYCL_RUNTIME
#endif
#define WITH_OPENCL_RUNTIME
#include <CL/cl.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

#ifdef WITH_SYCL_RUNTIME
sycl::kernel make_kernel(
        const dsl::kernel_t &kernel, sycl::context ctx, sycl::device dev);
#endif
#ifdef WITH_OPENCL_RUNTIME
cl_kernel make_kernel(
        const dsl::kernel_t &kernel, cl_context ctx, cl_device_id dev);
#endif

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
