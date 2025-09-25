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
#ifndef GPU_INTEL_JIT_DSL_RUNTIME_HPP
#define GPU_INTEL_JIT_DSL_RUNTIME_HPP

#include "gpu/intel/jit/codegen/codegen.hpp"
#include "gpu/intel/jit/dsl/dsl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

#ifdef WITH_SYCL_RUNTIME
inline ::sycl::kernel make_kernel(
        const kernel_t &kernel, ::sycl::context ctx, ::sycl::device dev) {
    return make_kernel(kernel.iface, kernel.body, kernel.options,
            kernel.debug_cfg, ctx, dev);
}
#endif
#ifdef WITH_OPENCL_RUNTIME
inline cl_kernel make_kernel(
        const kernel_t &kernel, cl_context ctx, cl_device_id dev) {
    return make_kernel(kernel.iface, kernel.body, kernel.options,
            kernel.debug_cfg, ctx, dev);
}
#endif

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
