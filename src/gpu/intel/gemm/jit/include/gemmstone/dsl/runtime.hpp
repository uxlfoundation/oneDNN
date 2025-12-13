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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_RUNTIME_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_RUNTIME_HPP

#include "gemmstone/dsl/dsl.hpp"

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
#include <vector>
#endif

#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
#include <sycl/sycl.hpp>
#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
#include <CL/cl.h>
#endif

GEMMSTONE_NAMESPACE_START
namespace dsl {

#ifdef GEMMSTONE_WITH_ASM_RUNTIME
std::string make_asm(const kernel_t &kernel);
#endif

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
std::vector<uint8_t> make_binary(const kernel_t &kernel);
#endif
#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
::sycl::kernel make_kernel(
        const kernel_t &kernel, ::sycl::context ctx, ::sycl::device dev);
#endif
#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
cl_kernel make_kernel(const kernel_t &kernel, cl_context ctx, cl_device_id dev);
#endif

} // namespace dsl
GEMMSTONE_NAMESPACE_END
#endif
