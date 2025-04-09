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

#ifndef GPU_GENERIC_ASYNCHRONOUS_VERBOSE_HPP
#define GPU_GENERIC_ASYNCHRONOUS_VERBOSE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_primitive.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

struct async_verbose_tracker {

    // initiates event tracking for SYCL devices for measuring kernel execution times
    void add_sycl_tracker(::sycl::event pevent, int jcnt) const {
        pevent.wait_and_throw();

        try {
            auto start_ms = pevent.get_profiling_info<
                    ::sycl::info::event_profiling::command_start>();
            auto end_ms = pevent.get_profiling_info<
                    ::sycl::info::event_profiling::command_end>();
            double exec_time_ms = (end_time - start_time);
            std::stringstream ss;
            ss << "kernel execution complete (j:" << jcnt << ")";
            VPROF(start_ms, primitive, exec, VERBOSE_profile, ss.str(),
                    exec_time_ms);
        } catch (const ::sycl::exception &e) {
            VERROR(common, common, "could tracking async profiling info");
        }
    }
}

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif