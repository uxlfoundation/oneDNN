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

#ifndef TEST_THREAD_HPP
#define TEST_THREAD_HPP

#include <functional>
#include <iostream>

#include "oneapi/dnnl/dnnl_config.h"

#ifdef COMMON_DNNL_THREAD_HPP
#error "src/common/dnnl_thread.hpp" was already included
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE

#if DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_SEQ
#error "DNNL_CPU_THREADING_RUNTIME is expected to be SEQ for GPU only configurations."
#endif

#undef DNNL_CPU_THREADING_RUNTIME

// Enable CPU threading layer for testing:
// - DPCPP: TBB
// - OCL: OpenMP
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_TBB
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP
#endif

#endif

// Here we define some types in global namespace to handle customized
// threading context for creation and execution
struct thr_ctx_t {
    int max_concurrency;
    int core_type;
    int nthr_per_core;

    bool operator==(const thr_ctx_t &rhs) const {
        return max_concurrency == rhs.max_concurrency
                && core_type == rhs.core_type
                && nthr_per_core == rhs.nthr_per_core;
    }
    bool operator!=(const thr_ctx_t &rhs) const { return !(*this == rhs); }
    void *get_interop_obj() const;
};

// This hack renames the namespaces used by threading functions for
// threadpool-related functions so that the calls to dnnl::impl::parallel*()
// from the test use a special testing threadpool.
//
// At the same time, the calls to dnnl::impl::parallel*() from within the
// library continue using the library version of these functions.
#define threadpool_utils testing_threadpool_utils
// Prohibit ITT API instrumentation
#undef DNNL_ENABLE_ITT_TASKS
#include "src/common/dnnl_thread.hpp"
#undef threadpool_utils

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
// Restore the original DNNL_CPU_THREADING_RUNTIME value.
#undef DNNL_CPU_THREADING_RUNTIME
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_SEQ
#endif

#ifndef COMMON_DNNL_THREAD_HPP
#error "src/common/dnnl_thread.hpp" has an unexpected header guard
#endif

std::ostream &operator<<(std::ostream &os, const thr_ctx_t &ctx);

const thr_ctx_t &get_default_thr_ctx();

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
namespace dnnl {

// Original threadpool utils are used by the scoped_tp_activation_t and thus
// need to be re-declared because of the hack above.
namespace impl {
namespace threadpool_utils {
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp);
void deactivate_threadpool();
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool();
int get_max_concurrency();
} // namespace threadpool_utils
} // namespace impl

namespace testing {

dnnl::threadpool_interop::threadpool_iface *get_threadpool(
        const thr_ctx_t &ctx = get_default_thr_ctx());

// Sets the testing threadpool as active for the lifetime of the object.
// Required for the tests that throw to work.
struct scoped_tp_activation_t {
    scoped_tp_activation_t(dnnl::threadpool_interop::threadpool_iface *tp_
            = get_threadpool()) {
        impl::threadpool_utils::activate_threadpool(tp_);
    }
    ~scoped_tp_activation_t() {
        impl::threadpool_utils::deactivate_threadpool();
    }
};

struct scoped_tp_deactivation_t {
    scoped_tp_deactivation_t() {
        impl::threadpool_utils::deactivate_threadpool();
    }
    ~scoped_tp_deactivation_t() {
        // we always use the same threadpool that is returned by `get_threadpool()`
        impl::threadpool_utils::activate_threadpool(get_threadpool());
    }
};

} // namespace testing
} // namespace dnnl
#endif

// These are free functions to allow running a function in a given threading
// context.
// A threading context is defined by:
// - number of threads
// - type of cores (TBB only)
// - threads per core (TBB only)

// Note: we have to differentiate creation and execution in thread
// context because of threadpool as it uses different mecanisms in
// both (in execution, tp is passed in stream)
//
// Definitions live in test_thread.cpp where the runtime-specific logic is
// handled inside a single version of each function.

int create_in_thr_ctx(const thr_ctx_t &ctx, const std::function<int()> &f);
// The function f shall take an interop obj as last argument
int execute_in_thr_ctx(const thr_ctx_t &ctx, const std::function<int()> &f);

// TBB runtime may crash when it is used under CTest. This is a known TBB
// limitation that can be worked around by doing explicit finalization.
// The API to do that was introduced in 2021.6.0. When using an older TBB
// runtime the crash may still happen.
// Appropriate header lives in a `src/common/dnnl_thread_tbb_proxy.hpp`.
void finalize_tbb();

#endif
