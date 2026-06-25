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

#ifndef TESTS_TEST_THREAD_DECL_HPP
#define TESTS_TEST_THREAD_DECL_HPP

// This header includes some essential declarations which are defined in
// test_thread.cpp as they all rely on functions from src/common/dnnl_thread.hpp
// and tangled macro dependency.
//
// It's cut off from test_thread.hpp for the purpose of de-tangling and getting
// rid of the transitive library dependency when using classes and functions
// declared below.
//
// If a test that included former 'test_thread.hpp' relies on the library
// internals, it must include correspondent headers explicitly.

#include "oneapi/dnnl/dnnl_config.h"

#include "tests/thread_context.hpp"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
namespace dnnl {
namespace testing {

dnnl::threadpool_interop::threadpool_iface *get_threadpool(
        const thr_ctx_t &ctx = get_default_thr_ctx());

// Sets the testing threadpool as active for the lifetime of the object.
// Required for the tests that throw to work.
struct scoped_tp_activation_t {
    scoped_tp_activation_t(
            dnnl::threadpool_interop::threadpool_iface *tp = get_threadpool());
    ~scoped_tp_activation_t();
};

struct scoped_tp_deactivation_t {
    scoped_tp_deactivation_t();
    ~scoped_tp_deactivation_t();
};

} // namespace testing
} // namespace dnnl
#endif

// TBB runtime may crash when it is used under CTest. This is a known TBB
// limitation that can be worked around by doing explicit finalization.
// The API to do that was introduced in 2021.6.0. When using an older TBB
// runtime the crash may still happen.
// Appropriate header lives in a `src/common/dnnl_thread_tbb_proxy.hpp`.
void finalize_tbb();

#endif
