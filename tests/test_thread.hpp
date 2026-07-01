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

#ifndef TESTS_TEST_THREAD_HPP
#define TESTS_TEST_THREAD_HPP

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

#endif
