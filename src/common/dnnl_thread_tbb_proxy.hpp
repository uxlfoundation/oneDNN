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

#ifndef COMMON_DNNL_THREAD_TBB_PROXY_HPP
#define COMMON_DNNL_THREAD_TBB_PROXY_HPP

// The purpose of the proxy header file is exactly to enable system_header
// diagnostics for TBB headers because they have multiple Wundef hits.
// Treating **this** file as system allows to avoid those warning hits.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC system_header
#elif defined(__clang__)
#pragma clang system_header
#endif

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

// API to do explicit finalization was introduced in 2021.6.
// Used in tests/test_thread.hpp. Keep all include files in a single place.
#if defined(TBB_INTERFACE_VERSION) && (TBB_INTERFACE_VERSION >= 12060)
#include "tbb/global_control.h"
#endif

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
