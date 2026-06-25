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

#ifndef TESTS_THREAD_CONTEXT_HPP
#define TESTS_THREAD_CONTEXT_HPP

#include <functional>
#include <iostream>

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

std::ostream &operator<<(std::ostream &os, const thr_ctx_t &ctx);

const thr_ctx_t &get_default_thr_ctx();

// These are free functions to allow running a function in a given threading
// context.
// A threading context is defined by:
// - number of threads
// - type of cores (TBB only)
// - threads per core (TBB only)

// Note: we have to differentiate creation and execution in thread context
// because of threadpool as it uses different mechanisms in both (in execution,
// tp is passed in stream).
//
// Definitions live in test_thread.cpp where the runtime-specific logic is
// handled inside a single version of each function.
int create_in_thr_ctx(const thr_ctx_t &ctx, const std::function<int()> &f);
// The function f shall take an interop obj as last argument
int execute_in_thr_ctx(const thr_ctx_t &ctx, const std::function<int()> &f);

#endif
