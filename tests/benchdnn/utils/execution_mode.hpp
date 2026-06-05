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

#ifndef UTILS_EXECUTION_MODE_HPP
#define UTILS_EXECUTION_MODE_HPP

#include <functional>
#include <sstream>

enum class execution_mode_t { direct, graph };

extern execution_mode_t execution_mode;
extern execution_mode_t default_execution_mode;

std::ostream &operator<<(std::ostream &s, execution_mode_t mode);

// RAII guard that temporarily overrides `execution_mode` and restores the
// original value when the guard goes out of scope.
struct execution_mode_guard_t {
    execution_mode_guard_t(execution_mode_t mode)
        : saved_mode_(execution_mode) {
        execution_mode = mode;
    }
    ~execution_mode_guard_t() { execution_mode = saved_mode_; }

    execution_mode_guard_t(const execution_mode_guard_t &) = delete;
    execution_mode_guard_t &operator=(const execution_mode_guard_t &) = delete;

private:
    execution_mode_t saved_mode_;
};

// Forwar declarations for `execute_in_graph_mode`.
struct engine_t;
struct stream_t;
struct res_t;

// Returns true if SYCL command graph execution mode is active.
bool use_sycl_graph_exec(const engine_t &engine);

// Executes `record_func` inside a SYCL command graph and replays it. Used when
// `execution_mode` is set to `graph`. Returns OK on success, FAIL on error
// (with `res->state` set to FAILED when `res` is not null).
int execute_in_graph_mode(
        stream_t &stream, const std::function<void()> &record_func, res_t *res);

#endif
