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

enum class execution_mode_t { direct, graph };

extern execution_mode_t execution_mode; // user execution mode

const char *execution_mode2str(execution_mode_t mode);

execution_mode_t str2execution_mode(const char *str);

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

// Returns true if SYCL command graph execution mode is active.
bool use_sycl_graph_exec();

#endif
