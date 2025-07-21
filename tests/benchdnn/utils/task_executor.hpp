/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
* Copyright 2025 Arm Ltd. and affiliates
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
#pragma once

#ifndef UTILS_TASK_EXECUTOR_HPP
#define UTILS_TASK_EXECUTOR_HPP

#include <map>

#include "utils/parallel.hpp"
#include "utils/task.hpp"

// A macro serves an unification purpose.
// It must be a macro due to `prb_t` type is unique per driver.
#define TASK_EXECUTOR_DECL_TYPES \
    using create_func_t = std::function<int( \
            std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, \
            const prb_t *, res_t *)>; \
    using check_func_t = std::function<int( \
            std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, \
            const prb_t *, res_t *)>; \
    using do_func_t = std::function<int( \
            const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, \
            const prb_t *, res_t *)>; \
    using driver_task_executor_t = task_executor_t<prb_t, perf_report_t, \
            create_func_t, check_func_t, do_func_t>;

extern int repeats_per_prb;

template <typename prb_t, typename perf_report_t, typename create_func_t,
        typename check_func_t, typename do_func_t>
struct task_executor_t {
    virtual ~task_executor_t() { assert(tasks_.empty()); }

    std::string get_impl_names(std::vector<res_t> &results) {
        std::string s;
        for (auto r : results) {
            s += r.impl_name + ": "
                    + std::to_string(
                            r.timer_map.perf_timer().ms(timer::timer_t::min))
                    + ", ";
        }
        return s.substr(0, s.size() - 2);
    }

    void submit(const prb_t &prb, const std::string &perf_template,
            const create_func_t &create_func, const check_func_t &check_func,
            const do_func_t &do_func) {
        static const int nthreads = benchdnn_get_max_threads();
        for (int r = 0; r < repeats_per_prb; r++) {
            tasks_.emplace_back(prb, perf_template, create_func, check_func,
                    do_func, get_idx());
            if (has_bench_mode_modifier(mode_modifier_t::par_create)
                    && static_cast<int>(tasks_.size()) < nthreads)
                continue;
            flush();
        }
    }

    void flush() {
        // Special case is needed for THREADPOOL RUNTIME. Both `Parallel_nd` and
        // `createit` calls activate threadpool which causes undesired behavior.
        if (tasks_.size() == 1)
            tasks_[0].create(/* in_parallel = */ false);
        else
            benchdnn_parallel_nd(tasks_.size(),
                    [&](int i) { tasks_[i].create(/* in_parallel = */ true); });

        // Check caches first to avoid filling cache with service reorders.
        for (auto &t : tasks_) {
            t.check();
        }

        for (auto &t : tasks_) {
            t.exec();
            if (bench_list) results_[t.prb().str()].emplace_back(t.res());
        }

        tasks_.clear();
    }

    std::string get_list_recommendation() {
        std::stringstream ss;
        ss << "\n--- New list recommendations ---\n";
        for (auto r : results_) {
            std::string original = get_impl_names(r.second);
            std::sort(r.second.begin(), r.second.end());
            std::string n = get_impl_names(r.second);
            if (original != n) {
                ss << r.first << "\nCurrent list: " << original
                   << "\nNew list: " << n << "\n";
            }
        }
        ss << "\n";
        return ss.str();
    }

    std::vector<task_t<prb_t, perf_report_t, create_func_t, check_func_t,
            do_func_t>>
            tasks_;
    std::map<std::string, std::vector<res_t>> results_;

    int get_idx() {
        static int idx = 0;
        return idx++;
    }
};

#endif
