/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include <algorithm>
#include <chrono>

#include "common.hpp"
#include "utils/timer.hpp"

namespace timer {

double ms_now() {
    auto timePointTmp
            = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

// TODO: remove me
#if !defined(BENCHDNN_USE_RDPMC) || defined(_WIN32)
#else
#endif

void timer_t::restart() {
    ms_.clear();
    start();
}

void timer_t::start() {
    ms_start_ = ms_now();
}

void timer_t::stop(int append_n_times) {
    stop(append_n_times, ms_now() - ms_start_);
}

void timer_t::stop(int append_n_times, double append_ms) {
    if (append_n_times <= 0) {
        // No measurements happened.
        return;
    } else if (append_n_times == 1) {
        ms_.push_back(append_ms);
    } else {
        for (auto i : append_n_times)
            ms_.push_back(append_ms / append_n_times);
    }

    // double d_ms = append_ms;

    // ms_[mode_t::avg] += d_ms;
    // ms_[mode_t::sum] += d_ms;

    // d_ms /= append_n_times;

    // ms_[mode_t::min] = times_ ? std::min(ms_[mode_t::min], d_ms) : d_ms;
    // ms_[mode_t::max] = times_ ? std::max(ms_[mode_t::max], d_ms) : d_ms;

    // times_ += append_n_times;
}

timer_t &timer_map_t::get_timer(const std::string &name) {
    auto it = timers.find(name);
    if (it != timers.end()) return it->second;
    // Set a new timer if requested one wasn't found
    auto res = timers.emplace(name, timer_t());
    return res.first->second;
}

const std::vector<service_timers_entry_t> &get_global_service_timers() {
    // `service_timers_entry_t` type for each entry is needed for old GCC 4.8.5,
    // otherwise, it reports "error: converting to ‘std::tuple<...>’ from
    // initializer list would use explicit constructor
    // ‘constexpr std::tuple<...>’.
    static const std::vector<service_timers_entry_t> global_service_timers = {
            service_timers_entry_t {
                    "create_pd", mode_bit_t::init, timer::names::cpd_timer},
            service_timers_entry_t {
                    "create_prim", mode_bit_t::init, timer::names::cp_timer},
            service_timers_entry_t {
                    "fill", mode_bit_t::exec, timer::names::fill_timer},
            service_timers_entry_t {
                    "execute", mode_bit_t::exec, timer::names::execute_timer},
            service_timers_entry_t {
                    "compute_ref", mode_bit_t::corr, timer::names::ref_timer},
            service_timers_entry_t {
                    "compare", mode_bit_t::corr, timer::names::compare_timer},
    };
    return global_service_timers;
}

} // namespace timer
