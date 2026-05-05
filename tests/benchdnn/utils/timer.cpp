/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <cmath>

#include "common.hpp"
#include "utils/timer.hpp"

namespace timer {

double ms_now() {
    auto timePointTmp
            = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

#if !defined(BENCHDNN_USE_RDPMC) || defined(_WIN32)
uint64_t ticks_now() {
    return (uint64_t)0;
}
#else
uint64_t ticks_now() {
    uint32_t eax, edx, ecx;

    ecx = (1 << 30) + 1;
    __asm__ volatile("rdpmc" : "=a"(eax), "=d"(edx) : "c"(ecx));

    return (uint64_t)eax | (uint64_t)edx << 32;
}
#endif

void timer_t::reset() {
    times_ = 0;
    for (int i = 0; i < n_modes; ++i)
        ticks_[i] = 0;
    ticks_start_ = 0;
    for (int i = 0; i < n_modes; ++i)
        ms_[i] = 0;
    ms_start_ = 0;
    ms_vec_.clear();

    start();
}

void timer_t::start() {
    ticks_start_ = ticks_now();
    ms_start_ = ms_now();
}

void timer_t::stop(int add_times, int64_t add_ticks, double add_ms) {
    if (add_times == 0) return;

    uint64_t d_ticks = add_ticks;
    double d_ms = add_ms;

    ticks_start_ += d_ticks;
    ms_start_ += d_ms;

    ms_[mode_t::avg] += d_ms;
    ms_[mode_t::sum] += d_ms;
    ticks_[mode_t::avg] += d_ticks;
    ticks_[mode_t::sum] += d_ticks;

    d_ticks /= add_times;
    d_ms /= add_times;

    ms_vec_.insert(ms_vec_.end(), add_times, d_ms);

    ms_[mode_t::min] = times_ ? std::min(ms_[mode_t::min], d_ms) : d_ms;
    ms_[mode_t::max] = times_ ? std::max(ms_[mode_t::max], d_ms) : d_ms;

    ticks_[mode_t::min]
            = times_ ? std::min(ticks_[mode_t::min], d_ticks) : d_ticks;
    ticks_[mode_t::max]
            = times_ ? std::max(ticks_[mode_t::max], d_ticks) : d_ticks;

    times_ += add_times;
}

void timer_t::stamp(int add_times) {
    stop(add_times, ticks_now() - ticks_start_, ms_now() - ms_start_);
}

void timer_t::filter_collection() {
    if (times_ <= 1) return;

    assert(ms_vec_.size() == times_);

    // First, drop the first half of all measurements. The motivation is second
    // half operates with stabilized frequency.
    size_t midpoint = ms_vec_.size() / 2;
    auto it_mid = ms_vec_.begin() + midpoint;
    ms_vec_.erase(ms_vec_.begin(), it_mid);
    if (ms_vec_.size() <= 1) return;

    // Then filter out "single-point" outliers that could appear even in a
    // second half of the run by pushing the bottom line of the peak time based
    // on bandwidth measurements in cold-cache mode.
    std::sort(ms_vec_.begin(), ms_vec_.end());

    // Remove up to 10% of fastest times.
    constexpr double outlier_percent = 0.10;

    // The idea is to measure the magnitude between values and if up to several
    // values are of bigger magnitude, drop them from the collection.
    //
    // The number of magnitudes is `outlier_percent` of the population round
    // down but, at least, one.
    //
    // For example, 25 samples will check two delta values between [0th, 1st]
    // and [1st, 2nd]. In case both delta will be outliers, e.g., 1.0, 1.1,
    // 1.21, [1.22...], both first values will be dropped.
    const size_t deltas_size = std::max(size_t(1),
            static_cast<size_t>(std::floor(outlier_percent * ms_vec_.size())));

    std::string msg;

    // The major magnitude is when `x_i = x_{i+1} * 1.04`.
    constexpr double magnitude_threshold = 1.04;
    size_t cut_point = SIZE_MAX;
    // For large collections demand that the minimal time has at least 1% of
    // representatives.
    size_t n_same_samples = 1;
    const size_t min_n_samples = std::max(
            size_t(1), static_cast<size_t>(std::floor(ms_vec_.size() * 0.01)));
    for (size_t i = 0; i < deltas_size; i++) {
        // It may happen there are more major magnitude jumps within
        // `outlier_percent` number of elements, drop as much values as allowed.
        if (ms_vec_[i + 1] >= ms_vec_[i] * magnitude_threshold) {
            cut_point = i;
            if (verbose >= 4) {
                msg += std::to_string(i) + ":" + std::to_string(ms_vec_[i + 1])
                        + "/" + std::to_string(ms_vec_[i]) + "="
                        + std::to_string(ms_vec_[i + 1] / ms_vec_[i]) + "; ";
            }
        }

        // Get to another metric - number of samples. If minimum is one,
        // nothing to do.
        if (min_n_samples == 1) continue;

        if (ms_vec_[i + 1] == ms_vec_[i]) {
            n_same_samples++;
            continue;
        }

        // If we just cut off by major value diff, it means there's no point
        // of cutting it by the number of samples. Restart the counter.
        if (cut_point == i) {
            n_same_samples = 1;
            continue;
        }

        // Values are not same, and diff val criterion didn't apply.
        // If number of samples is less then desired, drop them and restart the
        // counter.
        if (n_same_samples < min_n_samples) {
            cut_point = i;
            if (verbose >= 4) {
                msg += std::to_string(i) + ":" + std::to_string(ms_vec_[i])
                        + "(" + std::to_string(n_same_samples) + "); ";
            }
            n_same_samples = 1;
        }
    }

    if (cut_point < deltas_size) {
        ms_vec_.erase(ms_vec_.begin(), ms_vec_.begin() + cut_point + 1);
    }

    // After all undesired values discarded, re-compute stats.
    ms_[mode_t::sum] = 0;
    ms_[mode_t::min] = *ms_vec_.begin();
    ms_[mode_t::max] = *ms_vec_.rbegin();
    for (size_t i = 0; i < ms_vec_.size(); i++) {
        ms_[mode_t::sum] += ms_vec_[i];
    }
    ms_[mode_t::avg] = ms_[mode_t::sum];
    times_ = ms_vec_.size();
    // Frequency is not supported for filtering, explicitly zeroing it to avoid
    // potential misuse.
    for (int i = 0; i < n_modes; i++)
        ticks_[i] = 0;

    BENCHDNN_PRINT(4, "[TIMER]: MinSamples: %zu; Outliers: %zu; %s\n",
            min_n_samples, deltas_size, msg.c_str());
}

timer_t &timer_t::operator=(const timer_t &rhs) {
    if (this == &rhs) return *this;
    *this = timer_t(rhs);
    return *this;
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
