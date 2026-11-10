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

#ifndef UTILS_GPU_FREQ_HPP
#define UTILS_GPU_FREQ_HPP

#include <cstddef>
#include <vector>

// GPU frequency sampling via Level Zero Sysman (loader loaded at run time).
namespace gpu_freq {

struct sample_t {
    double ms; // Time since the window start.
    double mhz;
};

struct stats_t {
    double avg = 0, min = 0, max = 0;
    size_t n = 0;
};

// Samples GPU frequency while alive. No-op if no source is available.
struct session_t {
    session_t();
    ~session_t();
    session_t(const session_t &) = delete;
    session_t &operator=(const session_t &) = delete;

    // Stops sampling. Samples with `mhz <= 0` are invalid.
    std::vector<sample_t> stop();

private:
    bool active_ = true;
};

// Statistics of valid samples within the last `fraction` of the window, or of
// the last valid sample before it if there are none.
stats_t tail_stats(const std::vector<sample_t> &samples, double fraction);

// RP0 frequency in MHz, `0` if unknown.
double max_mhz();

} // namespace gpu_freq

#endif
