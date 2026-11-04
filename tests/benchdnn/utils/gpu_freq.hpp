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
#include <string>

// GPU frequency sampling for performance reporting.
//
// A background thread polls the actual GPU frequency of the test engine device
// while the performance loop runs. The average over samples taken while the
// GPU was busy is attributed to the measured executions, which makes the
// `%freq%`, `%clocks%`, `%ntime%`, and `%nflops%` perf report options
// meaningful on GPU without MDAPI.
//
// Sources:
// * `sysfs`: Linux DRM sysfs `act_freq` of compute GTs (xe and i915). This is
//   the same data GEOPM reports as `DRM::ACT_FREQ`. The value is zero when a GT
//   is in RC6, which is used to discard idle samples.
// * `ze`: Level Zero Sysman `zesFrequencyGetState()`. Works on Linux and
//   Windows. Idle samples can't be distinguished.
namespace gpu_freq {

enum class source_t { none, automatic, sysfs, ze };

extern source_t source; // `--gpu-freq-source` knob value.
extern const source_t default_source;

source_t str2source(const std::string &s);
const char *source2str(source_t s);

struct stats_t {
    size_t n_samples = 0; // All samples taken during a measurement window.
    size_t n_busy = 0; // Samples taken while the GPU was not idle.
    double avg_mhz = 0; // Statistics over busy samples.
    double min_mhz = 0;
    double max_mhz = 0;

    bool ok() const { return n_busy > 0 && avg_mhz > 0; }
};

// Returns `true` if a frequency source is available for the test engine.
bool is_available();
// Starts sampling. No-op if no source is available.
void start();
// Stops sampling and returns statistics collected since the last `start()`.
stats_t stop();
// Reference frequency in MHz (maximum non-overclocked frequency of the device)
// used to normalize performance. Returns `0` if unknown.
double ref_mhz();

} // namespace gpu_freq

#endif
