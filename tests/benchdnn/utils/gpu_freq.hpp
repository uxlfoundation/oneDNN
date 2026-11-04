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
#include <vector>

// GPU frequency sampling for performance reporting.
//
// A background thread polls the actual GPU frequency of the test engine device
// while the performance loop runs. Samples are mapped to individual executions
// to derive their frequency, which makes the `%freq%` and `%clocks%` perf
// report options meaningful on GPU.
//
// Sources:
// * `sysfs`: Linux DRM sysfs `act_freq` of compute GTs (xe and i915). The
//   value is zero when a GT is in RC6, which is used to discard idle samples.
// * `ze`: Level Zero Sysman `zesFrequencyGetState()`. Works on Linux and
//   Windows. Idle samples can't be distinguished.
namespace gpu_freq {

enum class source_t { none, automatic, sysfs, ze };

extern source_t source; // `--gpu-freq-source` knob value.
extern const source_t default_source;

source_t str2source(const std::string &s);
const char *source2str(source_t s);

struct sample_t {
    double ms; // Time since `start()`.
    double mhz; // `0` if the GPU was idle.
};

// Frequency samples collected over a measurement window, ordered by time. The
// first sample is taken at `start()` and the last one at `stop()`.
struct trace_t {
    std::vector<sample_t> samples;

    double window_ms() const;
    size_t n_busy() const;
    // Average frequency of non-idle samples within `[from_ms, to_ms]`. If no
    // samples fall into the interval, the closest non-idle sample to its
    // middle is used. Returns `0` if all samples are idle.
    double avg_mhz(double from_ms, double to_ms) const;
};

// Returns `true` if a frequency source is available for the test engine.
bool is_available();
// Starts sampling. No-op if no source is available.
void start();
// Stops sampling and returns samples collected since the last `start()`.
trace_t stop();
// Returns a frequency in MHz for each execution with a duration from
// `exec_ms`, given in the submission order, by mapping executions to the
// sampled window. Returns an empty vector if the trace has no data.
std::vector<double> attribute(
        const trace_t &trace, const std::vector<double> &exec_ms);
// Maximum (RP0) hardware frequency of the device in MHz, independent of user
// frequency limits. Returns `0` if unknown.
double max_mhz();

} // namespace gpu_freq

#endif
