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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#endif

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/gpu_freq.hpp"
#include "utils/system.hpp"

namespace gpu_freq {

namespace {

// Minimal subset of the Level Zero Sysman ABI (zes_api.h). The loader is
// loaded dynamically to avoid a build-time dependency for OpenCL builds.
namespace zes {
using result_t = int32_t;
using handle_t = void *;
using bool_t = uint8_t;
constexpr result_t success = 0;
constexpr int32_t stype_pci_properties = 0x2;
constexpr int32_t stype_freq_properties = 0x9;
constexpr int32_t stype_freq_state = 0x1b;
constexpr int32_t freq_domain_gpu = 0;

struct pci_properties_t {
    int32_t stype;
    void *pNext;
    uint32_t domain, bus, device, function;
    int32_t gen, width;
    int64_t max_bandwidth;
    bool_t have_bandwidth_counters, have_packet_counters, have_replay_counters;
};

struct freq_properties_t {
    int32_t stype;
    void *pNext;
    int32_t type;
    bool_t on_subdevice;
    uint32_t subdevice_id;
    bool_t can_control;
    bool_t is_throttle_event_supported;
    double min;
    double max;
};

struct freq_state_t {
    int32_t stype;
    const void *pNext;
    double current_voltage;
    double request;
    double tdp;
    double efficient;
    double actual;
    uint32_t throttle_reasons;
};

using init_f = result_t (*)(uint32_t);
using driver_get_f = result_t (*)(uint32_t *, handle_t *);
using device_get_f = result_t (*)(handle_t, uint32_t *, handle_t *);
using pci_get_properties_f = result_t (*)(handle_t, pci_properties_t *);
using enum_freq_domains_f = result_t (*)(handle_t, uint32_t *, handle_t *);
using freq_get_properties_f = result_t (*)(handle_t, freq_properties_t *);
using freq_get_state_f = result_t (*)(handle_t, freq_state_t *);
} // namespace zes

void *load_ze_symbol(const char *name) {
#ifdef _WIN32
    static HMODULE lib = LoadLibraryExA(
            "ze_loader.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!lib) return nullptr;
    return reinterpret_cast<void *>(GetProcAddress(lib, name));
#else
    static void *lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return nullptr;
    return dlsym(lib, name);
#endif
}

std::string bdf2str(uint32_t d, uint32_t b, uint32_t dev, uint32_t f) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x", d, b, dev, f);
    return buf;
}

// Reads the actual frequency of a device with `zesFrequencyGetState()` for
// every GPU frequency domain. A call costs ~50-60 us.
struct ze_reader_t {
    static std::unique_ptr<ze_reader_t> create(const std::string &bdf) {
#define LOAD(var, type, name) \
    auto var = reinterpret_cast<zes::type>(load_ze_symbol(name)); \
    if (!(var)) return nullptr;
        LOAD(init, init_f, "zesInit");
        LOAD(driver_get, driver_get_f, "zesDriverGet");
        LOAD(device_get, device_get_f, "zesDeviceGet");
        LOAD(pci_props, pci_get_properties_f, "zesDevicePciGetProperties");
        LOAD(enum_freq, enum_freq_domains_f, "zesDeviceEnumFrequencyDomains");
        LOAD(freq_props, freq_get_properties_f, "zesFrequencyGetProperties");
        LOAD(get_state, freq_get_state_f, "zesFrequencyGetState");
#undef LOAD
        if (init(0) != zes::success) return nullptr;

        uint32_t n_drv = 0;
        if (driver_get(&n_drv, nullptr) != zes::success) return nullptr;
        std::vector<zes::handle_t> drivers(n_drv);
        driver_get(&n_drv, drivers.data());

        std::vector<std::unique_ptr<ze_reader_t>> candidates;
        for (auto drv : drivers) {
            uint32_t n_dev = 0;
            device_get(drv, &n_dev, nullptr);
            std::vector<zes::handle_t> devices(n_dev);
            device_get(drv, &n_dev, devices.data());
            for (auto dev : devices) {
                zes::pci_properties_t pp {};
                pp.stype = zes::stype_pci_properties;
                std::string dev_bdf;
                if (pci_props(dev, &pp) == zes::success)
                    dev_bdf = bdf2str(
                            pp.domain, pp.bus, pp.device, pp.function);
                if (!bdf.empty() && dev_bdf != bdf) continue;

                std::unique_ptr<ze_reader_t> r(new ze_reader_t());
                r->get_state_ = get_state;
                r->bdf_ = dev_bdf;
                uint32_t n_freq = 0;
                enum_freq(dev, &n_freq, nullptr);
                std::vector<zes::handle_t> domains(n_freq);
                enum_freq(dev, &n_freq, domains.data());
                for (auto h : domains) {
                    zes::freq_properties_t fp {};
                    fp.stype = zes::stype_freq_properties;
                    if (freq_props(h, &fp) != zes::success) continue;
                    if (fp.type != zes::freq_domain_gpu) continue;
                    r->domains_.push_back(h);
                    r->max_mhz_ = std::max(r->max_mhz_, fp.max);
                }
                if (r->domains_.empty()) continue;
                candidates.push_back(std::move(r));
            }
        }
        if (candidates.size() != 1) return nullptr;
        return std::move(candidates[0]);
    }

    // Returns the average actual frequency over GPU domains in MHz, `0` if
    // unknown, or a negative value on error.
    double read_mhz() {
        double sum = 0;
        int n = 0;
        for (auto h : domains_) {
            zes::freq_state_t s {};
            s.stype = zes::stype_freq_state;
            if (get_state_(h, &s) != zes::success) return -1;
            if (s.actual > 0) {
                sum += s.actual;
                n++;
            }
        }
        return n ? sum / n : 0;
    }

    double max_mhz() const { return max_mhz_; }
    static constexpr int period_us = 1000;

    std::string info() const {
        return "ze sysman (" + bdf_ + "): " + std::to_string(domains_.size())
                + " GPU domain(s)";
    }

private:
    ze_reader_t() = default;

    zes::freq_get_state_f get_state_ = nullptr;
    std::vector<zes::handle_t> domains_;
    std::string bdf_;
    double max_mhz_ = 0;
};

// Sleeps for a polling period. On Windows, `std::this_thread::sleep_for` has
// a granularity of the system timer (up to 15.6 ms), so a high-resolution
// waitable timer is used instead.
struct period_timer_t {
    period_timer_t(int period_us) : period_us_(period_us) {
#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2
#endif
        // Not supported before Windows 10 1803; falls back to `sleep_for`.
        timer_ = CreateWaitableTimerExW(nullptr, nullptr,
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
#endif
    }
    ~period_timer_t() {
#ifdef _WIN32
        if (timer_) CloseHandle(timer_);
#endif
    }

    void wait() {
#ifdef _WIN32
        if (timer_) {
            LARGE_INTEGER due;
            // Relative time in 100-nanosecond intervals.
            due.QuadPart = -10LL * period_us_;
            if (SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE)
                    && WaitForSingleObject(timer_, INFINITE) == WAIT_OBJECT_0)
                return;
        }
#endif
        std::this_thread::sleep_for(std::chrono::microseconds(period_us_));
    }

private:
    int period_us_;
#ifdef _WIN32
    HANDLE timer_ = nullptr;
#endif
};

struct sampler_t {
    static sampler_t &get() {
        static sampler_t s;
        return s;
    }

    bool is_available() const { return (bool)reader_; }
    double max_mhz() const { return reader_ ? reader_->max_mhz() : 0; }

    void start() {
        if (!reader_ || running_) return;
        samples_.clear();
        stop_.store(false);
        running_ = true;
        start_ = clock_t::now();
        // A synchronous sample at start and another at stop guarantee at
        // least two samples even if the window is shorter than the thread
        // start-up or the polling period (e.g., a single short execution).
        sample();
        thread_ = std::thread([this]() {
            period_timer_t timer(ze_reader_t::period_us);
            while (true) {
                timer.wait();
                if (stop_.load(std::memory_order_relaxed)) break;
                sample();
            }
        });
    }

    trace_t stop() {
        if (!running_) return trace_t();
        stop_.store(true);
        thread_.join();
        running_ = false;
        // The GPU stays awake for a while after completion, so the final
        // sample still reflects the frequency of the last executions.
        sample();
        trace_t trace;
        trace.samples.swap(samples_);
        return trace;
    }

private:
    sampler_t() {
        if (!is_gpu()) return;
        const std::string bdf = get_gpu_pci_bdf();
        reader_ = ze_reader_t::create(bdf);

        if (reader_) {
            BENCHDNN_PRINT(2, "[GPU_FREQ] source: %s; max: %.0f MHz\n",
                    reader_->info().c_str(), reader_->max_mhz());
            samples_.reserve((
                    size_t)(4 * max_ms_per_prb * 1e3 / ze_reader_t::period_us));
        } else {
            BENCHDNN_PRINT(2,
                    "[GPU_FREQ] Warning: Level Zero Sysman is not available "
                    "for device \"%s\".\n",
                    bdf.c_str());
        }
    }

    ~sampler_t() {
        if (running_) {
            stop_.store(true);
            thread_.join();
        }
    }

    std::unique_ptr<ze_reader_t> reader_;
    std::thread thread_;
    std::atomic<bool> stop_ {false};
    bool running_ = false;
    using clock_t = std::chrono::steady_clock;

    void sample() {
        const double mhz = reader_->read_mhz();
        const double ms = std::chrono::duration<double, std::milli>(
                clock_t::now() - start_)
                                  .count();
        samples_.push_back({ms, mhz});
    }

    clock_t::time_point start_;
    std::vector<sample_t> samples_;
};

} // namespace

bool is_available() {
    return sampler_t::get().is_available();
}

void start() {
    sampler_t::get().start();
}

trace_t stop() {
    return sampler_t::get().stop();
}

double trace_t::window_ms() const {
    return samples.empty() ? 0 : samples.back().ms;
}

size_t trace_t::n_valid() const {
    size_t n = 0;
    for (const auto &s : samples)
        n += s.mhz > 0;
    return n;
}

double trace_t::avg_mhz(double from_ms, double to_ms) const {
    double sum = 0;
    size_t n = 0;
    const sample_t *nearest = nullptr;
    const double mid_ms = (from_ms + to_ms) / 2;
    for (const auto &s : samples) {
        if (s.mhz <= 0) continue;
        if (s.ms >= from_ms && s.ms <= to_ms) {
            sum += s.mhz;
            n++;
        }
        if (!nearest
                || std::abs(s.ms - mid_ms) < std::abs(nearest->ms - mid_ms))
            nearest = &s;
    }
    if (n) return sum / n;
    return nearest ? nearest->mhz : 0;
}

std::vector<double> attribute(
        const trace_t &trace, const std::vector<double> &exec_ms) {
    const size_t n = exec_ms.size();
    if (!n || !trace.n_valid()) return {};

    // Executions are assumed to run in submission order. The time not covered
    // by executions (submission latency, host overheads, gaps) is distributed
    // evenly in front of each execution. For device-bound runs it's small and
    // executions are placed back-to-back; for host-bound runs executions get
    // spread uniformly over the window.
    double busy_ms = 0;
    for (double ms : exec_ms)
        busy_ms += ms;
    const double gap_ms = std::max(0.0, trace.window_ms() - busy_ms) / n;

    std::vector<double> mhz(n);
    double t_ms = 0;
    for (size_t i = 0; i < n; i++) {
        t_ms += gap_ms;
        mhz[i] = trace.avg_mhz(t_ms, t_ms + exec_ms[i]);
        t_ms += exec_ms[i];
    }
    return mhz;
}

double max_mhz() {
    return sampler_t::get().max_mhz();
}

} // namespace gpu_freq
