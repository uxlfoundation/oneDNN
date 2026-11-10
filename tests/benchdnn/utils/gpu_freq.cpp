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
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/gpu_freq.hpp"
#include "utils/system.hpp"

#ifdef BENCHDNN_WITH_GPU_FREQ
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#endif

// Types only.
#include "level_zero/zes_api.h"
#endif

namespace gpu_freq {

#ifdef BENCHDNN_WITH_GPU_FREQ
namespace {

constexpr int period_us = 1000;

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

template <typename F>
F load_ze_fn(const char *name) {
    return reinterpret_cast<F>(load_ze_symbol(name));
}

// Reads the actual frequency of a device with `zesFrequencyGetState()` for
// every GPU frequency domain. A call costs ~50-60 us.
struct ze_reader_t {
    static std::unique_ptr<ze_reader_t> create(const std::string &bdf) {
        auto init = load_ze_fn<decltype(&zesInit)>("zesInit");
        auto driver_get = load_ze_fn<decltype(&zesDriverGet)>("zesDriverGet");
        auto device_get = load_ze_fn<decltype(&zesDeviceGet)>("zesDeviceGet");
        auto pci_props = load_ze_fn<decltype(&zesDevicePciGetProperties)>(
                "zesDevicePciGetProperties");
        auto enum_freq = load_ze_fn<decltype(&zesDeviceEnumFrequencyDomains)>(
                "zesDeviceEnumFrequencyDomains");
        auto freq_props = load_ze_fn<decltype(&zesFrequencyGetProperties)>(
                "zesFrequencyGetProperties");
        auto get_state = load_ze_fn<decltype(&zesFrequencyGetState)>(
                "zesFrequencyGetState");
        if (!init || !driver_get || !device_get || !pci_props || !enum_freq
                || !freq_props || !get_state)
            return nullptr;
        if (init(0) != ZE_RESULT_SUCCESS) return nullptr;

        uint32_t n_drv = 0;
        if (driver_get(&n_drv, nullptr) != ZE_RESULT_SUCCESS) return nullptr;
        std::vector<zes_driver_handle_t> drivers(n_drv);
        driver_get(&n_drv, drivers.data());

        std::unique_ptr<ze_reader_t> found;
        for (auto drv : drivers) {
            uint32_t n_dev = 0;
            device_get(drv, &n_dev, nullptr);
            std::vector<zes_device_handle_t> devices(n_dev);
            device_get(drv, &n_dev, devices.data());
            for (auto dev : devices) {
                zes_pci_properties_t pp {};
                pp.stype = ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
                char dev_bdf[16] = {};
                if (pci_props(dev, &pp) == ZE_RESULT_SUCCESS)
                    snprintf(dev_bdf, sizeof(dev_bdf), "%04x:%02x:%02x.%x",
                            pp.address.domain, pp.address.bus,
                            pp.address.device, pp.address.function);
                if (!bdf.empty() && dev_bdf != bdf) continue;

                std::unique_ptr<ze_reader_t> r(new ze_reader_t());
                r->get_state_ = get_state;
                uint32_t n_freq = 0;
                enum_freq(dev, &n_freq, nullptr);
                std::vector<zes_freq_handle_t> domains(n_freq);
                enum_freq(dev, &n_freq, domains.data());
                for (auto h : domains) {
                    zes_freq_properties_t fp {};
                    fp.stype = ZES_STRUCTURE_TYPE_FREQ_PROPERTIES;
                    if (freq_props(h, &fp) != ZE_RESULT_SUCCESS) continue;
                    if (fp.type != ZES_FREQ_DOMAIN_GPU) continue;
                    r->domains_.push_back(h);
                    r->max_mhz_ = std::max(r->max_mhz_, fp.max);
                }
                if (r->domains_.empty()) continue;
                if (found) return nullptr; // Ambiguous.
                found = std::move(r);
            }
        }
        return found;
    }

    // Average over GPU domains in MHz, `0` if unknown.
    double read_mhz() {
        double sum = 0;
        int n = 0;
        for (auto h : domains_) {
            zes_freq_state_t s {};
            s.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;
            if (get_state_(h, &s) != ZE_RESULT_SUCCESS) return 0;
            if (s.actual > 0) {
                sum += s.actual;
                n++;
            }
        }
        return n ? sum / n : 0;
    }

    double max_mhz() const { return max_mhz_; }
    size_t n_domains() const { return domains_.size(); }

private:
    ze_reader_t() = default;

    decltype(&zesFrequencyGetState) get_state_ = nullptr;
    std::vector<zes_freq_handle_t> domains_;
    double max_mhz_ = 0;
};

// `sleep_for` has ~15.6 ms granularity on Windows.
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

    void wait() const {
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

    double max_mhz() const { return reader_ ? reader_->max_mhz() : 0; }

    void start() {
        if (!reader_) return;
        if (thread_.joinable()) stop();
        samples_.clear();
        stop_.store(false);
        start_ = clk_t::now();
        // Sample at start and stop so short windows get >= 2 samples.
        sample();
        thread_ = std::thread([this]() {
            period_timer_t timer(period_us);
            while (true) {
                timer.wait();
                if (stop_.load()) break;
                sample();
            }
        });
    }

    std::vector<sample_t> stop() {
        if (!thread_.joinable()) return {};
        stop_.store(true);
        thread_.join();
        sample();
        return std::move(samples_);
    }

private:
    using clk_t = std::chrono::steady_clock;

    sampler_t() {
        if (!is_gpu()) return;
        const std::string bdf = get_gpu_pci_bdf();
        reader_ = ze_reader_t::create(bdf);
        if (reader_)
            BENCHDNN_PRINT(3,
                    "[GPU_FREQ] source: ze sysman (%s): %zu GPU domain(s); "
                    "max: %.0f MHz\n",
                    bdf.c_str(), reader_->n_domains(), reader_->max_mhz());
        else
            BENCHDNN_PRINT(2,
                    "[GPU_FREQ] Warning: Level Zero Sysman is not available "
                    "for device \"%s\".\n",
                    bdf.c_str());
    }

    ~sampler_t() { stop(); }

    void sample() {
        const double mhz = reader_->read_mhz();
        const double ms = std::chrono::duration<double, std::milli>(
                clk_t::now() - start_)
                                  .count();
        samples_.push_back({ms, mhz});
    }

    std::unique_ptr<ze_reader_t> reader_;
    std::thread thread_;
    std::atomic<bool> stop_ {false};
    clk_t::time_point start_;
    std::vector<sample_t> samples_;
};

} // namespace

void start() {
    sampler_t::get().start();
}

std::vector<sample_t> stop() {
    return sampler_t::get().stop();
}

double max_mhz() {
    return sampler_t::get().max_mhz();
}
#else
void start() {}

std::vector<sample_t> stop() {
    return {};
}

double max_mhz() {
    return 0;
}
#endif

stats_t tail_stats(const std::vector<sample_t> &samples, double fraction) {
    stats_t st;
    auto add = [&](double mhz) {
        st.min = st.n ? std::min(st.min, mhz) : mhz;
        st.max = st.n ? std::max(st.max, mhz) : mhz;
        st.avg += mhz;
        st.n++;
    };
    if (samples.empty()) return st;
    const double from_ms = samples.back().ms * (1 - fraction);
    const sample_t *last_before = nullptr;
    for (const auto &s : samples) {
        if (s.mhz <= 0) continue;
        if (s.ms < from_ms)
            last_before = &s;
        else
            add(s.mhz);
    }
    if (!st.n && last_before) add(last_before->mhz);
    if (st.n) st.avg /= st.n;
    return st;
}

session_t::session_t() {
    start();
}

session_t::~session_t() {
    if (active_) stop();
}

std::vector<sample_t> session_t::stop() {
    active_ = false;
    return gpu_freq::stop();
}

} // namespace gpu_freq
