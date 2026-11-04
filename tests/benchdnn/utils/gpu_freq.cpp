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
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include "windows.h"
#else
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/gpu_freq.hpp"
#include "utils/system.hpp"

namespace gpu_freq {

const source_t default_source = source_t::automatic;
source_t source = default_source;

source_t str2source(const std::string &s) {
    if (s == "auto") return source_t::automatic;
    if (s == "none") return source_t::none;
    if (s == "sysfs") return source_t::sysfs;
    if (s == "ze") return source_t::ze;
    BENCHDNN_PRINT(
            0, "Error: unsupported GPU frequency source \"%s\".\n", s.c_str());
    SAFE_V(FAIL);
    return source_t::none;
}

const char *source2str(source_t s) {
    switch (s) {
        case source_t::none: return "none";
        case source_t::automatic: return "auto";
        case source_t::sysfs: return "sysfs";
        case source_t::ze: return "ze";
    }
    return "unknown";
}

namespace {

// Reads the instantaneous actual frequency of a device.
struct reader_t {
    virtual ~reader_t() = default;
    // Returns the average actual frequency over the device compute domains in
    // MHz, `0` if all domains are idle, or a negative value on error.
    virtual double read_mhz() = 0;
    virtual double max_mhz() const = 0;
    virtual std::string info() const = 0;
    virtual int period_us() const = 0;
};

#ifdef __linux__
bool path_exists(const std::string &path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<std::string> list_dir(const std::string &path) {
    std::vector<std::string> names;
    DIR *dir = ::opendir(path.c_str());
    if (!dir) return names;
    for (dirent *e = ::readdir(dir); e != nullptr; e = ::readdir(dir)) {
        std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        names.push_back(n);
    }
    ::closedir(dir);
    std::sort(names.begin(), names.end());
    return names;
}

bool read_file_long(int fd, long &value) {
    char buf[32];
    ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return false;
    buf[n] = '\0';
    value = std::strtol(buf, nullptr, 10);
    return true;
}

bool read_path_long(const std::string &path, long &value) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    bool ok = read_file_long(fd, value);
    ::close(fd);
    return ok;
}

bool is_intel_device(const std::string &card_path) {
    int fd = ::open((card_path + "/device/vendor").c_str(), O_RDONLY);
    if (fd < 0) return false;
    char buf[16] = {};
    ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
    ::close(fd);
    return n >= 6 && std::string(buf, 6) == "0x8086";
}

// Polls `act_freq` of every compute GT of a DRM card. File descriptors stay
// open, so a sample costs a single `pread()` (~2 us) per GT.
struct sysfs_reader_t : public reader_t {
    struct gt_t {
        std::string name;
        int fd;
    };

    ~sysfs_reader_t() override {
        for (auto &gt : gts_)
            ::close(gt.fd);
    }

    static std::unique_ptr<reader_t> create(const std::string &bdf) {
        const std::string drm_root = "/sys/class/drm";
        std::vector<std::unique_ptr<sysfs_reader_t>> candidates;
        for (const auto &card : list_dir(drm_root)) {
            if (card.rfind("card", 0) != 0
                    || card.find('-') != std::string::npos)
                continue;
            const std::string card_path = drm_root + "/" + card;
            if (!is_intel_device(card_path)) continue;
            char real[PATH_MAX];
            if (!::realpath((card_path + "/device").c_str(), real)) continue;
            std::string dev_path = real;
            std::string card_bdf = dev_path.substr(dev_path.rfind('/') + 1);
            if (!bdf.empty() && card_bdf != bdf) continue;

            std::unique_ptr<sysfs_reader_t> r(new sysfs_reader_t());
            r->card_ = card + " (" + card_bdf + ")";
            r->add_xe_gts(card_path);
            if (r->gts_.empty()) r->add_i915_gts(card_path);
            if (r->gts_.empty()) continue;
            candidates.push_back(std::move(r));
        }
        // Without a BDF the device can't be matched reliably on multi-GPU
        // systems.
        if (candidates.size() != 1) return nullptr;
        return std::unique_ptr<reader_t>(candidates[0].release());
    }

    double read_mhz() override {
        double sum = 0;
        int n_busy = 0;
        for (const auto &gt : gts_) {
            long v = 0;
            if (!read_file_long(gt.fd, v)) return -1;
            if (v > 0) {
                sum += v;
                n_busy++;
            }
        }
        return n_busy ? sum / n_busy : 0;
    }

    double max_mhz() const override { return max_mhz_; }
    int period_us() const override { return 250; }

    std::string info() const override {
        std::string s = "sysfs " + card_ + ":";
        for (const auto &gt : gts_)
            s += " " + gt.name;
        return s;
    }

private:
    sysfs_reader_t() = default;

    void add_gt(const std::string &name, const std::string &act_path,
            const std::string &max_path) {
        int fd = ::open(act_path.c_str(), O_RDONLY);
        if (fd < 0) return;
        gts_.push_back({name, fd});
        long v = 0;
        if (read_path_long(max_path, v))
            max_mhz_ = std::max(max_mhz_, (double)v);
    }

    // xe: card/device/tileX/gtY/freq0/{act_freq,rp0_freq}. GTs without
    // rcs/ccs engines are media GTs and are skipped (same as GEOPM).
    void add_xe_gts(const std::string &card_path) {
        const std::string device = card_path + "/device";
        for (const auto &tile : list_dir(device)) {
            if (tile.rfind("tile", 0) != 0) continue;
            const std::string tile_path = device + "/" + tile;
            for (const auto &gt : list_dir(tile_path)) {
                const std::string gt_path = tile_path + "/" + gt;
                const std::string freq = gt_path + "/freq0";
                if (!path_exists(freq + "/act_freq")) continue;
                const std::string eng = gt_path + "/engines";
                if (path_exists(eng) && !path_exists(eng + "/rcs")
                        && !path_exists(eng + "/ccs"))
                    continue;
                add_gt(tile + "/" + gt, freq + "/act_freq", freq + "/rp0_freq");
            }
        }
    }

    // i915: card/gt/gtY/rps_act_freq_mhz or card/gt_act_freq_mhz.
    void add_i915_gts(const std::string &card_path) {
        const std::string gt_root = card_path + "/gt";
        for (const auto &gt : list_dir(gt_root)) {
            const std::string p = gt_root + "/" + gt;
            if (path_exists(p + "/rps_act_freq_mhz"))
                add_gt(gt, p + "/rps_act_freq_mhz", p + "/rps_RP0_freq_mhz");
        }
        if (gts_.empty() && path_exists(card_path + "/gt_act_freq_mhz"))
            add_gt("gt", card_path + "/gt_act_freq_mhz",
                    card_path + "/gt_RP0_freq_mhz");
    }

    std::string card_;
    std::vector<gt_t> gts_;
    double max_mhz_ = 0;
};
#endif

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

// Polls `zesFrequencyGetState()` for every GPU frequency domain of the device.
// A call costs ~50 us on Linux, so the polling period is larger than sysfs.
struct ze_reader_t : public reader_t {
    static std::unique_ptr<reader_t> create(const std::string &bdf) {
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
        return std::unique_ptr<reader_t>(candidates[0].release());
    }

    double read_mhz() override {
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

    double max_mhz() const override { return max_mhz_; }
    int period_us() const override { return 1000; }

    std::string info() const override {
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
        thread_ = std::thread([this]() {
            const auto period = std::chrono::microseconds(reader_->period_us());
            while (!stop_.load(std::memory_order_relaxed)) {
                samples_.push_back(reader_->read_mhz());
                std::this_thread::sleep_for(period);
            }
        });
    }

    stats_t stop() {
        stats_t st;
        if (!running_) return st;
        stop_.store(true);
        thread_.join();
        running_ = false;
        // The GPU stays awake for a while after completion, so the final
        // sample still reflects the frequency of the last executions. It
        // matters for windows shorter than the polling period.
        samples_.push_back(reader_->read_mhz());

        st.n_samples = samples_.size();
        double sum = 0;
        for (double f : samples_) {
            if (f <= 0) continue;
            st.min_mhz = st.n_busy ? std::min(st.min_mhz, f) : f;
            st.max_mhz = st.n_busy ? std::max(st.max_mhz, f) : f;
            sum += f;
            st.n_busy++;
        }
        if (st.n_busy) st.avg_mhz = sum / st.n_busy;
        return st;
    }

private:
    sampler_t() {
        if (!is_gpu() || source == source_t::none) return;
        const std::string bdf = get_gpu_pci_bdf();
#ifdef __linux__
        if (source == source_t::automatic || source == source_t::sysfs)
            reader_ = sysfs_reader_t::create(bdf);
#endif
        if (!reader_
                && (source == source_t::automatic || source == source_t::ze))
            reader_ = ze_reader_t::create(bdf);

        if (reader_) {
            BENCHDNN_PRINT(2, "[GPU_FREQ] source: %s; max: %.0f MHz\n",
                    reader_->info().c_str(), reader_->max_mhz());
            samples_.reserve(
                    (size_t)(4 * max_ms_per_prb * 1e3 / reader_->period_us()));
        } else {
            const int verbosity = source == source_t::automatic ? 2 : 0;
            BENCHDNN_PRINT(verbosity,
                    "[GPU_FREQ] Warning: no GPU frequency source \"%s\" "
                    "available for device \"%s\".\n",
                    source2str(source), bdf.c_str());
        }
    }

    ~sampler_t() {
        if (running_) {
            stop_.store(true);
            thread_.join();
        }
    }

    std::unique_ptr<reader_t> reader_;
    std::thread thread_;
    std::atomic<bool> stop_ {false};
    bool running_ = false;
    std::vector<double> samples_;
};

} // namespace

bool is_available() {
    return sampler_t::get().is_available();
}

void start() {
    sampler_t::get().start();
}

stats_t stop() {
    return sampler_t::get().stop();
}

double max_mhz() {
    return sampler_t::get().max_mhz();
}

} // namespace gpu_freq
