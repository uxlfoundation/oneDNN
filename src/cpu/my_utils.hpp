/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef MY_UTILS_HPP
#define MY_UTILS_HPP

#include "../common/c_types_map.hpp"
#include "../common/dnnl_thread.hpp"
#include "../common/memory_tracking.hpp"
#include "../common/utils.hpp"

#define USE_TIMERS

#ifdef USE_TIMERS
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "../common/verbose.hpp"
#define CALL_TIMER(x) (x)
#else
#define CALL_TIMER(x)
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace my_utils {

int read_stack(const char *title, int ithr);
int read_pagemap(
        const char *title, int ithr, size_t virt_addr, unsigned long virt_size);
bool get_env_num(const char *vname, int &value);
int is_debugged();

template <typename T>
bool get_env_value(const char *vname, T &value) {
    int res = static_cast<int>(value);
    bool status = get_env_num(vname, res);
    value = static_cast<T>(res);
    return status;
}

#ifdef USE_TIMERS
struct my_timer {
    static constexpr int CS = (64 * 64) / sizeof(double);
    static constexpr int MAXTHR = 1024;
    static constexpr int VSIZE = CS * MAXTHR;
    std::vector<double> v;
    std::vector<double> t;
    std::vector<uint64_t> c;
    //    std::vector<double> fr;
    double vs;
    double ts;
    int ncalls {0};
    std::string name;
    bool enabled {true};

    static constexpr bool use_rdtsc = true;
    static constexpr size_t tsc_freq = 2400000; // frequency is ~2Ghz
#ifdef _WIN32
#else
    struct timeval time;
#endif
    my_timer(const char *nm, bool enable = true)
        : v(VSIZE)
        , t(VSIZE)
        , c(VSIZE)
        /*   , fr(VSIZE) */
        , name(nm)
        , enabled(enable) {
        reset();
    }

    void reset() {
        //        printf("DEBUG: my_utils:reset :  name = %s \n", name.c_str());
        vs = -1000000.f;
        ts = -1000000.f;
        ncalls = 0;

        std::fill(v.begin(), v.end(), -1000000.f);
        std::fill(t.begin(), t.end(), -1000000.f);
        std::fill(c.begin(), c.end(), 0);
        //        std::fill(fr.begin(), fr.end(), 0);
    }

    ~my_timer() {
        //        print_all();
    }

    void print_all() {
        //        printf("DEBUG: my_utils:print_all :  name = %s \n", name.c_str());
        if (!enabled) return;
        if (ncalls <= 0) return;
        std::cout << "my_timer: " << name << " : ";
        printf("ncalls = %d", ncalls);
        double c_v = 0;
        double call_v = 0;
        double max_v = -1;
        int tot_thr = 0;
        for (int i = 0; i < MAXTHR; i++) {
            const auto thr_v = get_v(i);
            if (thr_v < -0.1f) continue;
            printf(" %d = %6.2f ", i, thr_v / tsc_freq);
            call_v += thr_v;
            c_v += get_c(i);
            max_v = thr_v > max_v ? thr_v : max_v;
            tot_thr++;
        }
        auto avg_v = tot_thr >= 0 ? call_v / tot_thr : 0;
        if (use_rdtsc) {
            avg_v /= tsc_freq;
            max_v /= tsc_freq;
        }
        printf(" : avg = %7.4f max = %7.4f total = %8.4f c_v = %8.1f\n", avg_v,
                max_v, avg_v * ncalls, c_v);
    }

    void new_call() {
        if (!enabled) return;
        ncalls++;
    }

    inline void start(int i = 0) {
        if (!enabled) return;
        //        const auto i = omp_get_thread_num();

        const auto idx = i * CS;
        t[idx] = get_ms();
        c[idx]++;

        //        printf("DEBUG: my_utils:start : %s:  idx = %d \n", name.c_str(), idx);
    }

    inline void stop(int i = 0, bool print = false) {
        if (!enabled) return;
        //        const auto i = omp_get_thread_num();

        const auto idx = i * CS;
        t[idx] = get_ms() - t[idx];
#if 0
        if (v[idx] < -1) v[idx] = 1e100;
        v[idx] = t[idx] < v[idx] ? t[idx] : v[idx];
#else
        if (v[idx] < -1) v[idx] = 0;
        v[idx] += t[idx];
#endif

        //        fr[idx] += get_tsc_frequency();
        if (print)
            printf("DEBUG: my_utils:stop : %s: t[%d] = %f v[%d] = %f\n",
                    name.c_str(), idx, t[idx], idx, v[idx]);
    }

    double get_v(int i) {
        if (!enabled) return 0;
        const auto idx = i * CS;
        return (ncalls != 0) ? (v[idx] / ncalls) : v[idx];
    }

    double get_c(int i) {
        if (!enabled) return 0;
        const auto idx = i * CS;
        return (ncalls != 0) ? (c[idx] / ncalls) : c[idx];
    }

#ifdef _WIN32
    inline double get_ms() { return 0; }
    inline uint64_t get_tsc_frequency() { return 0; }
#else
    inline double get_ms() {
        if (use_rdtsc) {
            uint32_t hi, lo;
            asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
            return (((uint64_t)hi) << 32) | lo;
        } else {
            gettimeofday(&time, nullptr);
            return 1e+3 * static_cast<double>(time.tv_sec)
                    + 1e-3 * static_cast<double>(time.tv_usec);
        }
    }

    inline uint64_t get_tsc_frequency() {
        uint32_t eax, ebx, ecx, edx;
        asm volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(0x15)
                     :);

        // ebx contains the denominator, ecx contains the numerator
        if (ecx == 0 || ebx == 0) {
            // TSC not supported or not reliable
            return 0;
        }

        uint64_t numerator = ecx;
        uint64_t denominator = ebx;

        return numerator / denominator;
    }
#endif
};

#define MY_TIMER(_n_, _enable_) my_timer _n_(#_n_, _enable_);

struct mt_caller {
    my_timer &mt;
    int ithr;
    mt_caller(my_timer &mt_) : mt(mt_), ithr(0) { CALL_TIMER(mt.start(ithr)); }

    mt_caller(my_timer &mt_, int ithr_) : mt(mt_), ithr(ithr_) {
        CALL_TIMER(mt.start(ithr));
    }
    ~mt_caller() { CALL_TIMER(mt.stop(ithr)); }
};
#else
struct my_timer {
    my_timer(const char *nm, bool enable = true) {}
};

struct mt_caller {
    mt_caller(my_timer &mt_) {}
    mt_caller(my_timer &mt, int ithr_) {}
    ~mt_caller() {}
};
#endif

} // namespace my_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
