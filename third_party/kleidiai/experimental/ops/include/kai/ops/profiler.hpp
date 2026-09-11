//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#define PROFILE_CALIBRATION 1
#define PROFILE_PREPA 2
#define PROFILE_PREPB 3
#define PROFILE_KERNEL 4
#define PROFILE_MERGE 5
#define PROFILE_ROWSUMS 6
#define PROFILE_QUANTIZE 7

#include "kai/ops/perf.hpp"

#ifndef NO_MULTI_THREADING
#include <mutex>
#endif

#undef NO_CALIBRATION
#define PULSE_ENABLE

#ifdef BARE_METAL
static inline void enable_cyclecounter() {
    uint64_t tmp;

    __asm __volatile (
        "mrs    %[tmp], pmcr_el0\n"
        "orr    %[tmp], %[tmp], #1\n"
        "msr    pmcr_el0, %[tmp]\n"
        : [tmp] "=r" (tmp)
    );
}

static inline void disable_cyclecounter() {
    uint64_t tmp;

    __asm __volatile (
        "mrs    %[tmp], pmcr_el0\n"
        "bic    %[tmp], %[tmp], #1\n"
        "msr    pmcr_el0, %[tmp]\n"
        : [tmp] "=r" (tmp)
    );
}

static inline uint64_t get_cyclecounter() {
    uint64_t retval;

    __asm __volatile (
        "mrs    %[retval], pmccntr_el0\n"
    : [retval] "=r" (retval));

    return retval;
}
#endif

namespace kai {
namespace ops {

#ifndef NO_MULTI_THREADING
extern std::mutex report_mutex;
#endif

class profiler {
private:
    static const int maxevents = 100000;
    unsigned long times[maxevents] = { };
    unsigned long units[maxevents] = { };
    int events[maxevents] = { };
    int currentevent=0;
    int countfd=0;

    class ScopedProfilerClass {
    private:
        profiler &_parent;
        bool legal=false;
#ifdef BARE_METAL
        uint64_t start_count;
#endif

    public:
        ScopedProfilerClass(profiler &prof, int i, unsigned long u) : _parent(prof) {
            if (prof.currentevent==maxevents)
                return;

            prof.events[prof.currentevent]=i;
            prof.units[prof.currentevent]=u;
            legal=true;
#ifndef BARE_METAL
            start_counter(prof.countfd);
#else
#ifdef PULSE_ENABLE
            enable_cyclecounter();
#endif
            start_count = get_cyclecounter();
#endif
        }

        ~ScopedProfilerClass() {
            if (!legal) return;

#ifndef BARE_METAL
            long long cycs = stop_counter(_parent.countfd);
#else
            long long cycs = get_cyclecounter() - start_count;
#ifdef PULSE_ENABLE
            disable_cyclecounter();
#endif
#endif
            _parent.times[_parent.currentevent++] = cycs;
        }
    };

public:
    profiler() {
#ifndef BARE_METAL
        countfd=open_cycle_counter();
#endif
        for (unsigned int i=0; i<100; i++) {
            auto p = ScopedProfiler(PROFILE_CALIBRATION, 1);
        }
    }

    ~profiler() {
        close(countfd);
        int tots[8];
        unsigned long counts[8];
        unsigned long tunits[8];
        const char * descs[] = { "Calibration", "Prepare A", "Prepare B", "Kernel", "Merge", "Row sums", "Requantize" };

        for (int i=1; i<8; i++) {
            tots[i] = 0;
            counts[i] = 0;
            tunits[i] = 0;
        }

        for (int i=0; i<currentevent; i++) {
//            printf("%10s: %ld\n", descs[events[i]-1], times[i]);
            tots[events[i]]++;
            counts[events[i]] += times[i];
            tunits[events[i]] += units[i];
        }

#ifdef NO_MULTI_THREADING
        printf("Profiled events:\n");
#else
        std::lock_guard<std::mutex> lock(report_mutex);
        printf("Profiled events (cpu %d):\n", sched_getcpu());
#endif

#ifndef NO_CALIBRATION
        int64_t calibrate_cycles = counts[1] / tots[1];

        for (int i=1; i<8; i++) {
            counts[i] -= tots[i] * calibrate_cycles;
        }

        // Suppress printing of calibration data.
        tots[1] = 0;
#endif

        unsigned long gtot=0;
        for (unsigned int i=1;i<8;i++) {
            gtot += counts[i];
        }

        printf("%20s      (%%)   %9s %9s %9s %12s %9s\n", "", "Events", "Total", "Average", "Bytes/MACs", "Per cycle");
        for (int i=1; i<8; i++) {
            if (tots[i]==0) {
                continue;
            }
            printf("%20s (%5.1f%%) : %9d %9ld %9ld %12lu %9.2f\n",descs[i-1],((float)counts[i]*100.0f/gtot),tots[i],counts[i],counts[i]/tots[i],tunits[i],(float)tunits[i]/counts[i]);
        }

        printf("%20s          :           %9ld\n","Grand total",gtot);
    }

    ScopedProfilerClass ScopedProfiler(int i, unsigned long u) {
        return ScopedProfilerClass(*this, i, u);
    }
};

}  // namespace ops
}  // namespace kai
