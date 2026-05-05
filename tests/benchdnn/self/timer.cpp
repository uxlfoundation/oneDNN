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

#include "utils/timer.hpp"

#include "self/self.hpp"

namespace self {

static int check_timer_filter() {
    {
        timer::timer_t t;
        constexpr int sample_size = 20;
        for (int i = 0; i < sample_size / 2; i++) {
            t.stop(1, 0, 0);
        }
        for (int i = 0; i < sample_size - sample_size / 2; i++) {
            t.stop(1, 0, 1000);
        }
        t.filter_collection();
        SELF_CHECK_EQ(t.ms(timer::timer_t::mode_t::min), 1000);
        SELF_CHECK_EQ(t.times(), sample_size - sample_size / 2);
    }
    {
        timer::timer_t t;
        constexpr int sample_size = 20;
        for (int i = 0; i < sample_size / 2; i++) {
            t.stop(1, 0, 0);
        }
        t.stop(1, 0, 500);
        for (int i = 1; i < sample_size - sample_size / 2; i++) {
            t.stop(1, 0, 1000);
        }
        t.filter_collection();
        SELF_CHECK_EQ(t.ms(timer::timer_t::mode_t::min), 1000);
        // 500us must be dropped.
        SELF_CHECK_EQ(t.times(), sample_size - sample_size / 2 - 1);
    }
    {
        timer::timer_t t;
        constexpr int sample_size = 21;
        for (int i = 0; i < sample_size / 2; i++) {
            t.stop(1, 0, 0);
        }
        t.stop(1, 0, 1000);
        t.stop(1, 0, 1039);
        for (int i = 2; i < sample_size - sample_size / 2; i++) {
            t.stop(1, 0, 2000);
        }
        t.filter_collection();
        SELF_CHECK_EQ(t.ms(timer::timer_t::mode_t::min), 1000);
        // 1000us mustn't be dropped.
        SELF_CHECK_EQ(t.times(), sample_size - sample_size / 2);
    }
    {
        timer::timer_t t;
        constexpr int sample_size = 21;
        for (int i = 0; i < sample_size / 2; i++) {
            t.stop(1, 0, 0);
        }
        t.stop(1, 0, 1000);
        t.stop(1, 0, 1040);
        for (int i = 2; i < sample_size - sample_size / 2; i++) {
            t.stop(1, 0, 2000);
        }
        t.filter_collection();
        SELF_CHECK_EQ(t.ms(timer::timer_t::mode_t::min), 1040);
        // 1000us must be dropped.
        SELF_CHECK_EQ(t.times(), sample_size - sample_size / 2 - 1);
    }
    {
        timer::timer_t t;
        constexpr int sample_size = 40;
        for (int i = 0; i < sample_size / 2; i++) {
            t.stop(1, 0, 0);
        }
        t.stop(1, 0, 1000);
        t.stop(1, 0, 1000);
        for (int i = 2; i < sample_size - sample_size / 2; i++) {
            t.stop(1, 0, 2000);
        }
        t.filter_collection();
        SELF_CHECK_EQ(t.ms(timer::timer_t::mode_t::min), 2000);
        // Both 1000us must be dropped.
        SELF_CHECK_EQ(t.times(), sample_size - sample_size / 2 - 2);
    }
    {
        timer::timer_t t;
        constexpr int sample_size = 1000;
        for (int i = 0; i < sample_size / 2; i++) {
            t.stop(1, 0, 0);
        }
        const int n_special_fill = (sample_size - sample_size / 2) / 100 - 1;
        for (int i = 0; i < n_special_fill; i++) {
            t.stop(1, 0, 1000);
        }
        for (int i = n_special_fill; i < sample_size - sample_size / 2; i++) {
            t.stop(1, 0, 1001);
        }
        t.filter_collection();
        SELF_CHECK_EQ(t.ms(timer::timer_t::mode_t::min), 1001);
        // Both 1000us must be dropped.
        SELF_CHECK_EQ(
                t.times(), sample_size - sample_size / 2 - n_special_fill);
    }
    return OK;
}

void timer() {
    RUN(check_timer_filter());
}

} // namespace self
