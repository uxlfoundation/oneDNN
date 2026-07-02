/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#include <vector>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "cpu/rv64/rvjit/rvjit_rvv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace rvjit;

// vpu_t::vector_latency(): vlenb/vlaneb, or 0 when vlaneb is unset

TEST(rvv_vpu, vector_latency_divides) {
    const rvv_t::vpu_t vpu(64, 16, LMUL::m8, LMUL::m1);
    EXPECT_EQ(vpu.vector_latency(), 4);
}

TEST(rvv_vpu, vector_latency_default) {
    const rvv_t::vpu_t vpu(64, 0, LMUL::m8, LMUL::m1);
    EXPECT_EQ(vpu.vector_latency(), 1);
}

// rvv_t forwards to its vpu_t/memory_t members

TEST(rvv, top_level_forwards) {
    const rvv_t model(rvv_t::vpu_t(64, 16, LMUL::m8, LMUL::m1),
            rvv_t::memory_t(32768, 64, 5));
    EXPECT_EQ(model.vector_latency(), 4);
    EXPECT_EQ(model.memory_latency(), 5);
}

// vpu_t::max_n_accumulators(): n = get_nvgroups(lmul_preference), minus 1 or 2

struct max_n_accumulators_case_t {
    LMUL lmul_preference;
    bool mixed_precision;
    int expected;
};

class max_n_accumulators_t
    : public ::testing::TestWithParam<max_n_accumulators_case_t> {};

TEST_P(max_n_accumulators_t, matches_formula) {
    const auto p = GetParam();
    const rvv_t::vpu_t vpu(64, 16, LMUL::m8, p.lmul_preference);
    const SEW sew_inp = p.mixed_precision ? SEW::e16 : SEW::e32;
    const SEW sew_acc = SEW::e32;
    EXPECT_EQ(vpu.max_n_accumulators(sew_inp, sew_acc), p.expected);
}

INSTANTIATE_TEST_SUITE_P(combos, max_n_accumulators_t,
        ::testing::ValuesIn(std::vector<max_n_accumulators_case_t> {
                // Uniform precision: reserve 2, except at m8 (reserve 1)
                {LMUL::m1, false, 16},
                {LMUL::m2, false, 14},
                {LMUL::m4, false, 6}, // matches conv's current VEC_MAX_UR
                {LMUL::m8, false, 3},
                // Mixed precision: reserve 1 regardless of LMUL
                {LMUL::m1, true, 16},
                {LMUL::m2, true, 15},
                {LMUL::m4, true, 7},
                {LMUL::m8, true, 3},
        }));

// from_params(): smoke test — must produce internally consistent, valid
// values without crashing, for a representative set of hardware parameters.
TEST(rvv, from_params_smoke) {
    const rvv_t model = rvv_t::from_params(
            /*vlen=*/256, /*cache_size=*/32768, /*cache_line_size=*/64,
            /*cache_latency=*/5);
    EXPECT_GT(model.vpu.vlenb, 0);
    EXPECT_GT(model.vpu.vlaneb, 0);
    EXPECT_GT(model.memory.cache_line_size, 0);
    EXPECT_GT(model.memory.cache_size, 0);
    EXPECT_GE(model.vpu.max_n_accumulators(SEW::e32, SEW::e32), 1);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
