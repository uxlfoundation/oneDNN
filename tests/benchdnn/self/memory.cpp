/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <cmath>
#include <set>

#include "dnnl_memory.hpp"

#include "self/self.hpp"

namespace self {

// Verifies that fill_for_perf_test() produces non-uniform, finite,
// nan-free, inf-free, and seed-varying valid data for mode=F.
static int check_fill_for_perf_test() {
    if (!DNNL_INTEL_GPU_RUNTIME_ENABLED || is_cpu(get_test_engine())) {
        BENCHDNN_PRINT(2, "%s\n",
                "Skipping fill_for_perf_test checks due to the use of a "
                "non-Intel GPU runtime or a CPU runtime.");
        return OK;
    }

    const int nelems = 1024;
    dnnl_dim_t dims {nelems};
    auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);

    // 1. Non-uniformity check: require at least 50% unique values
    {
        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        m.unmap();
        m.fill_for_perf_test(nelems * sizeof(float), 0);
        m.map();

        std::set<uint32_t> unique_vals_uint32_t;
        const auto *ptr_uint32_t = static_cast<const uint32_t *>(m);

        bool all_same = true;
        uint32_t first_val = ptr_uint32_t[0];
        for (int i = 0; i < nelems; i++) {
            // printf("%08X\n", ptr_uint32_t[i]);
            unique_vals_uint32_t.insert(ptr_uint32_t[i]);
            if (ptr_uint32_t[i] != first_val) all_same = false;
        }

        SELF_CHECK(!all_same,
                "fill_for_perf_test failed because all 32-bit samples are "
                "identical (val=0x%08X)",
                first_val);
        SELF_CHECK(
                unique_vals_uint32_t.size() > static_cast<size_t>(nelems / 2),
                "fill_for_perf_test failed because it produced too few unique "
                "32-bit values: %d",
                (int)unique_vals_uint32_t.size());
    }

    // 2. No NaN/Inf for any available FP type (mask 0xEEEEEEEE)
    {
        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        m.unmap();
        m.fill_for_perf_test(nelems * sizeof(float), 0);
        m.map();

        const auto *ptr_u32 = static_cast<const uint32_t *>(m);
        const auto *ptr_f32 = static_cast<const float *>(m);

        for (int i = 0; i < nelems; i++) {
            SELF_CHECK((ptr_u32[i] & 0x11111111u) == 0,
                    "fill_for_perf_test byte-mask invariant violated at index "
                    "%d: 0x%08X & 0x11111111 = 0x%08X",
                    i, ptr_u32[i], ptr_u32[i] & 0x11111111u);
            SELF_CHECK(std::isfinite(ptr_f32[i]),
                    "fill_for_perf_test produced non-finite f32 at index %d",
                    i);
        }
    }

    // 3. Different calls should produce different data (seed test)
    {
        dnn_mem_t m1(md, get_test_engine(), /* prefill = */ false);
        dnn_mem_t m2(md, get_test_engine(), /* prefill = */ false);
        m1.unmap();
        m2.unmap();
        m1.fill_for_perf_test(nelems * sizeof(float), 0);
        m2.fill_for_perf_test(nelems * sizeof(float), 0);
        m1.map();
        m2.map();
        const auto *p1 = static_cast<const uint32_t *>(m1);
        const auto *p2 = static_cast<const uint32_t *>(m2);
        int num_different = 0;
        for (int i = 0; i < nelems; i++)
            if (p1[i] != p2[i]) num_different++;
        // Require at least 50% of values to differ between two calls
        // if nelems set to default value of 1024.
        SELF_CHECK(num_different > nelems / 2,
                "Two fill_for_perf_test calls produced too similar data: "
                "only %d/%d values differ",
                num_different, nelems);
    }

    // 4. All initialized (tail leftover bytes should be initialized too)
    {
        const int nelems = 17;
        dnnl_dim_t dims {nelems};
        auto md = dnn_mem_t::init_md(1, &dims, dnnl_f16, tag::abx);

        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        const std::size_t total_bytes = nelems * sizeof(std::uint16_t);
        m.unmap();
        m.memset(0xFF, total_bytes, 0);
        m.fill_for_perf_test(total_bytes, 0);
        m.map();

        const auto *raw16 = static_cast<const uint16_t *>(m);
        int nan_count = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(nelems); ++i)
            if (raw16[i] == 0xFFFFu) nan_count++;

        // All values should be initialized and nan/inf free
        SELF_CHECK(nan_count == 0,
                "fill_for_perf_test left %d uninitialized values (0xFFFF)",
                nan_count);
    }

    return OK;
}

static int check_bool_operator() {
    dnnl_dim_t dims {1};
    auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);
    auto md0 = dnn_mem_t::init_md(0, &dims, dnnl_f32, tag::abx);
    {
        dnn_mem_t m;
        SELF_CHECK_EQ(bool(m), false);
    }
    {
        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, get_test_engine(), /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, &dims /* strides */, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, &dims /* strides */, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(md, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    return OK;
}

void memory() {
    RUN(check_bool_operator());
    RUN(check_fill_for_perf_test());
}

} // namespace self
