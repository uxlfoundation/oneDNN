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

#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"

#include "self/self.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace self {

static int check_bool_operator() {
    dnnl_dim_t dims {1};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
    DNN_SAFE(dnn_mem_t::init_md(md, 1, &dims, dnnl_f32, tag::abx), WARN);
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md0;
    DNN_SAFE(dnn_mem_t::init_md(md0, 0, &dims, dnnl_f32, tag::abx), WARN);
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

static int check_init_md_status() {
    dnnl_dims_t dims {2, 16};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
    SELF_CHECK_EQ(
            dnn_mem_t::init_md(md, 2, dims, dnnl_f32, "ab"), dnnl_success);
    // A tag with more letters than `ndims` must return an error instead of
    // terminating the run.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bad_md;
    SELF_CHECK_EQ(dnn_mem_t::init_md(bad_md, 2, dims, dnnl_f32, "abcd"),
            dnnl_invalid_arguments);
    return OK;
}

// Puts `n_nonzero` non-zeros at random padded positions of a blocked memory and
// expects check_zero_padding() to report exactly that many. The real area is
// filled with non-zeros, too, so that a check reaching outside of the padded
// area, or visiting a padded element more than once, breaks the count.
static int check_zero_padding_count(int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t dt, const std::string &tag, int n_nonzero) {
    dnn_mem_t mem(ndims, dims, dt, tag, get_test_engine(),
            /* prefill = */ false);
    SELF_CHECK_EQ(bool(mem), true);
    ::memset(mem.get_mapped_pointer<void>(), 0, mem.size());

    // Positions are enumerated over padded dims and translated with
    // md_off_v() to keep the reference independent from the checker.
    const auto &pdims = mem.padded_dims();
    std::vector<dnnl_dim_t> pad_offs;
    for (int64_t i = 0; i < mem.nelems(/* with_padded_dims = */ true); ++i) {
        dnnl_dims_t pos = {};
        bool is_pad = false;
        int64_t rem = i;
        for (int d = ndims - 1; d >= 0; --d) {
            pos[d] = rem % pdims[d];
            rem /= pdims[d];
            is_pad = is_pad || pos[d] >= dims[d];
        }
        const auto off = md_off_v(mem, pos, /* is_pos_padded = */ true);
        if (is_pad)
            pad_offs.push_back(off);
        else
            mem.set_elem(off, 1.f);
    }
    SELF_CHECK(n_nonzero <= static_cast<int>(pad_offs.size()),
            "%d non-zeros requested, %d padded elements in '%s'", n_nonzero,
            static_cast<int>(pad_offs.size()), tag.c_str());

    std::mt19937 rng(n_nonzero);
    std::shuffle(pad_offs.begin(), pad_offs.end(), rng);
    // Values stay small to be exact in every tested data type, s4 included.
    for (int i = 0; i < n_nonzero; ++i)
        mem.set_elem(pad_offs[i], 1.f + i % 3);

    int errors = -1;
    const int status = check_zero_padding(
            mem, DNNL_ARG_DST, /* res = */ nullptr, &errors);
    BENCHDNN_PRINT(1,
            "check_zero_padding: dt=%s tag=%s non-zeros=%d found=%d\n",
            dt2str(dt), tag.c_str(), n_nonzero, errors);
    SELF_CHECK_EQ(status, n_nonzero == 0 ? OK : FAIL);
    SELF_CHECK_EQ(errors, n_nonzero);
    return OK;
}

// Covers the 4-, 2- and 1-byte paths of the bitwise zero check, and the
// sub-byte path.
static const std::vector<dnnl_data_type_t> zero_padding_dts
        = {dnnl_f32, dnnl_bf16, dnnl_s8, dnnl_s4};

static int check_zero_padding_single_blocked() {
    // pdims: 2x16x3, 18 padded elements.
    const dnnl_dims_t dims {2, 13, 3};
    for (auto dt : zero_padding_dts) {
        for (int n : {0, 1, 5, 18})
            SAFE(check_zero_padding_count(3, dims, dt, "aBc16b", n), WARN);
    }
    return OK;
}

static int check_zero_padding_double_blocked() {
    // pdims: 8x16x2, 126 padded elements, 18 of them padded over both dims.
    const dnnl_dims_t dims {5, 13, 2};
    for (auto dt : zero_padding_dts) {
        for (int n : {0, 1, 17, 126})
            SAFE(check_zero_padding_count(3, dims, dt, "ABc4a8b", n), WARN);
    }
    return OK;
}

static int check_zero_padding_triple_blocked() {
    // Three inner blocks over two dims, as in `aBxC16c64b4c`.
    // pdims: 2x8x8x3, 294 padded elements.
    const dnnl_dims_t dims {2, 5, 3, 3};
    for (auto dt : zero_padding_dts) {
        for (int n : {0, 1, 29, 294})
            SAFE(check_zero_padding_count(4, dims, dt, "aBCd2c8b4c", n), WARN);
    }
    return OK;
}

void memory() {
    RUN(check_bool_operator());
    RUN(check_init_md_status());
    RUN(check_zero_padding_single_blocked());
    RUN(check_zero_padding_double_blocked());
    RUN(check_zero_padding_triple_blocked());
}

} // namespace self
