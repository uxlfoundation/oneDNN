/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef COMMON_GEMM_TYPES_HPP
#define COMMON_GEMM_TYPES_HPP

#include <assert.h>
#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/opdesc.hpp"

namespace dnnl {
namespace impl {

enum transpose_t { dnnl_notrans, dnnl_trans };

namespace transpose {
const transpose_t notrans = dnnl_notrans;
const transpose_t trans = dnnl_trans;
} // namespace transpose

enum offsetc_t { dnnl_fixed, dnnl_column, dnnl_row };

namespace offsetc {
const offsetc_t fixed = dnnl_fixed;
const offsetc_t column = dnnl_column;
const offsetc_t row = dnnl_row;
} // namespace offsetc

enum sum_ab_t { dnnl_sum_a_row, dnnl_sum_b_col, dnnl_sum_none };
namespace sum_ab {
const sum_ab_t sum_a_row = dnnl_sum_a_row;
const sum_ab_t sum_b_col = dnnl_sum_b_col;
const sum_ab_t sum_none = dnnl_sum_none;
} // namespace sum_ab

// A descriptor for a matrix multiplication (gemm) operation.
struct gemm_desc_t : public op_desc_t {
    gemm_desc_t() : op_desc_t(primitive_kind::gemm) {}

    std::unique_ptr<op_desc_t> clone() const override {
        return utils::make_unique<gemm_desc_t>(*this);
    }

    // Type for accumulating A*B.
    dnnl_data_type_t acc_type {};

    void set_inputs(const memory_desc_t &a, const memory_desc_t &b,
            const memory_desc_t &c, const memory_desc_t &bias) {
        a_desc_ = a;
        b_desc_ = b;
        c_desc_ = c;
        bias_desc_ = bias;
    }

    void set_sum_ab(sum_ab_t sum_ab, dnnl_data_type_t sum_ab_type) {
        sum_ab_ = sum_ab;
        sum_ab_type_ = sum_ab_type;
    }

    const memory_desc_t &a_md() const { return a_desc_; }
    const memory_desc_t &b_md() const { return b_desc_; }
    const memory_desc_t &c_md() const { return c_desc_; }
    const memory_desc_t &bias_md() const { return bias_desc_; }
    // Non-const overloads are for init-time format resolution only.
    // Orientation flips must go through gemm_pd_t::apply_swap_ab().
    memory_desc_t &a_md() { return a_desc_; }
    memory_desc_t &b_md() { return b_desc_; }
    memory_desc_t &c_md() { return c_desc_; }
    memory_desc_t &bias_md() { return bias_desc_; }

    sum_ab_t sum_ab() const { return sum_ab_; }
    dnnl_data_type_t sum_ab_type() const { return sum_ab_type_; }
    bool swap_ab() const { return swap_ab_; }

    // User-view accessors: return the slot mds as originally supplied,
    // regardless of apply_swap_ab() state.
    memory_desc_t a_md_user_view() const {
        memory_desc_t res = swap_ab_ ? b_desc_ : a_desc_;
        if (swap_ab_) transpose_mn_axes(res);
        return res;
    }
    memory_desc_t b_md_user_view() const {
        memory_desc_t res = swap_ab_ ? a_desc_ : b_desc_;
        if (swap_ab_) transpose_mn_axes(res);
        return res;
    }
    memory_desc_t c_md_user_view() const {
        memory_desc_t res = c_desc_;
        if (swap_ab_) transpose_mn_axes(res);
        return res;
    }
    memory_desc_t bias_md_user_view() const {
        memory_desc_t res = bias_desc_;
        if (swap_ab_) transpose_mn_axes(res);
        return res;
    }

    inline bool is_batched() const { return c_md().ndims >= 3; }

    static transpose_t get_trans(const memory_desc_t &md) {
        if (!md.ndims) return transpose::notrans; // arbitrary

        using namespace data_type;
        // Leading dimension must be byte-aligned for 4-bit types.
        bool is_4bit = utils::one_of(md.data_type, f4_e2m1, f4_e3m0, s4, u4);
        dim_t inner_m = md.dims[md.ndims - 2];
        dim_t inner_n = md.dims[md.ndims - 1];
        auto strides = md.format_desc.blocking.strides;
        dim_t notranspose_ld = inner_m > 1 ? strides[md.ndims - 2] : inner_n;
        if (is_4bit && notranspose_ld % 2 != 0) return transpose::trans;

        return inner_n != 1 && strides[md.ndims - 1] != 1 ? transpose::trans
                                                          : transpose::notrans;
    }

    transpose_t transa() const { return invert(get_trans(a_md())); }
    transpose_t transb() const { return invert(get_trans(b_md())); }
    transpose_t transc() const { return get_trans(c_md()); }
    transpose_t trans_bias() const { return get_trans(bias_md()); }

    dnnl_dim_t batch() const {
        // if ndims < 3, it should return 1
        int64_t batch = 1;
        const auto &c = c_md();
        for (int i = 0; i < c.ndims - 2; ++i) {
            if (is_runtime_value(c.dims[i]))
                return runtime_value_for<dnnl_dim_t>();
            batch *= c.dims[i];
        }
        return batch;
    }

    dnnl_dim_t m() const {
        const auto &c = c_md();
        return c.dims[c.ndims - 2];
    }
    dnnl_dim_t n() const {
        const auto &c = c_md();
        return c.dims[c.ndims - 1];
    }
    dnnl_dim_t k() const {
        const auto &a = a_md();
        return a.dims[a.ndims - 1];
    }

    static dnnl_dim_t get_stride(const memory_desc_t &md, int dim = 0) {
        return (dim >= md.ndims - 2 || md.dims[dim] == 1)
                ? 0
                : md.format_desc.blocking.strides[dim];
    }

    /** Stride between 2 matrices A in a batch. */
    dnnl_dim_t stride_a(int dim = 0) const {
        return get_stride(a_md(), dim);
    }
    /** Stride between 2 matrices B in a batch. */
    dnnl_dim_t stride_b(int dim = 0) const {
        return get_stride(b_md(), dim);
    }
    /** Stride between 2 matrices C in a batch. */
    dnnl_dim_t stride_c(int dim = 0) const { return get_stride(c_md(), dim); }

    // This assumes that one of the dimensions has strides 1
    static dnnl_dim_t get_ld(const memory_desc_t &md) {
        auto strides = md.format_desc.blocking.strides;
        assert(md.dims[md.ndims - 1] == 1 || strides[md.ndims - 1] == 1
                || md.dims[md.ndims - 2] == 1 || strides[md.ndims - 2] == 1);
        // Degenerate M=1 case: handle directly so get_trans() can stay
        // pure (its result is consumed raw by non-inverted callers).
        if (md.dims[md.ndims - 2] == 1 && md.dims[md.ndims - 1] > 1)
            return strides[md.ndims - 1];
        switch (get_trans(md)) {
            case transpose::trans:
                return md.dims[md.ndims - 1] > 1 ? strides[md.ndims - 1]
                                                 : md.dims[md.ndims - 2];
            default:
                return md.dims[md.ndims - 2] > 1 ? strides[md.ndims - 2]
                                                 : md.dims[md.ndims - 1];
        }
    }
    // Leading dimension of A.
    dnnl_dim_t lda() const { return get_ld(a_md()); }
    // Leading dimension of B.
    dnnl_dim_t ldb() const { return get_ld(b_md()); }
    // Leading dimension of C.
    dnnl_dim_t ldc() const { return get_ld(c_md()); }
    /** Leading dimension of bias. */
    dnnl_dim_t ld_bias() const { return get_ld(bias_md()); }

    // Type of matrix A.
    dnnl_data_type_t a_type() const { return a_md().data_type; }
    // Type of matrix B.
    dnnl_data_type_t b_type() const { return b_md().data_type; }
    // Type of matrix C.
    dnnl_data_type_t c_type() const { return c_md().data_type; }
    // Type of bias.
    dnnl_data_type_t bias_type() const { return bias_md().data_type; }
    int bias_mask() const {
        const auto &b = bias_md();
        assert(b.ndims <= 6);
        int mask = 0;
        for (int i = 0; i < b.ndims; i++)
            mask |= (b.dims[i] > 1) ? 1 << i : 0;
        return mask;
    }

    // Relabels the last two axes (m<->n) without reordering memory:
    // dims/strides swap, inner block sizes are preserved.
    static void transpose_mn_axes(memory_desc_t &md) {
        if (md.ndims < 2) return;
        const int i = md.ndims - 2, j = md.ndims - 1;
        std::swap(md.dims[i], md.dims[j]);
        std::swap(md.padded_dims[i], md.padded_dims[j]);
        std::swap(md.padded_offsets[i], md.padded_offsets[j]);
        if (md.format_kind == format_kind::blocked) {
            auto &blk = md.format_desc.blocking;
            std::swap(blk.strides[i], blk.strides[j]);
            for (int k = 0; k < blk.inner_nblks; ++k) {
                if (blk.inner_idxs[k] == i)
                    blk.inner_idxs[k] = j;
                else if (blk.inner_idxs[k] == j)
                    blk.inner_idxs[k] = i;
            }
        }
    }

private:
    // Only gemm_pd_t may flip orientation: its wrapper also reseeds
    // pd-cached quant mds.
    friend struct gemm_pd_t;

    // format_any mds must be resolved BEFORE this call: layout
    // heuristics read trans flags off the user dims.
    void apply_swap_ab() {
        std::swap(a_desc_, b_desc_);
        transpose_mn_axes(a_desc_);
        transpose_mn_axes(b_desc_);
        transpose_mn_axes(c_desc_);
        transpose_mn_axes(bias_desc_);
        if (sum_ab_ == dnnl_sum_a_row)
            sum_ab_ = dnnl_sum_b_col;
        else if (sum_ab_ == dnnl_sum_b_col)
            sum_ab_ = dnnl_sum_a_row;
        swap_ab_ = !swap_ab_;
    }

    memory_desc_t a_desc_;
    memory_desc_t b_desc_;
    memory_desc_t c_desc_;
    memory_desc_t bias_desc_;

    bool swap_ab_ = false;

    sum_ab_t sum_ab_ {};
    dnnl_data_type_t sum_ab_type_ {};

    static transpose_t invert(transpose_t t) {
        return t == transpose::trans ? transpose::notrans : transpose::trans;
    }
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_GEMM_TYPES_HPP
