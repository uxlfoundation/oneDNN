/*******************************************************************************
* Copyright 2025 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul_inner_product.hpp"

#include "common/memory_desc.hpp"
#include "cpu/cpu_primitive.hpp"

#include <algorithm>
#include <vector>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
format_tag_t pick_plain_tag(int ndims) {
    using namespace format_tag;
    return utils::pick(ndims - 2, ab, acb, acdb, acdeb);
}

format_tag_t pick_transposed_tag(int ndims) {
    using namespace format_tag;
    return utils::pick(ndims - 2, ba, cba, cdba, cdeba);
}

int inner_block(const memory_desc_t &md, int dim) {
    int block = 1;
    const auto &blk = md.format_desc.blocking;
    for (int i = 0; i < blk.inner_nblks; ++i)
        if (blk.inner_idxs[i] == dim) block *= blk.inner_blks[i];
    return block;
}

status_t init_unflattened_ip_weights_md(memory_desc_t &ip_wei_md,
        const memory_desc_t &mm_wei_md, const memory_desc_t &hint_wei_md,
        const memory_desc_t &src_md) {
    if (hint_wei_md.ndims < 3) return status::unimplemented;
    if (src_md.ndims != hint_wei_md.ndims) return status::unimplemented;
    if (mm_wei_md.ndims != 2) return status::unimplemented;
    if (src_md.format_kind != format_kind::blocked)
        return status::unimplemented;

    ip_wei_md = hint_wei_md;
    ip_wei_md.format_kind = format_kind::blocked;
    ip_wei_md.offset0 = 0;
    ip_wei_md.extra = mm_wei_md.extra;
    ip_wei_md.format_desc.blocking = blocking_desc_t {};

    for (int d = 0; d < ip_wei_md.ndims; ++d) {
        ip_wei_md.padded_dims[d] = ip_wei_md.dims[d];
        ip_wei_md.padded_offsets[d] = 0;
    }

    std::vector<dim_t> k_dims;
    k_dims.reserve(hint_wei_md.ndims - 1);
    for (dim_t d = 1; d < hint_wei_md.ndims; ++d)
        k_dims.push_back(d);

    // Preserve the source's physical K order when expanding matmul's flattened
    // K axis back to IP weight dimensions.
    const auto &src_blk = src_md.format_desc.blocking;
    std::sort(k_dims.begin(), k_dims.end(), [&](dim_t lhs, dim_t rhs) {
        return src_blk.strides[lhs] < src_blk.strides[rhs];
    });

    const dim_t O_dim = 0;
    const dim_t inner_k_dim = k_dims[0];
    const int interleaved_by = inner_block(mm_wei_md, 1);
    const int block_by = inner_block(mm_wei_md, 0);

    auto &blk = ip_wei_md.format_desc.blocking;
    blk.strides[inner_k_dim] = interleaved_by * block_by;
    ip_wei_md.padded_dims[inner_k_dim]
            = utils::rnd_up(ip_wei_md.dims[inner_k_dim], block_by);

    dim_t outer_stride = interleaved_by * ip_wei_md.padded_dims[inner_k_dim];
    for (size_t i = 1; i < k_dims.size(); ++i) {
        const dim_t k_dim = k_dims[i];
        blk.strides[k_dim] = outer_stride;
        outer_stride *= ip_wei_md.padded_dims[k_dim];
    }

    blk.strides[O_dim] = outer_stride;
    ip_wei_md.padded_dims[O_dim]
            = utils::rnd_up(ip_wei_md.dims[O_dim], interleaved_by);

    // Keep the nested matmul inner-block sequence intact. Only remap matmul's
    // 2D K/N indices to the corresponding unflattened IP dimensions.
    const auto &mm_blk = mm_wei_md.format_desc.blocking;
    for (int i = 0; i < mm_blk.inner_nblks; ++i) {
        const dim_t mm_inner_idx = mm_blk.inner_idxs[i];
        if (!utils::one_of(mm_inner_idx, 0, 1)) return status::unimplemented;
        blk.inner_idxs[blk.inner_nblks]
                = mm_inner_idx == 0 ? inner_k_dim : O_dim;
        blk.inner_blks[blk.inner_nblks] = mm_blk.inner_blks[i];
        ++blk.inner_nblks;
    }

    return status::success;
}

status_t translate_matmul_weights_md(memory_desc_t &ip_wei_md,
        const memory_desc_t &mm_wei_md, const memory_desc_t &hint_wei_md,
        const memory_desc_t &src_md) {
    if (mm_wei_md.ndims != 2) return status::unimplemented;

    int transpose_perm[DNNL_MAX_NDIMS] {};
    for (int d = 0; d < mm_wei_md.ndims; ++d)
        transpose_perm[d] = d;
    transpose_perm[0] = 1;
    transpose_perm[1] = 0;

    memory_desc_t transposed_mm_wei_md {};
    CHECK(memory_desc_permute_axes(
            transposed_mm_wei_md, mm_wei_md, transpose_perm));

    if (hint_wei_md.ndims == transposed_mm_wei_md.ndims) {
        ip_wei_md = transposed_mm_wei_md;
        return status::success;
    }

    const status_t unflatten_status = init_unflattened_ip_weights_md(
            ip_wei_md, mm_wei_md, hint_wei_md, src_md);
    if (unflatten_status == status::success) return unflatten_status;

    const status_t status = memory_desc_reshape(ip_wei_md, transposed_mm_wei_md,
            hint_wei_md.ndims, hint_wei_md.dims);
    return status == status::success ? status::success : status::unimplemented;
}

} // namespace

status_t init_inner_product_matmul_md(memory_desc_t &mm_md,
        const memory_desc_t &ip_md, format_tag_t tag, bool swap_dims) {
    auto p_dims = ip_md.dims;
    auto p_dim1 = utils::array_product(p_dims + 1, ip_md.ndims - 1);

    if (swap_dims) {
        dims_t dims_2d = {p_dim1, p_dims[0]};
        return memory_desc_init_by_tag(mm_md, 2, dims_2d, ip_md.data_type, tag);
    }

    dims_t dims_2d = {p_dims[0], p_dim1};
    return memory_desc_init_by_tag(mm_md, 2, dims_2d, ip_md.data_type, tag);
}

status_t matmul_inner_product_fwd_t::pd_t::init(engine_t *engine) {
    VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_INNER_PRODUCT(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    CHECK(init_matmul_params(engine));
    init_scratchpad();

    return status::success;
}

status_t matmul_inner_product_fwd_t::pd_t::init_matmul_params(
        engine_t *engine) {
    using namespace format_tag;

    auto src_tag = pick_plain_tag(src_md()->ndims);
    const auto wei_plain_tag = pick_plain_tag(weights_md()->ndims);
    const auto wei_transposed_tag = pick_transposed_tag(weights_md()->ndims);

    auto mm_wei_tag = format_tag::undef;
    const bool wei_md_is_any = weights_md_.format_kind == format_kind::any;

    VDISPATCH_INNER_PRODUCT(
            weights_md()->ndims <= 4, VERBOSE_UNSUPPORTED_TAG_S, "weights");

    if (src_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md_, src_md_.ndims, src_md_.dims,
                src_md_.data_type, src_tag));
    } else if (src_md()->ndims == 4 && memory_desc_matches_tag(*src_md(), abcd)
            && wei_md_is_any) {
        src_tag = abcd;
    } else {
        VDISPATCH_INNER_PRODUCT(memory_desc_matches_tag(*src_md(), src_tag),
                VERBOSE_UNSUPPORTED_TAG_S, "src");
    }

    if (wei_md_is_any) {
        const bool exact_bf16
                = utils::everyone_is(data_type::bf16, src_md()->data_type,
                        weights_md()->data_type, dst_md()->data_type);
        mm_wei_tag = exact_bf16 && !with_bias() ? ba : any;
    } else if (memory_desc_matches_tag(*weights_md(), wei_plain_tag)) {
        mm_wei_tag = ba;
    } else if (memory_desc_matches_tag(*weights_md(), wei_transposed_tag)) {
        mm_wei_tag = ab;
    } else {
        VDISPATCH_INNER_PRODUCT(false, VERBOSE_UNSUPPORTED_TAG_S, "weights");
    }

    if (dst_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(
                dst_md_, dst_md_.ndims, dst_md_.dims, dst_md_.data_type, ab));
    } else {
        VDISPATCH_INNER_PRODUCT(memory_desc_matches_tag(*dst_md(), ab),
                VERBOSE_UNSUPPORTED_TAG_S, "dst");
    }

    if (with_bias()) {
        if (bias_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(bias_md_, x));
        } else {
            VDISPATCH_INNER_PRODUCT(memory_desc_matches_tag(*weights_md(1), x),
                    VERBOSE_UNSUPPORTED_TAG_S, "bias");
        }
    }

    VDISPATCH_INNER_PRODUCT_SC(
            attr_.set_default_formats(dst_md(0)), VERBOSE_UNSUPPORTED_POSTOP);

    memory_desc_t mm_src_md {};
    memory_desc_t mm_wei_md {};
    memory_desc_t mm_dst_md {};
    const memory_desc_t ip_wei_md_hint = weights_md_;

    CHECK(init_inner_product_matmul_md(mm_src_md, *src_md(), ab));
    CHECK(init_inner_product_matmul_md(
            mm_wei_md, *weights_md(), mm_wei_tag, true));
    CHECK(init_inner_product_matmul_md(mm_dst_md, *dst_md(), ab));

    primitive_attr_t matmul_attr = *attr();
    const bool spatial_f32_bf16_fpmath = src_md()->ndims > 2
            && utils::everyone_is(data_type::f32, src_md()->data_type,
                    weights_md()->data_type, dst_md()->data_type)
            && attr()->fpmath_.mode_ == fpmath_mode::bf16;
    if (spatial_f32_bf16_fpmath)
        CHECK(matmul_attr.set_fpmath_mode(fpmath_mode::strict, false));

    if (!matmul_attr.scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
        const auto wei_mask = matmul_attr.scales_.get_mask(DNNL_ARG_WEIGHTS);
        if (wei_mask == 1) {
            VDISPATCH_INNER_PRODUCT_SC(matmul_attr.scales_.set(DNNL_ARG_WEIGHTS,
                                               1 << (mm_wei_md.ndims - 1)),
                    VERBOSE_UNSUPPORTED_ATTR);
        } else if (wei_mask > 0) {
            VDISPATCH_INNER_PRODUCT(false, VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
    }

    memory_desc_t mm_bia_md {};
    if (with_bias()) {
        dims_t mm_bia_dims = {1, weights_md(1)->dims[0]};
        CHECK(memory_desc_init_by_tag(
                mm_bia_md, 2, mm_bia_dims, weights_md(1)->data_type, ab));
    }

    VDISPATCH_INNER_PRODUCT_SC(
            ::dnnl::impl::create_matmul_pd(matmul_pd_, engine, &mm_src_md,
                    &mm_wei_md, with_bias() ? &mm_bia_md : nullptr, &mm_dst_md,
                    &matmul_attr, nullptr, matmul_reduce_kind::src),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");

    if (wei_md_is_any) {
        CHECK(translate_matmul_weights_md(weights_md_,
                *matmul_pd_->weights_md(), ip_wei_md_hint, *src_md()));
    }

    return status::success;
}

status_t matmul_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args = ctx.args();
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_->execute(matmul_ctx);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
