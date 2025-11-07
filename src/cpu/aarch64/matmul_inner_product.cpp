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

#include "cpu/cpu_primitive.hpp"

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

} // namespace

status_t create_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *dst_md,
        const memory_desc_t *bia_md, const primitive_attr_t *attr) {
    auto matmul_desc = matmul_desc_t();

    CHECK(matmul_desc_init(&matmul_desc, src_md, wei_md, bia_md, dst_md,
            nullptr, matmul_reduce_kind::src));

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&matmul_desc, attr, nullptr);

    while (it != it.end()) {
        matmul_pd = *(++it);
        if (!matmul_pd) return status::unimplemented;
        break;
    }

    return status::success;
}

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

    const auto src_tag = pick_plain_tag(src_md()->ndims);
    const auto wei_plain_tag = pick_plain_tag(weights_md()->ndims);
    const auto wei_transposed_tag = pick_transposed_tag(weights_md()->ndims);

    auto mm_wei_tag = format_tag::undef;

    if (src_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md_, src_md_.ndims, src_md_.dims,
                src_md_.data_type, src_tag));
    } else {
        VDISPATCH_INNER_PRODUCT(memory_desc_matches_tag(*src_md(), src_tag),
                VERBOSE_UNSUPPORTED_TAG_S, "src");
    }

    if (weights_md_.format_kind == format_kind::any) {
        // Request the IP weights in transposed form so the nested matmul sees
        // a regular KxN matrix and can stay on kai's non-transposed B
        // path.
        CHECK(memory_desc_init_by_tag(weights_md_, weights_md_.ndims,
                weights_md_.dims, weights_md_.data_type, wei_transposed_tag));
        mm_wei_tag = ab;
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

    CHECK(init_inner_product_matmul_md(mm_src_md, *src_md(), ab));
    CHECK(init_inner_product_matmul_md(
            mm_wei_md, *weights_md(), mm_wei_tag, true));
    CHECK(init_inner_product_matmul_md(mm_dst_md, *dst_md(), ab));

    primitive_attr_t matmul_attr = *attr();
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
            create_matmul_pd(matmul_pd_, engine, &mm_src_md, &mm_wei_md,
                    &mm_dst_md, with_bias() ? &mm_bia_md : nullptr,
                    &matmul_attr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");

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
