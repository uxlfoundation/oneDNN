/*******************************************************************************
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

#include "cpu/aarch64/kai_convolution_base.hpp"

#include <algorithm>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace data_type;
using namespace kai_utils;

namespace {

bool bias_ok(const kai_convolution_fwd_base_t::pd_t &pd) {
    return !pd.with_bias()
            || pd.invariant_bia_md()->data_type == pd.dst_md()->data_type
            || pd.invariant_bia_md()->data_type == pd.gemm_dst_dt();
}

bool dense_layout(const memory_desc_t &md, format_tag_t tag) {
    const memory_desc_wrapper mdw(md);
    return mdw.is_plain() && mdw.is_dense() && mdw.offset0() == 0
            && mdw.matches_tag(tag);
}

bool datatypes_ok(const cpu_convolution_fwd_pd_t &pd) {
    const auto src_dt = pd.invariant_src_md()->data_type;
    const auto wei_dt = pd.invariant_wei_md()->data_type;
    const auto dst_dt = pd.invariant_dst_md()->data_type;

    return (src_dt == f32 && wei_dt == f32 && dst_dt == f32)
            || (src_dt == bf16 && wei_dt == bf16
                    && utils::one_of(dst_dt, bf16, f32))
            || (src_dt == f16 && wei_dt == f16
                    && utils::one_of(dst_dt, f16, f32));
}

double weight_reorder_work(const kai_convolution_fwd_base_t::pd_t &pd) {
    return static_cast<double>(pd.OC()) * pd.IC() * pd.KH() * pd.KW();
}

double kernel_execute_work(const kai_convolution_fwd_base_t::pd_t &pd) {
    return weight_reorder_work(pd) * pd.MB() * pd.OH() * pd.OW();
}

bool accumulation_mode_ok(const cpu_convolution_fwd_pd_t &pd) {
    const auto acc_mode = pd.attr()->acc_mode_;
    if (utils::one_of(acc_mode, accumulation_mode::strict,
                accumulation_mode::relaxed, accumulation_mode::any,
                accumulation_mode::f32))
        return true;

    if (acc_mode == accumulation_mode::f16) {
        return utils::everyone_is(data_type::f16,
                pd.invariant_src_md()->data_type,
                pd.invariant_wei_md()->data_type,
                pd.invariant_dst_md()->data_type);
    }

    return false;
}

} // namespace

std::unique_ptr<kai::ops::IGemmCommon>
kai_convolution_fwd_base_t::pd_t::create_kai_gemm() const {
    // Supported source and weight types match.
    return kai_utils::create_kai_gemm(
            *args_, gemm_src_dt_, gemm_src_dt_, gemm_dst_dt());
}

data_type_t kai_convolution_fwd_base_t::pd_t::gemm_dst_dt() const {
    return with_bias() && invariant_bia_md()->data_type == data_type::f32
                    && utils::one_of(dst_md()->data_type, data_type::f16,
                            data_type::bf16)
            ? data_type::f32
            : dst_md()->data_type;
}

bool kai_convolution_fwd_base_t::pd_t::is_1x1(
        const cpu_convolution_fwd_pd_t &pd) {
    return pd.KH() == 1 && pd.KW() == 1;
}

bool kai_convolution_fwd_base_t::pd_t::has_zero_top_left_padding(
        const cpu_convolution_fwd_pd_t &pd) {
    return pd.padT() == 0 && pd.padL() == 0;
}

bool kai_convolution_fwd_base_t::pd_t::source_pixels_in_bounds(
        const cpu_convolution_fwd_pd_t &pd) {
    return pd.OH() > 0 && pd.OW() > 0 && (pd.OH() - 1) * pd.KSH() < pd.IH()
            && (pd.OW() - 1) * pd.KSW() < pd.IW();
}

status_t kai_convolution_fwd_base_t::pd_t::init_common(const engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_CONV(!with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");
    VDISPATCH_CONV(datatypes_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    const auto supported_attrs = primitive_mask_t::fpmath_mode
            | primitive_mask_t::accumulation_mode | primitive_mask_t::post_ops;
    VDISPATCH_CONV(
            attr()->has_default_values(supported_attrs, dst_md()->data_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(accumulation_mode_ok(*this),
            "accumulation mode is not valid for the data type combination");
    // Leave weights `any` until kernel selection
    // determines their final fixed format or plain HWIO layout.
    if (src_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md_, format_tag::nhwc));
    if (dst_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md_, format_tag::nhwc));
    if (with_bias() && bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, format_tag::x));
    VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(dense_layout(src_md_, format_tag::nhwc),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dense_layout(dst_md_, format_tag::nhwc),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    if (weights_md_.format_kind != format_kind::any) {
        VDISPATCH_CONV(dense_layout(weights_md_, format_tag::hwio),
                VERBOSE_UNSUPPORTED_TAG_S, "weights");
    }
    VDISPATCH_CONV(
            IMPLICATION(with_bias(), dense_layout(bias_md_, format_tag::x)),
            VERBOSE_UNSUPPORTED_TAG_S, "bias");

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);
    gemm_src_dt_ = src_md()->data_type;
    return status::success;
}

void kai_convolution_fwd_base_t::pd_t::init_gemm_args(unsigned int m,
        unsigned int k, unsigned int k_sections, unsigned int batches,
        unsigned int multi, bool indirect) {
    const int num_threads = threads_for_kernel_execute(
            kernel_execute_work(*this), dnnl_get_current_num_threads());
    args_ = std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), m, OC(), k,
            k_sections, batches, multi, indirect, kai::ops::Activation(),
            num_threads, weights_md_.format_kind == format_kind::any,
            use_fast_mode(*src_md(), *attr()));
}

status_t kai_convolution_fwd_base_t::pd_t::init_gemm(const engine_t *engine) {
    VDISPATCH_CONV(num_sum_post_ops(attr_.post_ops_) <= 1,
            "supports at most one sum post op");
    const auto post_ops_fusion
            = create_post_ops_fusion(attr_.post_ops_, !with_bias());
    CHECK(post_ops.init(engine, attr_.post_ops_, *dst_md(),
            post_ops_fusion.fallback_start_index));
    if (post_ops.len() > 0) name_ += "+post_ops_fallback";
    args_->_act = post_ops_fusion.activation;
    args_->_accumulate = post_ops_fusion.accumulate;

    std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
    const bool fixed_format_failed = args_->_fixed_format
            && (!kernel
                    || !is_fixed_format(kernel->get_config().weight_format));

    if (fixed_format_failed) {
        args_->_fixed_format = false;
        kernel = create_kai_gemm();
    }
    VDISPATCH_CONV(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    kai::ops::GemmConfig kernel_cfg = kernel->get_config();

    if (args_->_fixed_format) {
        constexpr dim_t O_dim = 0;
        constexpr dim_t I_dim = 1;
        constexpr dim_t H_dim = 2;
        constexpr dim_t W_dim = 3;
        weights_md_.data_type = gemm_src_dt_;
        weight_format_to_memory_desc(weights_md_, kernel_cfg.weight_format,
                I_dim, O_dim, {W_dim, H_dim});
    } else if (weights_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md_, format_tag::hwio));
    }

    auto scratchpad = scratchpad_registry().registrar();
    if (kernel->get_working_size() != 0) {
        // Match ACL's GEMM workspace alignment. Cache-line alignment alone
        // can regress interleaved convolutions on large packed-input buffers.
        // KleidiAI reports the workspace size in bytes.
        scratchpad.book<char>(memory_tracking::names::key_gemm_asm_tmp_buffer,
                kernel->get_working_size(), PAGE_4K);
    }

    if (!args_->_fixed_format && kernel->B_is_pretransposed()) {
        scratchpad.book<char>(memory_tracking::names::key_conv_permuted_weights,
                kernel->get_B_pretransposed_array_size());
    }

    if (gemm_dst_dt() != dst_md()->data_type) {
        memory_desc_t tmp_dst_md;
        CHECK(memory_desc_init_by_tag(tmp_dst_md, dst_md()->ndims,
                dst_md()->dims, gemm_dst_dt(), format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(dst_reorder_pd_, engine,
                                  &tmp_dst_md, dst_md()),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "dst reorder");

        const memory_desc_wrapper tmp_dst_d(&tmp_dst_md);
        scratchpad.book<char>(
                memory_tracking::names::key_conv_ncsp_dst, tmp_dst_d.size());
        scratchpad.book(memory_tracking::names::key_nested,
                dst_reorder_pd_->scratchpad_registry());
    }

    if (post_ops.has_sum()) {
        const memory_desc_wrapper dst_d(dst_md());
        scratchpad.book<char>(
                memory_tracking::names::key_generic_acc, dst_d.size());
    }

    post_ops.init_scratchpad(scratchpad);

    return status::success;
}

status_t kai_convolution_fwd_base_t::init_common(engine_t *engine) {
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_, engine));
    post_ops_ = pd()->post_ops;
    CHECK(post_ops_.init_primitives(engine));
    return status::success;
}

status_t kai_convolution_fwd_base_t::execute_reorder(const exec_ctx_t &ctx,
        const primitive_t &reorder, const memory_desc_t *src_md,
        const memory_desc_t *dst_md, const void *src, void *dst,
        memory_tracking::key_t nested_key) {
    auto *engine = ctx.stream()->engine();
    std::unique_ptr<memory_t, memory_deleter_t> src_mem(new memory_t(
            engine, src_md, use_runtime_ptr, const_cast<void *>(src)));
    std::unique_ptr<memory_t, memory_deleter_t> dst_mem(
            new memory_t(engine, dst_md, use_runtime_ptr, dst));
    exec_args_t reorder_args;
    reorder_args[DNNL_ARG_SRC] = {src_mem.get(), true};
    reorder_args[DNNL_ARG_DST] = {dst_mem.get(), false};
    exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));
    auto *nested_grantor = memory_tracking::create_nested_grantor(
            ctx.get_scratchpad_grantor(), nested_key,
            reorder.pd()->scratchpad_registry());
    reorder_ctx.set_scratchpad_grantor(nested_grantor);
    return reorder.execute(reorder_ctx);
}

status_t kai_convolution_fwd_base_t::prepare(kernel_call_args_t &args) const {
    const auto &ctx = args.ctx;
    const auto &pd = args.pd;
    auto &kernel = args.kernel;
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const int requested_threads = std::min(
            pd.gemm_args()._maxthreads, dnnl_get_current_num_threads());
    args.thread_partition = make_thread_partition(
            requested_threads, kernel.get_window_size());
    kernel.set_nthreads(args.thread_partition.nthr);

    args.src_base = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    args.wei_base = const_cast<void *>(raw_wei);
    if (!pd.gemm_args()._fixed_format && kernel.B_is_pretransposed()) {
        args.wei_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
        parallel_pretranspose_B_array(kernel, args.wei_base, raw_wei,
                args.ldb(), 0, false, args.thread_partition.nthr);
    }

    args.bias_base = pd.with_bias() ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
                                    : nullptr;
    if (dst_reorder_) {
        args.kernel_dst_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_ncsp_dst);
    } else if (post_ops_.has_sum()) {
        args.kernel_dst_base
                = scratchpad.get<void>(memory_tracking::names::key_generic_acc);
    } else {
        args.kernel_dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    }

    if (kernel.get_working_size() != 0) {
        kernel.set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }
    return status::success;
}

status_t kai_convolution_fwd_base_t::execute_kernel(
        const kernel_call_args_t &args) const {
    parallel_execute(
            args.kernel, args.kernel.get_window_size(), args.thread_partition);
    const auto &ctx = args.ctx;
    const auto &pd = args.pd;
    auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    void *post_ops_src = args.kernel_dst_base;
    if (dst_reorder_) {
        post_ops_src = post_ops_.has_sum()
                ? ctx.get_scratchpad_grantor().get<void>(
                          memory_tracking::names::key_generic_acc)
                : dst;
        CHECK(execute_reorder(ctx, *dst_reorder_, pd.dst_reorder_pd_->src_md(),
                pd.dst_md(), args.kernel_dst_base, post_ops_src,
                memory_tracking::names::key_nested));
    }
    if (post_ops_.len() > 0)
        CHECK(post_ops_.execute(
                ctx, post_ops_src, post_ops_.has_sum() ? dst : nullptr));
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
