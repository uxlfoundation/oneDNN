/*******************************************************************************
* Copyright 2021-2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/kai_matmul.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/gemm_based_common.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"

#include <functional>
#include <memory>
#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace data_type;
using namespace kai_utils;

#define VCHECK_MATMUL_EXEC(cond, msg, ...) \
    VCONDCHECK(primitive, exec, check, matmul, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

namespace {

bool batch_dims_have_default_order(const memory_desc_wrapper &mdw) {
    assert(mdw.is_blocking_desc());

    if (mdw.ndims() <= 2) return true;

    const auto &dims = mdw.dims();
    const auto ndims = mdw.ndims();
    const auto &strides = mdw.strides();

    dim_t expected_stride = dims[ndims - 1] * dims[ndims - 2];
    for (int i = ndims - 3; i >= 0; --i) {
        if (strides[i] != expected_stride) return false;
        expected_stride *= dims[i];
    }

    return true;
}

bool batch_dims_match(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {
    if (lhs.ndims() != rhs.ndims()) return false;

    for (int i = 0; i < lhs.ndims() - 2; ++i) {
        if (lhs.dims()[i] != rhs.dims()[i]) return false;
    }

    return true;
}

bool batch_dims_are_all_one(const memory_desc_wrapper &mdw) {
    for (int i = 0; i < mdw.ndims() - 2; ++i) {
        if (mdw.dims()[i] != 1) return false;
    }

    return true;
}

int get_innermost_batch_stride(const memory_desc_t *md) {
    return md->ndims > 2 ? md->format_desc.blocking.strides[md->ndims - 3] : 0;
}

double kernel_execute_work(const kai_matmul_t::pd_t &pd) {
    return static_cast<double>(pd.M()) * pd.N() * pd.K() * pd.batch();
}

} //namespace

std::unique_ptr<kai::ops::IGemmCommon>
kai_matmul_t::pd_t::create_kai_gemm_dequant(
        const kai::ops::DequantizeFloat &dequant) const {
    return kai_utils::create_kai_gemm_dequant(
            *args_, kai_src_dt_, kai_weights_dt_, kai_dst_dt_, dequant);
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_matmul_t::pd_t::create_kai_gemm() const {
    return kai_utils::create_kai_gemm(
            *args_, kai_src_dt_, kai_weights_dt_, kai_dst_dt_);
}

bool kai_matmul_t::pd_t::fixed_format() const {
    return args_ && args_->_fixed_format;
}

int kai_matmul_t::pd_t::kernel_maxthreads() const {
    return args_->_maxthreads;
}

status_t kai_matmul_t::pd_t::init(const engine_t *engine) {

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper wei_d(weights_md());
    const memory_desc_wrapper dst_d(dst_md());
    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);

    const bool weights_md_was_any
            = weights_md()->format_kind == format_kind::any;
    const bool weights_md_may_be_fixed_format
            = weights_md()->format_kind == format_kind::blocked
            && weights_md()->format_desc.blocking.inner_nblks > 0;

    auto sdt = src_md()->data_type;
    auto wdt = weights_md()->data_type;
    auto ddt = dst_md()->data_type;
    const bool fast_mode = use_fast_mode(*src_md(), *attr());

    kai_src_dt_ = sdt;
    kai_weights_dt_ = wdt;
    kai_dst_dt_ = ddt;

    if (types::is_integral_dt(sdt) && types::is_integral_dt(wdt)
            && !types::is_integral_dt(ddt)) {
        kai_gemm_type_ = kai_gemm_type::dequant;
    } else {
        kai_gemm_type_ = kai_gemm_type::noquant;
    }

    // Quant workflows do not yet support fixed format, in this case set_default_formats() will set
    // wtag, and we will do a reorder in execute
    bool try_fixed_format = false;
    if ((weights_md_was_any || weights_md_may_be_fixed_format)
            && kai_gemm_type_ == kai_gemm_type::noquant) {
        try_fixed_format = true;
    }

    // Note that this may change formats
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    if (with_bias() && bias_md_.format_kind == format_kind::any) {
        VDISPATCH_MATMUL_SC(memory_desc_init_by_strides(bias_md_, nullptr),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
    }

    if (try_fixed_format && fast_mode
            && utils::everyone_is(data_type::f32, sdt, ddt)
            && wdt == data_type::bf16) {
        kai_weights_dt_ = data_type::f32;
    }

    const memory_desc_wrapper bia_d(weights_md(1));
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_MATMUL(
            cpu::matmul::gemm_based::check_gemm_input_format(*src_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_MATMUL(helper.transA() == 'N', VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_MATMUL(
            cpu::matmul::gemm_based::check_gemm_output_format(*dst_md())
                    || cpu::matmul::gemm_based::check_gemm_input_format(
                            *dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_MATMUL(helper.transC() == 'N', VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_MATMUL(batch_dims_have_default_order(src_d),
            "src batch dimensions must be in order");
    VDISPATCH_MATMUL(batch_dims_have_default_order(wei_d),
            "weights batch dimensions must be in order");
    VDISPATCH_MATMUL(batch_dims_have_default_order(dst_d),
            "dst batch dimensions must be in order");
    VDISPATCH_MATMUL(
            IMPLICATION(with_bias(),
                    is_bias_1xN()
                            && cpu::matmul::gemm_based::
                                    check_gemm_output_format(*weights_md(1))
                            && bia_d.data_type() == ddt),
            VERBOSE_UNSUPPORTED_BIAS_CFG);

    const bool is_gemv = utils::everyone_is(2, src_md()->ndims, dst_md()->ndims)
            && (src_md()->dims[0] == 1 || weights_md()->dims[1] == 1);

    // If this is a GEMV and we have SVE, we use brgemm_matmul because it has an optimized path
    VDISPATCH_MATMUL(
            !(is_gemv && mayiuse(sve_128)), "falling brgemm_matmul for GEMV");

    if (kai_gemm_type_ == kai_gemm_type::noquant) {
        using smask_t = primitive_attr_t::skip_mask_t;
        VDISPATCH_MATMUL(
                attr()->has_default_values(smask_t::fpmath_mode
                        | smask_t::accumulation_mode | smask_t::post_ops),
                VERBOSE_UNSUPPORTED_ATTR);
    } else {
        using smask_t = primitive_attr_t::skip_mask_t;
        // TODO: implement zero_points
        VDISPATCH_MATMUL(utils::one_of(ddt, f32), VERBOSE_UNSUPPORTED_DT_CFG);
        VDISPATCH_MATMUL(
                attr()->has_default_values(smask_t::scales | smask_t::post_ops,
                        dst_md()->data_type),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_MATMUL(attr()->scales_.has_default_values(DNNL_ARG_SRC)
                        || attr()->scales_.get_mask(DNNL_ARG_SRC) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
                        || attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(attr()->scales_.has_default_values(DNNL_ARG_DST),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    batch_mode_ = batch_mode::none;
    unsigned int nbatches = 1;
    unsigned int nmulti = 1;
    broadcast_src_batch_ = false;

    if (helper.batched()) {
        const bool src_matches_dst = batch_dims_match(src_d, dst_d);
        const bool wei_matches_dst = batch_dims_match(wei_d, dst_d);
        const bool src_all_one = batch_dims_are_all_one(src_d);
        const bool wei_all_one = batch_dims_are_all_one(wei_d);

        const bool can_use_batches = wei_all_one && src_matches_dst;
        const bool can_use_multis
                = wei_matches_dst && (src_matches_dst || src_all_one);

        if (can_use_batches) {
            batch_mode_ = batch_mode::batches;
            nbatches = static_cast<unsigned int>(helper.batch());
        } else if (can_use_multis) {
            batch_mode_ = batch_mode::multis;
            nmulti = static_cast<unsigned int>(helper.batch());
            broadcast_src_batch_ = src_all_one;
        } else {
            VDISPATCH_MATMUL(false,
                    "only supports batch dims that are fully shared or "
                    "fully varying");
        }
    }

    auto scratchpad = scratchpad_registry().registrar();

    unsigned int sections = 1;
    bool indirect = false;

    if (types::is_integral_dt(ddt)) {
        VDISPATCH_MATMUL(attr_.post_ops_.len() == 0,
                "no post op support for integral dt");
    }
    VDISPATCH_MATMUL(num_sum_post_ops(attr_.post_ops_) <= 1,
            "supports at most one sum post op");
    const post_ops_fusion_t post_ops_fusion
            = create_post_ops_fusion(attr_.post_ops_, !with_bias());
    VDISPATCH_MATMUL(
            !post_ops_fusion.has_fallback(attr_.post_ops_) || ddt == f32,
            "post ops must be fused into KAI unless dst is f32");
    CHECK(post_ops_fallback_.init(engine, attr_.post_ops_, *dst_md(),
            post_ops_fusion.fallback_start_index));

    const int max_threads = dnnl_get_current_num_threads();
    const int num_threads = threads_for_kernel_execute(
            kernel_execute_work(*this), max_threads);

    args_ = std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), M(), N(), K(),
            sections, nbatches, nmulti, indirect, post_ops_fusion.activation,
            num_threads, try_fixed_format, fast_mode,
            post_ops_fusion.accumulate);

    std::unique_ptr<kai::ops::IGemmCommon> kernel = nullptr;

    const auto make_kernel = [&]() -> std::unique_ptr<kai::ops::IGemmCommon> {
        // Create an kai object, this is where we enforce the datatype combination
        if (is_dequant()) {
            // Non-trivial placeholder value, because the value is only provided at runtime
            kai::ops::DequantizeFloat dequant(0.5);
            return create_kai_gemm_dequant(dequant);
        }
        return create_kai_gemm();
    };

    kernel = make_kernel();
    // If fixed format construction failed, try a non-fixed format kernel
    if (!kernel && try_fixed_format && weights_md_was_any) {
        args_->_fixed_format = false;
        kernel = make_kernel();
    }
    VDISPATCH_MATMUL(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    // Now that we have decided between fixed format and plain weights, check the format
    VDISPATCH_MATMUL(fixed_format()
                    || cpu::matmul::gemm_based::check_gemm_input_format(
                            *weights_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    // We need information from the concrete kernel config, but we do not overwrite cfg
    // because it can cause a different kernel to be constructed in execute. This should be
    // fixed in KleidiAI
    kai::ops::GemmConfig kernel_cfg = kernel->get_config();

    kai_pack_weights_ = !fixed_format() && kernel->B_is_pretransposed();
    VDISPATCH_MATMUL(fixed_format() || helper.transB() == 'N',
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    if (fixed_format()) {
        // Logical dimension indices
        dim_t innermost_dim = weights_md_.ndims - 1;
        dim_t N_dim = innermost_dim;
        dim_t K_dim = innermost_dim - 1;

        // The logical indices of dimensions related to the batch, ordered from
        // innermost to outermost
        std::vector<dim_t> batch_dims = {};
        for (dim_t i = K_dim - 1; i >= 0; --i)
            batch_dims.push_back(i);

        if (weights_md_was_any) {
            weight_format_to_memory_desc(weights_md_, kernel_cfg.weight_format,
                    K_dim, N_dim, {}, batch_dims);
        } else {
            VDISPATCH_MATMUL(memory_desc_matches_weight_format(weights_md_,
                                     kernel_cfg.weight_format, K_dim, N_dim, {},
                                     batch_dims),
                    VERBOSE_UNSUPPORTED_TAG_S, "weights");
        }
    }

    if (kernel->get_working_size() != 0)
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                kernel->get_working_size(), 1);

    // KleidiAI names its B packing transform "pretranspose". Keep the oneDNN
    // flag named as packing so it is distinct from layout reorders such as
    // fixed format.
    if (kai_pack_weights_)
        scratchpad.book(memory_tracking::names::key_matmul_wei_trans,
                kernel->get_B_pretransposed_array_size(), 1);

    if (post_ops_fallback_.has_sum()) {
        const memory_desc_wrapper tmp_dst_d(dst_md());
        scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                tmp_dst_d.size(), 1, 64, 64);
    }
    post_ops_fallback_.init_scratchpad(scratchpad);

    return status::success;
}

status_t kai_matmul_t::init(engine_t *engine) {
    post_ops_fallback_ = pd()->post_ops_fallback();
    CHECK(post_ops_fallback_.init_primitives(engine));
    return status::success;
}

status_t kai_matmul_t::create_resource(
        engine_t *engine, resource_mapper_t &mapper) const {
    UNUSED(engine);
    UNUSED(mapper);
    return status::success;
}

status_t kai_matmul_t::execute(const exec_ctx_t &ctx) const {

    // Construct kernel identically to how we did in pd_t::init so that
    // workspace and format is identical
    std::unique_ptr<kai::ops::IGemmCommon> kernel = nullptr;
    if (pd()->is_dequant()) {
        DEFINE_ARG_SCALES_BUFFER(src_scale, DNNL_ARG_SRC);
        DEFINE_ARG_SCALES_BUFFER(wei_scale, DNNL_ARG_WEIGHTS);
        kai::ops::DequantizeFloat dequant(src_scale[0] * wei_scale[0]);
        kernel = pd()->create_kai_gemm_dequant(dequant);
    } else {
        kernel = pd()->create_kai_gemm();
    }
    if (!kernel) return status::runtime_error;

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    // We cannot increase the number of threads, otherwise we could overflow the
    // scratchpad which was fixed at pd_t::init
    const int num_threads = std::min(
            pd()->kernel_maxthreads(), dnnl_get_current_num_threads());
    const kai::ops::ndrange_t window_size = kernel->get_window_size();
    const auto thread_partition
            = make_thread_partition(num_threads, window_size);

    kernel->set_nthreads(thread_partition.nthr);

    auto src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    VCHECK_MATMUL_EXEC(src_base != nullptr, "%s (src)", VERBOSE_NULL_ARG);

    auto raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    VCHECK_MATMUL_EXEC(raw_wei != nullptr, "%s (weights)", VERBOSE_NULL_ARG);
    void *wei_base = const_cast<void *>(raw_wei);

    const auto dst_arg = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    VCHECK_MATMUL_EXEC(dst_arg != nullptr, "%s (dst)", VERBOSE_NULL_ARG);
    auto dst_base = dst_arg;

    if (post_ops_fallback_.has_sum()) {
        dst_base = scratchpad.get<void>(
                memory_tracking::names::key_matmul_dst_in_acc_dt);
    }

    const void *bias_base = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;
    VCHECK_MATMUL_EXEC(!pd()->with_bias() || bias_base != nullptr, "%s (bias)",
            VERBOSE_NULL_ARG);

    if (pd()->kai_pack_weights()) {
        wei_base = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
    }

    const void *kernel_src_base = src_base;
    const void *kernel_raw_wei = raw_wei;
    void *kernel_wei_base = wei_base;

    const memory_desc_t *kernel_src_md = pd()->src_md();
    const memory_desc_t *kernel_wei_md = pd()->weights_md();
    const memory_desc_t *kernel_dst_md = pd()->dst_md();

    const memory_desc_wrapper src_d(kernel_src_md);
    const memory_desc_wrapper wei_d(kernel_wei_md);
    const memory_desc_wrapper dst_d(kernel_dst_md);
    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);

    // Leading dimensions of our matrices are the strides of the first non-dense dimensions (second logical)
    auto ld_src = kernel_src_md->format_desc.blocking
                          .strides[kernel_src_md->ndims - 2];
    auto ld_dst = kernel_dst_md->format_desc.blocking
                          .strides[kernel_dst_md->ndims - 2];

    // With fixed format, weights are already packed in kai's expected
    // layout, so the row stride follows the innermost logical dimension.
    // Otherwise rely on matmul_helper_t for the kernel-facing descriptor.
    auto ld_wei = pd()->fixed_format()
            ? kernel_wei_md->format_desc.blocking
                      .strides[kernel_wei_md->ndims - 1]
            : helper.ldb();

    const int src_batch_stride = get_innermost_batch_stride(kernel_src_md);
    const int wei_batch_stride = get_innermost_batch_stride(kernel_wei_md);
    const int dst_batch_stride = get_innermost_batch_stride(kernel_dst_md);

    const int batch_stride_a = pd()->is_batches() ? src_batch_stride : 0;
    const int multi_stride_a = pd()->is_multis()
            ? (pd()->broadcast_src_batch() ? 0 : src_batch_stride)
            : 0;
    const int multi_stride_b = pd()->is_multis() ? wei_batch_stride : 0;
    const int batch_stride_c = pd()->is_batches() ? dst_batch_stride : 0;
    const int multi_stride_c = pd()->is_multis() ? dst_batch_stride : 0;

    if (pd()->kai_pack_weights()) {
        parallel_pretranspose_B_array(*kernel, kernel_wei_base, kernel_raw_wei,
                ld_wei, multi_stride_b, false, thread_partition.team_nthr);
    }

    if (kernel->get_working_size() != 0) {
        kernel->set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }

    kernel->set_arrays_generic(kernel_src_base, ld_src, batch_stride_a,
            multi_stride_a, kernel_wei_base, ld_wei, multi_stride_b, dst_base,
            ld_dst, batch_stride_c, multi_stride_c, bias_base, 0);

    parallel_execute(*kernel, window_size, thread_partition);

    if (post_ops_fallback_.len() > 0) {
        if (post_ops_fallback_.has_sum()) {
            CHECK(post_ops_fallback_.execute(ctx, dst_base, dst_arg));
        } else {
            CHECK(post_ops_fallback_.execute(ctx, dst_base));
        }
    }

    return status::success;
}

#undef VCHECK_MATMUL_EXEC

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
