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

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace data_type;
using namespace kai_utils;

namespace {
inline bool is_fixed_format_fast_math(const kai::ops::WeightFormat &wf) {
    return (static_cast<int>(wf) >> 4) & 0x1;
}

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

} //namespace

// Helper to make if statements clearer
bool kai_matmul_t::pd_t::swd_dt(
        data_type_t s, data_type_t w, data_type_t d) const {
    auto sdt = src_md()->data_type;
    auto wdt = weights_md()->data_type;
    auto ddt = dst_md()->data_type;

    return (sdt == s && wdt == w && ddt == d);
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_matmul_t::pd_t::create_kai_gemm_dequant(
        const kai::ops::DequantizeFloat &dequant) const {
    return kai_utils::create_kai_gemm_dequant(*_args, _cfg.get(),
            src_md()->data_type, weights_md()->data_type, dst_md()->data_type,
            dequant);
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_matmul_t::pd_t::create_kai_gemm() const {
    return kai_utils::create_kai_gemm(*_args, _cfg.get(), src_md()->data_type,
            weights_md()->data_type, dst_md()->data_type);
}

status_t kai_matmul_t::pd_t::init(engine_t *engine) {

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper wei_d(weights_md());
    const memory_desc_wrapper dst_d(dst_md());
    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);

    auto sdt = src_md()->data_type;
    auto wdt = weights_md()->data_type;
    auto ddt = dst_md()->data_type;

    _cfg = std::make_shared<kai::ops::GemmConfig>();

    if (types::is_integral_dt(sdt) && types::is_integral_dt(wdt)
            && !types::is_integral_dt(ddt)) {
        kai_gemm_type_ = kai_gemm_type::dequant;
    } else {
        kai_gemm_type_ = kai_gemm_type::noquant;
    }

    // TODO: handle the case where the specific blocked format is passed in
    // Note that this may enable reusing prefill reordered weights for decode
    // TODO: handle ba, ab ect
    _fixed_format = false;
    // Quant workflows do not yet support fixed format, in this case set_default_formats() will set
    // wtag, and we will do a reorder in execute
    const bool bf16_fast_math = use_bf16_fast_math(attr(), sdt, wdt, ddt);

    if (wei_d.format_kind() == format_kind::any && !bf16_fast_math
            && kai_gemm_type_ == kai_gemm_type::noquant) {
        _cfg->weight_format = kai::ops::WeightFormat::ANY;
        _fixed_format = true;
    }

    // Note that this may change formats
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    if (with_bias() && bias_md_.format_kind == format_kind::any) {
        VDISPATCH_MATMUL_SC(memory_desc_init_by_strides(bias_md_, nullptr),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
    }
    const memory_desc_wrapper bia_d(weights_md(1));
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_MATMUL(
            cpu::matmul::gemm_based::check_gemm_input_format(*src_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_MATMUL(
            cpu::matmul::gemm_based::check_gemm_output_format(*dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_MATMUL(_fixed_format
                    || cpu::matmul::gemm_based::check_gemm_input_format(
                            *weights_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");
    VDISPATCH_MATMUL(
            helper.transA() == 'N', "kai only supports non-transposed src");
    VDISPATCH_MATMUL(helper.transC() == 'N', "kai only supports row-major dst");
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
    _ag_nbatches = 1;
    _ag_nmulti = 1;
    _src_broadcast_batch_dims = false;

    if (helper.batched()) {
        const bool src_matches_dst = batch_dims_match(src_d, dst_d);
        const bool wei_matches_dst = batch_dims_match(wei_d, dst_d);
        const bool src_all_one = batch_dims_are_all_one(src_d);
        const bool wei_all_one = batch_dims_are_all_one(wei_d);

        const bool can_use_batches = wei_all_one && src_matches_dst;
        const bool can_use_multis
                = wei_matches_dst && (src_matches_dst || src_all_one);

        VDISPATCH_MATMUL(can_use_batches || can_use_multis,
                "kai only supports batch dims that are fully shared or "
                "fully varying");

        if (can_use_batches) {
            batch_mode_ = batch_mode::batches;
            _ag_nbatches = static_cast<unsigned int>(helper.batch());
        } else {
            batch_mode_ = batch_mode::multis;
            _ag_nmulti = static_cast<unsigned int>(helper.batch());
            _src_broadcast_batch_dims = src_all_one;
        }
    }

    auto scratchpad = scratchpad_registry().registrar();

    unsigned int sections = 1;
    bool indirect = false;

    const bool fast_mode = bf16_fast_math
            || (src_md()->data_type == data_type::f16
                    && utils::one_of(attr()->acc_mode_,
                            accumulation_mode::relaxed,
                            accumulation_mode::f16));

    int nfused_post_ops = 0;

    bool accumulate = false;
    if (types::is_integral_dt(ddt)){
        VDISPATCH_MATMUL(attr_.post_ops_.len() == 0, "no post op support for integral dt");
    }
    if (attr_.post_ops_.contain(primitive_kind::sum, 0)) {
        VDISPATCH_MATMUL(attr_.post_ops_.entry_[0].sum.scale == 1.0f,
                "sum post op scale must be 1 (no scale)");
        VDISPATCH_MATMUL(attr_.post_ops_.entry_[0].sum.zero_point == 0,
                "sum post op zero point must be 0 (no shift)");
        accumulate = true;
        nfused_post_ops = 1;
    }

    kai::ops::Activation act = kai::ops::Activation::Type::None;
    if (attr_.post_ops_.len() > nfused_post_ops
            && attr_.post_ops_.entry_[nfused_post_ops].kind
                    == primitive_kind::eltwise) {
        const auto &eltwise_po
                = attr_.post_ops_.entry_[nfused_post_ops].eltwise;
        if (eltwise_po.alg == alg_kind::eltwise_relu
                && eltwise_po.alpha == 0.0f) {
            act = kai::ops::Activation(kai::ops::Activation::Type::ReLU);
            nfused_post_ops += 1;
        } else if (utils::one_of(eltwise_po.alg, alg_kind::eltwise_clip,
                           alg_kind::eltwise_clip_v2)
                && eltwise_po.alpha == 0.0f && eltwise_po.beta > 0.0f) {
            // kai::ops::BoundedReLU is between 0 and beta
            act = kai::ops::Activation(kai::ops::Activation::Type::BoundedReLU,
                    eltwise_po.beta, 0.0f);
            nfused_post_ops += 1;
        } else {
            // TODO: implement post-op fallback
            VDISPATCH_MATMUL(false, "unsupported post-op");
        }
    }
    VDISPATCH_MATMUL(attr_.post_ops_.len() == nfused_post_ops,
            "unsupported post-op");

    unsigned int num_threads = dnnl_get_current_num_threads();
    _args = std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), M(), N(), K(),
            sections, _ag_nbatches, _ag_nmulti, indirect, act, num_threads,
            _fixed_format, fast_mode, accumulate, _cfg.get());

    std::unique_ptr<kai::ops::IGemmCommon> kernel = nullptr;

    // Create an kai object, this is where we enforce the datatype combination
    if (is_dequant()) {
        // Non-trivial placeholder value, because the value is only provided at runtime
        kai::ops::DequantizeFloat dequant(0.5);
        kernel = create_kai_gemm_dequant(dequant);
    } else {
        kernel = create_kai_gemm();
    }
    VDISPATCH_MATMUL(kernel, VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(_fixed_format || helper.transB() == 'N'
                    || (kernel->B_is_pretransposed()
                            && kernel->B_pretranspose_supports_transpose()),
            "kai only supports transposed weights when pretransposition "
            "is available");

    // Copy the resulting config object constructed from kernel
    _cfg = std::make_shared<kai::ops::GemmConfig>(kernel->get_config());

    if (_fixed_format) {
        // Logical dimension indices
        dim_t innermost_dim = weights_md_.ndims - 1;
        dim_t N_dim = innermost_dim;
        dim_t K_dim = innermost_dim - 1;

        // The logical indices of dimensions related to the batch, ordered from
        // innermost to outermost
        std::vector<dim_t> batch_dims = {};
        for (dim_t i = K_dim - 1; i >= 0; --i)
            batch_dims.push_back(i);

        weight_format_to_memory_desc(
                weights_md_, _cfg->weight_format, K_dim, N_dim, {}, batch_dims);
    }

    if (kernel->get_working_size() != 0)
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                kernel->get_working_size(), 1);

    if (!_fixed_format) {
        // Pretranspose any kernel that expects B in kai's packed layout.
        // Whether the pretranspose routine also accepts a transposed source is
        // a separate capability from whether pretranspose is required at all.
        _run_weight_reorder = kernel->B_is_pretransposed();

        if (_run_weight_reorder)
            scratchpad.book(memory_tracking::names::key_matmul_wei_trans,
                    kernel->get_B_pretransposed_array_size(), 1);
    }

    // TODO: extend post ops to take a pointer?
    return status::success;
}

status_t kai_matmul_t::init(engine_t *engine) {
    return status::success;
}

status_t kai_matmul_t::execute(const exec_ctx_t &ctx) const {

    std::unique_ptr<kai::ops::IGemmCommon> _kernel = nullptr;
    if (pd()->is_dequant()) {
        DEFINE_ARG_SCALES_BUFFER(src_scale, DNNL_ARG_SRC);
        DEFINE_ARG_SCALES_BUFFER(wei_scale, DNNL_ARG_WEIGHTS);
        kai::ops::DequantizeFloat dequant(src_scale[0] * wei_scale[0]);
        _kernel = pd()->create_kai_gemm_dequant(dequant);
    } else {
        _kernel = pd()->create_kai_gemm();
    }
    if (!_kernel) return status::runtime_error;

    if (get_verbose(verbose_t::profile_externals)) {
        std::cout << "profile_externals: " << _kernel->get_config().filter
                  << std::endl;
    }

    const auto scratchpad = ctx.get_scratchpad_grantor();

    const kai::ops::ndrange_t window_size = _kernel->get_window_size();
    const int num_windows = static_cast<int>(window_size.total_size());
    int num_threads = std::min(num_windows, dnnl_get_current_num_threads());

    unsigned int row_parts = num_threads;
    unsigned int col_parts = 1;
    if (window_size.get_size(1) > 1) {
        row_parts = split_window_2d(num_threads, window_size);
        col_parts = num_threads / row_parts;

        const unsigned int max_threads_2d
                = std::min(row_parts, window_size.get_size(0))
                * std::min(col_parts, window_size.get_size(1));
        if (max_threads_2d < static_cast<unsigned int>(num_threads)) {
            row_parts = std::min(row_parts, window_size.get_size(0));
            col_parts = std::min(col_parts, window_size.get_size(1));
            num_threads = static_cast<int>(max_threads_2d);
        }
    }

    _kernel->set_nthreads(num_threads);

    auto src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);

    auto raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto wei_base = CTX_OUT_MEM(void *, DNNL_ARG_WEIGHTS);
    if (pd()->_run_weight_reorder) {
        wei_base = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
    }

    auto dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias_base = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);

    // Leading dimensions of our matrices are the strides of the first non-dense dimensions (second logical)
    auto ld_src
            = pd()->src_md()
                      ->format_desc.blocking.strides[pd()->src_md()->ndims - 2];
    auto ld_dst
            = pd()->dst_md()
                      ->format_desc.blocking.strides[pd()->dst_md()->ndims - 2];

    // With fixed format, weights are already packed in kai's expected
    // layout, so the row stride follows the innermost logical dimension.
    // Otherwise rely on matmul_helper_t so transposed raw weights use the
    // correct leading dimension during kai pretranspose.
    auto ld_wei = pd()->_fixed_format
            ? pd()->weights_md()
                      ->format_desc.blocking
                      .strides[pd()->weights_md()->ndims - 1]
            : helper.ldb();

    const int src_batch_stride = get_innermost_batch_stride(pd()->src_md());
    const int wei_batch_stride = get_innermost_batch_stride(pd()->weights_md());
    const int dst_batch_stride = get_innermost_batch_stride(pd()->dst_md());

    const int batch_stride_a = pd()->is_batches() ? src_batch_stride : 0;
    const int multi_stride_a = pd()->is_multis()
            ? (pd()->_src_broadcast_batch_dims ? 0 : src_batch_stride)
            : 0;
    const int multi_stride_b = pd()->is_multis() ? wei_batch_stride : 0;
    const int batch_stride_c = pd()->is_batches() ? dst_batch_stride : 0;
    const int multi_stride_c = pd()->is_multis() ? dst_batch_stride : 0;

    // Do we need to modify ld_wei after this?
    if (pd()->_run_weight_reorder) {
        const bool transposed
                = wei_d.matches_one_of_tag(format_tag::ba) == format_tag::ba;
        const unsigned int wsize = _kernel->get_B_pretranspose_window_size();

        if (pd()->swd_dt(data_type::bf16, data_type::bf16, data_type::bf16)) {
            _kernel->pretranspose_B_array_generic(
                    wei_base, raw_wei, ld_wei, multi_stride_b, transposed);
        } else {
            parallel(num_threads, [&](int ithr, int nthr) {
                const unsigned int start = (ithr * wsize) / nthr;
                const unsigned int end = ((ithr + 1) * wsize) / nthr;

                if (start < end) {
                    _kernel->pretranspose_B_array_part_generic(wei_base,
                            raw_wei, ld_wei, multi_stride_b, transposed, start,
                            end);
                }
            });
        }
    }

    if (_kernel->get_working_size() != 0) {
        _kernel->set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }

    _kernel->set_arrays_generic(src_base, ld_src, batch_stride_a,
            multi_stride_a, wei_base, ld_wei, multi_stride_b, dst_base, ld_dst,
            batch_stride_c, multi_stride_c, bias_base, 0);

    parallel(num_threads, [&](int ithr, int nthr) {
        unsigned int row_start = 0;
        unsigned int row_end = window_size.get_size(0);
        unsigned int col_start = 0;
        unsigned int col_end = window_size.get_size(1);

        unsigned int thread_row = ithr;
        unsigned int thread_col = 0;
        if (col_parts > 1) {
            balance2D(nthr, ithr, window_size.get_size(0), row_start, row_end,
                    window_size.get_size(1), col_start, col_end, col_parts);
            thread_row = ithr % row_parts;
            thread_col = ithr / row_parts;
        } else {
            balance211(window_size.get_size(0), nthr, ithr, row_start, row_end);
        }

        kai::ops::ndcoord_t thread_locator {{thread_row, row_parts},
                {thread_col, col_parts}, {0, 1}, {0, 1}, {0, 1}, {0, 1}};
        kai::ops::ndcoord_t win {{row_start, row_end - row_start},
                {col_start, col_end - col_start}, {0, window_size.get_size(2)},
                {0, window_size.get_size(3)}, {0, window_size.get_size(4)},
                {0, window_size.get_size(5)}};

        _kernel->execute(win, thread_locator, ithr);
    });

    return status::success;
}

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
