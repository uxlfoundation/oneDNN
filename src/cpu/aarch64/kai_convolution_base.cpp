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
#include "common/utils.hpp"

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

bool bias_ok(const cpu_convolution_fwd_pd_t &pd) {
    return !pd.with_bias()
            || pd.invariant_bia_md()->data_type == pd.dst_md()->data_type;
}

format_tag_t exact_src_dst_tag(const memory_desc_t &md) {
    using namespace format_tag;
    if (md.format_kind != format_kind::blocked || md.ndims != 4)
        return format_tag::undef;

    const auto &strides = md.format_desc.blocking.strides;
    const dim_t c = md.dims[1];
    const dim_t h = md.dims[2];
    const dim_t w = md.dims[3];
    if (strides[0] == c * h * w && strides[1] == h * w && strides[2] == w
            && strides[3] == 1)
        return nchw;
    if (strides[0] == h * w * c && strides[1] == 1 && strides[2] == w * c
            && strides[3] == c)
        return nhwc;

    return format_tag::undef;
}

format_tag_t src_dst_tag(const memory_desc_t &md) {
    using namespace format_tag;
    const auto exact_tag = exact_src_dst_tag(md);
    if (exact_tag != format_tag::undef) return exact_tag;
    return memory_desc_matches_one_of_tag(md, nhwc, nchw);
}

bool regular_swd_ok(const cpu_convolution_fwd_pd_t &pd) {
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

int threads_for_work(double work, double min_work_per_thread, int max_threads) {
    if (max_threads <= 1) return 1;
    const double parallel_work_threshold = min_work_per_thread * max_threads;
    return work >= parallel_work_threshold ? max_threads : 1;
}

} // namespace

bool kai_convolution_fwd_base_t::pd_t::swd_dt(
        data_type_t s, data_type_t w, data_type_t d) const {
    return src_md()->data_type == s && weights_md()->data_type == w
            && dst_md()->data_type == d;
}

bool kai_convolution_fwd_base_t::pd_t::set_default_formats() {
    using namespace format_tag;
    if (src_md_.format_kind == format_kind::any
            && dst_md_.format_kind == format_kind::any) {
        return set_default_formats_common(nhwc, hwio, nhwc);
    }

    const auto src_tag = src_dst_tag(src_md_);
    const auto dst_tag = src_dst_tag(dst_md_);
    if (src_md_.format_kind == format_kind::any && dst_tag != format_tag::undef)
        CHECK(memory_desc_init_by_tag(src_md_, dst_tag));
    else if (dst_md_.format_kind == format_kind::any
            && src_tag != format_tag::undef)
        CHECK(memory_desc_init_by_tag(dst_md_, src_tag));

    const auto resolved_src_tag = src_dst_tag(src_md_);
    const auto resolved_dst_tag = src_dst_tag(dst_md_);
    if (resolved_src_tag == format_tag::undef
            || resolved_dst_tag == format_tag::undef
            || resolved_src_tag != resolved_dst_tag)
        return false;

    if (weights_md_.format_kind == format_kind::any) {
        const format_tag_t wei_tag = resolved_src_tag == nchw ? ihwo : hwio;
        CHECK(memory_desc_init_by_tag(weights_md_, wei_tag));
    }

    if (with_bias() && bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, x));

    return true;
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_convolution_fwd_base_t::pd_t::create_kai_gemm(int max_threads) const {
    const auto weights_dt = gemm_weights_dt_ == data_type::undef
            ? weights_md()->data_type
            : gemm_weights_dt_;
    return kai_utils::create_kai_gemm(*args_, cfg_.get(), src_md()->data_type,
            weights_dt, dst_md()->data_type, max_threads);
}

unsigned int kai_convolution_fwd_base_t::pd_t::gemm_m() const {
    return static_cast<unsigned int>(MB() * OH() * OW());
}

unsigned int kai_convolution_fwd_base_t::pd_t::gemm_k() const {
    return static_cast<unsigned int>(IC() * KH() * KW());
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_src_layout_ok() const {
    return src_channels_last_ || IC() == 1;
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_kernel_ok() const {
    return KH() == 1 && KW() == 1;
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_padding_ok() const {
    return padT() == 0 && padL() == 0;
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_output_samples_in_bounds()
        const {
    return OH() > 0 && OW() > 0 && (OH() - 1) * KSH() < IH()
            && (OW() - 1) * KSW() < IW();
}

status_t kai_convolution_fwd_base_t::pd_t::init(engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;

    fixed_format_ = weights_md_.format_kind == format_kind::any;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_CONV(!with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");
    VDISPATCH_CONV(regular_swd_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_CONV(attr()->has_default_values(primitive_mask_t::fpmath_mode
                                   | primitive_mask_t::accumulation_mode
                                   | primitive_mask_t::post_ops,
                           dst_md()->data_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    const auto src_tag = src_dst_tag(src_md_);
    const auto dst_tag = src_dst_tag(dst_md_);
    const auto wei_tag = memory_desc_matches_one_of_tag(
            weights_md_, format_tag::hwio, format_tag::ihwo);

    VDISPATCH_CONV(src_md_.format_kind == format_kind::any
                    || utils::one_of(
                            src_tag, format_tag::nhwc, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dst_md_.format_kind == format_kind::any
                    || utils::one_of(
                            dst_tag, format_tag::nhwc, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(weights_md_.format_kind == format_kind::any
                    || wei_tag == format_tag::hwio
                    || wei_tag == format_tag::ihwo,
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);

    const bool fast_mode = use_fast_mode(*src_md(), *attr());
    src_channels_last_ = src_tag != format_tag::nchw;
    dst_channels_last_ = dst_tag != format_tag::nchw;
    const bool weights_hwio = wei_tag == format_tag::hwio;
    VDISPATCH_CONV(
            src_channels_last_ == weights_hwio || (KH() == 1 && KW() == 1),
            "requires weights K dimension order to match src");
    wei_k_stride_dim_ = weights_hwio ? 1 : 3;
    const size_t src_dt_size = types::data_type_size(src_md()->data_type);

    VDISPATCH_CONV(num_sum_post_ops(attr_.post_ops_) <= 1,
            "supports at most one sum post op");
    const bool allow_sum_fusion = dst_channels_last_ && !with_bias();
    const auto post_ops_fusion
            = create_post_ops_fusion(attr_.post_ops_, allow_sum_fusion);
    CHECK(post_ops.init(engine, attr_.post_ops_, *dst_md(),
            post_ops_fusion.fallback_start_index));
    has_post_ops_fallback_ = post_ops_fusion.has_fallback(attr_.post_ops_);

    name_ = impl_base_name();
    if (has_post_ops_fallback_) name_ += "+post_ops_fallback";

    cfg_ = std::make_shared<kai::ops::GemmConfig>();
    args_.reset();
    gemm_weights_dt_ = weights_md()->data_type;
    run_weight_reorder_ = false;
    use_dst_reorder_ = false;
    dst_reorder_pd_.reset();
    tmp_dst_md_ = memory_desc_t {};

    if (uses_indirect_gemm())
        VDISPATCH_CONV(src_channels_last_, VERBOSE_UNSUPPORTED_TAG_S, "src");
    CHECK(init_datapath(engine));

    if (fixed_format_) cfg_->weight_format = kai::ops::WeightFormat::ANY;

    const auto make_args = [&]() {
        return std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), gemm_m(),
                OC(), gemm_k(), gemm_k_sections(), gemm_n_batches(),
                gemm_n_multi(), uses_indirect_gemm(),
                post_ops_fusion.activation, dnnl_get_current_num_threads(),
                fixed_format_, fast_mode, post_ops_fusion.accumulate,
                cfg_.get());
    };

    args_ = make_args();
    std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
    const bool fixed_format_failed = fixed_format_
            && (!kernel
                    || !is_fixed_format(kernel->get_config().weight_format));

    if (fixed_format_failed) {
        fixed_format_ = false;
        cfg_ = std::make_shared<kai::ops::GemmConfig>();
        args_ = make_args();
        kernel = create_kai_gemm();
    }
    VDISPATCH_CONV(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    cfg_ = std::make_shared<kai::ops::GemmConfig>(kernel->get_config());
    // Some generated filters do not match the impl list, so it ends up rejecting
    // the second time around. This could be removed if this is fixed in KleidiAI.
    cfg_->filter.clear();

    if (fixed_format_) {
        constexpr dim_t O_dim = 0;
        constexpr dim_t I_dim = 1;
        constexpr dim_t H_dim = 2;
        constexpr dim_t W_dim = 3;
        weight_format_to_memory_desc(
                weights_md_, cfg_->weight_format, I_dim, O_dim, {W_dim, H_dim});
    }

    run_weight_reorder_ = !fixed_format_ && kernel->B_is_pretransposed();

    auto scratchpad = scratchpad_registry().registrar();
    if (kernel->get_working_size() != 0) {
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                kernel->get_working_size(), 1);
    }

    if (run_weight_reorder_) {
        scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
                kernel->get_B_pretransposed_array_size(), 1);
    }

    if (!dst_channels_last_) {
        CHECK(memory_desc_init_by_tag(tmp_dst_md_, dst_md()->ndims,
                dst_md()->dims, dst_md()->data_type, format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(dst_reorder_pd_, engine,
                                  &tmp_dst_md_, dst_md()),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "dst reorder");
        use_dst_reorder_ = true;

        const memory_desc_wrapper tmp_dst_d(&tmp_dst_md_);
        scratchpad.book(memory_tracking::names::key_conv_ncsp_dst,
                tmp_dst_d.size(), 1, 64, 64);
        scratchpad.book(memory_tracking::names::key_nested,
                dst_reorder_pd_->scratchpad_registry());
    }

    if (post_ops.has_sum()) {
        const memory_desc_wrapper dst_d(dst_md());
        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.size(),
                1, 64, 64);
    }

    book_datapath_scratchpad(scratchpad, src_dt_size);

    post_ops.init_scratchpad(scratchpad);

    return status::success;
}

status_t kai_convolution_fwd_base_t::init(engine_t *engine) {
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_, engine));
    return status::success;
}

status_t kai_convolution_fwd_base_t::execute(const exec_ctx_t &ctx) const {
    const auto *pd = this->pd();
    const auto *src_md = pd->src_md();
    const auto *dst_md = pd->dst_md();
    const auto *weights_md = pd->weights_md();
    const bool run_weight_reorder = pd->run_weight_reorder_;
    const bool with_bias = pd->with_bias();
    const bool dst_channels_last = pd->dst_channels_last_;
    const bool use_dst_reorder = pd->use_dst_reorder_;
    const bool fixed_format = pd->fixed_format_;
    const bool has_post_ops_fallback = pd->has_post_ops_fallback_;
    const int wei_k_stride_dim = pd->wei_k_stride_dim_;
    const auto &post_ops = pd->post_ops;
    const bool post_ops_has_sum = post_ops.has_sum();
    const dim_t OC = pd->OC();
    const dim_t OH = pd->OH();
    const dim_t OW = pd->OW();
    const auto scratchpad = ctx.get_scratchpad_grantor();

    constexpr double min_kernel_work_per_thread = 4 * 1024;
    const int max_threads = dnnl_get_current_num_threads();
    // KAI kernels are configured for either one thread or the runtime maximum.
    int kernel_num_threads = threads_for_work(
            kernel_execute_work(*pd), min_kernel_work_per_thread, max_threads);

    std::unique_ptr<kai::ops::IGemmCommon> kernel
            = pd->create_kai_gemm(kernel_num_threads);
    if (!kernel) return status::runtime_error;

    if (get_verbose(verbose_t::profile_externals)) {
        std::cout << "profile_externals: " << kernel->get_config().filter
                  << std::endl;
    }

    const kai::ops::ndrange_t window_size = kernel->get_window_size();
    const int num_windows = static_cast<int>(window_size.total_size());
    kernel_num_threads = std::max(1, std::min(num_windows, kernel_num_threads));

    unsigned int row_parts = kernel_num_threads;
    unsigned int col_parts = 1;
    if (window_size.get_size(1) > 1) {
        row_parts = split_window_2d(kernel_num_threads, window_size);
        col_parts = kernel_num_threads / row_parts;

        const unsigned int max_threads_2d
                = std::min(row_parts, window_size.get_size(0))
                * std::min(col_parts, window_size.get_size(1));
        if (max_threads_2d < static_cast<unsigned int>(kernel_num_threads)) {
            row_parts = std::min(row_parts, window_size.get_size(0));
            col_parts = std::min(col_parts, window_size.get_size(1));
            kernel_num_threads = static_cast<int>(max_threads_2d);
        }
    }

    kernel->set_nthreads(kernel_num_threads);

    const auto *src_base = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *wei_base = const_cast<void *>(raw_wei);
    if (run_weight_reorder) {
        wei_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
    }

    void *dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias_base
            = with_bias ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS) : nullptr;

    constexpr int src_n_dim = 0;
    constexpr int src_c_dim = 1;
    constexpr int src_h_dim = 2;
    constexpr int src_w_dim = 3;
    constexpr int dst_n_dim = 0;
    constexpr int dst_h_dim = 2;
    constexpr int dst_w_dim = 3;
    constexpr int wei_o_dim = 0;

    const int ld_src
            = static_cast<int>(src_md->format_desc.blocking.strides[src_w_dim]);
    const int ld_dst = dst_channels_last
            ? static_cast<int>(dst_md->format_desc.blocking.strides[dst_w_dim])
            : static_cast<int>(OC);
    const int ld_wei = static_cast<int>(fixed_format
                    ? weights_md->format_desc.blocking.strides[wei_o_dim]
                    : weights_md->format_desc.blocking
                              .strides[wei_k_stride_dim]);

    const int src_batch_stride
            = static_cast<int>(src_md->format_desc.blocking.strides[src_n_dim]);
    const int src_channel_stride
            = static_cast<int>(src_md->format_desc.blocking.strides[src_c_dim]);
    const int src_h_stride
            = static_cast<int>(src_md->format_desc.blocking.strides[src_h_dim]);
    const int dst_h_stride = dst_channels_last
            ? static_cast<int>(dst_md->format_desc.blocking.strides[dst_h_dim])
            : static_cast<int>(OW * OC);
    const int dst_batch_stride = dst_channels_last
            ? static_cast<int>(dst_md->format_desc.blocking.strides[dst_n_dim])
            : static_cast<int>(OH * OW * OC);
    const size_t src_dt_size = types::data_type_size(src_md->data_type);
    const size_t src_col_stride_bytes
            = static_cast<size_t>(ld_src) * src_dt_size;
    const size_t src_channel_stride_bytes
            = static_cast<size_t>(src_channel_stride) * src_dt_size;
    const size_t src_h_stride_bytes
            = static_cast<size_t>(src_h_stride) * src_dt_size;
    const size_t src_batch_stride_bytes
            = static_cast<size_t>(src_batch_stride) * src_dt_size;
    void *kernel_dst_base = nullptr;
    if (dst_channels_last) {
        kernel_dst_base = post_ops_has_sum
                ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
                : dst_base;
    } else {
        kernel_dst_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_ncsp_dst);
    }

    if (run_weight_reorder) {
        constexpr double min_reorder_work_per_thread = 4 * 1024;
        const int reorder_num_threads
                = threads_for_work(weight_reorder_work(*pd),
                        min_reorder_work_per_thread, max_threads);
        const unsigned int wsize = kernel->get_B_pretranspose_window_size();
        parallel(reorder_num_threads, [&](int ithr, int nthr) {
            const unsigned int start = (ithr * wsize) / nthr;
            const unsigned int end = ((ithr + 1) * wsize) / nthr;
            if (start < end) {
                kernel->pretranspose_B_array_part_generic(
                        wei_base, raw_wei, ld_wei, 0, false, start, end);
            }
        });
    }

    if (kernel->get_working_size() != 0) {
        kernel->set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }

    CHECK(setup_kernel_arrays(kernel_call_args_t {ctx, *pd, *kernel, scratchpad,
            kernel_num_threads, src_base, wei_base, kernel_dst_base, bias_base,
            ld_src, ld_wei, ld_dst, src_h_stride, src_batch_stride,
            dst_h_stride, dst_batch_stride, src_dt_size, src_col_stride_bytes,
            src_channel_stride_bytes, src_h_stride_bytes,
            src_batch_stride_bytes}));

    parallel(kernel_num_threads, [&](int ithr, int nthr) {
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

        kernel->execute(win, thread_locator, ithr);
    });

    void *post_ops_src = kernel_dst_base;
    if (!dst_channels_last) {
        post_ops_src = post_ops_has_sum
                ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
                : dst_base;

        if (!use_dst_reorder || !dst_reorder_) return status::runtime_error;

        auto *engine = ctx.stream()->engine();
        std::unique_ptr<memory_t, memory_deleter_t> tmp_dst_mem(new memory_t(
                engine, &pd->tmp_dst_md_, use_runtime_ptr, kernel_dst_base));
        std::unique_ptr<memory_t, memory_deleter_t> dst_mem(
                new memory_t(engine, dst_md, use_runtime_ptr, post_ops_src));

        exec_args_t reorder_args;
        reorder_args[DNNL_ARG_SRC] = {tmp_dst_mem.get(), true};
        reorder_args[DNNL_ARG_DST] = {dst_mem.get(), false};
        exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));

        auto *nested_grantor = memory_tracking::create_nested_grantor(
                ctx.get_scratchpad_grantor(),
                memory_tracking::names::key_nested,
                dst_reorder_->pd()->scratchpad_registry());
        reorder_ctx.set_scratchpad_grantor(nested_grantor);
        CHECK(dst_reorder_->execute(reorder_ctx));
    }

    if (has_post_ops_fallback) {
        if (post_ops_has_sum)
            CHECK(post_ops.execute(ctx, post_ops_src, dst_base));
        else
            CHECK(post_ops.execute(ctx, post_ops_src));
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
