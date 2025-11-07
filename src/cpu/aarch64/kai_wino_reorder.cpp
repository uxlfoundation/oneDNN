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

#include "cpu/aarch64/kai_wino_reorder.hpp"

#include <cstring>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/kai_utils.hpp"
#include "cpu/aarch64/kai_wino_utils.hpp"

#include "kai/ops/conv/winograd.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

bool is_kai_wino_desc(const memory_desc_wrapper &mdw) {
    return mdw.is_wino_desc()
            && mdw.wino_desc().wino_format
            == wino_memory_format_t::wino_wei_aaIOoi;
}

template <typename data_t>
bool init_wino_impl(kai::ops::winograd::WinogradImpl &wino_impl,
        const kai::ops::ConvolutionArgs &conv_args, kai::ops::GemmConfig *cfg,
        int max_threads, int alpha) {
    kai::ops::winograd::WinogradConfig wino_cfg;
    wino_cfg.output_rows = alpha - conv_args.kernel_shape.rows + 1;
    wino_cfg.output_cols = alpha - conv_args.kernel_shape.cols + 1;
    return kai::ops::winograd::get_implementation<data_t>(wino_impl,
            kai_utils::get_cpu_info(), conv_args, max_threads, false, &wino_cfg,
            cfg);
}

template <typename src_data_t, typename wino_data_t>
status_t execute_reorder(
        const kai_wino_reorder_t::pd_t *pd, const exec_ctx_t &ctx) {
    const memory_desc_wrapper src_d(pd->src_md());
    const memory_desc_wrapper dst_d(pd->dst_md());
    const auto &wds = pd->wino_impl_->winograd_spec;
    const auto &wd = dst_d.wino_desc();
    const dim_t OC = src_d.dims()[0];
    const dim_t IC = src_d.dims()[1];
    const dim_t KH = src_d.dims()[2];
    const dim_t KW = src_d.dims()[3];
    const dim_t n_matrices = kai_wino_utils::n_matrices(wd);

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const auto *src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_FROM);
    auto *dst = CTX_OUT_MEM(wino_data_t *, DNNL_ARG_TO);
    auto *plain = scratchpad.get<wino_data_t>(
            memory_tracking::names::key_reorder_wino_plain);
    auto *wino = scratchpad.get<wino_data_t>(
            memory_tracking::names::key_reorder_wino_transform_space);

    std::memset(plain, 0, pd->plain_weights_size_);
    parallel_nd(KH, KW, IC, OC, [&](dim_t kh, dim_t kw, dim_t ic, dim_t oc) {
        const auto plain_off = ((kh * KW + kw) * IC + ic) * OC + oc;
        plain[plain_off]
                = static_cast<wino_data_t>(src[src_d.off(oc, ic, kh, kw)]);
    });

    std::memset(wino, 0, pd->wino_weights_size_);
    const size_t plain_row_stride = static_cast<size_t>(KW * IC * OC);
    const size_t plain_col_stride = static_cast<size_t>(IC * OC);
    const size_t plain_ic_stride = static_cast<size_t>(OC);
    const int nthreads = dnnl_get_current_num_threads();
    parallel(nthreads, [&](int ithr, int nthr) {
        pd->wino_impl_->weight_transform->execute(*pd->conv_args_, plain,
                plain_row_stride, plain_col_stride, plain_ic_stride, wino, wds,
                ithr, nthr);
    });

    std::memset(dst, 0, dst_d.size());
    const memory_desc_wrapper packed_d(&pd->packed_md_);
    parallel_nd(n_matrices, IC, OC, [&](dim_t matrix, dim_t ic, dim_t oc) {
        dst[packed_d.off(matrix, ic, oc)] = wino[matrix * wds.weight_ld_matrix
                + ic * wds.weight_ld_row + oc];
    });

    return status::success;
}

} // namespace

status_t kai_wino_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper dst_d(dst_md_);
    VDISPATCH_REORDER(
            src_d.ndims() == 4, VERBOSE_BAD_NDIMS, "src", src_d.ndims());
    VDISPATCH_REORDER(
            src_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_REORDER(
            is_kai_wino_desc(dst_d), VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_REORDER((src_d.data_type() == dst_d.data_type())
                    || (src_d.data_type() == data_type::f32
                            && dst_d.data_type() == data_type::f16),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(
            utils::one_of(dst_d.data_type(), data_type::f32, data_type::f16),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REORDER(!src_d.has_runtime_dims_or_strides()
                    && !dst_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_REORDER(!src_d.has_zero_dim(), VERBOSE_EMPTY_TENSOR, "");

    const auto &wd = dst_d.wino_desc();
    VDISPATCH_REORDER(src_d.dims()[0] == wd.oc && src_d.dims()[1] == wd.ic
                    && src_d.dims()[2] == wd.r && src_d.dims()[3] == wd.r,
            "inconsistent KAI Winograd weights descriptor");
    VDISPATCH_REORDER(wd.alpha > wd.r, "invalid KAI Winograd tile");

    cfg_ = std::make_shared<kai::ops::GemmConfig>();
    conv_args_ = std::make_shared<kai::ops::ConvolutionArgs>(1,
            kai::ops::Shape2D {static_cast<unsigned int>(wd.alpha),
                    static_cast<unsigned int>(wd.alpha)},
            static_cast<unsigned int>(wd.ic), 0, 0,
            kai::ops::Shape2D {static_cast<unsigned int>(wd.alpha - wd.r + 1),
                    static_cast<unsigned int>(wd.alpha - wd.r + 1)},
            static_cast<unsigned int>(wd.oc),
            kai::ops::Shape2D {static_cast<unsigned int>(wd.r),
                    static_cast<unsigned int>(wd.r)});
    wino_impl_ = std::make_shared<kai::ops::winograd::WinogradImpl>();
    bool supported = false;
    if (dst_d.data_type() == data_type::f32) {
        supported = init_wino_impl<float>(*wino_impl_, *conv_args_, cfg_.get(),
                dnnl_get_current_num_threads(), wd.alpha);
    } else if (dst_d.data_type() == data_type::f16) {
        supported = init_wino_impl<__fp16>(*wino_impl_, *conv_args_, cfg_.get(),
                dnnl_get_current_num_threads(), wd.alpha);
    }
    VDISPATCH_REORDER(supported, "unsupported KAI Winograd reorder");

    packed_md_ = kai_wino_utils::make_packed_weights_desc(dst_md_);
    plain_weights_size_ = static_cast<size_t>(src_d.dims()[0] * src_d.dims()[1]
                                  * src_d.dims()[2] * src_d.dims()[3])
            * types::data_type_size(dst_d.data_type());
    wino_weights_size_ = wino_impl_->winograd_spec.weight_matrix_size_bytes;

    init_scratchpad();
    return status::success;
}

void kai_wino_reorder_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_reorder_wino_plain,
            plain_weights_size_, 1, 64, 64);
    scratchpad.book(memory_tracking::names::key_reorder_wino_transform_space,
            wino_weights_size_, 1, 64, 64);
}

status_t kai_wino_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto _pd = std::unique_ptr<pd_t>(new pd_t(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md));
    if (!_pd) return status::out_of_memory;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    CHECK(_pd->init_scratchpad_md());
    return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd.release());
}

status_t kai_wino_reorder_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->src_md()->data_type == data_type::f32
            && pd()->dst_md()->data_type == data_type::f32)
        return execute_reorder<float, float>(pd(), ctx);
    if (pd()->src_md()->data_type == data_type::f32
            && pd()->dst_md()->data_type == data_type::f16)
        return execute_reorder<float, __fp16>(pd(), ctx);
    if (pd()->src_md()->data_type == data_type::f16
            && pd()->dst_md()->data_type == data_type::f16)
        return execute_reorder<__fp16, __fp16>(pd(), ctx);
    return status::runtime_error;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
