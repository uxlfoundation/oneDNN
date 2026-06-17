/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
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

#include "cpu/rv64/rvv_dwconv.hpp"

#include <algorithm>
#include <vector>

#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "cpu/rv64/jit_rvv_dwconv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

namespace {

static dim_t tensor_nhwc_elems(dim_t n, dim_t h, dim_t w, dim_t c) {
    return n * h * w * c;
}

static void pack_input_nhwc(const float16_t *src,
        const memory_desc_wrapper &src_d, dim_t n, dim_t ih, dim_t iw,
        dim_t channels, dim_t padded_h, dim_t padded_w, dim_t t_pad,
        dim_t l_pad, float16_t *packed) {
    const float16_t zero(0.0f);
    std::fill(packed,
            packed + tensor_nhwc_elems(1, padded_h, padded_w, channels), zero);

    for (dim_t h = 0; h < ih; ++h) {
        for (dim_t w = 0; w < iw; ++w) {
            float16_t *dst = packed
                    + ((h + t_pad) * padded_w + (w + l_pad)) * channels;
            const float16_t *src_ptr = src + src_d.off(n, 0, h, w);
            std::copy_n(src_ptr, channels, dst);
        }
    }
}

static void pack_weights_goihw(const float16_t *weights,
        const memory_desc_wrapper &wei_d, dim_t groups, dim_t oc_per_group,
        std::vector<float16_t> &packed) {
    packed.resize(oc_per_group * 3 * 3 * groups);
    for (dim_t oc = 0; oc < oc_per_group; ++oc) {
        float16_t *oc_base = packed.data() + oc * 9 * groups;
        for (dim_t kh = 0; kh < 3; ++kh) {
            for (dim_t kw = 0; kw < 3; ++kw) {
                float16_t *k_base = oc_base + (kh * 3 + kw) * groups;
                for (dim_t g = 0; g < groups; ++g)
                    k_base[g] = weights[wei_d.off(g, oc, 0, kh, kw)];
            }
        }
    }
}

static void prepare_bias(const void *bias, bool bias_is_f32, dim_t channels,
        std::vector<float> &bias_fp32) {
    if (bias == nullptr) {
        bias_fp32.clear();
        return;
    }

    bias_fp32.resize(channels);
    if (bias_is_f32) {
        const auto *bias_data = static_cast<const float *>(bias);
        std::copy_n(bias_data, channels, bias_fp32.begin());
    } else {
        const auto *bias_data = static_cast<const float16_t *>(bias);
        for (dim_t c = 0; c < channels; ++c)
            bias_fp32[c] = static_cast<float>(bias_data[c]);
    }
}

} // namespace

status_t rvv_dwconv_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto *src = CTX_IN_MEM(const float16_t *, DNNL_ARG_SRC);
    const auto *wei = CTX_IN_MEM(const float16_t *, DNNL_ARG_WEIGHTS);
    const auto *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto *dst = CTX_OUT_MEM(float16_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const convolution_desc_t *cd = pd()->desc();
    const dim_t mb = src_d.dims()[0];
    const dim_t ih = src_d.dims()[2];
    const dim_t iw = src_d.dims()[3];
    const dim_t groups = wei_d.dims()[0];
    const dim_t oc_per_group = wei_d.dims()[1];
    const dim_t oh = dst_d.dims()[2];
    const dim_t ow = dst_d.dims()[3];
    const dim_t t_pad = cd->padding[0][0];
    const dim_t l_pad = cd->padding[0][1];
    const dim_t b_pad = cd->padding[1][0];
    const dim_t r_pad = cd->padding[1][1];
    const dim_t padded_h = ih + t_pad + b_pad;
    const dim_t padded_w = iw + l_pad + r_pad;
    const dim_t stride_h = cd->strides[0];

    std::vector<float16_t> packed_weights;
    pack_weights_goihw(wei, wei_d, groups, oc_per_group, packed_weights);

    std::vector<float> bias_fp32;
    const bool bias_is_f32 = pd()->with_bias()
            && pd()->weights_md(1)->data_type == data_type::f32;
    prepare_bias(bias, bias_is_f32, groups * oc_per_group, bias_fp32);

    parallel_nd(mb, oc_per_group, [&](dim_t n, dim_t oc) {
        std::vector<float16_t> packed_src(
                tensor_nhwc_elems(1, padded_h, padded_w, groups));
        pack_input_nhwc(src, src_d, n, ih, iw, groups, padded_h, padded_w,
                t_pad, l_pad, packed_src.data());

        jit_rvv_dwconv_kernel_t::call_params_t args;
        args.lhs = packed_src.data();
        args.lhs_stride_0 = padded_w * groups * (dim_t)sizeof(float16_t);
        args.lhs_stride_1 = groups * (dim_t)sizeof(float16_t);
        args.rhs = packed_weights.data() + oc * 9 * groups;
        args.rhs_stride_0 = 3 * groups * (dim_t)sizeof(float16_t);
        args.rhs_stride_1 = groups * (dim_t)sizeof(float16_t);
        args.out = dst + dst_d.off(n, oc, 0, 0);
        args.out_stride_0
                = dst_d.blocking_desc().strides[2] * (dim_t)sizeof(float16_t);
        args.out_stride_1
                = dst_d.blocking_desc().strides[3] * (dim_t)sizeof(float16_t);
        args.h = oh;
        args.w = ow;
        args.c = groups;
        args.ratio_bytes = oc_per_group * (dim_t)sizeof(float16_t);
        args.bias
                = bias_fp32.empty() ? nullptr : bias_fp32.data() + oc * groups;
        static const jit_rvv_dwconv_kernel_t kernel_s1(1);
        static const jit_rvv_dwconv_kernel_t kernel_s2(2);
        const auto &kernel = stride_h == 1 ? kernel_s1 : kernel_s2;
        kernel(&args);
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
