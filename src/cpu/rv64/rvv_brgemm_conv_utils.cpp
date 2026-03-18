/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/rvv_brgemm_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace prop_kind;
using namespace data_type;

namespace brgemm_convolution_utils {

status_t init_conf(brgemm_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    const cpu_isa_t isa = v;

    if (!one_of(cd.prop_kind, forward_training, forward_inference))
        return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    jcp.isa = isa;
    jcp.prop_kind = cd.prop_kind;
    jcp.ndims = ndims;
    jcp.nthr = nthreads;

    jcp.mb = src_d.dims()[0];
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;

    // Spatial dimensions.
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims >= 4) ? src_d.dims()[ndims - 2] : 1;
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;
    jcp.ow = dst_d.dims()[ndims - 1];

    // Kernel dimensions.
    const int goff = with_groups ? 1 : 0;
    jcp.kd = (ndims == 5) ? weights_d.dims()[goff + 2] : 1;
    jcp.kh = (ndims >= 4) ? weights_d.dims()[goff + ndims - 2] : 1;
    jcp.kw = weights_d.dims()[goff + ndims - 1];

    // Strides.
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims >= 4) ? cd.strides[ndims - 4] : 1;
    jcp.stride_w = cd.strides[ndims - 3];

    // Dilations.
    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims >= 4) ? cd.dilates[ndims - 4] : 0;
    jcp.dilate_w = cd.dilates[ndims - 3];

    // Padding.
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims >= 4) ? cd.padding[0][ndims - 4] : 0;
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.back_pad = (ndims == 5) ? cd.padding[1][0] : 0;
    jcp.b_pad = (ndims >= 4) ? cd.padding[1][ndims - 4] : 0;
    jcp.r_pad = cd.padding[1][ndims - 3];

    // Data types — only f32 supported.
    jcp.src_dt = src_d.data_type();
    jcp.wei_dt = weights_d.data_type();
    jcp.dst_dt = dst_d.data_type();
    jcp.bia_dt = bias_d.ndims() ? bias_d.data_type() : data_type::undef;

    if (!everyone_is(f32, jcp.src_dt, jcp.wei_dt, jcp.dst_dt))
        return status::unimplemented;
    if (jcp.bia_dt != data_type::undef && jcp.bia_dt != f32)
        return status::unimplemented;

    // Reject when output width or per-group input channels are too small
    // for per-kernel-position GEMM calls to be efficient.
    if (jcp.ow < 20 || jcp.ic < 16) return status::unimplemented;

    jcp.with_bias = jcp.bia_dt != data_type::undef;
    jcp.with_sum = attr.post_ops_.find(primitive_kind::sum) != -1;

    // Source and destination: NSPC (channel-last) only.
    const auto dat_tag = pick(ndims - 3, nwc, nhwc, ndhwc);

    if (src_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
    else if (!src_d.matches_one_of_tag(dat_tag))
        return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
    else if (!dst_d.matches_one_of_tag(dat_tag))
        return status::unimplemented;

    // Weights: plain spatial × IC × OC layout (same as gemm conv NSPC).
    const auto wei_tag = with_groups ? pick(ndims - 3, wigo, hwigo, dhwigo)
                                     : pick(ndims - 3, wio, hwio, dhwio);

    memory_desc_t want_wei_md = weights_md;
    CHECK(memory_desc_init_by_tag(want_wei_md, wei_tag));

    if (weights_md.format_kind == format_kind::any) {
        weights_md = want_wei_md;
    } else {
        if (want_wei_md != weights_md) return status::unimplemented;
    }

    if (bias_d.ndims() && bias_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));

    jcp.src_tag = dat_tag;
    jcp.dst_tag = dat_tag;
    jcp.wei_tag = wei_tag;

    return status::success;
}

} // namespace brgemm_convolution_utils

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
