/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/gemm/jit/pd.hpp"
#include "common/c_types_map.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

using namespace intel::jit;

namespace {

// Obtain dimension count for gemmstone (common scales give count 0).
int quant_entry_ndims(
        const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx) {
    if (entry.has_default_values()) return -1;
    if (qmd.ndims < 2) return 0;

    // Count the number of nontrivial (dim > 1) dimensions present
    int count = 0;
    for (int i = qmd.ndims - 2; i < qmd.ndims; ++i) {
        if (qmd.dims[i] > 1) { count++; }
    }

    // for gemmstone, 1D quantization implies a full column vector
    // (i.e. not on the K dimension). If quantization varies over K,
    // we have to send these as 2D
    if (k_idx >= 0 && count == 1 && qmd.dims[k_idx] > 1) return 2;

    return count;
}
} // anonymous namespace

status_t pd_t::init_post_ops() {
    using namespace primitive_kind;
    using namespace alg_kind;
    using namespace data_type;

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: init_post_ops() **** >");

    const auto d = desc();

    // Examine post-ops and remember binary srcs.
    post_ops_ = attr()->post_ops_;
    binary_srcs_.reserve(post_ops_.len() + 4);

    bool ok = true;
    int prelu_count = 0;
    const int num_orig_postops = post_ops_.len();
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &e = post_ops_.entry_[i];
        switch (e.kind) {
            case binary:
                ok &= supported_binary_op(e.binary.alg)
                        && is_md_gemm_compatible_plain_format(
                                &e.binary.src1_desc);
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::binary, int(i)});
                non_scale_po_ = true;
                break;
            case sum:
                ok &= !with_sum_;
                with_sum_ = true;
                sum_at_begin_ = (i == 0);
                binary_srcs_.push_back(binary_src_t {binary_src_t::none, 0});
                beta_ = e.sum.scale;
                break;
            case eltwise:
                ok &= eltwise_injector_f32_is_supported(e.eltwise.alg);
                binary_srcs_.push_back(binary_src_t {binary_src_t::none, 0});
                non_scale_po_ = true;
                break;
            case prelu:
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::prelu, int(i)});
                ok &= get_prelu_md(e.prelu.mask, dst_md()->dims, prelu_wei_md,
                              dst_md()->ndims)
                        == status::success;
                prelu_count++;
                ok &= prelu_count <= 1;
                non_scale_po_ = true;
                break;
            default: return status::unimplemented;
        }
    }

    if (!ok) return status::unimplemented;

    //VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: init_post_ops() **** try convert scales to postops");

    // If scales are present, convert them and any bias to binary post-ops.
    //   Exception: 2D scales.
    // Also convert bias to binary post-op if dst zp are present.
    const auto &a_scales = attr()->scales_.get(DNNL_ARG_A);
    const auto &b_scales = attr()->scales_.get(DNNL_ARG_B);
    const auto &c_scales = attr()->scales_.get(DNNL_ARG_C);

    bias_via_binary_ = (desc()->bias_type() != data_type::undef)
            && (d->bias_desc.ndims >= 1 || !a_scales.has_default_values()
                    || !b_scales.has_default_values()
                    || !attr()->zero_points_.has_default_values(DNNL_ARG_C));
    if (bias_via_binary_) {
        CHECK(post_ops_.prepend_binary(binary_add, &d->bias_desc));
        binary_srcs_.insert(
                binary_srcs_.begin(), binary_src_t {binary_src_t::bias, 0});
    }
    non_scale_po_ |= bias_via_binary_;

    auto maybe_convert_scales_to_postop
            = [this](const memory_desc_t &scale_md, int arg, data_type_t dt,
                      bool mx, bool &converted) -> status_t {
        //VDEBUGINFO(4, primitive, gemm_jit_pd,"MY: init_post_ops() **** maybe_convert_scales_to_postop >>>>>");

        auto ndims = desc()->c_desc.ndims;
        // Scales on A/B can be converted to postops if
        // the scales md has K=1
        converted = false;
        int inner_dim = (arg == DNNL_ARG_A ? ndims - 2 : ndims - 1);
        bool convert = (scale_md.dims[inner_dim] <= 1) || (arg == DNNL_ARG_C);
        convert &= !mx;
        if (convert) {
            if (arg == DNNL_ARG_C) {
                CHECK(post_ops_.append_binary(binary_div, &scale_md));
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::scales, arg});
            } else {
                CHECK(post_ops_.prepend_binary(binary_mul, &scale_md));
                binary_srcs_.insert(binary_srcs_.begin(),
                        binary_src_t {binary_src_t::scales, arg});
            }
            converted = true;
        }
        //VDEBUGINFO(4, primitive, gemm_jit_pd,"MY: maybe_convert_scales_to_postop : converted = %d",converted);
        return status::success;
    };

    if (!a_scales.has_default_values() && !a_scales.is_host_scalar()) {
        //VDEBUGINFO(4, primitive, gemm_jit_pd,"MY: init_post_ops() **** ! a_scales has_default");
        // Host scalar scale will be converted to Alpha
        bool converted;
        CHECK(maybe_convert_scales_to_postop(a_scale_md_, DNNL_ARG_A,
                a_scales.get_data_type(), a_scales.is_mx(), converted));
        if (converted) asc_dims_ = -1;
    }

    if (!b_scales.has_default_values() && !b_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(b_scale_md_, DNNL_ARG_B,
                b_scales.get_data_type(), b_scales.is_mx(), converted));
        if (converted) bsc_dims_ = -1;
    }

    bool try_c_scale = !c_scales.is_host_scalar()
            || (c_scales.is_host_scalar() && num_orig_postops > 0);
    if (!c_scales.has_default_values() && try_c_scale) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(c_scale_md_, DNNL_ARG_C,
                c_scales.get_data_type(), c_scales.is_mx(), converted));
        // Conversion of dst scales to post ops is currently supported for all
        // cases supported in the library.
        gpu_assert(converted || c_scales.is_mx())
                << "Unable to convert dst scales to a post op";
    }

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: init_post_ops() < ****");
    return status::success;
}

bool pd_t::dy_quant_enabled() {
    const auto d = desc();
    using namespace data_type;
    bool all_f8 = (utils::one_of(d->a_type(), f8_e5m2, f8_e4m3)
            && utils::one_of(d->b_type(), f8_e5m2, f8_e4m3)
            && utils::one_of(d->c_type(), f8_e5m2, f8_e4m3, f16, bf16, f32));
    return (utils::one_of(d->c_type(), f32, f16, bf16, u8, s8)
                   && utils::one_of(d->a_type(), u8, s8, s4, u4)
                   && utils::one_of(d->b_type(), u8, s8))
            || all_f8;
}

bool pd_t::wei_decomp() {
    const auto d = desc();
    using namespace data_type;
    return (utils::one_of(d->c_type(), f32, f16, bf16, f8_e5m2, f8_e4m3)
                   && utils::one_of(d->a_type(), u8, s8, s4, u4, f8_e4m3,
                           f8_e5m2, f4_e2m1, f4_e3m0)
                   && utils::one_of(
                           d->b_type(), f16, f32, bf16, f8_e5m2, f8_e4m3))
            && types::data_type_bits(d->a_type())
            < types::data_type_bits(d->b_type())
            && attr()->mayiconvert(d->a_type(), f32);
}

bool pd_t::quant_enabled() {
    return wei_decomp() || dy_quant_enabled();
}

status_t pd_t::init_attrs() {

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: init_attrs @@@@@ >");

    wei_decomp_ = wei_decomp();
    dy_quant_enabled_ = dy_quant_enabled();
    quant_enabled_ = quant_enabled();
    const auto &d = desc();

    const auto &attr_zps = attr()->zero_points_;
    const auto a_zps = attr_zps.get(DNNL_ARG_A);
    const auto b_zps = attr_zps.get(DNNL_ARG_B);
    const auto c_zps = attr_zps.get(DNNL_ARG_C);

    const auto &attr_gs = attr()->precomputed_reductions_;
    const auto a_gs = attr_gs.get(DNNL_ARG_A);
    const auto b_gs = attr_gs.get(DNNL_ARG_B);

    const auto &scales = attr()->scales_;
    const auto a_scales = scales.get(DNNL_ARG_A);
    const auto b_scales = scales.get(DNNL_ARG_B);
    const auto c_scales = scales.get(DNNL_ARG_C);

    cmask_a_ = a_zps.get_mask();
    cmask_b_ = b_zps.get_mask();
    cmask_c_ = c_zps.get_mask();

    // Swap descriptors to follow column major format
    CHECK(a_zps.get_md(a_zp_md_, d->b_desc));
    CHECK(b_zps.get_md(b_zp_md_, d->a_desc));
    CHECK(a_gs.get_md(a_gs_md_, d->b_desc));
    CHECK(b_gs.get_md(b_gs_md_, d->a_desc));
    CHECK(a_scales.get_md(a_scale_md_, desc_.b_desc));
    CHECK(b_scales.get_md(b_scale_md_, desc_.a_desc));
    CHECK(c_scales.get_md(c_scale_md_, desc_.c_desc));

    auto ndims = d->c_desc.ndims;
    ao_dims_ = quant_entry_ndims(a_zps, a_zp_md_, ndims - 2);
    bo_dims_ = quant_entry_ndims(b_zps, b_zp_md_, ndims - 1);
    ag_dims_ = quant_entry_ndims(a_gs, a_gs_md_, ndims - 2);
    bg_dims_ = quant_entry_ndims(b_gs, b_gs_md_, ndims - 1);
    asc_dims_ = quant_entry_ndims(a_scales, a_scale_md_, ndims - 2);
    bsc_dims_ = quant_entry_ndims(b_scales, b_scale_md_, ndims - 1);
    csc_dims_ = quant_entry_ndims(c_scales, c_scale_md_, -1);

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: @@@@@ : ao_dims_ = %d", ao_dims_);


    a_scales_type_ = a_scales.get_data_type();
    if (!a_zps.has_default_groups()) {
        a_zp_group_k_ = a_zps.get_group(0);
        a_zp_group_m_ = a_zps.get_group(1);
    }
    if (!a_gs.has_default_groups()) {
        a_gs_group_k_ = a_gs.get_group(0);
        a_gs_group_m_ = a_gs.get_group(1);
    }
    if (!a_scales.has_default_groups()) {
        a_scales_group_k_ = a_scales.get_group(0);
        a_scales_group_m_ = a_scales.get_group(1);
    }

    b_scales_type_ = b_scales.get_data_type();
    if (!b_zps.has_default_groups()) {
        b_zp_group_n_ = b_zps.get_group(0);
        b_zp_group_k_ = b_zps.get_group(1);
    }
    if (!b_gs.has_default_groups()) {
        b_gs_group_n_ = b_gs.get_group(0);
        b_gs_group_k_ = b_gs.get_group(1);
    }
    if (!b_scales.has_default_groups()) {
        b_scales_group_n_ = b_scales.get_group(0);
        b_scales_group_k_ = b_scales.get_group(1);
    }
    c_scales_type_ = c_scales.get_data_type();
    if (!c_scales.has_default_groups()) {
        c_scales_group_m_ = c_scales.get_group(1);
        c_scales_group_n_ = c_scales.get_group(0);
        with_mx_scale_ = c_scales.is_mx();
    }

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: init_attrs < @@@@@");
    return status::success;
}

bool pd_t::zp_ok() {
    using namespace data_type;
    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: zp_ok ______ >");
    auto &attr_zps = attr()->zero_points_;
    auto &a_zps = attr_zps.get(DNNL_ARG_A);
    auto &b_zps = attr_zps.get(DNNL_ARG_B);

    // INT4 ZPs on SRC do not expand the range in a meaningful way, skipping
    if (utils::one_of((swap_ab() ? a_zps : b_zps).get_data_type(), s4, u4)){
        VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
        return false;
    }

    int ndims = desc()->a_desc.ndims;
    const bool a_int4 = utils::one_of(desc()->a_type(), s4, u4);
    const bool b_int4 = utils::one_of(desc()->b_type(), s4, u4);
    const bool weights_upconversion = wei_decomp_
            || ((swap_ab() ? b_int4 : a_int4) && dy_quant_enabled_);

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : a_int4 = %d b_int4 = %d weights_upconversion = %d",a_int4,b_int4,weights_upconversion);

    bool allow = gpu_utils::dev_getenv("ALLOW_HOST_SCALAR", true);
    if (attr_zps.has_host_scalars() && !allow) {
        VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
        return false;
    }

    if (!a_zps.has_default_values()) {
        VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !a_zps.has_default_values()");
        // Groups determine supported masks.
        if (!a_zps.has_default_groups()) {
            VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !a_zps.has_default_groups()");
            if (!valid_2d_mask(
                        cmask_a_, ndims, !swap_ab() && weights_upconversion)){
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
            const auto a_q2d_group_n = a_zps.get_group(1);
            // Non-trivial N group unsupported.
            if (a_q2d_group_n != 1) {
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
            // Zero points with non-trivial groups only supported with
            // precomputed reductions or when target tensor is being dequantized.
            if (attr()->precomputed_reductions_.has_default_values(DNNL_ARG_B)
                    && dy_quant_enabled_ && b_int4 && !a_int4 && a_zp_2d()){
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
        } else {
            VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !a_zps.has_default_groups() else");
            if (!utils::one_of(cmask_a_, 0, mask_per_oc, mask_per_ic)){
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
            // Weights zp can only be performantly enabled during upconversion
            // for cases that perform decompression.
            if (!wei_decomp_ && !a_int4 && a_scales_2d()) {
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
        }
    }

    if (!b_zps.has_default_values()) {
        VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !b_zps.has_default_values()");
        // Groups determine supported masks.
        if (!b_zps.has_default_groups()) {
            VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !b_zps.has_default_groups()");
            if (!valid_2d_mask(
                        cmask_b_, ndims, swap_ab() && weights_upconversion)){
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
            const auto b_q2d_group_n = b_zps.get_group(0);
            // Non-trivial M group unsupported.
            if (!utils::one_of(b_q2d_group_n, 1, desc()->n())) return false;
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            if (dy_quant_enabled_ && a_int4 && !b_int4 && b_zp_2d()){
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
        } else {
            VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !b_zps.has_default_groups() else");
            if (!utils::one_of(
                        cmask_b_, 0, mask_scalar, mask_per_oc | mask_per_ic)){
                VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
                return false;
            }
        }
    }

    if (!attr_zps.has_default_values(DNNL_ARG_C)) {
        VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ : !attr_zps.has_default_values(DNNL_ARG_C)");
        if (!utils::one_of(cmask_c_, 0, mask_scalar, mask_per_oc)) {
            VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: ______ < false");
            return false;
        }
    }

    VDEBUGINFO(4, primitive, gemm_jit_pd, "MY: zp_ok < ______");
    return true;
}

bool pd_t::gs_ok() {
    auto &attr_gs = attr()->precomputed_reductions_;

    if (!attr_gs.has_default_values(DNNL_ARG_DST)) { return false; }

    bool with_a_group_sums_ = !attr_gs.has_default_values(DNNL_ARG_A);
    bool with_b_group_sums_ = !attr_gs.has_default_values(DNNL_ARG_B);

    if ((attr_gs.get_data_type(DNNL_ARG_A) != data_type::s32)
            && with_a_group_sums_) {
        return false;
    }
    if ((attr_gs.get_data_type(DNNL_ARG_B) != data_type::s32)
            && with_b_group_sums_) {
        return false;
    }
    if (swap_ab_) std::swap(with_a_group_sums_, with_b_group_sums_);

    return true;
}

bool pd_t::scales_ok() {
    const auto &scales = attr()->scales_;
    int ndims = desc()->a_desc.ndims;
    using namespace data_type;

    for (auto s : {DNNL_ARG_A, DNNL_ARG_B, DNNL_ARG_C}) {
        if (scales.has_default_values(s) || scales.get(s).is_host_scalar())
            continue;
        const auto &x_scales = scales.get(s);

        auto mask = x_scales.get_mask();
        if (!(utils::one_of(mask, 0, mask_scalar, mask_per_oc, mask_per_ic)
                    || (utils::one_of(s, DNNL_ARG_A, DNNL_ARG_B)
                            && !x_scales.has_default_groups()
                            && valid_2d_mask(mask, ndims))
                    || (s == DNNL_ARG_C && !x_scales.has_default_groups()
                            && with_mx_scale() && valid_2d_mask(mask, ndims))))
            return false;

        // Nontrivial groups are only supported across one GEMM dimension.
        // Nontrivial: 1 < group size < dim size
        if (!x_scales.has_default_groups()) {
            const memory_desc_t *md = nullptr;
            switch (s) {
                // Swap descriptors to follow column major format
                case DNNL_ARG_A: md = &desc()->b_desc; break;
                case DNNL_ARG_B: md = &desc()->a_desc; break;
                case DNNL_ARG_C: md = &desc()->c_desc; break;
            }
            if (!md) gpu_error_not_expected();
            int count = 0;
            for (int i = 0; i < 2; i++) {
                int gs = x_scales.get_group(i);
                int dim = md->dims[md->ndims - 2 + i];
                if (1 < gs && gs < dim) count++;
            }
            if (count > 1) return false;

            // Dynamic Dst Quant only supported with `1x32` groups.
            if (s == DNNL_ARG_C && with_mx_scale() && x_scales.get_group(0) != 1
                    && x_scales.get_group(1) != 32)
                return false;
        }
    }

    return true;
}

bool pd_t::valid_2d_mask(int mask, int ndims, bool per_tensor_ok) {
    return (mask == full_tensor_mask() && per_tensor_ok)
            || utils::one_of(mask, (1 << (ndims - 1)),
                    (1 << (ndims - 1)) + (1 << (ndims - 2)));
}

dim_t pd_t::ld_binary(int idx) const {
    switch (binary_srcs_[idx].type) {
        case binary_src_t::binary: {
            const auto &entry = post_ops_.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return gemm_desc_t::get_ld(entry.binary.src1_desc);
        }
        case binary_src_t::bias: return desc()->ld_bias();
        case binary_src_t::prelu: {
            return gemm_desc_t::get_ld(prelu_wei_md);
        }

        default: return 1;
    }
}

dim_t pd_t::stride_binary(int idx, int stride) const {
    switch (binary_srcs_[idx].type) {
        case binary_src_t::binary:
        case binary_src_t::scales:
        case binary_src_t::bias: {
            const auto &entry = post_ops_.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return gemm_desc_t::get_stride(entry.binary.src1_desc, stride);
        }
        case binary_src_t::prelu: {
            return gemm_desc_t::get_stride(prelu_wei_md, stride);
        }
        default: return 0;
    }
}

dim_t pd_t::eff_scale_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, DNNL_ARG_A, DNNL_ARG_B));
    auto scale_md
            = ((DNNL_ARG_A == arg) ^ swap_ab()) ? a_scale_md_ : b_scale_md_;
    gpu_assert(memory_desc_wrapper(scale_md).is_plain())
            << "Expected plain scale_md_";
    if (scale_md.dims[idx] == 1) return 0;
    return scale_md.format_desc.blocking.strides[idx];
}

dim_t pd_t::eff_zp_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, DNNL_ARG_A, DNNL_ARG_B));
    auto zp_md = ((DNNL_ARG_A == arg) ^ swap_ab()) ? a_zp_md_ : b_zp_md_;
    gpu_assert(memory_desc_wrapper(zp_md).is_plain())
            << "Expected plain zp_md_";
    if (zp_md.dims[idx] == 1) return 0;
    return zp_md.format_desc.blocking.strides[idx];
}

dim_t pd_t::eff_gs_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, DNNL_ARG_A, DNNL_ARG_B));
    auto gs_md = ((DNNL_ARG_A == arg) ^ swap_ab()) ? a_gs_md_ : b_gs_md_;
    gpu_assert(memory_desc_wrapper(gs_md).is_plain())
            << "Expected plain gs_md_";
    if (gs_md.dims[idx] == 1) return 0;
    return gs_md.format_desc.blocking.strides[idx];
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
