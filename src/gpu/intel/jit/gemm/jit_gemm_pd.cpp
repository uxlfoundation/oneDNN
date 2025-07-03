/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/jit/gemm/jit_gemm_pd.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

status_t jit_gemm_pd_t::init_post_ops() {
    using namespace primitive_kind;
    using namespace alg_kind;
    using namespace data_type;

    const auto d = desc();

    // Examine post-ops and remember binary srcs.
    post_ops_ = attr()->post_ops_;
    binary_srcs_.reserve(post_ops_.len() + 4);

    bool ok = true;
    int prelu_count = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &e = post_ops_.entry_[i];
        switch (e.kind) {
            case binary:
                ok &= supported_binary_op(e.binary.alg)
                        && is_md_gemm_compatible_plain_format(
                                &e.binary.src1_desc);
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::binary, int(i)});
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
                break;
            case prelu:
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::prelu, int(i)});
                ok &= get_prelu_md(e.prelu.mask, dst_md()->dims, prelu_wei_md,
                              dst_md()->ndims)
                        == status::success;
                prelu_count++;
                ok &= prelu_count <= 1;
                break;
            default: return status::unimplemented;
        }
    }

    if (!ok) return status::unimplemented;

    // If scales are present, convert them and any bias to binary post-ops.
    //   Exception: 2D scales.
    // Also convert bias to binary post-op if dst zp are present.
    const auto *wei_scales = &attr()->scales_.get(DNNL_ARG_WEIGHTS);
    const auto *src_scales = &attr()->scales_.get(DNNL_ARG_SRC);
    const auto *c_scales = &attr()->scales_.get(DNNL_ARG_DST);

    bias_via_binary_ = (desc()->bias_type() != data_type::undef)
            && (d->bias_desc.ndims >= 1 || !wei_scales->has_default_values()
                    || !src_scales->has_default_values()
                    || !attr()->zero_points_.has_default_values(DNNL_ARG_DST));
    if (bias_via_binary_) {
        auto status = post_ops_.prepend_binary(binary_add, &d->bias_desc);
        if (status != status::success) return status;
        binary_srcs_.insert(
                binary_srcs_.begin(), binary_src_t {binary_src_t::bias, 0});
    }

    if (!wei_scales->has_default_values()) {
        const auto &mask = wei_scales->get_mask();
        bool convert = (mask == 0 || math::is_pow2(mask));
        if (!wei_scales->has_default_groups())
            convert |= (wei_scales->get_group(0) >= d->k());
        if (convert) {
            dim_t dims = {(mask > 0) ? d->m() : 1};
            CHECK(memory_desc_init_by_tag(wei_scales_md, 1, &dims,
                    wei_scales->get_data_type(), format_tag::a));

            CHECK(post_ops_.prepend_binary(binary_mul, &wei_scales_md));

            binary_srcs_.insert(binary_srcs_.begin(),
                    binary_src_t {binary_src_t::scales, DNNL_ARG_WEIGHTS});
            asc_dims_ = -1;
        }
    }
    if (!src_scales->has_default_values()) {
        const auto &mask = src_scales->get_mask();
        bool convert = (mask == 0);
        if (!src_scales->has_default_groups()) {
            convert |= (src_scales->get_group(1) >= d->k());
        }
        if (convert) {
            if (mask == 0) {
                dim_t dims = 1;
                CHECK(memory_desc_init_by_tag(src_scales_md, 1, &dims,
                        src_scales->get_data_type(), format_tag::a));
            } else if (!src_scales->has_default_groups()) {
                // TODO: is it inverted?
                int n_group = src_scales->get_group(0);
                int k_group = src_scales->get_group(1);
                int ndims = d->c_desc.ndims;
                std::vector<dim_t> dims;
                for (int i = ndims - 3; i >= 0; --i) {
                    if (mask & (i + 1)) dims.push_back(d->c_desc.dims[i]);
                }
                dims.push_back((mask & (d->batch() > 1 ? 2 : 1))
                                ? d->n() / n_group
                                : 1);
                dims.push_back(d->k() / k_group);
                CHECK(memory_desc_init_by_tag(src_scales_md, (int)dims.size(),
                        dims.data(), src_scales->get_data_type(),
                        get_abx_tag((int)dims.size())));
            } else {
                dim_t dims[] = {d->n(), 1};
                CHECK(memory_desc_init_by_tag(src_scales_md, 2, dims,
                        src_scales->get_data_type(), format_tag::ab));
            }

            CHECK(post_ops_.prepend_binary(binary_mul, &src_scales_md));

            binary_srcs_.insert(binary_srcs_.begin(),
                    binary_src_t {binary_src_t::scales, DNNL_ARG_SRC});
            bsc_dims_ = -1;
        }
    }
    if (!c_scales->has_default_values()) {
        const auto &mask = c_scales->get_mask();
        bool convert = (mask == 0 || math::is_pow2(mask));
        if (!c_scales->has_default_groups())
            convert |= (c_scales->get_group(0) >= d->m());
        if (convert) {
            ok = ok && (mask == 0 || mask == (1 << (d->c_desc.ndims - 1)));
            dim_t dims = {(mask > 0) ? d->m() : 1};
            CHECK(memory_desc_init_by_tag(c_scales_md, 1, &dims,
                    c_scales->get_data_type(), format_tag::a));

            CHECK(post_ops_.append_binary(binary_div, &c_scales_md));

            binary_srcs_.push_back(
                    binary_src_t {binary_src_t::scales, DNNL_ARG_DST});
        }
    }

    return status::success;
}

// Obtain dimension count for gemmstone (common scales give count 0).
int jit_gemm_pd_t::quant_entry_ndims(
        const quant_entry_t &entry, const memory_desc_t &md) const {
    if (entry.has_default_values()) return -1;
    int mask = entry.get_mask();
    if (mask == 0) return 0;
    if (entry.has_default_groups()) return mask > 0;
    int count = 0;
    for (int i = md.ndims - 1; i >= 0; --i) {
        if (!(mask & (1 << i))) continue;
        bool batch_dim = i < batch_dims();
        if ((batch_dim && md.dims[i] > 1)
                || (!batch_dim
                        && (md.dims[i] / entry.get_group(i - batch_dims()) > 1
                                || count)))
            ++count;
    }
    return count;
}

int jit_gemm_pd_t::quant_entry_group_prod(const quant_entry_t &attr) const {
    int ret = 1;
    if (attr.has_default_groups()) return ret;
    for (int i = 0; i < 2; ++i)
        ret *= attr.get_group(i);
    return ret;
}

bool jit_gemm_pd_t::dy_quant_enabled() {
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

bool jit_gemm_pd_t::wei_decomp() {
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

bool jit_gemm_pd_t::quant_enabled() {
    return wei_decomp() || dy_quant_enabled();
}

void jit_gemm_pd_t::init_attrs() {
    wei_decomp_ = wei_decomp();
    dy_quant_enabled_ = dy_quant_enabled();
    quant_enabled_ = quant_enabled();
    const auto &d = desc();

    auto &attr_zps = attr()->zero_points_;
    cmask_a_ = attr_zps.get(DNNL_ARG_A).get_mask();
    cmask_b_ = attr_zps.get(DNNL_ARG_B).get_mask();
    cmask_c_ = attr_zps.get(DNNL_ARG_C).get_mask();

    // Swap descriptors to follow column major format.
    ao_dims_ = quant_entry_ndims(attr_zps.get(DNNL_ARG_A), d->b_desc);
    bo_dims_ = quant_entry_ndims(attr_zps.get(DNNL_ARG_B), d->a_desc);
    if (wei_zp_2d()) { wei_q2d_group_k_ = attr_zps.get_group(DNNL_ARG_A, 0); }
    if (src_zp_2d()) { src_q2d_group_k_ = attr_zps.get_group(DNNL_ARG_B, 0); }

    const auto &scales = attr()->scales_;
    const auto *src_scales = &attr()->scales_.get(DNNL_ARG_B);
    const auto *wei_scales = &attr()->scales_.get(DNNL_ARG_A);
    asc_dims_ = quant_entry_ndims(scales.get(DNNL_ARG_A), d->b_desc);
    bsc_dims_ = quant_entry_ndims(scales.get(DNNL_ARG_B), d->a_desc);
    wei_scales_group_k_ = wei_scales->get_group(0);
    src_scales_group_k_ = src_scales->get_group(1);

    wei_scales_type_ = wei_scales->get_data_type();
    if (wei_scales_2d()) {
        if (!wei_zp_2d()) wei_q2d_group_k_ = wei_scales->get_group(0);
    }

    src_scales_type_ = src_scales->get_data_type();
    if (src_scales_2d()) { src_q2d_group_k_ = src_scales->get_group(1); }
}

bool jit_gemm_pd_t::zp_ok() {
    auto &attr_zps = attr()->zero_points_;
    int ndims = desc()->a_desc.ndims;
    const auto d = desc();
    using namespace data_type;

    if (!attr_zps.has_default_values(DNNL_ARG_A)) {
        // Groups determine supported masks.
        if (!attr_zps.has_default_groups(DNNL_ARG_A)) {
            if (!valid_2d_mask(cmask_a_, ndims, false)) return false;
            const auto wei_q2d_group_n = attr_zps.get_group(DNNL_ARG_A, 1);
            // Non-trivial N group unsupported.
            if (wei_q2d_group_n != 1) return false;
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            if (dy_quant_enabled_ && !utils::one_of(d->a_type(), s4, u4)
                    && wei_zp_2d())
                return false;
        } else {
            if (!utils::one_of(cmask_a_, 0, mask_per_oc, mask_per_ic))
                return false;
            // Weights zp can only be performantly enabled during upconversion
            // for cases that perform decompression.
            if (!wei_decomp_ && !utils::one_of(d->a_type(), s4, u4)
                    && wei_scales_2d())
                return false;
        }
    }

    if (!attr_zps.has_default_values(DNNL_ARG_B)) {
        // Groups determine supported masks.
        if (!attr_zps.has_default_groups(DNNL_ARG_B)) {
            if (!valid_2d_mask(cmask_b_, ndims, false)) return false;

            const auto src_q2d_group_n = attr_zps.get_group(DNNL_ARG_B, 0);
            zp_group_k_b_ = attr_zps.get_group(DNNL_ARG_B, 1);
            // Non-trivial M group unsupported.
            if (!utils::one_of(src_q2d_group_n, 1, desc()->n())) return false;
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            if (dy_quant_enabled_ && !utils::one_of(d->b_type(), s4, u4)
                    && src_zp_2d())
                return false;
        } else {
            if (!utils::one_of(
                        cmask_b_, 0, mask_scalar, mask_per_oc | mask_per_ic))
                return false;
        }
    }

    if (!attr_zps.has_default_values(DNNL_ARG_C)) {
        if (!utils::one_of(cmask_c_, 0, mask_scalar, mask_per_oc)) return false;
    }

    return true;
}

bool jit_gemm_pd_t::scales_ok() {
    const auto *wei_scales = &attr()->scales_.get(DNNL_ARG_A);
    const auto *src_scales = &attr()->scales_.get(DNNL_ARG_B);
    int ndims = desc()->a_desc.ndims;
    using namespace data_type;

    for (auto s : {DNNL_ARG_A, DNNL_ARG_B, DNNL_ARG_C}) {
        if (attr()->scales_.has_default_values(s)) continue;

        auto mask = attr()->scales_.get_mask(s);
        if (!(utils::one_of(mask, 0, mask_scalar, mask_per_oc, mask_per_ic)
                    || (s == DNNL_ARG_A && !wei_scales->has_default_groups()
                            && valid_2d_mask(mask, ndims))
                    || (s == DNNL_ARG_B && !src_scales->has_default_groups()
                            && valid_2d_mask(mask, ndims))))
            return false;
    }

    if (wei_scales_2d()) {
        if (wei_q2d_group_k_ != wei_scales_group_k_) return false;
        // Non-trivial N group unsupported.
        if (wei_scales->get_group(1) != 1) return false;
    }

    if (src_scales_2d()) {
        int cmask_b_sc_ = attr()->scales_.get_mask(DNNL_ARG_B);
        if (!dy_quant_enabled_
                || (!utils::one_of(eff_a_type(), s4, u4)
                        && ((cmask_b_sc_ != full_tensor_mask())
                                || bsc_dims_ > 2)))
            return false;
    } else {
        if (!src_scales->has_default_values() && src_scales->get_mask() != 0
                && wei_scales_group_k_ > desc()->k())
            return false;
    }

    return true;
}

bool jit_gemm_pd_t::valid_2d_mask(int mask, int ndims, bool per_tensor_ok) {
    return (mask == full_tensor_mask() && per_tensor_ok)
            || utils::one_of(mask, (1 << (ndims - 1)),
                    (1 << (ndims - 1)) + (1 << (ndims - 2)));
}

dim_t jit_gemm_pd_t::ld_binary(int idx) const {
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

dim_t jit_gemm_pd_t::stride_binary(int idx, int stride) const {
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

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
