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
#include "common/primitive_attr_quant.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
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
        return status::success;
    };

    if (!a_scales.has_default_values() && !a_scales.is_host_scalar()) {
        // Host scalar scale will be converted to Alpha
        bool converted;
        CHECK(maybe_convert_scales_to_postop(a_scale_md_, DNNL_ARG_A,
                a_scales.get_data_type(), a_scales.is_mx(), converted));
        if (converted) a_quant.scale_ndims = -1;
    }

    if (!b_scales.has_default_values() && !b_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(b_scale_md_, DNNL_ARG_B,
                b_scales.get_data_type(), b_scales.is_mx(), converted));
        if (converted) b_quant.scale_ndims = -1;
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
    a_quant.zp_ndims = quant_entry_ndims(a_zps, a_zp_md_, ndims - 2);
    b_quant.zp_ndims = quant_entry_ndims(b_zps, b_zp_md_, ndims - 1);
    a_quant.gs_ndims = quant_entry_ndims(a_gs, a_gs_md_, ndims - 2);
    b_quant.gs_ndims = quant_entry_ndims(b_gs, b_gs_md_, ndims - 1);
    a_quant.scale_ndims = quant_entry_ndims(a_scales, a_scale_md_, ndims - 2);
    b_quant.scale_ndims = quant_entry_ndims(b_scales, b_scale_md_, ndims - 1);
    c_quant.scale_ndims = quant_entry_ndims(c_scales, c_scale_md_, -1);

    a_quant.scales_type = a_scales.get_data_type();
    a_quant.zp_type = a_zps.get_data_type();
    a_quant.gs_type = a_gs.get_data_type();
    a_quant.force_gs = !a_gs.has_default_values();
    a_quant.zp_host_scalar = a_zp_host_scalar();
    // XXX, gemmstone support: if multiple grouped quantization attributes exist
    // for one matrix, they must have the same group size
    const auto &set_a_groups
            = [](quant_params &quant, const quant_entry_t &entry) -> status_t {
        int k_grp = entry.get_group(0);
        int m_grp = entry.get_group(1);
        if (quant.group_k > 0 && quant.group_k != k_grp)
            return status::unimplemented;
        quant.group_k = k_grp;
        if (quant.group_m > 0 && quant.group_m != m_grp)
            return status::unimplemented;
        quant.group_m = m_grp;
        return status::success;
    };
    if (!a_zps.has_default_groups()) CHECK(set_a_groups(a_quant, a_zps));
    if (!a_gs.has_default_groups()) CHECK(set_a_groups(a_quant, a_gs));
    if (!a_scales.has_default_groups()) CHECK(set_a_groups(a_quant, a_scales));

    b_quant.scales_type = b_scales.get_data_type();
    b_quant.zp_type = b_zps.get_data_type();
    b_quant.gs_type = b_gs.get_data_type();
    b_quant.force_gs = !b_gs.has_default_values();
    b_quant.zp_host_scalar = b_zp_host_scalar();
    const auto &set_b_groups
            = [](quant_params &quant, const quant_entry_t &entry) -> status_t {
        int n_grp = entry.get_group(0);
        int k_grp = entry.get_group(1);
        if (quant.group_n > 0 && quant.group_n != n_grp)
            return status::unimplemented;
        quant.group_n = n_grp;
        if (quant.group_k > 0 && quant.group_k != k_grp)
            return status::unimplemented;
        quant.group_k = k_grp;
        return status::success;
    };
    if (!b_zps.has_default_groups()) CHECK(set_b_groups(b_quant, b_zps));
    if (!b_gs.has_default_groups()) CHECK(set_b_groups(b_quant, b_gs));
    if (!b_scales.has_default_groups()) CHECK(set_b_groups(b_quant, b_scales));

    c_quant.scales_type = c_scales.get_data_type();
    if (!c_scales.has_default_groups()) {
        c_quant.group_m = c_scales.get_group(1);
        c_quant.group_n = c_scales.get_group(0);
        with_mx_scale_ = c_scales.is_mx();
    }
    return status::success;
}

bool pd_t::zp_ok() {
    using namespace data_type;
    auto &attr_zps = attr()->zero_points_;
    if (attr_zps.has_default_values()) return true;
    auto &a_zps = attr_zps.get(DNNL_ARG_A);
    auto &b_zps = attr_zps.get(DNNL_ARG_B);
    auto &c_zps = attr_zps.get(DNNL_ARG_C);

    // INT4 ZPs on SRC do not expand the range in a meaningful way, skipping
    if (utils::one_of(b_zps.get_data_type(), s4, u4)) return false;

    int ndims = desc()->a_desc.ndims;
    const bool a_int4 = utils::one_of(desc()->a_type(), s4, u4);
    const bool b_int4 = utils::one_of(desc()->b_type(), s4, u4);
    const bool weights_upconversion
            = wei_decomp_ || (a_int4 && dy_quant_enabled_);

    // Host scalar ZPs supported only for A & B
    if (c_zps.is_host_scalar()) return false;

    if (!a_zps.has_default_values()) {
        // Groups determine supported masks.
        if (!a_zps.has_default_groups()) {
            if (!valid_2d_mask(cmask_a_, ndims, weights_upconversion))
                return false;
            const auto a_q2d_group_n = a_zps.get_group(1);
            // Non-trivial N group unsupported.
            if (a_q2d_group_n != 1) return false;
            // Zero points with non-trivial groups only supported with
            // precomputed reductions or when target tensor is being dequantized.
            if (attr()->precomputed_reductions_.has_default_values(DNNL_ARG_B)
                    && dy_quant_enabled_ && b_int4 && !a_int4 && a_zp_2d())
                return false;
        } else {
            if (!utils::one_of(cmask_a_, 0, mask_per_oc, mask_per_ic))
                return false;
            // Weights zp can only be performantly enabled during upconversion
            // for cases that perform decompression.
            if (!wei_decomp_ && !a_int4 && a_scales_2d()) return false;
        }
    }

    if (!b_zps.has_default_values()) {
        // Groups determine supported masks.
        if (!b_zps.has_default_groups()) {
            if (!valid_2d_mask(cmask_b_, ndims, false)) return false;
            const auto b_q2d_group_n = b_zps.get_group(0);
            // Non-trivial M group unsupported.
            if (!utils::one_of(b_q2d_group_n, 1, desc()->n())) return false;
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            if (dy_quant_enabled_ && a_int4 && !b_int4 && b_zp_2d())
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

bool pd_t::gs_ok() {
    auto &attr_gs = attr()->precomputed_reductions_;
    if (attr_gs.has_default_values()) return true;

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

    return true;
}

bool pd_t::scales_ok() {
    const auto &scales = attr()->scales_;
    if (scales.has_default_values()) return true;
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

        if (!x_scales.has_default_groups()) {
            // Dynamic Dst Quant only supported with `1x32` groups.
            if (s == DNNL_ARG_C && with_mx_scale() && x_scales.get_group(0) != 1
                    && x_scales.get_group(1) != 32)
                return false;
            // Other dynamic quant unsupported
            if (x_scales.is_dynamic()) return false;
        }
    }

    return true;
}

bool pd_t::valid_2d_mask(int mask, int ndims, bool per_tensor_ok) {
    return (mask == full_tensor_mask() && per_tensor_ok)
            || utils::one_of(mask, (1 << (ndims - 1)),
                    (1 << (ndims - 1)) + (1 << (ndims - 2)));
}

status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops_) {
    using namespace gemmstone;
    problem.postOps = std::move(post_ops_);
    const auto &post_ops = problem.postOps;

    if (post_ops.len() > 0) {

        size_t po_count = post_ops.len();
        problem.Tbinary.reserve(po_count);
        problem.binary.reserve(po_count);
        problem.postOps.binaryRow = {};
        problem.postOps.binaryCol = {};
        problem.postOps.binaryBatch = {};
        problem.postOps.binaryTrans = {};

        if (problem.Ta == Type::f16) problem.Ts = Type::f32;
        if (problem.Ta.isF8() || problem.Tb.isF8()) problem.Ts = Type::f32;

        for (size_t i = 0; i < po_count; i++) {
            const auto &entry = post_ops[i];
            if (!entry.is_binary()) {
                problem.Tbinary.push_back(Type::invalid);
                problem.binary.push_back(MatrixAddressing {});
                continue;
            }

            auto &src_rmd = entry.as_binary().src1_desc;

            auto T = convert_dnnl_to_kernel_type(src_rmd.dt);
            bool is_multi_row = (src_rmd.broadcast_mask & 1) == 0;
            bool is_multi_col = (src_rmd.broadcast_mask & 2) == 0;

            bool is_compatible = src_rmd.inner_layout.empty();
            if (!is_compatible) return status::unimplemented;

            bool trans = is_multi_row && !src_rmd.inner_dim.is_innermost();

            problem.Tbinary.push_back(T);
            problem.postOps.binaryRow[i] = is_multi_row;
            problem.postOps.binaryCol[i] = is_multi_col;
            problem.postOps.binaryBatch[i] = src_rmd.ndims() >= 3;
            problem.postOps.binaryTrans[i] = trans;

            MatrixAddressing atype;
            atype.layout = trans ? MatrixLayout::T : MatrixLayout::N;
            atype.crosspack = 1;
            atype.packSize = 0;
            atype.setAlignment(T.size());

            problem.binary.push_back(atype);
        }
    }

    return status::success;
}

status_t pd_t::init_GEMMProblem(
        gemmstone::GEMMProblem &problem, const intel::engine_t *engine) const {
    // Set up problem structure.
    using namespace gemmstone;
    problem = {};

    auto hw = convert_dnnl_arch_to_ngen(engine->device_info()->gpu_arch());
    bool has_systolic
            = engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            || engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate);

    auto a_type = get_type(DNNL_ARG_A);
    auto b_type = get_type(DNNL_ARG_B);
    if (swap_ab_) std::swap(a_type, b_type);

    auto m = desc()->m();
    auto n = desc()->n();
    auto k = desc()->k();
    if (swap_ab_) std::swap(m, n);

    auto lda = ld(DNNL_ARG_A);
    auto ldb = ld(DNNL_ARG_B);
    if (swap_ab_) std::swap(lda, ldb);

    auto trans_a = eff_transa();
    auto align_a = nstl::max(eff_align_a(), (int)types::data_type_size(a_type));
    auto a_size = (trans_a ? m : k) * lda * types::data_type_size(a_type);

    auto trans_b = eff_transb();
    auto align_b = nstl::max(eff_align_b(), (int)types::data_type_size(b_type));
    auto b_size = (trans_b ? k : n) * ldb * types::data_type_size(b_type);

    bool int_acc = utils::one_of(a_type, data_type::s8, data_type::u8);
    int_acc &= !(a_grouped() || b_grouped());
    auto c_type = desc()->c_type();
    auto align_c
            = nstl::max(this->align_c(), (int)types::data_type_size(c_type));
    auto ldc = desc()->ldc();
    auto c_size = n * ldc * types::data_type_size(c_type);

    auto co_type = with_bias() ? desc()->bias_type()
            : with_sum_ab()    ? desc()->sum_ab_type
            : int_acc          ? data_type::s32
                               : desc()->c_type();

    // Choose accumulation data type.
    auto acc_type = int_acc
            ? data_type::s32
            : (utils::one_of(data_type::f64, a_type, b_type) ? data_type::f64
                                                             : data_type::f32);

    bool with_binary = (post_ops_.find(primitive_kind::binary) != -1)
            || (post_ops_.find(primitive_kind::prelu) != -1);

    bool need_x32_acc = with_binary || !IMPLICATION(with_sum_, sum_at_begin_);

    switch (attr()->acc_mode_) {
        case accumulation_mode::any:
            if (!need_x32_acc) acc_type = data_type::undef;
            break;
        case accumulation_mode::f16: acc_type = data_type::f16; break;
        case accumulation_mode::f32: acc_type = data_type::f32; break;
        case accumulation_mode::s32: acc_type = data_type::s32; break;
        default: break;
    }
    if (wei_decomp_) { acc_type = data_type::f32; }

    auto trans_co = trans_bias();
    if (swap_ab_) trans_co = !trans_co;
    auto dst_sround = with_sround_;
    bool c_offset = with_c_zero_points();
    bool bias = with_bias();

    jit::quant_params a_quant = this->a_quant;
    jit::quant_params b_quant = this->b_quant;

    if (swap_ab()) {
        std::swap(a_quant, b_quant);
        std::swap(a_quant.group_m, a_quant.group_n);
        std::swap(b_quant.group_m, b_quant.group_n);
    }

    problem.Ta = problem.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem.Tb = problem.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem.Ts = problem.Tc;
    problem.Tao = convert_dnnl_to_kernel_type(a_quant.zp_type);
    problem.Tbo = convert_dnnl_to_kernel_type(b_quant.zp_type);
    problem.Tco = convert_dnnl_to_kernel_type(co_type);
    problem.A.layout = trans_a ? MatrixLayout::T : MatrixLayout::N;
    problem.B.layout = trans_b ? MatrixLayout::T : MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;
    problem.A.crosspack = problem.B.crosspack = problem.C.crosspack = 1;
    problem.A.packSize = problem.B.packSize = problem.C.packSize = 0;
    problem.A.setAlignment(align_a);
    problem.B.setAlignment(align_b);
    problem.C.setAlignment(align_c);

    // Consolidate specialization logic to limit large buffer configurations
    bool needA64 = std::max({a_size, b_size, c_size})
            > std::numeric_limits<uint32_t>::max();
    problem.A.needA64 = needA64;
    problem.B.needA64 = needA64;
    problem.C.needA64 = needA64;

    if (batch_dims() > 0) {
        problem.batch = BatchMode::Strided;
        problem.batchDims = batch_dims();
    }
    if (a_quant.zp_ndims >= 0 || a_quant.zp_host_scalar)
        problem.aOffset = ABOffset::Calc;
    if (b_quant.zp_ndims >= 0 || b_quant.zp_host_scalar)
        problem.bOffset = ABOffset::Calc;
    problem.aoPtrDims = a_quant.zp_host_scalar ? -1 : a_quant.zp_ndims;
    problem.boPtrDims = b_quant.zp_host_scalar ? -1 : b_quant.zp_ndims;
    problem.AO.layout = MatrixLayout::N;
    problem.BO.layout
            = (problem.bOffset2D()) ? MatrixLayout::N : MatrixLayout::T;
    problem.AO.crosspack = problem.BO.crosspack = 1;
    problem.AO.packSize = problem.BO.packSize = 0;
    problem.A_scale = problem.Ag = problem.AO;
    problem.B_scale = problem.Bg = problem.BO;
    if (a_quant.zp_type != data_type::undef)
        problem.AO.setAlignment(int(types::data_type_size(a_quant.zp_type)));
    if (b_quant.zp_type != data_type::undef)
        problem.BO.setAlignment(int(types::data_type_size(b_quant.zp_type)));

    problem.asPtrDims = a_quant.scale_ndims;
    problem.bsPtrDims = b_quant.scale_ndims;
    problem.aqGroupK = a_quant.group_k;
    problem.bqGroupK = b_quant.group_k;
    problem.aqGroupM = a_quant.group_m;
    problem.bqGroupN = b_quant.group_n;
    if (a_quant.scales_type != data_type::undef) {
        problem.Ta_scale = convert_dnnl_to_kernel_type(a_quant.scales_type);
        problem.A_scale.layout = swap_ab() ? MatrixLayout::T : MatrixLayout::N;
        problem.A_scale.setAlignment(
                int(types::data_type_size(a_quant.scales_type)));
    }
    if (b_quant.scales_type != data_type::undef) {
        problem.Tb_scale = convert_dnnl_to_kernel_type(b_quant.scales_type);
        problem.B_scale.layout = swap_ab() ? MatrixLayout::T : MatrixLayout::N;
        problem.B_scale.setAlignment(
                int(types::data_type_size(b_quant.scales_type)));
    }

    if (c_quant.scales_type != data_type::undef) {
        problem.csPtrDims = c_quant.scale_ndims;
        problem.cMXScale = with_mx_scale_;
        problem.Tc_scale = convert_dnnl_to_kernel_type(c_quant.scales_type);
        problem.cqGroupM = c_quant.group_m;
        problem.cqGroupN = c_quant.group_n;
    }

    if (problem.Ta_ext.isInt4() && problem.Tb_ext.isInt8()
            && a_quant.zp_ndims >= 0)
        problem.Ta = Type::s8;
    if (problem.Tb_ext.isInt4() && problem.Ta_ext.isInt8()
            && b_quant.zp_ndims >= 0)
        problem.Tb = Type::s8;

    if (problem.Ta.isInteger()) problem.Ts = Type::f32;

    if (alpha() == 1.0f) problem.alpha = alpha();
    if (beta() == 0.0f || beta() == 1.0f) problem.beta = beta();

    gpu_post_ops_t gpu_post_ops;
    CHECK(gpu_post_ops_t::make(
            gpu_post_ops, post_ops_, dst_md(), get_post_op_specializations()));

    CHECK(transfer_post_ops(problem, std::move(gpu_post_ops)));
    if (swap_ab()) problem.postOps.transpose();

    auto reduce_ab = sum_ab();
    if (c_offset || bias || reduce_ab != sum_ab::sum_none) {
        assert(!(c_offset && bias));
        if (bias) problem.cOffset = COffset::Pre;
        if (c_offset) problem.cOffset = COffset::Post;
        problem.CO.crosspack = 1;
        problem.CO.alignment = problem.C.alignment;
        problem.CO.layout = trans_co ? MatrixLayout::T : MatrixLayout::N;
    }

    problem.sumA = (reduce_ab == sum_ab::sum_b_col);
    problem.sumB = (reduce_ab == sum_ab::sum_a_row);
    if (swap_ab_) std::swap(problem.sumA, problem.sumB);
    problem.forceGroupSumsA = a_quant.force_gs;
    problem.forceGroupSumsB = b_quant.force_gs;

    problem.postOps.cStochasticRound = dst_sround;

    if (problem.needsAGroupSums() || problem.needsBGroupSums())
        problem.autoTypeConversions(hw, has_systolic);

    if (problem.needsAGroupSums()) {
        data_type_t gs_dt = a_quant.gs_type == data_type::undef
                ? data_type::s32
                : a_quant.gs_type;
        problem.Tag = convert_dnnl_to_kernel_type(gs_dt);
        problem.Ag.layout = MatrixLayout::N;
        problem.Ag.setAlignment(problem.Tag.paddedSize());
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
    }
    if (problem.needsBGroupSums()) {
        data_type_t gs_dt = b_quant.gs_type == data_type::undef
                ? data_type::s32
                : b_quant.gs_type;
        problem.Tbg = convert_dnnl_to_kernel_type(gs_dt);
        problem.Bg.layout = MatrixLayout::N;
        problem.Bg.setAlignment(problem.Tbg.paddedSize());
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
    }
    return status::success;
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
