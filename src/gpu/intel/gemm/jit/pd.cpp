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
#include "gpu/intel/gemm/utils.hpp"
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

// Obtain dimension count for gemmstone (common scales give count 0).
int pd_t::quant_entry_ndims(
        const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx) {
    if (entry.has_default_values()) return -1;
    if (qmd.ndims < 2) return 0;

    // Count the number of nontrivial (dim > 1) dimensions present
    int count = 0;
    for (int i = qmd.ndims - 2; i < qmd.ndims; ++i) {
        if (qmd.dims[i] > 1) { count++; }
    }

    if (count == 0) return 0;

    // for gemmstone, 1D quantization implies a full column vector
    // (i.e. not on the K dimension). If quantization varies over K,
    // we have to send these as 2D
    if (k_idx >= 0 && count == 1 && qmd.dims[k_idx] > 1) return 2;

    // If M/N quantization requires broadcast it is unsupported
    // by post-ops and must be submitted as 2D.
    int gcount = 0;
    for (int i = 0; i < 2; ++i) {
        if (entry.get_group(i) > 1) { gcount++; }
    }

    if (gcount == 2) return 2;

    return count;
}

status_t pd_t::init_post_ops(impl::engine_t *engine) {
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
                if (e.binary.alg == binary_prelu) {
                    // Canonicalized prelu: user buffer still keyed under
                    // DNNL_ARG_WEIGHTS, so route via prelu.
                    binary_srcs_.push_back(
                            binary_src_t {binary_src_t::prelu, int(i)});
                    prelu_wei_md = e.binary.src1_desc;
                    prelu_count++;
                    ok &= prelu_count <= 1;
                } else {
                    ok &= supported_binary_op(e.binary.alg)
                            && is_md_gemm_compatible_plain_format(
                                    &e.binary.src1_desc);
                    binary_srcs_.push_back(
                            binary_src_t {binary_src_t::binary, int(i)});
                }
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
                // Expected to be canonicalized to binary upstream.
                VDISPATCH_GEMM(false,
                        "%s: prelu post-op not canonicalized",
                        VERBOSE_UNSUPPORTED_POSTOP);
                break;
            default: VDISPATCH_GEMM(false, VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    VDISPATCH_GEMM(ok, VERBOSE_UNSUPPORTED_POSTOP);

    // If scales are present, convert them and any bias to binary post-ops.
    //   Exception: 2D scales.
    // Also convert bias to binary post-op if dst zp are present.
    const auto &a_scales = attr()->scales_.get(gemm_arg::A);
    const auto &b_scales = attr()->scales_.get(gemm_arg::B);
    const auto &c_scales = attr()->scales_.get(gemm_arg::C);

    bias_via_binary_ = (desc()->bias_type() != data_type::undef)
            && (d->bias_md().ndims >= 1 || !a_scales.has_default_values()
                    || !b_scales.has_default_values()
                    || !attr()->zero_points_.has_default_values(gemm_arg::C));
    if (bias_via_binary_) {
        VDISPATCH_GEMM_SC(post_ops_.prepend_binary(binary_add, &d->bias_md()),
                "%s: bias via binary post-op", VERBOSE_UNSUPPORTED_POSTOP);
        binary_srcs_.insert(
                binary_srcs_.begin(), binary_src_t {binary_src_t::bias, 0});
    }
    non_scale_po_ |= bias_via_binary_;

    auto maybe_convert_scales_to_postop
            = [this, engine](const memory_desc_t &scale_md, int arg,
                      int scale_ndims, bool mx, bool &converted) -> status_t {
        auto ndims = desc()->c_md().ndims;
        // Scales on A/B can be converted to postops if the scales md has
        // K=1 and M/N is not bcast.
        converted = false;
        if (scale_ndims > 1) return status::success;
        int inner_dim = (arg == gemm_arg::A ? ndims - 1 : ndims - 2);
        bool convert = (scale_md.dims[inner_dim] <= 1) || (arg == gemm_arg::C);
        convert &= !mx;
        if (convert) {
            if (arg == gemm_arg::C) {
                VDISPATCH_GEMM_SC(
                        post_ops_.append_binary(binary_div, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::scales, arg});
            } else {
                VDISPATCH_GEMM_SC(
                        post_ops_.prepend_binary(binary_mul, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
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
        CHECK(maybe_convert_scales_to_postop(a_scale_md_, gemm_arg::A,
                a_scale_ndims(), a_scales.is_mx(), converted));
        if (converted) a_scale_ndims_override_ = -1;
    }

    if (!b_scales.has_default_values() && !b_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(b_scale_md_, gemm_arg::B,
                b_scale_ndims(), b_scales.is_mx(), converted));
        if (converted) b_scale_ndims_override_ = -1;
    }

    // Host-scalar c_scale folds into alpha at exec; when there are
    // additive post-ops (incl. bias_via_binary_), convert to a
    // binary_div post-op so dst_scale also scales bias.
    bool try_c_scale = !c_scales.is_host_scalar() || num_orig_postops > 0
            || bias_via_binary_;
    if (!c_scales.has_default_values() && try_c_scale) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(c_scale_md_, gemm_arg::C,
                c_scale_ndims(), c_scales.is_mx(), converted));
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
    auto c_ok = utils::one_of(d->c_type(), f32, f16, bf16, f8_e5m2, f8_e4m3);
    if (!c_ok) return false;
    auto int_low = [](data_type_t t) {
        return utils::one_of(t, u8, s8, s4, u4, f8_e4m3, f8_e5m2, f4_e2m1,
                f4_e3m0);
    };
    auto float_hi = [](data_type_t t) {
        return utils::one_of(t, f16, f32, bf16, f8_e5m2, f8_e4m3);
    };
    // Symmetric across swap_ab: int weights paired with float src in
    // either slot.
    auto int_t = int_low(d->a_type()) ? d->a_type()
            : int_low(d->b_type())    ? d->b_type()
                                      : data_type::undef;
    auto float_t = float_hi(d->a_type()) ? d->a_type()
            : float_hi(d->b_type())      ? d->b_type()
                                         : data_type::undef;
    if (int_t == data_type::undef || float_t == data_type::undef
            || int_t == float_t)
        return false;
    return types::data_type_bits(int_t) < types::data_type_bits(float_t)
            && attr()->mayiconvert(int_t, f32);
}

bool pd_t::quant_enabled() {
    return wei_decomp() || dy_quant_enabled();
}

status_t pd_t::init_attrs(impl::engine_t *engine) {
    wei_decomp_ = wei_decomp();
    dy_quant_enabled_ = dy_quant_enabled();
    quant_enabled_ = quant_enabled();

    const auto &attr_zps = attr()->zero_points_;
    const auto a_zps = attr_zps.get(gemm_arg::A);
    const auto b_zps = attr_zps.get(gemm_arg::B);
    const auto c_zps = attr_zps.get(gemm_arg::C);

    const auto &attr_gs = attr()->precomputed_reductions_;
    const auto a_gs = attr_gs.get(gemm_arg::A);
    const auto b_gs = attr_gs.get(gemm_arg::B);

    const auto &scales = attr()->scales_;
    const auto a_scales = scales.get(gemm_arg::A);
    const auto b_scales = scales.get(gemm_arg::B);
    const auto c_scales = scales.get(gemm_arg::C);

    cmask_a_ = a_zps.get_mask();
    cmask_b_ = b_zps.get_mask();
    cmask_c_ = c_zps.get_mask();

    // XXX, gemmstone support: if multiple grouped quantization attributes
    // exist for one matrix, they must have the same group size (default
    // unset == 0).
    const auto set_if_consistent
            = [this, engine](int &dst, int new_dim, int arg) -> status_t {
        VDISPATCH_GEMM(utils::one_of(dst, 0, new_dim),
                "%s: %s quantization attrs with different group sizes",
                VERBOSE_UNSUPPORTED_ATTR, arg2str(arg).c_str());
        dst = new_dim;
        return status::success;
    };
    // a_md has K at ndims-1, kernel-M at ndims-2.
    const auto add_a = [&](const quant_entry_t &entry) -> status_t {
        if (entry.has_default_groups()) return status::success;
        CHECK(set_if_consistent(
                a_group_m_, into<int>(entry.get_group(0)), gemm_arg::A));
        CHECK(set_if_consistent(
                a_group_k_, into<int>(entry.get_group(1)), gemm_arg::A));
        return status::success;
    };
    CHECK(add_a(a_zps));
    CHECK(add_a(a_gs));
    CHECK(add_a(a_scales));

    // b_md has K at ndims-2, kernel-N at ndims-1.
    const auto add_b = [&](const quant_entry_t &entry) -> status_t {
        if (entry.has_default_groups()) return status::success;
        CHECK(set_if_consistent(
                b_group_k_, into<int>(entry.get_group(0)), gemm_arg::B));
        CHECK(set_if_consistent(
                b_group_n_, into<int>(entry.get_group(1)), gemm_arg::B));
        return status::success;
    };
    CHECK(add_b(b_zps));
    CHECK(add_b(b_gs));
    CHECK(add_b(b_scales));

    if (!c_scales.has_default_groups()) {
        c_group_m_ = into<int>(c_scales.get_group(0));
        c_group_n_ = into<int>(c_scales.get_group(1));
        with_mx_scale_ = c_scales.is_mx();
    }

    return status::success;
}

status_t pd_t::zp_ok(impl::engine_t *engine) {
    using namespace data_type;
    auto &attr_zps = attr()->zero_points_;
    if (attr_zps.has_default_values()) return status::success;
    auto &a_zps = attr_zps.get(gemm_arg::A);
    auto &b_zps = attr_zps.get(gemm_arg::B);
    auto &c_zps = attr_zps.get(gemm_arg::C);

    int ndims = desc()->a_md().ndims;
    const bool a_int4 = utils::one_of(desc()->a_type(), s4, u4);
    const bool b_int4 = utils::one_of(desc()->b_type(), s4, u4);
    const bool weights_upconversion
            = wei_decomp_ || (a_int4 && dy_quant_enabled_);

    if (!a_zps.has_default_values()) {
        // Groups determine supported masks.
        if (!a_zps.has_default_groups()) {
            VDISPATCH_GEMM(valid_2d_mask(cmask_a_, ndims, weights_upconversion),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto a_q2d_group_n = a_zps.get_group(1);
            // Non-trivial N group unsupported.
            VDISPATCH_GEMM(a_q2d_group_n == 1,
                    "%s: Grouped N dimension on A matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Zero points with non-trivial groups only supported with
            // precomputed reductions or when target tensor is being dequantized.
            bool has_prB = !attr()->precomputed_reductions_.has_default_values(
                    gemm_arg::B);
            // TODO: Re-examine this condition
            bool is_dequantized = !dy_quant_enabled_ || !b_int4 || a_int4;
            VDISPATCH_GEMM(IMPLICATION(a_zp_2d(), is_dequantized || has_prB),
                    "%s: Nontrivial groups on A matrix, and no precomputed "
                    "reductions or dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_GEMM(utils::one_of(cmask_a_, 0, mask_per_oc, mask_per_ic),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            // Weights zp can only be performantly enabled during upconversion
            // for cases that perform decompression.
            VDISPATCH_GEMM(IMPLICATION(a_scales_2d(),
                                   !(b_int4 && !wei_decomp_ && !a_int4)),
                    "%s: 2D scales on A matrix, but no weights "
                    "decompression",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!b_zps.has_default_values()) {
        // INT4 ZPs on SRC do not expand the range in a meaningful way, skipping
        VDISPATCH_GEMM(!utils::one_of(b_zps.get_data_type(), s4, u4),
                VERBOSE_UNSUPPORTED_ZP_CFG);

        // Groups determine supported masks.
        if (!b_zps.has_default_groups()) {
            VDISPATCH_GEMM(valid_2d_mask(cmask_b_, ndims, false),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto b_q2d_group_n = b_zps.get_group(0);
            // Non-trivial M group unsupported.
            VDISPATCH_GEMM(utils::one_of(b_q2d_group_n, 1, desc()->n()),
                    "%s: Nontrivial N groups on B matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            // TODO: Re-examine this condition
            bool is_dequantized = !dy_quant_enabled_ || !a_int4 || b_int4;
            VDISPATCH_GEMM(IMPLICATION(b_zp_2d(), is_dequantized),
                    "%s: Grouped B zero points, and no dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_GEMM(utils::one_of(cmask_b_, 0, mask_scalar,
                                   mask_per_oc | mask_per_ic),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!attr_zps.has_default_values(gemm_arg::C)) {
        VDISPATCH_GEMM(
                IMPLICATION(!c_zps.is_host_scalar(),
                        utils::one_of(cmask_c_, 0, mask_scalar, mask_per_oc)),
                "%s: unsupported C mask", VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    return status::success;
}

status_t pd_t::gs_ok(impl::engine_t *engine) {
    auto &attr_gs = attr()->precomputed_reductions_;
    if (attr_gs.has_default_values()) return status::success;

    VDISPATCH_GEMM(attr_gs.has_default_values(gemm_arg::C),
            VERBOSE_UNSUPPORTED_PR_CFG);

    bool with_a_group_sums_ = !attr_gs.has_default_values(gemm_arg::A);
    bool with_b_group_sums_ = !attr_gs.has_default_values(gemm_arg::B);

    VDISPATCH_GEMM(IMPLICATION(with_a_group_sums_,
                           attr_gs.get_data_type(gemm_arg::A) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_GEMM(IMPLICATION(with_b_group_sums_,
                           attr_gs.get_data_type(gemm_arg::B) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);

    return status::success;
}

status_t pd_t::scales_ok(impl::engine_t *engine) {
    const auto &scales = attr()->scales_;
    if (scales.has_default_values()) return status::success;
    int ndims = desc()->a_md().ndims;
    using namespace data_type;

    for (auto s : {gemm_arg::A, gemm_arg::B}) {
        if (scales.has_default_values(s) || scales.get(s).is_host_scalar())
            continue;
        const auto &x_scales = scales.get(s);

        auto mask = x_scales.get_mask();
        bool supportedMask
                = utils::one_of(mask, 0, mask_scalar, mask_per_oc, mask_per_ic)
                || (!x_scales.has_default_groups()
                        && valid_2d_mask(mask, ndims));
        VDISPATCH_GEMM(supportedMask, "%s: unsupported A/B mask",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    const auto &dst_scales = scales.get(gemm_arg::C);
    if (!dst_scales.has_default_values() && !dst_scales.is_host_scalar()) {
        auto mask = dst_scales.get_mask();
        bool supportedMask
                = utils::one_of(mask, 0, mask_scalar, mask_per_oc, mask_per_ic)
                || (!dst_scales.has_default_groups() && with_mx_scale()
                        && valid_2d_mask(mask, ndims));
        VDISPATCH_GEMM(supportedMask, "%s: unsupported C mask",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    if (!dst_scales.has_default_values() && with_mx_scale()) {
        // Dynamic Dst Quant only supported with `1x32` groups.
        VDISPATCH_GEMM(dst_scales.get_group(0) == 1
                        && dst_scales.get_group(1) == 32
                        && arch_ >= compute::gpu_arch_t::xe_hpc,
                "%s: unsupported mx_scale groups",
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        // M+N dimensions must have trivial strides for Dynamic Dst Quant
        auto md = &desc()->c_md();
        auto strides = md->format_desc.blocking.strides;
        VDISPATCH_GEMM(strides[md->ndims - 1] == 1,
                "%s: unsupported mx_scale strides",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_GEMM(strides[md->ndims - 2] == md->dims[md->ndims - 1],
                "%s: unsupported mx_scale strides",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    return status::success;
}

bool pd_t::valid_2d_mask(int mask, int ndims, bool per_tensor_ok) {
    return (mask == full_tensor_mask() && per_tensor_ok)
            || utils::one_of(mask, (1 << (ndims - 1)),
                    (1 << (ndims - 1)) + (1 << (ndims - 2)));
}

status_t transfer_post_ops(gemmstone::GEMMProblem &problem,
        gpu_post_ops_t &&post_ops_) {
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
            constexpr int row_bit = 2;
            constexpr int col_bit = 1;
            bool is_multi_row = (src_rmd.broadcast_mask & row_bit) == 0;
            bool is_multi_col = (src_rmd.broadcast_mask & col_bit) == 0;

            bool is_compatible = src_rmd.inner_layout.empty();
            if (!is_compatible) return status::unimplemented;

            bool trans = src_rmd.inner_dim.is_innermost();

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

    problem.product = get_ngen_product(*engine->device_info());
    bool has_systolic
            = engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            || engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate);

    auto a_type = get_type(gemm_arg::A);
    auto b_type = get_type(gemm_arg::B);

    auto m = desc()->m();
    auto n = desc()->n();
    auto k = desc()->k();

    auto align_a = align(gemm_arg::A);
    auto align_b = align(gemm_arg::B);

    auto lda = ld(gemm_arg::A);
    auto ldb = ld(gemm_arg::B);

    auto trans_a = this->trans_a();
    auto trans_b = this->trans_b();

    align_a = nstl::max(align_a, (int)types::data_type_size(a_type));
    auto a_size = (trans_a ? m : k) * lda * types::data_type_size(a_type);

    align_b = nstl::max(align_b, (int)types::data_type_size(b_type));
    auto b_size = (trans_b ? k : n) * ldb * types::data_type_size(b_type);

    bool int_acc = utils::one_of(a_type, data_type::s8, data_type::u8)
            || (types::is_integral_dt(a_type) && types::is_integral_dt(b_type));
    int_acc &= !(a_grouped() || b_grouped());
    auto c_type = desc()->c_type();
    auto align_c
            = nstl::max(align(gemm_arg::C), (int)types::data_type_size(c_type));
    auto ldc = desc()->ldc();
    auto c_size = n * ldc * types::data_type_size(c_type);

    auto co_type = with_bias() ? desc()->bias_type()
            : with_sum_ab()    ? desc()->sum_ab_type()
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

    auto trans_co = !trans_bias();
    auto dst_sround = with_sround_;
    bool c_offset = with_c_zero_points();
    bool bias = with_bias();

    problem.Ta = problem.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem.Tb = problem.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem.Ts = problem.Tc;
    problem.Tao = convert_dnnl_to_kernel_type(a_zp_dt());
    problem.Tbo = convert_dnnl_to_kernel_type(b_zp_dt());
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
    if (a_zp_ndims() >= 0 || a_zp_host_scalar())
        problem.aOffset = ABOffset::Calc;
    if (b_zp_ndims() >= 0 || b_zp_host_scalar())
        problem.bOffset = ABOffset::Calc;
    problem.aoPtrDims = a_zp_host_scalar() ? -1 : a_zp_ndims();
    problem.boPtrDims = b_zp_host_scalar() ? -1 : b_zp_ndims();
    problem.asPtrDims = a_scale_ndims();
    problem.bsPtrDims = b_scale_ndims();

    problem.AO.layout = problem.BO.layout = MatrixLayout::N;
    problem.AO.crosspack = problem.BO.crosspack = 1;
    problem.AO.packSize = problem.BO.packSize = 0;
    problem.A_scale = problem.Ag = problem.AO;
    problem.B_scale = problem.Bg = problem.BO;

    // Only the late-scale (integer-DPAS) path needs scale md layout
    // aligned with A/B; 4-bit and wei_decomp leave layout to the kernel.
    const bool any_4bit = utils::one_of(a_type, data_type::s4,
                                  data_type::u4, data_type::f4_e2m1,
                                  data_type::f4_e3m0)
            || utils::one_of(b_type, data_type::s4, data_type::u4,
                    data_type::f4_e2m1, data_type::f4_e3m0);
    const bool late_scale_path = !wei_decomp_ && !any_4bit;
    if (problem.aScale2D() && trans_a && late_scale_path)
        problem.A_scale.layout = MatrixLayout::T;
    if (problem.bScale2D() && trans_b && late_scale_path)
        problem.B_scale.layout = MatrixLayout::T;

    // 1D tensors can be treated as either transposition - choose the one that
    // allows block loads (i.e. A -> N and B -> T)
    if (!problem.bOffset2D()) problem.BO.layout = MatrixLayout::T;
    if (!problem.bScale2D()) problem.B_scale.layout = MatrixLayout::T;
    if (b_gs_ndims() < 2) problem.Bg.layout = MatrixLayout::T;

    if (a_zp_dt() != data_type::undef)
        problem.AO.setAlignment(int(types::data_type_size(a_zp_dt())));
    if (b_zp_dt() != data_type::undef)
        problem.BO.setAlignment(int(types::data_type_size(b_zp_dt())));
    problem.aqGroupK = a_group_k_;
    problem.bqGroupK = b_group_k_;
    problem.aqGroupM = a_group_m_;
    problem.bqGroupN = b_group_n_;
    if (a_scale_dt() != data_type::undef) {
        problem.Ta_scale = convert_dnnl_to_kernel_type(a_scale_dt());
        problem.A_scale.setAlignment(
                int(types::data_type_size(a_scale_dt())));
    }
    if (b_scale_dt() != data_type::undef) {
        problem.Tb_scale = convert_dnnl_to_kernel_type(b_scale_dt());
        problem.B_scale.setAlignment(
                int(types::data_type_size(b_scale_dt())));
    }

    if (c_scale_dt() != data_type::undef) {
        problem.csPtrDims = c_scale_ndims();
        problem.cMXScale = with_mx_scale_;
        problem.Tc_scale = convert_dnnl_to_kernel_type(c_scale_dt());
        problem.cqGroupM = c_group_m_;
        problem.cqGroupN = c_group_n_;
    }

    if (problem.Ta_ext.isInt4() && problem.Tb_ext.isInt8()
            && a_zp_ndims() >= 0)
        problem.Ta = Type::s8;
    if (problem.Tb_ext.isInt4() && problem.Ta_ext.isInt8()
            && b_zp_ndims() >= 0)
        problem.Tb = Type::s8;

    if (problem.Ta.isInteger()) problem.Ts = Type::f32;

    if (alpha() == 1.0f) problem.alpha = (int)alpha();
    if (beta() == 0.0f || beta() == 1.0f) problem.beta = (int)beta();

    gpu_post_ops_t gpu_post_ops;
    CHECK(gpu_post_ops_t::make(
            gpu_post_ops, post_ops_, dst_md(), get_post_op_specializations()));

    CHECK(transfer_post_ops(problem, std::move(gpu_post_ops)));

    auto reduce_ab = sum_ab();
    if (c_offset || bias || reduce_ab != sum_ab::sum_none) {
        assert(!(c_offset && bias));
        if (bias) problem.cOffset = COffset::Pre;
        if (c_offset) problem.cOffset = COffset::Post;
        problem.CO.crosspack = 1;
        problem.CO.alignment = problem.C.alignment;
        problem.CO.layout = trans_co ? MatrixLayout::T : MatrixLayout::N;
        problem.coPtrDims = c_zp_host_scalar() ? -1 : c_zp_ndims();
    }

    problem.sumA = (reduce_ab == sum_ab::sum_a_row);
    problem.sumB = (reduce_ab == sum_ab::sum_b_col);
    problem.forceGroupSumsA = a_force_gs();
    problem.forceGroupSumsB = b_force_gs();

    problem.postOps.cStochasticRound = dst_sround;

    if (problem.needsAGroupSums() || problem.needsBGroupSums())
        problem.autoTypeConversions(has_systolic);

    if (problem.needsAGroupSums()) {
        data_type_t gs_dt = a_gs_dt() == data_type::undef
                ? data_type::s32
                : a_gs_dt();
        problem.Tag = convert_dnnl_to_kernel_type(gs_dt);
        problem.Ag.setAlignment(problem.Tag.paddedSize());
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
    }
    if (problem.needsBGroupSums()) {
        data_type_t gs_dt = b_gs_dt() == data_type::undef
                ? data_type::s32
                : b_gs_dt();
        problem.Tbg = convert_dnnl_to_kernel_type(gs_dt);
        problem.Bg.setAlignment(problem.Tbg.paddedSize());
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
    }
    // Disable bdpas with unsupported k dim.
    // TODO: Enable 2D block, masking scale loads.
    if (problem.nativeBDPAS()) {
        if (((!problem.Ta.isF4() || !problem.Tb.isF4()) || k % 64 == 0))
            problem.bdpasEnabled = true;
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

dim_t pd_t::scale_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, gemm_arg::A, gemm_arg::B));
    const memory_desc_t *md_ptr
            = (arg == gemm_arg::A) ? &a_scale_md_ : &b_scale_md_;
    gpu_assert(memory_desc_wrapper(md_ptr).is_plain())
            << "Expected plain scale_md_";
    if (md_ptr->dims[idx] == 1) return 0;
    return md_ptr->format_desc.blocking.strides[idx];
}

dim_t pd_t::zp_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, gemm_arg::A, gemm_arg::B));
    const memory_desc_t *md_ptr = (arg == gemm_arg::A) ? &a_zp_md_ : &b_zp_md_;
    gpu_assert(memory_desc_wrapper(md_ptr).is_plain())
            << "Expected plain zp_md_";
    if (md_ptr->dims[idx] == 1) return 0;
    return md_ptr->format_desc.blocking.strides[idx];
}

dim_t pd_t::gs_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, gemm_arg::A, gemm_arg::B));
    const memory_desc_t *md_ptr = (arg == gemm_arg::A) ? &a_gs_md_ : &b_gs_md_;
    gpu_assert(memory_desc_wrapper(md_ptr).is_plain())
            << "Expected plain gs_md_";
    if (md_ptr->dims[idx] == 1) return 0;
    return md_ptr->format_desc.blocking.strides[idx];
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
