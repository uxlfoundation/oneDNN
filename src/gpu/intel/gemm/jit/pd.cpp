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

#include <utility>

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

bool a_grouped(const kernel_config_t &cfg) {
    // Group sizes live in cfg.problem (populated in Phase A, kernel-oriented
    // after swap_fold). The sole caller (init_GEMMProblem) is post-swap, so
    // cfg.problem.aqGroup* matches the old post-swap cfg.a_quant.group_*.
    bool k_grouped = 1 < cfg.problem.aqGroupK && cfg.problem.aqGroupK < cfg.k;
    bool m_grouped = 1 < cfg.problem.aqGroupM && cfg.problem.aqGroupM < cfg.m;
    return k_grouped || m_grouped;
}

bool b_grouped(const kernel_config_t &cfg) {
    bool k_grouped = 1 < cfg.problem.bqGroupK && cfg.problem.bqGroupK < cfg.k;
    bool n_grouped = 1 < cfg.problem.bqGroupN && cfg.problem.bqGroupN < cfg.n;
    return k_grouped || n_grouped;
}

} // anonymous namespace

status_t pd_t::init(impl::engine_t *engine, compute::gpu_arch_t arch) {
    arch_ = arch;

    // Phase A — populate cfg in user orientation. The validators (scales_ok,
    // zp_ok, gs_ok) are interleaved with the populators here because
    // init_post_ops assumes those user-attr cases the validators reject have
    // already been filtered (e.g. unsupported 2D dst scales would otherwise
    // trip an init_post_ops assertion).
    CHECK(init_attrs(cfg_, engine));
    CHECK(scales_ok(cfg_, engine));
    CHECK(zp_ok(cfg_, engine));
    CHECK(gs_ok(cfg_, engine));
    CHECK(init_post_ops(cfg_, engine));

    return status::success;
}

status_t pd_t::init_post_ops(kernel_config_t &cfg, impl::engine_t *engine) {
    using namespace primitive_kind;
    using namespace alg_kind;
    using namespace data_type;

    const auto d = desc();

    // Examine post-ops and remember binary srcs. The converted list lives only
    // in this transient `po`; its lowered form is committed to cfg.problem.postOps
    // at the end of this function. The per-entry runtime residue (source md +
    // arg routing) is captured into cfg.binary_srcs[*] as we go.
    post_ops_t po = attr()->post_ops_;
    cfg.binary_srcs.reserve(po.len() + 4);

    bool ok = true;
    bool seen_sum = false;
    int prelu_count = 0;
    const int num_orig_postops = po.len();
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        switch (e.kind) {
            case binary:
                ok &= supported_binary_op(e.binary.alg)
                        && is_md_gemm_compatible_plain_format(
                                &e.binary.src1_desc);
                cfg.binary_srcs.push_back(
                        binary_src_t {binary_src_t::binary, int(i)});
                cfg.binary_srcs.back().src_md = e.binary.src1_desc;
                cfg.non_scale_po = true;
                break;
            case sum:
                ok &= !seen_sum;
                seen_sum = true;
                cfg.binary_srcs.push_back(binary_src_t {binary_src_t::none, 0});
                cfg.beta = e.sum.scale;
                break;
            case eltwise:
                ok &= eltwise_injector_f32_is_supported(e.eltwise.alg);
                cfg.binary_srcs.push_back(binary_src_t {binary_src_t::none, 0});
                cfg.non_scale_po = true;
                break;
            case prelu: {
                memory_desc_t prelu_md = {};
                ok &= get_prelu_md(e.prelu.mask, dst_md()->dims, prelu_md,
                              dst_md()->ndims)
                        == status::success;
                cfg.binary_srcs.push_back(
                        binary_src_t {binary_src_t::prelu, int(i)});
                cfg.binary_srcs.back().src_md = prelu_md;
                prelu_count++;
                ok &= prelu_count <= 1;
                cfg.non_scale_po = true;
                break;
            }
            default: VDISPATCH_GEMM(false, VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    VDISPATCH_GEMM(ok, VERBOSE_UNSUPPORTED_POSTOP);

    // If scales are present, convert them and any bias to binary post-ops.
    //   Exception: 2D scales.
    // Also convert bias to binary post-op if dst zp are present.
    const auto &a_scales = attr()->scales_.get(DNNL_ARG_A);
    const auto &b_scales = attr()->scales_.get(DNNL_ARG_B);
    const auto &c_scales = attr()->scales_.get(DNNL_ARG_C);

    // Transient: bias is folded into a binary post-op (it never outlives this
    // function — `with_bias` below captures the persistent signal).
    const bool bias_via_binary = (desc()->bias_type() != data_type::undef)
            && (d->bias_desc.ndims >= 1 || !a_scales.has_default_values()
                    || !b_scales.has_default_values()
                    || !attr()->zero_points_.has_default_values(DNNL_ARG_C));
    if (bias_via_binary) {
        VDISPATCH_GEMM_SC(po.prepend_binary(binary_add, &d->bias_desc),
                "%s: bias via binary post-op", VERBOSE_UNSUPPORTED_POSTOP);
        cfg.binary_srcs.insert(
                cfg.binary_srcs.begin(), binary_src_t {binary_src_t::bias, 0});
        cfg.binary_srcs.front().src_md = d->bias_desc;
    }
    cfg.non_scale_po |= bias_via_binary;

    // Bias-cfg derivatives. `with_bias` is false when bias was
    // converted to a binary post-op above; `ld_bias` is still needed
    // by `ld_binary` in that case (the bias is still the binary
    // post-op's source), so populate it whenever a bias exists.
    if (desc()->bias_type() != data_type::undef) {
        cfg.with_bias = !bias_via_binary;
        unsigned char to_cmask[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        assert(unsigned(desc()->bias_mask()) < 8);
        cfg.bias_cmask = cfg.with_bias ? to_cmask[desc()->bias_mask() & 7] : -1;
        cfg.ld_bias = desc()->ld_bias();
    }

    auto maybe_convert_scales_to_postop
            = [this, &cfg, &po, engine](const memory_desc_t &scale_md, int arg,
                      int scale_ndims, bool mx, bool &converted) -> status_t {
        auto ndims = desc()->c_desc.ndims;
        // Scales on A/B can be converted to postops if
        // the scales md has K=1 and M/N is not bcast.
        converted = false;
        if (scale_ndims > 1) return status::success;
        int inner_dim = (arg == DNNL_ARG_A ? ndims - 2 : ndims - 1);
        bool convert = (scale_md.dims[inner_dim] <= 1) || (arg == DNNL_ARG_C);
        convert &= !mx;
        if (convert) {
            if (arg == DNNL_ARG_C) {
                VDISPATCH_GEMM_SC(po.append_binary(binary_div, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                cfg.binary_srcs.push_back(
                        binary_src_t {binary_src_t::scales, arg});
                cfg.binary_srcs.back().src_md = scale_md;
            } else {
                VDISPATCH_GEMM_SC(po.prepend_binary(binary_mul, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                cfg.binary_srcs.insert(cfg.binary_srcs.begin(),
                        binary_src_t {binary_src_t::scales, arg});
                cfg.binary_srcs.front().src_md = scale_md;
            }
            converted = true;
        }
        return status::success;
    };

    // cfg.problem.{as,bs,cs}PtrDims were populated from scale_ndims in
    // init_attrs; converting scales to post-ops drops the A/B scale arg, so
    // reset the corresponding ptr-dims to -1 (matches the old scale_ndims=-1).
    if (!a_scales.has_default_values() && !a_scales.is_host_scalar()) {
        // Host scalar scale will be converted to Alpha
        bool converted;
        CHECK(maybe_convert_scales_to_postop(cfg.a_scale_md, DNNL_ARG_A,
                cfg.problem.asPtrDims, a_scales.is_mx(), converted));
        if (converted) cfg.problem.asPtrDims = -1;
    }

    if (!b_scales.has_default_values() && !b_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(cfg.b_scale_md, DNNL_ARG_B,
                cfg.problem.bsPtrDims, b_scales.is_mx(), converted));
        if (converted) cfg.problem.bsPtrDims = -1;
    }

    bool try_c_scale = !c_scales.is_host_scalar()
            || (c_scales.is_host_scalar() && num_orig_postops > 0);
    if (!c_scales.has_default_values() && try_c_scale) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(cfg.c_scale_md, DNNL_ARG_C,
                cfg.problem.csPtrDims, c_scales.is_mx(), converted));
        // Conversion of dst scales to post ops is currently supported for all
        // cases supported in the library.
        gpu_assert(converted || c_scales.is_mx())
                << "Unable to convert dst scales to a post op";
    }

    // Commit the converted post-ops to the kernel projection in user
    // orientation; swap_fold's cfg.problem.transpose() folds postOps/binary
    // afterward. The transient `po` is not stored on the cfg — only the
    // lowered problem.postOps + binary_srcs (with per-entry src_md) survive.
    gpu_post_ops_t gpu_post_ops;
    CHECK(gpu_post_ops_t::make(
            gpu_post_ops, po, dst_md(), get_post_op_specializations()));
    CHECK(transfer_post_ops(cfg.problem, std::move(gpu_post_ops)));

    return status::success;
}

status_t pd_t::seed_problem(kernel_config_t &cfg) {
    const auto d = desc();

    // Seed the cfg's user-orientation A/B/C scalars. Phase B (pad_lda +
    // swap_fold) reads/mutates these in place.
    cfg.m = d->m();
    cfg.n = d->n();
    cfg.k = d->k();
    cfg.lda = d->lda();
    cfg.ldb = d->ldb();
    cfg.ldc = d->ldc();

    // Seed the embedded problem's matrix types and A/B/C/CO orientation
    // (user orientation); swap_fold's transpose() folds them downstream.
    // cfg.{a,b,c}_type()/trans_a()/trans_b() project them back to dnnl, and
    // init_GEMMProblem consumes them via `problem = cfg.problem`.
    using gemmstone::MatrixLayout;
    auto to_layout
            = [](bool t) { return t ? MatrixLayout::T : MatrixLayout::N; };
    cfg.problem.Ta_ext = convert_dnnl_to_kernel_type(d->a_type());
    cfg.problem.Tb_ext = convert_dnnl_to_kernel_type(d->b_type());
    cfg.problem.Tc_ext = convert_dnnl_to_kernel_type(d->c_type());
    cfg.problem.A.layout = to_layout(d->transa() == dnnl_trans);
    cfg.problem.B.layout = to_layout(d->transb() == dnnl_trans);
    cfg.problem.C.layout = MatrixLayout::N;
    cfg.problem.CO.layout = to_layout(d->trans_bias() == dnnl_trans);

    // The post-op binary metadata (problem.postOps/binary/Tbinary) is committed
    // by init_post_ops (in user orientation, before this seed runs); swap_fold's
    // cfg.problem.transpose() folds it along with the A/B/C/CO orientation set
    // above.

    return status::success;
}

bool pd_t::dy_quant_enabled() const {
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

bool pd_t::wei_decomp() const {
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

status_t pd_t::init_attrs(kernel_config_t &cfg, impl::engine_t *engine) {
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

    // Swap descriptors to follow column major format
    VDISPATCH_GEMM_SC(a_zps.get_md(cfg.a_zp_md, d->b_desc),
            VERBOSE_DESC_CREATION_FAIL, "A zero points");
    VDISPATCH_GEMM_SC(b_zps.get_md(cfg.b_zp_md, d->a_desc),
            VERBOSE_DESC_CREATION_FAIL, "B zero points");
    VDISPATCH_GEMM_SC(c_zps.get_md(cfg.c_zp_md, d->c_desc),
            VERBOSE_DESC_CREATION_FAIL, "C zero points");
    VDISPATCH_GEMM_SC(a_gs.get_md(cfg.a_gs_md, d->b_desc),
            VERBOSE_DESC_CREATION_FAIL, "A group sums");
    VDISPATCH_GEMM_SC(b_gs.get_md(cfg.b_gs_md, d->a_desc),
            VERBOSE_DESC_CREATION_FAIL, "B group sums");
    VDISPATCH_GEMM_SC(a_scales.get_md(cfg.a_scale_md, desc_.b_desc),
            VERBOSE_DESC_CREATION_FAIL, "A scales");
    VDISPATCH_GEMM_SC(b_scales.get_md(cfg.b_scale_md, desc_.a_desc),
            VERBOSE_DESC_CREATION_FAIL, "B scales");
    VDISPATCH_GEMM_SC(c_scales.get_md(cfg.c_scale_md, desc_.c_desc),
            VERBOSE_DESC_CREATION_FAIL, "C scales");

    // Collapse the per-side quant inputs directly into cfg.problem (the kernel
    // projection) in user orientation; swap_fold transposes it in Phase B via
    // problem.transpose(). Done here in Phase A so the validators (which run
    // before init_post_ops) read cfg.problem rather than raw quant scalars.
    // init_post_ops re-derives the A/B scale ptr-dims if it converts scales
    // into post-ops. Matrix types (Ta/Tb/Tc_ext) are seeded separately in
    // gen_t::pd_t::init (still undef here).
    auto &p = cfg.problem;
    auto ndims = d->c_desc.ndims;

    // A/B zero points. A host-scalar zp folds aoPtrDims/boPtrDims to -1.
    const int a_zp_ndims = quant_entry_ndims(a_zps, cfg.a_zp_md, ndims - 2);
    const int b_zp_ndims = quant_entry_ndims(b_zps, cfg.b_zp_md, ndims - 1);
    const bool a_zp_hs = a_zp_host_scalar();
    const bool b_zp_hs = b_zp_host_scalar();
    p.Tao = convert_dnnl_to_kernel_type(a_zps.get_data_type());
    p.Tbo = convert_dnnl_to_kernel_type(b_zps.get_data_type());
    if (a_zp_ndims >= 0 || a_zp_hs) p.aOffset = gemmstone::ABOffset::Calc;
    if (b_zp_ndims >= 0 || b_zp_hs) p.bOffset = gemmstone::ABOffset::Calc;
    p.aoPtrDims = a_zp_hs ? -1 : a_zp_ndims;
    p.boPtrDims = b_zp_hs ? -1 : b_zp_ndims;

    // A/B scales.
    p.asPtrDims = quant_entry_ndims(a_scales, cfg.a_scale_md, ndims - 2);
    p.bsPtrDims = quant_entry_ndims(b_scales, cfg.b_scale_md, ndims - 1);
    if (a_scales.get_data_type() != data_type::undef)
        p.Ta_scale = convert_dnnl_to_kernel_type(a_scales.get_data_type());
    if (b_scales.get_data_type() != data_type::undef)
        p.Tb_scale = convert_dnnl_to_kernel_type(b_scales.get_data_type());

    // A/B group sums (the gs type is derived later from the gs md in
    // init_GEMMProblem, so it is not seeded here).
    p.agPtrDims = quant_entry_ndims(a_gs, cfg.a_gs_md, ndims - 2);
    p.bgPtrDims = quant_entry_ndims(b_gs, cfg.b_gs_md, ndims - 1);
    p.forceGroupSumsA = !a_gs.has_default_values();
    p.forceGroupSumsB = !b_gs.has_default_values();

    // C-side quant (swap-invariant). coPtrDims carries the RAW c-zp
    // dimensionality (no host-scalar fold): gen_t folds a host-scalar c-zp to
    // -1 itself in init_GEMMProblem, while xe_hp_systolic consumes it directly
    // (it rejects host-scalar zps upstream, so raw == folded there).
    p.coPtrDims = quant_entry_ndims(c_zps, cfg.c_zp_md, -1);
    if (c_scales.get_data_type() != data_type::undef) {
        p.csPtrDims = quant_entry_ndims(c_scales, cfg.c_scale_md, -1);
        p.Tc_scale = convert_dnnl_to_kernel_type(c_scales.get_data_type());
        if (!c_scales.has_default_groups()) {
            p.cqGroupM = into<int>(c_scales.get_group(1));
            p.cqGroupN = into<int>(c_scales.get_group(0));
        }
    }

    p.sumA = (d->sum_ab == sum_ab::sum_b_col);
    p.sumB = (d->sum_ab == sum_ab::sum_a_row);

    // XXX, gemmstone support: if multiple grouped quantization attributes exist
    // for one matrix, they must have the same group size (default/unset is 0).
    const auto &set_if_consistent
            = [this, engine](int &group, int new_group, int arg) -> status_t {
        VDISPATCH_GEMM(utils::one_of(group, 0, new_group),
                "%s: %s quantization attrs with different group sizes",
                VERBOSE_UNSUPPORTED_ATTR, arg2str(arg).c_str());
        group = new_group;
        return status::success;
    };
    // g0/g1 receive get_group(0)/get_group(1): for A that is (K, M), for B
    // (N, K) — matching the original per-side group mapping.
    const auto &set_groups
            = [&set_if_consistent](int &g0, int &g1, const quant_entry_t &entry,
                      int arg) -> status_t {
        if (entry.has_default_groups()) return status::success;
        CHECK(set_if_consistent(g0, into<int>(entry.get_group(0)), arg));
        CHECK(set_if_consistent(g1, into<int>(entry.get_group(1)), arg));
        return status::success;
    };
    CHECK(set_groups(p.aqGroupK, p.aqGroupM, a_zps, DNNL_ARG_A));
    CHECK(set_groups(p.aqGroupK, p.aqGroupM, a_gs, DNNL_ARG_A));
    CHECK(set_groups(p.aqGroupK, p.aqGroupM, a_scales, DNNL_ARG_A));
    CHECK(set_groups(p.bqGroupN, p.bqGroupK, b_zps, DNNL_ARG_B));
    CHECK(set_groups(p.bqGroupN, p.bqGroupK, b_gs, DNNL_ARG_B));
    CHECK(set_groups(p.bqGroupN, p.bqGroupK, b_scales, DNNL_ARG_B));

    return status::success;
}

status_t pd_t::zp_ok(const kernel_config_t &cfg, impl::engine_t *engine) {
    using namespace data_type;
    auto &attr_zps = attr()->zero_points_;
    if (attr_zps.has_default_values()) return status::success;
    auto &a_zps = attr_zps.get(DNNL_ARG_A);
    auto &b_zps = attr_zps.get(DNNL_ARG_B);
    auto &c_zps = attr_zps.get(DNNL_ARG_C);

    int ndims = desc()->a_desc.ndims;
    const bool a_int4 = utils::one_of(desc()->a_type(), s4, u4);
    const bool b_int4 = utils::one_of(desc()->b_type(), s4, u4);
    const bool weights_upconversion
            = wei_decomp() || (a_int4 && dy_quant_enabled());

    // cfg.problem was populated in init_attrs (pre-swap, user orientation), so
    // these match the old raw zp_ndims>=2 / scale_ndims>1 reads. A host-scalar
    // zp is never 2D, so the aoPtrDims fold is irrelevant to aOffset2D().
    const bool a_zp_2d = cfg.problem.aOffset2D();
    const bool b_zp_2d = cfg.problem.bOffset2D();
    const bool a_scales_2d = cfg.problem.aScale2D();

    if (!a_zps.has_default_values()) {
        // Groups determine supported masks.
        if (!a_zps.has_default_groups()) {
            VDISPATCH_GEMM(
                    valid_2d_mask(cmask_a(), ndims, weights_upconversion),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto a_q2d_group_n = a_zps.get_group(1);
            // Non-trivial N group unsupported.
            VDISPATCH_GEMM(a_q2d_group_n == 1,
                    "%s: Grouped N dimension on A matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Zero points with non-trivial groups only supported with
            // precomputed reductions or when target tensor is being dequantized.
            bool has_prB = !attr()->precomputed_reductions_.has_default_values(
                    DNNL_ARG_B);
            // TODO: Re-examine this condition
            bool is_dequantized = !dy_quant_enabled() || !b_int4 || a_int4;
            VDISPATCH_GEMM(IMPLICATION(a_zp_2d, is_dequantized || has_prB),
                    "%s: Nontrivial groups on A matrix, and no precomputed "
                    "reductions or dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_GEMM(
                    utils::one_of(cmask_a(), 0, mask_per_oc, mask_per_ic),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            // Weights zp can only be performantly enabled during upconversion
            // for cases that perform decompression.
            VDISPATCH_GEMM(IMPLICATION(a_scales_2d,
                                   !(b_int4 && !wei_decomp() && !a_int4)),
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
            VDISPATCH_GEMM(valid_2d_mask(cmask_b(), ndims, false),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto b_q2d_group_n = b_zps.get_group(0);
            // Non-trivial M group unsupported.
            VDISPATCH_GEMM(utils::one_of(b_q2d_group_n, 1, desc()->n()),
                    "%s: Nontrivial N groups on B matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            // TODO: Re-examine this condition
            bool is_dequantized = !dy_quant_enabled() || !a_int4 || b_int4;
            VDISPATCH_GEMM(IMPLICATION(b_zp_2d, is_dequantized),
                    "%s: Grouped B zero points, and no dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_GEMM(utils::one_of(cmask_b(), 0, mask_scalar,
                                   mask_per_oc | mask_per_ic),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!attr_zps.has_default_values(DNNL_ARG_C)) {
        VDISPATCH_GEMM(
                IMPLICATION(!c_zps.is_host_scalar(),
                        utils::one_of(cmask_c(), 0, mask_scalar, mask_per_oc)),
                "%s: unsupported C mask", VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    return status::success;
}

status_t pd_t::gs_ok(const kernel_config_t &cfg, impl::engine_t *engine) {
    auto &attr_gs = attr()->precomputed_reductions_;
    if (attr_gs.has_default_values()) return status::success;

    VDISPATCH_GEMM(attr_gs.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_PR_CFG);

    bool with_a_group_sums_ = !attr_gs.has_default_values(DNNL_ARG_A);
    bool with_b_group_sums_ = !attr_gs.has_default_values(DNNL_ARG_B);

    VDISPATCH_GEMM(IMPLICATION(with_a_group_sums_,
                           attr_gs.get_data_type(DNNL_ARG_A) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_GEMM(IMPLICATION(with_b_group_sums_,
                           attr_gs.get_data_type(DNNL_ARG_B) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);

    return status::success;
}

status_t pd_t::scales_ok(const kernel_config_t &cfg, impl::engine_t *engine) {
    const auto &scales = attr()->scales_;
    if (scales.has_default_values()) return status::success;
    int ndims = desc()->a_desc.ndims;
    using namespace data_type;

    for (auto s : {DNNL_ARG_A, DNNL_ARG_B}) {
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

    const auto &dst_scales = scales.get(DNNL_ARG_C);
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
        auto md = &desc()->c_desc;
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

            // Build each binary entry in USER (un-swapped) orientation. The
            // A/B swap is folded once, downstream, by swap_fold's
            // cfg.problem.transpose(): PostOpsProblem::transpose() swaps
            // Row<->Col and flips binaryTrans; MatrixAddressing::transpose()
            // flips N<->T. binaryBatch and Tbinary are swap-invariant.
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

status_t pd_t::init_GEMMProblem(gemmstone::GEMMProblem &problem,
        const intel::engine_t *engine, const kernel_config_t &cfg) const {
    // Set up problem structure.
    using namespace gemmstone;
    // Seed from the embedded, swap-folded kernel projection. The migrated
    // scalar fields (Ta/Tb/Tc_ext, Tao/Tbo, Ta/Tb/Tc_scale, a/b offset
    // modes + ptrDims, aq/bq/cq groups, sumA/sumB, forceGroupSums, c-scale
    // ptrDims) come from cfg.problem; addressing/layout/alignment and derived
    // register types (Ta/Tb/Tc/Ts/Tco) are computed below and overwrite the
    // seed's defaults.
    problem = cfg.problem;

    problem.product = get_ngen_product(*engine->device_info());
    bool has_systolic
            = engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            || engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate);

    // Leading-dim / batch-stride alignment. cfg.{lda,ldb,a_type,b_type} are
    // already swap-folded to kernel orientation; the desc() batch strides are
    // user-frame, so select the effective side via cfg.swap_ab. (Formerly
    // precomputed into cfg.align_* in jit.hpp before swap_fold; now derived here
    // so the alignment is not persistent cfg state.)
    const int a_batch_arg = cfg.swap_ab ? DNNL_ARG_B : DNNL_ARG_A;
    const int b_batch_arg = cfg.swap_ab ? DNNL_ARG_A : DNNL_ARG_B;
    auto align_a = nstl::max(align(cfg.lda, cfg.a_type(), a_batch_arg),
            (int)types::data_type_size(cfg.a_type()));
    auto a_size = (cfg.trans_a() ? cfg.m : cfg.k) * cfg.lda
            * types::data_type_size(cfg.a_type());

    auto align_b = nstl::max(align(cfg.ldb, cfg.b_type(), b_batch_arg),
            (int)types::data_type_size(cfg.b_type()));
    auto b_size = (cfg.trans_b() ? cfg.k : cfg.n) * cfg.ldb
            * types::data_type_size(cfg.b_type());

    bool int_acc = utils::one_of(cfg.a_type(), data_type::s8, data_type::u8)
            || (types::is_integral_dt(cfg.a_type())
                    && types::is_integral_dt(cfg.b_type()));
    int_acc &= !(a_grouped(cfg) || b_grouped(cfg));
    auto align_c = nstl::max(align(cfg.ldc, cfg.c_type(), DNNL_ARG_C),
            (int)types::data_type_size(cfg.c_type()));
    auto c_size = cfg.n * cfg.ldc * types::data_type_size(cfg.c_type());

    auto co_type = cfg.with_bias ? desc()->bias_type()
            : with_sum_ab()      ? desc()->sum_ab_type
            : int_acc            ? data_type::s32
                                 : cfg.c_type();

    // Choose accumulation data type.
    auto acc_type = int_acc
            ? data_type::s32
            : (utils::one_of(data_type::f64, cfg.a_type(), cfg.b_type())
                              ? data_type::f64
                              : data_type::f32);

    // problem.postOps is already populated (carried by `problem = cfg.problem`)
    // and lowers prelu into a binary, so hasBinaryPostOp() covers the old
    // find(binary)||find(prelu) (converted A/B/C scales + bias included, as
    // before).
    bool with_binary = problem.hasBinaryPostOp();

    bool need_x32_acc = with_binary || !IMPLICATION(with_sum(), sum_at_begin());
    auto acc_mode = attr()->acc_mode_;

    // Strict acc mode default.
    auto Tacc_mode = problem.Tc_ext.isFP() ? data_type::f32 : data_type::s32;

    // Initialize non-strict acc mode.
    switch (acc_mode) {
        case accumulation_mode::any:
            if (!need_x32_acc) Tacc_mode = data_type::undef;
            break;
        case accumulation_mode::f16: Tacc_mode = data_type::f16; break;
        case accumulation_mode::f32: Tacc_mode = data_type::f32; break;
        case accumulation_mode::s32: Tacc_mode = data_type::s32; break;
        default: break;
    }

    // Minimum precision type for applying post-ops based on acc mode.
    // Limits use of atomic add.
    problem.Tacc_min = convert_dnnl_to_kernel_type(Tacc_mode);

    if (acc_mode != accumulation_mode::strict) acc_type = Tacc_mode;

    if (wei_decomp()) { acc_type = data_type::f32; }

    auto dst_sround = with_sround();
    bool c_offset = with_c_zero_points();
    bool bias = cfg.with_bias;

    // Ta_ext/Tb_ext/Tc_ext/Tao/Tbo seeded from cfg.problem; register types
    // and the C-offset type are derived here.
    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;
    problem.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem.Ts = problem.Tc;
    problem.Tco = convert_dnnl_to_kernel_type(co_type);
    // A/B.layout are already the swap-folded values (seeded in gen_t::pd_t::init
    // and carried by `problem = cfg.problem`); only force C back to N (the cfg's
    // C.layout may have been flipped by swap_fold's transpose() but C is never
    // transposed).
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
    // aOffset/bOffset + ao/bo/as/bs PtrDims seeded from cfg.problem.

    // Default A-side layout is N, B-side T (one-shot 1D-friendly default).
    // Under swap_ab, the previous swap_ab post-pass called .transpose() on
    // AO/BO/etc; we instead flip the default layout here so no post-pass is
    // needed.
    const auto layout_N = cfg.swap_ab ? MatrixLayout::T : MatrixLayout::N;
    const auto layout_T = cfg.swap_ab ? MatrixLayout::N : MatrixLayout::T;

    problem.AO.layout = problem.BO.layout = layout_N;
    problem.AO.crosspack = problem.BO.crosspack = 1;
    problem.AO.packSize = problem.BO.packSize = 0;
    problem.A_scale = problem.Ag = problem.AO;
    problem.B_scale = problem.Bg = problem.BO;

    // 1D tensors can be treated as either transposition - choose the one that
    // allows block loads (i.e. A -> N and B -> T)
    if (!problem.bOffset2D()) problem.BO.layout = layout_T;
    if (!problem.bScale2D()) problem.B_scale.layout = layout_T;
    if (problem.bgPtrDims < 2) problem.Bg.layout = layout_T;

    // Group sizes, offset/scale types + ptrDims, and cq groups are all seeded
    // from cfg.problem. Offset/scale-matrix alignments and cMXScale are derived
    // here from the (already-seeded) kernel quant types; paddedSize() gives the
    // external byte size (== the old data_type_size), matching the Ag/Bg
    // alignment below. Type::invalid marks an absent type (undef -> invalid).
    if (problem.Tao != Type::invalid)
        problem.AO.setAlignment(problem.Tao.paddedSize());
    if (problem.Tbo != Type::invalid)
        problem.BO.setAlignment(problem.Tbo.paddedSize());
    if (problem.Ta_scale != Type::invalid)
        problem.A_scale.setAlignment(problem.Ta_scale.paddedSize());
    if (problem.Tb_scale != Type::invalid)
        problem.B_scale.setAlignment(problem.Tb_scale.paddedSize());

    if (problem.Tc_scale != Type::invalid) problem.cMXScale = with_mx_scale();

    // Mixed s8/s4 DPAS support:
    // - Xe3p: Not supported, require s4->s8 upconversion
    // - pre-Xe3p: supported, but only when s4 matrix doesn't have zero points
    bool has_s8s4_dpas = getCore(problem.product.family) != ngen::HW::Xe3p;
    // "s4 matrix has no zero points" — the old check was zp_ndims < 0, i.e.
    // no zp at all. That maps to aOffset == None (NOT aoPtrDims < 0, which a
    // host-scalar zp also satisfies but must still force upconversion).
    if (problem.Ta_ext.isInt4() && problem.Tb_ext.isInt8()) {
        bool s8s4_dpas_ok
                = has_s8s4_dpas && (problem.aOffset == ABOffset::None);
        if (!s8s4_dpas_ok) problem.Ta = Type::s8;
    }
    if (problem.Tb_ext.isInt4() && problem.Ta_ext.isInt8()) {
        bool s8s4_dpas_ok
                = has_s8s4_dpas && (problem.bOffset == ABOffset::None);
        if (!s8s4_dpas_ok) problem.Tb = Type::s8;
    }

    if (problem.Ta.isInteger()) problem.Ts = Type::f32;

    // Post-op accumulation-type bump, relocated from transfer_post_ops (which
    // now runs pre-swap in gen_t::pd_t::init, before register types exist).
    // problem.postOps is already populated and carried by `problem = cfg.problem`;
    // gate on it to match the old `post_ops.len() > 0` guard. Reads the final
    // register Ta/Tb (post s4/s8 upconversion above) — identical to the old
    // placement inside transfer_post_ops.
    if (problem.postOps.len() > 0) {
        if (problem.Ta == Type::f16) problem.Ts = Type::f32;
        if (problem.Ta.isF8() || problem.Tb.isF8()) problem.Ts = Type::f32;
    }

    if (alpha() == 1.0f) problem.alpha = (int)alpha();
    if (cfg.beta == 0.0f || cfg.beta == 1.0f) problem.beta = (int)cfg.beta;

    // post-ops (problem.postOps/binary/Tbinary) were built and swap-folded
    // upstream; only the runtime-derived stochastic-round flag remains (below).

    if (c_offset || bias || problem.sumA || problem.sumB) {
        assert(!(c_offset && bias));
        if (bias) problem.cOffset = COffset::Pre;
        if (c_offset) problem.cOffset = COffset::Post;
        problem.CO.crosspack = 1;
        problem.CO.alignment = problem.C.alignment;
        // CO.layout is the swap-folded value seeded in gen_t::pd_t::init and
        // carried by `problem = cfg.problem`.
        // coPtrDims already holds the raw c-zp dimensionality (seeded in
        // init_attrs); fold a host-scalar c-zp to -1 here.
        if (c_zp_host_scalar()) problem.coPtrDims = -1;
    }

    // sumA/sumB and forceGroupSumsA/B seeded from cfg.problem.

    problem.postOps.cStochasticRound = dst_sround;

    if (problem.needsAGroupSums() || problem.needsBGroupSums())
        problem.autoTypeConversions(has_systolic);

    if (problem.needsAGroupSums()) {
        // Group-sum type: read from the (already swap-folded) group-sum md;
        // default to s32 when absent. Mirrors the old cfg.a_qr.gs_type source.
        data_type_t gs_dt = memory_desc_wrapper(cfg.a_gs_md).data_type();
        if (gs_dt == data_type::undef) gs_dt = data_type::s32;
        problem.Tag = convert_dnnl_to_kernel_type(gs_dt);
        problem.Ag.setAlignment(problem.Tag.paddedSize());
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
    }
    if (problem.needsBGroupSums()) {
        data_type_t gs_dt = memory_desc_wrapper(cfg.b_gs_md).data_type();
        if (gs_dt == data_type::undef) gs_dt = data_type::s32;
        problem.Tbg = convert_dnnl_to_kernel_type(gs_dt);
        problem.Bg.setAlignment(problem.Tbg.paddedSize());
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
    }
    // Disable bdpas with unsupported k dim.
    // TODO: Enable 2D block, masking scale loads.
    if (problem.nativeBDPAS()) {
        if (((!problem.Ta.isF4() || !problem.Tb.isF4()) || cfg.k % 64 == 0))
            problem.bdpasEnabled = true;
    }

    return status::success;
}

namespace {
// Per-batch-dim stride of a single (swap-folded, kernel-oriented) quant md.
// The md is already in kernel orientation, so the caller selects the side by
// passing the right md — there is no DNNL_ARG to get wrong.
dim_t quant_md_stride(const memory_desc_t &md, int idx) {
    gpu_assert(memory_desc_wrapper(&md).is_plain())
            << "Expected plain quant md";
    if (md.dims[idx] == 1) return 0;
    return md.format_desc.blocking.strides[idx];
}
} // namespace

dim_t pd_t::a_scale_stride(int idx) const {
    return quant_md_stride(cfg_.a_scale_md, idx);
}
dim_t pd_t::b_scale_stride(int idx) const {
    return quant_md_stride(cfg_.b_scale_md, idx);
}
dim_t pd_t::a_zp_stride(int idx) const {
    return quant_md_stride(cfg_.a_zp_md, idx);
}
dim_t pd_t::b_zp_stride(int idx) const {
    return quant_md_stride(cfg_.b_zp_md, idx);
}
dim_t pd_t::a_gs_stride(int idx) const {
    return quant_md_stride(cfg_.a_gs_md, idx);
}
dim_t pd_t::b_gs_stride(int idx) const {
    return quant_md_stride(cfg_.b_gs_md, idx);
}

dim_t kernel_config_t::ld_binary(int idx) const {
    const auto &src = binary_srcs[idx];
    switch (src.type) {
        case binary_src_t::binary: return gemm_desc_t::get_ld(src.src_md);
        case binary_src_t::bias: return ld_bias;
        case binary_src_t::prelu: return gemm_desc_t::get_ld(src.src_md);
        default: return 1;
    }
}

dim_t kernel_config_t::stride_binary(int idx, int stride) const {
    const auto &src = binary_srcs[idx];
    switch (src.type) {
        case binary_src_t::binary:
        case binary_src_t::scales:
        case binary_src_t::bias:
        case binary_src_t::prelu:
            return gemm_desc_t::get_stride(src.src_md, stride);
        default: return 0;
    }
}

void pad_lda(kernel_config_t &cfg, bool swap) {
    // Runs in user orientation (before swap_fold). When swap_ab is on,
    // the gen_t path can replace a non-transposed A of width 1 with a
    // transposed A of width k; mirrors the historical jit.hpp logic.
    if (swap && !cfg.trans_a() && cfg.m == 1) {
        cfg.problem.A.layout = gemmstone::MatrixLayout::T;
        cfg.lda = cfg.k;
    }

    // Pad leading dimensions in case of a single row/column.
    if ((cfg.k == 1 && !cfg.trans_a()) || (cfg.m == 1 && cfg.trans_a()))
        cfg.lda = utils::rnd_up(cfg.lda, 16);
    if ((cfg.n == 1 && !cfg.trans_b()) || (cfg.k == 1 && cfg.trans_b()))
        cfg.ldb = utils::rnd_up(cfg.ldb, 16);
}

void swap_fold(kernel_config_t &cfg, bool swap) {
    cfg.swap_ab = swap;
    if (!swap) return;
    // Swapping A and B is equivalent to transposing both. The orientation
    // (A/B/CO.layout) and matrix types (Ta/Tb_ext) ride on cfg.problem and are
    // swapped+flipped by problem.transpose() below, so trans_a()/trans_b()/
    // a_type()/b_type() follow automatically — no separate fields to flip.
    std::swap(cfg.m, cfg.n);
    std::swap(cfg.lda, cfg.ldb);
    // After swap, the kernel's A-side reads from the matmul B's quant state.
    // The entire quant subset rides on cfg.problem (transposed below); nothing
    // quant-related is stored separately on the cfg anymore. sumA/sumB are
    // swapped by problem.transpose() below (no separate cfg.sum_ab to flip).
    // Quant memory descriptors follow the swap of the quant state.
    std::swap(cfg.a_scale_md, cfg.b_scale_md);
    std::swap(cfg.a_zp_md, cfg.b_zp_md);
    std::swap(cfg.a_gs_md, cfg.b_gs_md);
    // c_scale_md, c_zp_md, binary_srcs (per-entry src_md) are swap-invariant.

    // Fold the embedded problem to kernel orientation. transpose() swaps
    // A/B types, addressing, offsets, scale/offset/group-sum ptrDims
    // (ao/bo, as/bs, ag/bg), group sizes (aqGroupM<->bqGroupN, aqGroupK<->
    // bqGroupK), sumA/sumB, postOps and binaries. It does NOT swap two
    // A/B-paired things:
    //   - forceGroupSumsA/B: populated pre-swap, so swap them explicitly here.
    //   - Tag/Tbg (group-sum types): not seeded into cfg.problem at all; they
    //     are derived post-swap in init_GEMMProblem from the already-swapped
    //     group-sum mds (cfg.a_gs_md/cfg.b_gs_md, swapped above), so no swap is
    //     needed here. If they ever get seeded into cfg.problem in init_attrs,
    //     add a swap for them too.
    cfg.problem.transpose();
    std::swap(cfg.problem.forceGroupSumsA, cfg.problem.forceGroupSumsB);
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
