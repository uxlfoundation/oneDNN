/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#include <cassert>
#include <cstring>
#include <limits>
#include <vector>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_uni_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// The x64 binary primitive's set. The rv64 injector itself realizes the wider
// aarch64 set (get_all_strategies_supported_by_injector adds per_mb_spatial/
// per_mb_w via the per-lane gather), but as on x64 the binary primitive does
// not advertise those: such a post-op rhs classifies as unsupported here and
// the primitive defers to ref.
static bcast_set_t get_supported_postops_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
}

static bool compare_layouts(const memory_desc_wrapper &src0_md,
        const memory_desc_wrapper &src1_md) {
    const strides_t &strides0 = src0_md.blocking_desc().strides;
    const strides_t &strides1 = src1_md.blocking_desc().strides;
    const dims_t &dims0 = src0_md.dims();
    const dims_t &dims1 = src1_md.dims();
    const int ndims = src0_md.ndims();

    bool is_bcast = false;
    for (int d = 1; d < ndims; d++)
        is_bcast = is_bcast || dims0[d] != dims1[d];
    if (is_bcast) return true;

    bool same_layouts = true;
    // For batch size == 1, the first dimension is ignored for stride checks,
    // as non-contiguous strides in this dimension do not affect correctness.
    int start_dim = (dims0[0] == 1 && dims1[0] == 1) ? 1 : 0;
    for (int d = start_dim; d < ndims; ++d)
        same_layouts = same_layouts && strides0[d] == strides1[d];
    return same_layouts;
}

static dim_t get_different_layout_stride(
        const strides_t &strides0, const strides_t &strides1, const int ndims) {
    for (int d = 0; d < ndims; d++)
        if (strides0[d] == 1) return strides1[d];
    return strides1[ndims - 1];
}

static dim_t get_outer_dims_product(
        const strides_t &strides0, const dims_t &dims, const int ndims) {
    // nchw:nhwc->nchw
    if (strides0[1] == 1) return dims[1];
    // nhwc:nchw->nhwc
    else if (strides0[ndims - 1] == 1)
        return utils::array_product(dims + 2, ndims - 2);
    else
        return dims[ndims - 1];
}

// Runtime-dispatch analog of x64's data_type_supported(dtype, isa): this
// primitive is pure JIT registered via CPU_INSTANCE_RV64, so the dtype gate
// carries the mayiuse checks for the f16 (zvfh) and bf16 (zvfbfwma) paths.
// f32/s32/s8/u8 are always supported (as on x64).
static bool data_type_supported(const data_type_t dtype) {
    switch (dtype) {
        case data_type::f16: return mayiuse(zvfh);
        case data_type::bf16: return mayiuse(zvfbfwma);
        case data_type::f32:
        case data_type::s32:
        case data_type::s8:
        case data_type::u8: return true;
        default: return false;
    }
}

static bool data_format_supported(const memory_desc_wrapper &mdw) {
    if (mdw.is_plain()) return true;
    // The {16, 8, 4} channel-block family x64 supports on its widest isa;
    // RVV's VLA code has no per-isa block split.
    const auto blk_size = mdw.blocking_desc().inner_blks[0];
    return utils::one_of(blk_size, 16, 8, 4);
}

status_t jit_uni_binary_t::pd_t::init(const engine_t *engine) {
    UNUSED(engine);
    using sm = primitive_attr_t::skip_mask_t;

    conf_.dst_type = dst_md()->data_type;
    conf_.src0_type = src_md(0)->data_type;
    conf_.src1_type = src_md(1)->data_type;

    memory_desc_wrapper dst_md_(dst_md());
    memory_desc_wrapper src0_md_(src_md(0));
    memory_desc_wrapper src1_md_(src_md(1));

    const auto &po = attr()->post_ops_;
    const int elt_idx = po.find(primitive_kind::eltwise);
    conf_.is_i8 = utils::one_of(conf_.dst_type, data_type::s8, data_type::u8);

    // RVV is not part of the rv64gc baseline: gate the whole primitive on the
    // V extension at runtime.
    VDISPATCH_BINARY(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

    // The supported algorithms are checked explicitly (instead of accepting
    // whatever the common layer validated) to avoid silently taking on new,
    // unsupported algorithms in the future; the set matches x64 (aarch64
    // additionally excludes binary_select).
    const bool alg_ok = utils::one_of(desc()->alg_kind, alg_kind::binary_add,
            alg_kind::binary_sub, alg_kind::binary_mul, alg_kind::binary_div,
            alg_kind::binary_ge, alg_kind::binary_gt, alg_kind::binary_le,
            alg_kind::binary_lt, alg_kind::binary_eq, alg_kind::binary_ne,
            alg_kind::binary_max, alg_kind::binary_min,
            alg_kind::binary_select);
    VDISPATCH_BINARY(alg_ok, VERBOSE_BAD_ALGORITHM);

    VDISPATCH_BINARY(
            data_type_supported(conf_.dst_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BINARY(
            data_type_supported(conf_.src0_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BINARY(
            data_type_supported(conf_.src1_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BINARY(data_format_supported(src0_md_), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_BINARY_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_BINARY(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src0");
    // x64's non-int8 rule: src0 and dst share one memory desc (layout and
    // dtype). i8 (quantized) dst may differ from src0 (e.g. f32 src0 -> s8 dst).
    VDISPATCH_BINARY(IMPLICATION(!conf_.is_i8, src0_md_ == dst_md_),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");
    bool use_standard_strategy = is_applicable();
    VDISPATCH_BINARY(attr()->has_default_values(sm::post_ops | sm::scales),
            VERBOSE_UNSUPPORTED_ATTR);
    // Resolve formats before classifying the post-op chain: the post-op
    // binary rhs may arrive as `any`, and post_ops_ok() compares the rhs
    // layout to dst -- x64's ordering, so a dst-shaped (per_element) rhs is
    // not spuriously rejected.
    VDISPATCH_BINARY_SC(
            attr_.set_default_formats(dst_md(0)), VERBOSE_UNSUPPORTED_POSTOP);

    // All operations over blocking descriptors should have md initialized.
    conf_.is_src_different_layouts = !compare_layouts(src0_md_, src1_md_);
    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    po, src0_md_, get_supported_postops_bcast_strategies());
    const auto &src0_blocking = src0_md_.blocking_desc();
    // The synchronized driver is efficient for flat tensor/scalar work, but
    // slices a plain broadcast into short x64-shaped calls. RVV can retain a
    // longer VLA run by coalescing the physical dimensions instead.
    const bool generic_plain_bcast_strategy = src0_md_.is_plain()
            && src1_md_.is_plain() && !is_tensor_op()
            && src1_md_.nelems(false) != 1;
    const bool generic_postops_strategy
            = (elt_idx != -1 && !dst_md_.is_dense()
                      && !cpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                              po.entry_[elt_idx].eltwise))
            || (postops_per_oc_broadcast_exists
                    && (conf_.is_src_different_layouts
                            || (!src0_md_.is_plain()
                                    && src0_blocking.inner_nblks != 1)));
    if (generic_plain_bcast_strategy || generic_postops_strategy)
        use_standard_strategy = false;
    const cpu_isa_t postops_isa
            = mayiuse(zvfbfwma) ? zvfbfwma : (mayiuse(zvfh) ? zvfh : v);
    VDISPATCH_BINARY(
            post_ops_ok(attr(), src_md(0), dst_md(),
                    use_standard_strategy ? conf_.is_src_different_layouts
                                          : false,
                    postops_isa, !use_standard_strategy),
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_BINARY(
            (conf_.is_i8 || elt_idx == -1 || !use_standard_strategy
                    || IMPLICATION(!dst_md_.is_dense(),
                            cpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                                    po.entry_[elt_idx].eltwise))),
            "unsupported datatype or sparse configuration");
    VDISPATCH_BINARY(IMPLICATION((!attr()->scales_.has_default_values()),
                             check_scales_mask()),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    conf_.do_scale_src0 = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
    conf_.do_scale_src1 = !attr()->scales_.has_default_values(DNNL_ARG_SRC_1);
    const auto sum_idx = po.find(primitive_kind::sum);
    conf_.do_sum = sum_idx != -1 && po.entry_[sum_idx].sum.scale != 0.f;
    conf_.with_eltwise = po.find(primitive_kind::eltwise) != -1;
    conf_.with_binary = po.find(primitive_kind::binary) != -1;
    conf_.with_postops
            = conf_.with_binary || conf_.with_eltwise || conf_.do_sum;
    conf_.sum_scale = conf_.do_sum ? po.entry_[sum_idx].sum.scale : 0.f;

    if (!use_standard_strategy) {
        conf_.use_generic_strategy = true;
        VDISPATCH_BINARY(init_generic_conf(),
                "not applicable for current implementation");
        // Generic calls are already sliced at a physical run boundary. Keep
        // post-op RHS addressing in the general gather mode instead of routing
        // it through a channel-aligned x64 strategy.
        conf_.op_type = op_t::none;
        conf_.bcast_type
                = conf_.generic_scalar_inner ? bcast_t::scalar : bcast_t::none;
        conf_.broadcast_src1_value = conf_.generic_scalar_inner;
        conf_.use_stride_src1 = !conf_.generic_scalar_inner;
        conf_.is_src_different_layouts = conf_.src1_stride > 1;
        conf_.outer_dims = conf_.generic_inner;
    } else {
        conf_.postops_per_oc_broadcast_exists = postops_per_oc_broadcast_exists;
        conf_.op_type = get_op_type(src0_md_);

        const auto &bcast_dims = broadcast_dims();
        conf_.bcast_type = is_tensor_op()
                ? bcast_t::none
                : get_bcast_type(src1_md_, bcast_dims);
        // op_type only matters for broadcasted operation
        VDISPATCH_BINARY(IMPLICATION(conf_.bcast_type != bcast_t::none,
                                 conf_.op_type != op_t::none),
                "unsupported src0 layout for broadcast operation");
        // src1 addressing mode (x64 parity): a single value broadcast across the
        // run, or advancing 1:1 with it (else a fixed src1 vector maps to each run,
        // which the driver slices for).
        conf_.broadcast_src1_value
                = (conf_.op_type == op_t::n_c_spatial
                          && conf_.bcast_type == bcast_t::per_c)
                || (utils::one_of(
                            conf_.op_type, op_t::n_spatial_c, op_t::c_blocked)
                        && conf_.bcast_type == bcast_t::per_w)
                || conf_.bcast_type == bcast_t::scalar;
        conf_.use_stride_src1 = !conf_.broadcast_src1_value
                && (utils::one_of(
                            conf_.bcast_type, bcast_t::none, bcast_t::per_batch)
                        || (conf_.op_type == op_t::n_spatial_c
                                && conf_.bcast_type == bcast_t::per_c)
                        || (conf_.op_type == op_t::n_c_spatial
                                && conf_.bcast_type == bcast_t::per_w));

        const auto ndims = src0_md_.ndims();
        if (conf_.is_src_different_layouts) {
            const auto &strides0 = src0_md_.blocking_desc().strides;
            const auto &strides1 = src1_md_.blocking_desc().strides;
            conf_.src1_stride
                    = get_different_layout_stride(strides0, strides1, ndims);
            conf_.outer_dims
                    = get_outer_dims_product(strides0, src0_md_.dims(), ndims);
        }
        if (conf_.bcast_type == bcast_t::per_w) {
            for (int d = 2; d < ndims; ++d)
                conf_.not_bcasted_sp_dims += !bcast_dims[d];
        }
    }

    if (is_ternary_op()) {
        conf_.is_ternary_op = true;
        // The common binary descriptor forces the condition src2 to s8; keep
        // the check explicit (x64 validates the src2 dtype the same way).
        VDISPATCH_BINARY(
                src_md(2)->data_type == data_type::s8, VERBOSE_UNSUPPORTED_DT);
    }

    return status::success;
}

op_t jit_uni_binary_t::pd_t::get_op_type(const memory_desc_wrapper &src0_d) {
    const auto &strides = src0_d.blocking_desc().strides;
    const auto ndims = src0_d.ndims();

    if (!src0_d.is_plain() && src0_d.blocking_desc().inner_idxs[0] == 1)
        return op_t::c_blocked;
    else if (strides[1] == 1)
        return op_t::n_spatial_c;
    else if (strides[0] >= strides[1]
            && IMPLICATION(ndims >= 3, strides[1] >= strides[2]))
        return op_t::n_c_spatial;
    return op_t::none;
}

bool jit_uni_binary_t::pd_t::is_only_dim0_bcasted(
        const dims_t &bcast_dims, const int ndims) {
    bool only_dim0_bcasted = true;
    for (int d = 1; d < ndims; d++)
        only_dim0_bcasted = only_dim0_bcasted && bcast_dims[d] == 0;
    return only_dim0_bcasted;
}

// non-blocked: nxc || ncx
bool jit_uni_binary_t::pd_t::is_format_non_blocked(
        const memory_desc_wrapper &mdw) const {
    const auto &dims = mdw.dims();
    const auto &strides = mdw.blocking_desc().strides;
    const auto &ndims = mdw.ndims();

    const bool is_ncx
            = IMPLICATION(strides[0] != 0,
                      strides[0] >= utils::array_product(dims + 1, ndims - 1))
            && IMPLICATION(ndims >= 3 && strides[1] != 0,
                    strides[1] >= utils::array_product(dims + 2, ndims - 2))
            && IMPLICATION(ndims >= 4 && strides[2] != 0,
                    strides[2] >= utils::array_product(dims + 3, ndims - 3))
            && IMPLICATION(ndims >= 5 && strides[3] != 0,
                    strides[3] >= utils::array_product(dims + 4, ndims - 4))
            && IMPLICATION(strides[ndims - 1] != 0, strides[ndims - 1] == 1);
    const bool is_nxc
            = IMPLICATION(strides[0] != 0,
                      strides[0] >= utils::array_product(dims + 1, ndims - 1))
            && IMPLICATION(ndims >= 3 && strides[2] != 0,
                    strides[2] >= dims[1]
                                    * utils::array_product(dims + 3, ndims - 3))
            && IMPLICATION(ndims >= 4 && strides[3] != 0,
                    strides[3] >= dims[1]
                                    * utils::array_product(dims + 4, ndims - 4))
            && IMPLICATION(ndims >= 5 && strides[4] != 0,
                    strides[4] >= dims[1]
                                    * utils::array_product(dims + 5, ndims - 5))
            && IMPLICATION(strides[1] != 0, strides[1] == 1);
    return is_nxc || is_ncx;
}

bcast_t jit_uni_binary_t::pd_t::get_bcast_type(
        const memory_desc_wrapper &src1_d, const dims_t &bcast_dims) {
    if (src1_d.nelems() == 1)
        return bcast_t::scalar;
    else if (bcast_dims[1] == 1)
        return bcast_t::per_w;
    else if (is_only_dim0_bcasted(bcast_dims, src1_d.ndims()))
        return bcast_t::per_batch;
    else
        return bcast_t::per_c;
}

bool jit_uni_binary_t::pd_t::alg_preserves_zero() const {
    using namespace utils;
    using namespace alg_kind;
    return utils::one_of(desc()->alg_kind, binary_add, binary_max, binary_min,
            binary_mul, binary_sub, binary_ge, binary_gt, binary_le, binary_lt,
            binary_eq, binary_ne, binary_select);
}

bool jit_uni_binary_t::pd_t::check_scales_mask() const {
    const std::vector<int> supported_args = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};
    return attr_scales_ok(supported_args);
}

bool jit_uni_binary_t::pd_t::is_bcast_pattern(const dims_t &bcast_dims,
        const dim_t ndims, const dim_t N_bcast, const dim_t C_bcast,
        const dim_t W_bcast) const {
    return bcast_dims[0] == N_bcast && bcast_dims[1] == C_bcast
            && bcast_dims[ndims - 1] == W_bcast;
}

bool jit_uni_binary_t::pd_t::is_bcast_allowed(const int ndims) const {
    // supported cases: NxCxDxHxW:{NxCx1x1x1,1xCx1x1x1,Nx1xDxHxW,Nx1x1xHxW,
    //                            Nx1x1x1xW,1xCxDxHxW,1x1xDxHxW,1x1x1xHxW,
    //                            1x1x1x1xW,1x1x1x1x1}
    const auto &bcast_dims = broadcast_dims();
    // check if there is continuous broadcast between non-broadcast dims
    // if next_bcast_expected == 1, not broadcast dim not met
    int next_bcast_expected = 1;
    bool sp_not_bcasted = true;
    bool ok = true;
    for (int d = 2; d < ndims; ++d) {
        if (bcast_dims[d] == 0)
            next_bcast_expected = 0;
        else
            sp_not_bcasted = false;
        ok = ok && bcast_dims[d] == next_bcast_expected;
    }

#define BCAST_PATTERN(N, C, W, condition) \
    (is_bcast_pattern(bcast_dims, ndims, N, C, W) && (condition))
    if (ndims > 2)
        ok = ok
                && (BCAST_PATTERN(0, 1, 0, true) || BCAST_PATTERN(1, 1, 0, true)
                        || BCAST_PATTERN(1, 0, 0, sp_not_bcasted)
                        || BCAST_PATTERN(0, 0, 1, !!next_bcast_expected)
                        || BCAST_PATTERN(1, 0, 1, !!next_bcast_expected)
                        || BCAST_PATTERN(1, 1, 1, !!next_bcast_expected));
#undef BCAST_PATTERN
    return ok;
}

// check for different src formats with same dims
// broadcast can be accepted if src_dim == src1_dims (1 == 1)
bool jit_uni_binary_t::pd_t::is_different_layouts_allowed(
        const memory_desc_wrapper &src0_d,
        const memory_desc_wrapper &src1_d) const {
    const dims_t &src0_dims = src0_d.dims();
    const dims_t &src1_dims = src1_d.dims();
    const int ndims = src0_d.ndims();

    bool without_bcast = true;
    for (int d = 0; d < ndims; d++)
        without_bcast = without_bcast && src0_dims[d] == src1_dims[d];
    if (!without_bcast) return false;

    // allow nchw:nhwc and nhwc:nchw and disable for blocked layouts
    return src0_d.is_plain() && src1_d.is_plain()
            && is_format_non_blocked(src0_d) && is_format_non_blocked(src1_d);
}

bool jit_uni_binary_t::pd_t::is_applicable() {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper src2_d(src_md(2));
    const memory_desc_wrapper dst_d(dst_md());
    const auto ndims = src0_d.ndims();

    // check density first to avoid same non-dense src0 and src1 to pass
    // the next check
    bool ok = src0_d.is_dense(true) && src1_d.is_dense(true)
            && dst_d.is_dense(true);
    ok = ok
            && IMPLICATION(
                    is_ternary_op(), src2_d.similar_to(src0_d, true, false, 0));
    if (!ok) return false;

    // Keep x64's padded-tensor family: a single blocking with block size <= 16.
    const auto &blk_d = dst_d.blocking_desc();
    if (!dst_d.is_dense()
            && (blk_d.inner_nblks > 1 || blk_d.inner_blks[0] > 16))
        return false;

    const bool is_src_different_layouts = !compare_layouts(src0_d, src1_d);
    const bool different_layouts_allowed
            = is_different_layouts_allowed(src0_d, src1_d);
    if (!conf_.is_i8) {
        // Non-i8: padded tensors require a zero-preserving algorithm. (x64
        // computes the padded lanes and relies on 0 OP 0 == 0; the rv64 driver
        // explicitly zeroes the padded tail, but mirrors the accept surface.)
        const bool has_padding = utils::one_of(true,
                src0_d.nelems(true) != src0_d.nelems(false),
                src1_d.nelems(true) != src1_d.nelems(false),
                dst_d.nelems(true) != dst_d.nelems(false));
        ok = IMPLICATION(has_padding, alg_preserves_zero());
        if (!ok) return false;

        // full tensor operation
        bool same_dims = true;
        const auto &src0_dims = src0_d.dims();
        const auto &src1_dims = src1_d.dims();
        for (int d = 0; d < ndims; d++)
            same_dims = same_dims && src0_dims[d] == src1_dims[d];
        if (same_dims
                && IMPLICATION(
                        is_src_different_layouts, different_layouts_allowed))
            return true;
    } else {
        const dim_t C = ndims >= 2 ? src0_d.dims()[1] : 1;
        const bool has_oc_tail = C != src0_d.padded_dims()[1];
        const bool has_outer_dims_tail = is_src_different_layouts
                && get_outer_dims_product(src0_d.blocking_desc().strides,
                        src0_d.dims(), src0_d.ndims());

        // Disable compare operations when blocked tag with tail. Tail
        // processing is not supported and the comparison overwrites the output.
        if (utils::one_of(desc()->alg_kind, alg_kind::binary_ge,
                    alg_kind::binary_gt, alg_kind::binary_le,
                    alg_kind::binary_lt, alg_kind::binary_eq,
                    alg_kind::binary_ne)
                && (has_oc_tail || has_outer_dims_tail))
            return false;

        // The kernel walks src0/src1/dst in flat physical lockstep, so an i8
        // dst -- whose md is allowed to differ from src0's (f32 src0 -> s8 dst)
        // -- must still share src0's physical layout; only the dtype may
        // differ. This has to precede the full-tensor early return below:
        // otherwise a same-shaped src1 lets a differently-laid-out dst through
        // and the three walks desynchronize. It also keeps the original
        // meaning for the broadcast path (source0 broadcast not supported).
        if (!src0_d.similar_to(dst_d, true, false, 0)) return false;

        // full tensor operation
        if (src0_d.similar_to(src1_d, true, false, 0)
                || different_layouts_allowed)
            return true;
    }

    // broadcast or different layouts operation
    if (!(is_bcast_allowed(ndims)
                && IMPLICATION(
                        is_src_different_layouts, different_layouts_allowed)))
        return false;

    // only nspc and ncsp formats are supported for bcast
    if (src0_d.is_plain() && src1_d.is_plain())
        return is_format_non_blocked(src0_d) && is_format_non_blocked(src1_d)
                && get_op_type(src0_d) != op_t::none;

    // blocked formats
    if (!conf_.is_i8) {
        // x64 additionally requires the channel block to equal the machine simd
        // width; the VLA analog keeps the {16, 8, 4} single-C-block family (the
        // driver handles the exact src1/dst block compatibility incl. a scalar
        // src1).
        const auto valid_bd = [&](const memory_desc_wrapper &mdw) {
            const auto &bd = mdw.blocking_desc();
            return bd.inner_nblks == 1
                    && utils::one_of(bd.inner_blks[0], 16, 8, 4)
                    && bd.inner_idxs[0] == 1;
        };
        return valid_bd(src0_d) && valid_bd(src1_d);
    } else {
        const auto &bd0 = src0_d.blocking_desc();
        const auto &bd1 = src1_d.blocking_desc();
        const auto &bcast_dims = broadcast_dims();
        // disable blocked tag for source1 when W is not broadcast
        return bd0.strides[1] == 1 && bd0.inner_nblks == 0
                && IMPLICATION(
                        bcast_dims[ndims - 1] == 0, bd1.inner_nblks == 0);
    }
}

bool jit_uni_binary_t::pd_t::init_generic_conf() {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());
    if (!dst_d.is_dense(true) || !src0_d.is_dense(true)
            || !src1_d.is_dense(true))
        return false;
    if (!src0_d.similar_to(dst_d, true, false)) return false;
    if (is_ternary_op()) {
        const memory_desc_wrapper src2_d(src_md(2));
        if (!src2_d.is_dense(true)
                || !src2_d.similar_to(src0_d, true, false, 0))
            return false;
    }

    conf_.generic_ndims = dst_d.ndims();
    conf_.generic_total = dst_d.nelems(false);
    const bool has_padding = dst_d.nelems(true) != dst_d.nelems(false);

    // Flat tensor and scalar cases keep one long VLA pass per thread.
    if (!has_padding && src1_d.nelems(false) == 1) {
        conf_.generic_whole = true;
        conf_.generic_scalar_inner = true;
        conf_.generic_inner = conf_.generic_total;
        return true;
    }
    if (!has_padding && src1_d.similar_to(dst_d, true, false)) {
        conf_.generic_whole = true;
        conf_.generic_inner = conf_.generic_total;
        return true;
    }

    // Build a physical-dimension iteration plan. dst is walked in contiguous
    // order; each src1 dimension either broadcasts (stride 0) or follows its
    // memory descriptor. A single channel inner block is represented as an
    // extra unit-stride physical dimension.
    const auto &dst_bd = dst_d.blocking_desc();
    const auto &src1_bd = src1_d.blocking_desc();
    const bool src1_scalar = src1_d.nelems(false) == 1;
    if (dst_bd.inner_nblks > 1) return false;

    const int blocked_dim = dst_bd.inner_nblks == 1 ? dst_bd.inner_idxs[0] : -1;
    const dim_t block = dst_bd.inner_nblks == 1 ? dst_bd.inner_blks[0] : 1;
    const bool src1_blocked = src1_bd.inner_nblks == 1 && blocked_dim >= 0
            && src1_bd.inner_idxs[0] == blocked_dim
            && src1_bd.inner_blks[0] == block;
    if (src1_bd.inner_nblks != 0 && !src1_blocked && !src1_scalar) return false;

    const dim_t *dst_dims = dst_d.dims();
    const dim_t *src1_dims = src1_d.dims();
    const auto &dst_strides = dst_bd.strides;
    const auto &src1_strides = src1_bd.strides;
    for (int d = 0; d < conf_.generic_ndims; ++d)
        if (src1_dims[d] != 1 && src1_dims[d] != dst_dims[d]) return false;

    if (has_padding) {
        if (blocked_dim != 1 || block > 16 || (block & (block - 1)) != 0)
            return false;
        for (int d = 0; d < conf_.generic_ndims; ++d)
            if (d != blocked_dim && dst_d.padded_dims()[d] != dst_dims[d])
                return false;
        if (dst_d.padded_dims()[blocked_dim]
                != utils::rnd_up(dst_dims[blocked_dim], block))
            return false;
        conf_.generic_tail = dst_dims[blocked_dim] % block;
    }

    struct physical_dim_t {
        dim_t size;
        dim_t src1_stride;
        dim_t dst_stride;
        int logical_dim;
    } physical_dims[DNNL_MAX_NDIMS + 1];
    int nphysical = 0;
    for (int d = 0; d < conf_.generic_ndims; ++d) {
        const dim_t outer_size = d == blocked_dim
                ? utils::div_up(dst_dims[d], block)
                : dst_dims[d];
        if (outer_size < 1) return false;
        const dim_t src1_stride = src1_dims[d] == 1
                ? 0
                : (d == blocked_dim && !src1_blocked ? src1_strides[d] * block
                                                     : src1_strides[d]);
        physical_dims[nphysical++]
                = {outer_size, src1_stride, dst_strides[d], d};
    }
    for (int a = 0; a < nphysical; ++a)
        for (int b = a + 1; b < nphysical; ++b)
            if (physical_dims[b].dst_stride > physical_dims[a].dst_stride) {
                const auto tmp = physical_dims[a];
                physical_dims[a] = physical_dims[b];
                physical_dims[b] = tmp;
            }
    if (blocked_dim >= 0) {
        const dim_t src1_inner_stride = src1_dims[blocked_dim] == 1
                ? 0
                : (src1_blocked ? 1 : src1_strides[blocked_dim]);
        physical_dims[nphysical++] = {block, src1_inner_stride, 1, blocked_dim};
    }

    // Coalesce adjacent physical dimensions while both dst and src1 remain
    // uniform. Preserve enough outer runs to expose all available threads and
    // keep padded channel blocks separate for explicit tail zeroing.
    const dim_t min_outer = dnnl_get_max_threads();
    while (conf_.generic_tail == 0 && nphysical >= 2) {
        const auto inner = physical_dims[nphysical - 1];
        const auto outer = physical_dims[nphysical - 2];
        if (outer.dst_stride != inner.dst_stride * inner.size) break;
        const bool both_broadcast
                = outer.src1_stride == 0 && inner.src1_stride == 0;
        const bool uniform = inner.src1_stride != 0
                && outer.src1_stride == inner.src1_stride * inner.size;
        if (!both_broadcast && !uniform) break;
        dim_t outer_runs = 1;
        for (int d = 0; d + 2 < nphysical; ++d)
            outer_runs *= physical_dims[d].size;
        if (outer_runs < min_outer) break;
        physical_dims[nphysical - 2].size = outer.size * inner.size;
        physical_dims[nphysical - 2].src1_stride = inner.src1_stride;
        physical_dims[nphysical - 2].dst_stride = inner.dst_stride;
        --nphysical;
    }

    const auto &inner = physical_dims[nphysical - 1];
    if (inner.dst_stride != 1) return false;
    conf_.generic_inner = inner.size;
    conf_.generic_scalar_inner = inner.src1_stride == 0;
    conf_.src1_stride = nstl::max((dim_t)1, inner.src1_stride);
    conf_.generic_n_outer = 1;
    conf_.generic_ndims = nphysical;
    for (int d = 0; d < nphysical - 1; ++d) {
        conf_.generic_outer_dims[d] = physical_dims[d].size;
        conf_.generic_src1_strides[d] = physical_dims[d].src1_stride;
        conf_.generic_n_outer *= physical_dims[d].size;
        if (conf_.generic_tail != 0
                && physical_dims[d].logical_dim == blocked_dim)
            conf_.generic_tail_axis = d;
    }
    if (conf_.generic_tail != 0 && conf_.generic_tail_axis < 0) return false;
    conf_.generic_src1_same_layout = src1_d.similar_to(dst_d, true, false);
    return true;
}

bool jit_uni_binary_t::post_ops_ok(const primitive_attr_t *attr,
        const memory_desc_wrapper &src0_d, const memory_desc_wrapper &dst_d,
        const bool is_src_different_layouts, const cpu_isa_t isa,
        const bool use_generic_strategy) {
    using namespace injector;
    using namespace primitive_kind;

    const auto &p = attr->post_ops_;
    const auto supported_strategies = get_supported_postops_bcast_strategies();
    // The kernel supplies 4 eltwise aux groups (heavy algs available) and
    // keeps v0 dead across the inject point, so the select (ternary) post-op
    // is enabled (x64 accepts it unconditionally).
    if (!injector::post_ops_ok(post_ops_ok_args_t(isa, {binary, eltwise, sum},
                p, &dst_d, false /*sum_at_pos_0_only*/,
                false /*sum_requires_scale_one*/, true /*sum_requires_zp_zero*/,
                true /*sum_requires_same_params*/, supported_strategies,
                4 /*n_vaux*/, true /*allow_binary_select*/)))
        return false;

    // data type of int8 dst is allowed to differ from src0 unless there is a sum postop
    if (p.find(primitive_kind::sum) != -1) {
        if (src0_d.data_type() != dst_d.data_type()) return false;
        // the in-kernel sum reads the old dst back at the dst dtype
        for (int i = 0; i < p.len(); i++) {
            if (!p.entry_[i].is_sum(false, false)) continue;
            const auto &s = p.entry_[i].sum;
            if (s.dt != data_type::undef && s.dt != dst_d.data_type())
                return false;
        }
    }

    // no prelu support
    if (p.find(primitive_kind::prelu) != -1) return false;

    const auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };
    const bool is_i8
            = utils::one_of(dst_d.data_type(), data_type::s8, data_type::u8);

    for (int i = 0; i < p.len(); i++) {
        if (is_binary(i)) {
            const auto &post_ops_mem = p.entry_[i].binary.src1_desc;
            const bool is_src1_xf16 = utils::one_of(
                    post_ops_mem.data_type, data_type::f16, data_type::bf16);
            // x64 parity: an i8 dst rejects an xf16 binary post-op rhs.
            if (is_i8 && is_src1_xf16) return false;
            // TODO: eliminate in favor of check in injectors::post_ops_ok
            // (conditions are slightly different, need to check corner cases)
            if (get_rhs_arg_broadcasting_strategy(
                        post_ops_mem, dst_d, supported_strategies)
                    == broadcasting_strategy_t::no_broadcast) {
                const memory_desc_wrapper post_op_mem_d(post_ops_mem);
                if (!post_op_mem_d.similar_to(dst_d, true, false)) return false;
            }
        }
    }

    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    p, src0_d, supported_strategies);
    if (!use_generic_strategy && postops_per_oc_broadcast_exists
            && is_src_different_layouts)
        return false;

    const bool blocked_format = !src0_d.is_plain() && src0_d.is_blocking_desc();

    if (!use_generic_strategy && postops_per_oc_broadcast_exists
            && blocked_format) {
        /*
         * check blocking_desc consistency: with a per_oc rhs the per-C driver
         * slicing and the injector's channel recovery assume the only inner
         * block is C (x64 additionally pins it to the machine simd width; the
         * VLA analog keeps the {16, 8, 4} family, as in is_applicable()).
         */
        const auto &blocking_desc = src0_d.blocking_desc();
        if (blocking_desc.inner_nblks != 1
                || !utils::one_of(blocking_desc.inner_blks[0], 16, 8, 4)
                || blocking_desc.inner_idxs[0] != 1)
            return false;
    }

    const dim_t n_dims = src0_d.ndims();
    const dim_t &oc = n_dims >= 2 ? src0_d.dims()[1] : 1;

    /*
     * TODO: Remove limitation supporting tail with blocked format for i8i8
     */
    const dim_t blksize
            = blocked_format ? src0_d.blocking_desc().inner_blks[0] : 1;
    const bool blocked_tail = p.len() && blocked_format && oc % blksize;

    return binary_injector::binary_args_broadcast_supported(
                   p, src0_d, get_supported_postops_bcast_strategies())
            && IMPLICATION(utils::one_of(src0_d.data_type(), data_type::s8,
                                   data_type::u8),
                    !blocked_tail);
}

jit_uni_binary_t::jit_uni_binary_t(const pd_t *apd) : primitive_t(apd) {}

status_t jit_uni_binary_t::init(engine_t *engine) {
    UNUSED(engine);
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_binary_kernel_t(
                    pd(), pd()->get_conf(), false /*tail_kernel*/)));

    // The tail kernel stores the valid lanes of each channel block and zeroes
    // the padded tail in-kernel. x64 builds it only for a non-i8 dst, yet its
    // execute() selects the tail kernel without looking at the dst dtype, so an
    // i8 dst with a blocked oc tail dereferences a null kernel there. The rv64
    // tail kernel is dtype-generic (store_vector covers s8/u8, zero_pad_tail
    // writes zero BYTES), so build it whenever the dispatch can ask for it --
    // the creation predicate must equal the execute-side one.
    const memory_desc_wrapper src0_d(pd_->src_md(0));
    const auto oc = src0_d.ndims() >= 2 ? src0_d.dims()[1] : 1;

    if (op_t::c_blocked == pd()->get_conf().op_type
            && oc % src0_d.blocking_desc().inner_blks[0]) {
        CHECK(safe_ptr_assign(kernel_tail_,
                new jit_uni_binary_kernel_t(
                        pd(), pd()->get_conf(), true /*tail_kernel*/)));
        CHECK(kernel_tail_->create_kernel());
    }

    return kernel_->create_kernel();
}

void jit_uni_binary_t::execute_no_bcast_strategy(const data_t *src0,
        const data_t *src1, const data_t *src2, data_t *dst, float scale0,
        float scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        const bcast_t bcast_type) const {
    const auto conf = pd()->get_conf();
    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));

    const size_t es0 = types::data_type_size(conf.src0_type);
    const size_t es1 = types::data_type_size(conf.src1_type);
    const size_t esd = types::data_type_size(conf.dst_type);
    const size_t es2 = conf.is_ternary_op
            ? types::data_type_size(pd()->src_md(2)->data_type)
            : 0;
    const void *const *rhs_ptrs = post_ops_binary_rhs_arg_vec.empty()
            ? nullptr
            : post_ops_binary_rhs_arg_vec.data();

    auto call = [&](dim_t off0, dim_t off1, dim_t offd, dim_t work) {
        jit_uni_binary_args_t p = {};
        p.src0 = src0 + off0 * es0;
        p.src1 = src1 + off1 * es1;
        if (conf.is_ternary_op) p.src2 = src2 + offd * es2;
        p.dst = dst + offd * esd;
        p.post_ops_binary_rhs_arg_vec = rhs_ptrs;
        p.work_amount = work;
        p.dst_orig = dst;
        p.scales_src0 = scale0;
        p.scales_src1 = scale1;
        p.sum_scale = conf.sum_scale;
        (*kernel_)(&p);
    };

    if (conf.is_src_different_layouts) {
        // Different plain layouts (nchw:nhwc): src0/dst are contiguous, src1
        // is read with the element stride conf.src1_stride (vlse). x64
        // batches whole-run chunks per thread and lets the kernel iterate the
        // runs, resetting src1 to the next contiguous element after each
        // stride range; the rv64 kernel does the same with outer_dims and
        // src1_stride baked as codegen constants (instead of x64's indices
        // vector + runtime stride-range register). Consecutive runs start at
        // consecutive src1 elements within a batch, so a chunk's src1 base is
        // batch base + first run index.
        const dim_t batch = src0_d.dims()[0];
        const dim_t run_len = conf.outer_dims;
        const dim_t batch_stride = src1_d.blocking_desc().strides[0];
        const dim_t nelems_per_batch = src0_d.nelems(true) / batch;
        const dim_t runs_per_batch = nelems_per_batch / run_len;

        const int nthr = dnnl_get_current_num_threads();
        const dim_t thr_per_run_group = nstl::min(
                nstl::max((dim_t)nthr / batch, (dim_t)1), runs_per_batch);

        // Compute strategy:
        // Iterate over batch and over runs. Divide number of threads by batch
        // size and limit it by the number of runs to parallelize over them
        // when needed (x64's thr_per_nelems_group).
        parallel_nd(batch, thr_per_run_group, [&](dim_t b, dim_t run_group) {
            dim_t start = 0, end = 0;
            balance211(
                    runs_per_batch, thr_per_run_group, run_group, start, end);
            if (start >= end) return;

            const dim_t off = b * nelems_per_batch + start * run_len;
            call(off, b * batch_stride + start, off, (end - start) * run_len);
        });
        return;
    }

    // Plain no-broadcast (or point broadcast): one flat pass, split by threads.
    const dim_t nelems = src0_d.nelems(true);
    const bool point_broadcast = bcast_type == bcast_t::scalar;
    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start = 0, end = 0;
        balance211(nelems, nthr, ithr, start, end);
        if (start >= end) return;
        call(start, point_broadcast ? 0 : start, start, end - start);
    });
}

void jit_uni_binary_t::execute_bcast_per_batch_strategy(const data_t *src0,
        const data_t *src1, const data_t *src2, data_t *dst, float scale0,
        float scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec) const {
    const auto conf = pd()->get_conf();
    const memory_desc_wrapper src0_d(pd()->src_md(0));

    const size_t es0 = types::data_type_size(conf.src0_type);
    const size_t es1 = types::data_type_size(conf.src1_type);
    const size_t esd = types::data_type_size(conf.dst_type);
    const size_t es2 = conf.is_ternary_op
            ? types::data_type_size(pd()->src_md(2)->data_type)
            : 0;
    const void *const *rhs_ptrs = post_ops_binary_rhs_arg_vec.empty()
            ? nullptr
            : post_ops_binary_rhs_arg_vec.data();

    const dim_t MB = src0_d.dims()[0];
    const dim_t nelems_per_b = src0_d.nelems(true) / MB;

    // Compute strategy: src1 is one per-batch block reused for every batch;
    // parallelize over MB and chunks of the per-batch elements (src1 advances
    // 1:1 within a chunk, resetting per batch).
    const dim_t nthr
            = nstl::min(nelems_per_b, (dim_t)dnnl_get_current_num_threads());
    parallel_nd(MB, nthr, [&](dim_t b, dim_t ithr) {
        dim_t start = 0, end = 0;
        balance211(nelems_per_b, nthr, ithr, start, end);
        if (start >= end) return;
        const dim_t off = b * nelems_per_b + start;
        jit_uni_binary_args_t p = {};
        p.src0 = src0 + off * es0;
        p.src1 = src1 + start * es1;
        if (conf.is_ternary_op) p.src2 = src2 + off * es2;
        p.dst = dst + off * esd;
        p.post_ops_binary_rhs_arg_vec = rhs_ptrs;
        p.work_amount = end - start;
        p.dst_orig = dst;
        p.scales_src0 = scale0;
        p.scales_src1 = scale1;
        p.sum_scale = conf.sum_scale;
        (*kernel_)(&p);
    });
}

void jit_uni_binary_t::execute_bcast_per_c_strategy(const data_t *src0,
        const data_t *src1, const data_t *src2, data_t *dst, float scale0,
        float scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        const op_t op_type, const bcast_t bcast_type,
        const bool blocked_oc_tail) const {
    const auto conf = pd()->get_conf();
    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const size_t es0 = types::data_type_size(conf.src0_type);
    const size_t es1 = types::data_type_size(conf.src1_type);
    const size_t esd = types::data_type_size(conf.dst_type);
    const size_t es2 = conf.is_ternary_op
            ? types::data_type_size(pd()->src_md(2)->data_type)
            : 0;
    const void *const *rhs_ptrs = post_ops_binary_rhs_arg_vec.empty()
            ? nullptr
            : post_ops_binary_rhs_arg_vec.data();

    const auto ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const dim_t MB = dims[0];
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t SP = ndims >= 3 ? utils::array_product(dims + 2, ndims - 2) : 1;
    const auto &bcast_dims = pd()->broadcast_dims();

    const dim_t nelems_slice_src0
            = utils::array_product(src0_d.padded_dims() + 1, ndims - 1);
    const dim_t nelems_slice_src1 = bcast_type == bcast_t::none
            ? nelems_slice_src0
            : ((bcast_dims[0] == 0)
                              ? utils::array_product(
                                        src1_d.padded_dims() + 1, ndims - 1)
                              : 0);

    auto call = [&](binary_kernel_t *kernel, dim_t off0, dim_t off1, dim_t offd,
                        dim_t work) {
        jit_uni_binary_args_t p = {};
        p.src0 = src0 + off0 * es0;
        p.src1 = src1 + off1 * es1;
        if (conf.is_ternary_op) p.src2 = src2 + offd * es2;
        p.dst = dst + offd * esd;
        p.post_ops_binary_rhs_arg_vec = rhs_ptrs;
        p.work_amount = work;
        p.dst_orig = dst;
        p.scales_src0 = scale0;
        p.scales_src1 = scale1;
        p.sum_scale = conf.sum_scale;
        (*kernel)(&p);
    };

    if (op_type == op_t::c_blocked) {
        const dim_t c_block = dst_d.blocking_desc().inner_blks[0];
        const dim_t C_blocks = utils::div_up(dst_d.padded_dims()[1], c_block);
        // Compute strategy:
        // Each block is individual - parallel over MB and C_blocks safely.
        // One kernel call covers the whole SP run (x64 granularity); the
        // kernel steps one channel block per iteration, re-reading the fixed
        // per_c src1 vector each block (x64's offt_src1_ == 0). The last
        // block goes to kernel_tail_, which stores the valid lanes and zeroes
        // the padded tail in-kernel.
        const auto src1_off = [&](dim_t mb, dim_t C_blk, dim_t off) -> dim_t {
            switch (bcast_type) {
                case bcast_t::scalar: return mb * nelems_slice_src1;
                case bcast_t::per_batch: return C_blk * SP * c_block;
                case bcast_t::none: return off;
                default: return mb * nelems_slice_src1 + C_blk * c_block;
            }
        };
        parallel_nd(MB, C_blocks, [&](dim_t mb, dim_t C_blk) {
            const dim_t off = mb * nelems_slice_src0 + C_blk * SP * c_block;
            auto *kernel = blocked_oc_tail && C_blk == (C_blocks - 1)
                    ? kernel_tail_.get()
                    : kernel_.get();
            call(kernel, off, src1_off(mb, C_blk, off), off, SP * c_block);
        });
    } else if (op_type == op_t::n_spatial_c) {
        // Each line of channels is individual, parallel over MB and spatial.
        // src1 (per_c) is the C-vector reused per spatial position, advancing
        // 1:1 with the run (use_stride_src1).
        const auto src1_off = [&](dim_t mb, dim_t off) -> dim_t {
            switch (bcast_type) {
                case bcast_t::per_batch: return off - mb * nelems_slice_src0;
                case bcast_t::none: return off;
                default: return mb * nelems_slice_src1;
            }
        };
        parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
            const dim_t off = mb * nelems_slice_src0 + sp * C;
            call(kernel_.get(), off, src1_off(mb, off), off, C);
        });
    } else if (op_type == op_t::n_c_spatial) {
        // Each line of spatial is individual, parallel over MB and C. src1
        // (per_c) is one value per channel broadcast over the SP run
        // (broadcast_src1_value).
        const auto src1_off = [&](dim_t mb, dim_t c, dim_t off) -> dim_t {
            switch (bcast_type) {
                case bcast_t::scalar: return mb * nelems_slice_src1;
                case bcast_t::per_batch: return c * SP;
                case bcast_t::none: return off;
                default: return mb * nelems_slice_src1 + c;
            }
        };
        parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
            const dim_t off = mb * nelems_slice_src0 + c * SP;
            call(kernel_.get(), off, src1_off(mb, c, off), off, SP);
        });
    }
}

void jit_uni_binary_t::execute_bcast_per_w_strategy(const data_t *src0,
        const data_t *src1, const data_t *src2, data_t *dst, float scale0,
        float scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        const op_t op_type, const bool blocked_oc_tail) const {
    const auto conf = pd()->get_conf();
    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const size_t es0 = types::data_type_size(conf.src0_type);
    const size_t es1 = types::data_type_size(conf.src1_type);
    const size_t esd = types::data_type_size(conf.dst_type);
    const size_t es2 = conf.is_ternary_op
            ? types::data_type_size(pd()->src_md(2)->data_type)
            : 0;
    const void *const *rhs_ptrs = post_ops_binary_rhs_arg_vec.empty()
            ? nullptr
            : post_ops_binary_rhs_arg_vec.data();

    const auto ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const auto &bcast_dims = pd()->broadcast_dims();
    const int not_bcasted_sp_dims = conf.not_bcasted_sp_dims;
    const dim_t MB = dims[0];
    const dim_t SP_no_bcast = ndims >= 3
            ? utils::array_product(
                      dims + (ndims - not_bcasted_sp_dims), not_bcasted_sp_dims)
            : 1;
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t SP = ndims >= 3 ? utils::array_product(dims + 2, ndims - 2) : 1;
    const dim_t N = SP / SP_no_bcast; // broadcasted spatial dims
    const dim_t nelems_slice_src0
            = utils::array_product(src0_d.padded_dims() + 1, ndims - 1);

    auto call = [&](binary_kernel_t *kernel, dim_t off0, dim_t off1, dim_t offd,
                        dim_t work) {
        jit_uni_binary_args_t p = {};
        p.src0 = src0 + off0 * es0;
        p.src1 = src1 + off1 * es1;
        if (conf.is_ternary_op) p.src2 = src2 + offd * es2;
        p.dst = dst + offd * esd;
        p.post_ops_binary_rhs_arg_vec = rhs_ptrs;
        p.work_amount = work;
        p.dst_orig = dst;
        p.scales_src0 = scale0;
        p.scales_src1 = scale1;
        p.sum_scale = conf.sum_scale;
        (*kernel)(&p);
    };

    if (op_type == op_t::c_blocked) {
        const dim_t c_block = dst_d.blocking_desc().inner_blks[0];
        const dim_t C_blocks = utils::div_up(dst_d.padded_dims()[1], c_block);
        // Each (mb, C_blk, n, sp) block: one src1 value (per_w, broadcast over
        // C) broadcast across the c_block channels (broadcast_src1_value).
        // The per_w src1 value is real data, so the last block's padded lanes
        // must not compute 0 OP src1: they go to kernel_tail_, which stores
        // the valid lanes and zeroes the padded tail in-kernel (x64 shape).
        parallel_nd(MB, C_blocks, N, SP_no_bcast,
                [&](dim_t mb, dim_t C_blk, dim_t n, dim_t sp) {
            const dim_t off = mb * nelems_slice_src0
                    + c_block * (C_blk * SP + n * SP_no_bcast + sp);
            const dim_t s1 = bcast_dims[0] == 1
                    ? sp * c_block
                    : (mb * SP_no_bcast + sp) * c_block;
            auto *kernel = blocked_oc_tail && C_blk == (C_blocks - 1)
                    ? kernel_tail_.get()
                    : kernel_.get();
            call(kernel, off, s1, off, c_block);
        });
    } else if (op_type == op_t::n_spatial_c) {
        // Each line of channels: one src1 value (per_w) broadcast over C
        // (broadcast_src1_value).
        parallel_nd(MB, N, SP_no_bcast, [&](dim_t mb, dim_t n, dim_t sp) {
            const dim_t off
                    = mb * nelems_slice_src0 + n * SP_no_bcast * C + sp * C;
            const dim_t s1 = bcast_dims[0] == 1 ? sp : mb * SP_no_bcast + sp;
            call(kernel_.get(), off, s1, off, C);
        });
    } else if (op_type == op_t::n_c_spatial) {
        // Each line of width: the src1 W-vector advances 1:1 over SP_no_bcast
        // (use_stride_src1), reused across C and the broadcasted spatial dims.
        parallel_nd(MB, C, N, [&](dim_t mb, dim_t c, dim_t n) {
            const dim_t off = mb * nelems_slice_src0 + c * N * SP_no_bcast
                    + n * SP_no_bcast;
            const dim_t s1 = bcast_dims[0] == 1 ? 0 : mb * SP_no_bcast;
            call(kernel_.get(), off, s1, off, SP_no_bcast);
        });
    }
}

void jit_uni_binary_t::execute_generic_strategy(const data_t *src0,
        const data_t *src1, const data_t *src2, data_t *dst, float scale0,
        float scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec) const {
    const auto conf = pd()->get_conf();
    const size_t es0 = types::data_type_size(conf.src0_type);
    const size_t es1 = types::data_type_size(conf.src1_type);
    const size_t esd = types::data_type_size(conf.dst_type);
    const size_t es2 = conf.is_ternary_op
            ? types::data_type_size(pd()->src_md(2)->data_type)
            : 0;
    const void *const *rhs_ptrs = post_ops_binary_rhs_arg_vec.empty()
            ? nullptr
            : post_ops_binary_rhs_arg_vec.data();

    auto call = [&](dim_t off0, dim_t off1, dim_t offd, dim_t work) {
        jit_uni_binary_args_t p = {};
        p.src0 = src0 + off0 * es0;
        p.src1 = src1 + off1 * es1;
        if (conf.is_ternary_op) p.src2 = src2 + offd * es2;
        p.dst = dst + offd * esd;
        p.post_ops_binary_rhs_arg_vec = rhs_ptrs;
        p.work_amount = work;
        p.scales_src0 = scale0;
        p.scales_src1 = scale1;
        p.sum_scale = conf.sum_scale;
        p.dst_orig = dst;
        (*kernel_)(&p);
    };

    if (conf.generic_whole) {
        parallel(0, [&](int ithr, int nthr) {
            dim_t start = 0, end = 0;
            balance211(conf.generic_total, nthr, ithr, start, end);
            if (start >= end) return;
            call(start, conf.generic_scalar_inner ? 0 : start, start,
                    end - start);
        });
        return;
    }

    parallel(0, [&](int ithr, int nthr) {
        dim_t begin = 0, end = 0;
        balance211(conf.generic_n_outer, nthr, ithr, begin, end);
        for (dim_t outer = begin; outer < end; ++outer) {
            dim_t src1_off = 0;
            dim_t rem = outer;
            bool tail_run = false;
            for (int d = conf.generic_ndims - 2; d >= 0; --d) {
                const dim_t dim = conf.generic_outer_dims[d];
                const dim_t index = rem % dim;
                rem /= dim;
                src1_off += index * conf.generic_src1_strides[d];
                if (d == conf.generic_tail_axis) tail_run = index == dim - 1;
            }
            if (conf.generic_src1_same_layout)
                src1_off = outer * conf.generic_inner;
            const dim_t run = tail_run ? conf.generic_tail : conf.generic_inner;
            const dim_t dst_off = outer * conf.generic_inner;
            call(dst_off, src1_off, dst_off, run);
            if (run < conf.generic_inner)
                std::memset(dst + (dst_off + run) * esd, 0,
                        (conf.generic_inner - run) * esd);
        }
    });
}

status_t jit_uni_binary_t::execute(const exec_ctx_t &ctx) const {
    auto src0 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_0);
    auto src1 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const auto &post_ops = pd()->attr()->post_ops_;
    const auto &post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(post_ops, ctx);

    const auto conf = pd()->get_conf();

    // Per-tensor scale values are resolved once here and passed by value (x64
    // forwards the scale pointers to the kernel instead).
    float scale0 = 1.f, scale1 = 1.f;
    if (conf.do_scale_src0)
        scale0 = *CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0);
    if (conf.do_scale_src1)
        scale1 = *CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1);

    // Honor md offsets (dense submemory views).
    src0 += (size_t)pd()->src_md(0)->offset0
            * types::data_type_size(conf.src0_type);
    src1 += (size_t)pd()->src_md(1)->offset0
            * types::data_type_size(conf.src1_type);
    dst += (size_t)pd()->dst_md()->offset0
            * types::data_type_size(conf.dst_type);

    const data_t *src2 = nullptr;
    if (conf.is_ternary_op) {
        src2 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_2);
        src2 += (size_t)pd()->src_md(2)->offset0
                * types::data_type_size(pd()->src_md(2)->data_type);
    }

    const auto op_type = conf.op_type;
    const auto bcast_type = conf.bcast_type;
    const bool point_broadcast = bcast_type == bcast_t::scalar;
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const dim_t C = dst_d.ndims() >= 2 ? dst_d.dims()[1] : 1;
    const bool with_postops = !post_ops.entry_.empty();
    const bool has_oc_tail = op_type == op_t::c_blocked
            && (C % dst_d.blocking_desc().inner_blks[0]);
    const bool point_broadcast_no_oc_tail = point_broadcast && !has_oc_tail;
    const auto alg = pd()->desc()->alg_kind;

    // Use the tail-handling per_c/per_w strategies for compare ops with
    // oc_tail and blocked format due to overwriting the padded lanes, and
    // whenever post-ops could write them (x64 routes the same cases to its
    // kernel_tail_; the rv64 driver computes the valid lanes and zeroes the
    // padded tail itself).
    const bool vector_overwrite = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    const bool blocked_oc_tail = op_type == op_t::c_blocked && has_oc_tail
            && (with_postops || point_broadcast || bcast_type == bcast_t::per_w
                    || vector_overwrite);
    // init()'s creation predicate must cover every case this dispatch routes to
    // kernel_tail_ (it is strictly weaker: c_blocked && oc % blk).
    assert(IMPLICATION(blocked_oc_tail, kernel_tail_ != nullptr));

    // x64 also forces the per_c/per_w strategies whenever a per_oc post-op
    // rhs exists so its per-call channel addressing stays valid; the rv64
    // kernel uses the same channel-aligned addressing under the identical
    // routing (see conf). An op_t::none src0 cannot be routed (no per_c
    // strategy branch would run) and keeps the gather-addressed flat path;
    // x64's per_w-rhs condition is not needed (the per-lane gather stays in
    // bounds without the expanded scratchpad copy).
    const bool route_postops_per_oc
            = conf.postops_per_oc_broadcast_exists && op_type != op_t::none;

    if (conf.use_generic_strategy)
        execute_generic_strategy(src0, src1, src2, dst, scale0, scale1,
                post_ops_binary_rhs_arg_vec);
    else if ((bcast_type == bcast_t::none || point_broadcast_no_oc_tail)
            && !route_postops_per_oc && !blocked_oc_tail)
        execute_no_bcast_strategy(src0, src1, src2, dst, scale0, scale1,
                post_ops_binary_rhs_arg_vec, bcast_type);
    else if (bcast_type == bcast_t::per_batch && !route_postops_per_oc
            && !blocked_oc_tail)
        execute_bcast_per_batch_strategy(src0, src1, src2, dst, scale0, scale1,
                post_ops_binary_rhs_arg_vec);
    else if (bcast_type == bcast_t::per_w)
        execute_bcast_per_w_strategy(src0, src1, src2, dst, scale0, scale1,
                post_ops_binary_rhs_arg_vec, op_type, blocked_oc_tail);
    else
        execute_bcast_per_c_strategy(src0, src1, src2, dst, scale0, scale1,
                post_ops_binary_rhs_arg_vec, op_type, bcast_type,
                blocked_oc_tail);

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
