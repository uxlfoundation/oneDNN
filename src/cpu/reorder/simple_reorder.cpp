/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "cpu/reorder/simple_reorder.hpp"

#include <cstring>
#include <vector>

#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
// Everything past dim 0 is dense when the span of those dimensions equals the
// number of elements they hold, leaving dim 0 as the only dimension allowed to
// carry a stride gap.
// TODO: consider merging with `memory_desc_wrapper::is_dense`. It is kept
// separate for now because its implementation differs and unifying them would
// require a whole-library validation that is out of scope here.
bool is_dense_no_dim_0(const memory_desc_wrapper &data_d) {
    return static_cast<dim_t>(data_d.span(1)) == data_d.nelems(false, 1);
}
} // namespace

status_t simple_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    using namespace data_type;

    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const memory_desc_wrapper input_d(src_md());
    const memory_desc_wrapper output_d(dst_md());

    // TODO: enable checks once ready.
    // VDISPATCH_REORDER(platform::has_data_type_support(input_d.data_type()),
    //         VERBOSE_UNSUPPORTED_DT);
    // VDISPATCH_REORDER(platform::has_data_type_support(output_d.data_type()),
    //         VERBOSE_UNSUPPORTED_DT);

    // These types are not supported in this implementation as they would
    // require special handling on a common infrastructure side (loading and
    // storing 64-bit values for int64_t and double properly), which doesn't
    // seem to bring any value to the library at this point, thus, just making
    // sure they are not expected here.
    VDISPATCH_REORDER(!utils::one_of(input_d.data_type(), f64, s64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(!utils::one_of(output_d.data_type(), f64, s64),
            VERBOSE_UNSUPPORTED_DT);

    // Sub-byte data types require dedicated bit-level handling which is a
    // prerogative of `ref_reorder` only. Reject them before any kernel is
    // selected so no specialization has to worry about them.
    VDISPATCH_REORDER(types::data_type_bits(input_d.data_type()) >= 8,
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(types::data_type_bits(output_d.data_type()) >= 8,
            VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_REORDER(
            input_d.is_blocking_desc() && !input_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    // The dst may carry an additional (compensation) buffer; only the
    // `plain_to_dense_comp` kernel consumes it, the other kernels reject it
    // in their own `is_*` checks.
    VDISPATCH_REORDER(output_d.is_blocking_desc(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    auto gpu_zp = memory_extra_flags::compensation_gpu_conv_asymmetric_src;
    VDISPATCH_REORDER(!(dst_md()->extra.flags & gpu_zp),
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    // Select a concrete kernel by probing each specialization's applicability.
    // The first match wins; sub-calls dispatch on `kernel_kind_` at execute
    // time. If nothing matches, the problem is not supported.
    kernel_kind_ = kernel_kind_t::unknown;
    if (is_direct_copy(engine, input_d, output_d, attr()) == status::success)
        kernel_kind_ = kernel_kind_t::direct_copy;
    else if (is_plain_to_blocked(engine, input_d, output_d, attr())
            == status::success)
        kernel_kind_ = kernel_kind_t::plain_to_blocked;
    else if (is_blocked_to_blocked(engine, input_d, output_d, attr())
            == status::success)
        kernel_kind_ = kernel_kind_t::blocked_to_blocked;
    else if (is_plain_to_dense_comp(engine, input_d, output_d, attr())
            == status::success)
        kernel_kind_ = kernel_kind_t::plain_to_dense_comp;

    VDISPATCH_REORDER(kernel_kind_ != kernel_kind_t::unknown,
            VERBOSE_IMPL_HEURISTIC_FAIL, "no applicable kernel kind");

    return status::success;
}

status_t simple_reorder_t::pd_t::is_direct_copy(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Only per-tensor (mask 0), default data type and default groups scales,
    // plus an optional sum post-op, are supported; zero points are not. The
    // skip mask implicitly rejects non-default zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(
                              skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        if (!attr->scales_.has_default_values(arg))
            VDISPATCH_REORDER(
                    attr->scales_.get_mask(arg) == 0, VERBOSE_UNSUPPORTED_ATTR);
    }

    // `direct_copy` requires identical src/dst layouts for every dimension but
    // the outermost one, which may carry a stride gap (e.g. a padded dim 0).
    // The dim-0-contiguous case is naturally covered as a sub-case.
    VDISPATCH_REORDER(is_dense_no_dim_0(input_d),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(is_dense_no_dim_0(output_d),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
    VDISPATCH_REORDER(input_d.similar_to(output_d, true, false, 1),
            VERBOSE_TENSOR_FORMAT_MISMATCH, "src", "dst");

    // Compensation buffers are handled by `plain_to_dense_comp` only.
    VDISPATCH_REORDER(!output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    return status::success;
}

status_t simple_reorder_t::pd_t::is_plain_to_blocked(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Only per-tensor (mask 0), default data type and default groups scales,
    // plus an optional sum post-op, are supported; zero points are not. The
    // skip mask implicitly rejects non-default zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(
                              skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        if (!attr->scales_.has_default_values(arg))
            VDISPATCH_REORDER(
                    attr->scales_.get_mask(arg) == 0, VERBOSE_UNSUPPORTED_ATTR);
    }

    // `plain_to_blocked` converts between a plain layout and a blocked one
    // whose inner blocks cover one or two distinct logical dims (a dim may be
    // blocked more than once, e.g. `8a16b2a`). Exactly one side must be plain.
    const bool src_plain = input_d.is_plain();
    const bool dst_plain = output_d.is_plain();
    VDISPATCH_REORDER(src_plain != dst_plain, VERBOSE_TENSOR_FORMAT_MISMATCH,
            "src", "dst");

    const auto &blocked_d = src_plain ? output_d : input_d;
    const char *blocked_arg = src_plain ? "dst" : "src";
    const auto &blk = blocked_d.blocking_desc();
    VDISPATCH_REORDER(blk.inner_nblks >= 1, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT,
            blocked_arg);

    // Collect the distinct blocked logical dims. One or two are supported;
    // three or more (`d1 == -2`) are not.
    dim_t d0 = blk.inner_idxs[0], d1 = -1;
    for (int i = 1; i < blk.inner_nblks; ++i) {
        const dim_t d = blk.inner_idxs[i];
        if (d == d0 || d == d1) continue;
        d1 = d1 == -1 ? d : -2;
    }
    VDISPATCH_REORDER(d1 != -2, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
    if (d1 >= 0 && d0 > d1) nstl::swap(d0, d1);

    const int ndims = blocked_d.ndims();
    if (d1 == -1) {
        // A single inner block on dim 0 or dim 1 (e.g. nchw <-> nChw16c).
        // `ndims >= 3` guarantees the innermost logical dim is distinct from
        // the blocked one, which the kernel iterates as its contiguous inner
        // loop.
        VDISPATCH_REORDER(utils::one_of(d0, 0, 1),
                VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
        VDISPATCH_REORDER(ndims >= 3 && ndims <= 6,
                VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
    } else {
        // Two blocked dims anchored at dims {0,1} (plain weights, e.g.
        // OIhw16i16o) or at dims {1,2} with a leading group dim (grouped
        // weights, e.g. gOIhw16i16o). This mirrors the `_AB`/`_BC` layouts.
        // `ndims >= d1 + 2` guarantees a trailing non-blocked innermost dim.
        const bool ab = d0 == 0 && d1 == 1;
        const bool bc = d0 == 1 && d1 == 2;
        VDISPATCH_REORDER(
                ab || bc, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
        VDISPATCH_REORDER(ndims >= d1 + 2 && ndims <= (ab ? 5 : 6),
                VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
    }

    // Compensation buffers are handled by `plain_to_dense_comp` only.
    VDISPATCH_REORDER(!output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    return status::success;
}

// TODO: remove this kernel kind, along with its `execute_*` counterpart and the
// dispatch wiring in `init`/`execute`, once blocked activation layouts are
// dropped from the library.
status_t simple_reorder_t::pd_t::is_blocked_to_blocked(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Only per-tensor (mask 0), default data type and default groups scales,
    // plus an optional sum post-op, are supported; zero points are not. The
    // skip mask implicitly rejects non-default zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(
                              skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        if (!attr->scales_.has_default_values(arg))
            VDISPATCH_REORDER(
                    attr->scales_.get_mask(arg) == 0, VERBOSE_UNSUPPORTED_ATTR);
    }

    // `blocked_to_blocked` re-blocks a single blocked logical dim (the channel
    // dim) between two blocked activation layouts whose inner blocks differ in
    // size (e.g. nChw8c <-> nChw16c). Both sides must be blocked exactly once,
    // over the same single logical dim, with different block sizes. Equal block
    // sizes would make the layouts identical and are served by `direct_copy`.
    VDISPATCH_REORDER(!input_d.is_plain() && !output_d.is_plain(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    const auto &iblk = input_d.blocking_desc();
    const auto &oblk = output_d.blocking_desc();
    VDISPATCH_REORDER(iblk.inner_nblks == 1 && oblk.inner_nblks == 1,
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    // The channel dim is dim 1 for every blocked activation layout (nCw/nChw/
    // nCdhw). Anchoring on it keeps this kernel scoped to activations.
    VDISPATCH_REORDER(iblk.inner_idxs[0] == 1 && oblk.inner_idxs[0] == 1,
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(iblk.inner_blks[0] != oblk.inner_blks[0],
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    VDISPATCH_REORDER(input_d.ndims() == output_d.ndims(),
            VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
    VDISPATCH_REORDER(input_d.ndims() >= 3 && input_d.ndims() <= 5,
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    // Compensation buffers are handled by `plain_to_dense_comp` only.
    VDISPATCH_REORDER(!output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    return status::success;
}

status_t simple_reorder_t::pd_t::is_plain_to_dense_comp(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    using namespace data_type;
    using namespace memory_extra_flags;

    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // This kernel owns the s8 convolution-weights reorders that must fill an
    // appended compensation buffer: s8s8 and/or asymmetric-src (zero-point).
    // An optional scale adjustment may accompany them; any other extra flag is
    // out of scope.
    //
    // Dispatch is purely on runtime blocking, so the canonical forms are
    // handled directly.
    const auto &extra = output_d.extra();
    const bool req_s8s8 = extra.flags & compensation_conv_s8s8;
    const bool req_asymm = extra.flags & compensation_conv_asymmetric_src;
    VDISPATCH_REORDER(
            req_s8s8 || req_asymm, VERBOSE_UNSUPPORTED_MD_FLAG, "extra");
    const uint64_t supported = compensation_conv_s8s8
            | compensation_conv_asymmetric_src | scale_adjust;
    VDISPATCH_REORDER(
            !(extra.flags & ~supported), VERBOSE_UNSUPPORTED_MD_FLAG, "extra");
    // When both compensations are present they must share the same mask so a
    // single comp index space addresses both buffers.
    VDISPATCH_REORDER(
            IMPLICATION(req_s8s8 && req_asymm,
                    extra.compensation_mask == extra.asymm_compensation_mask),
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    VDISPATCH_REORDER(
            input_d.is_plain(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    // The dst may be plain (e.g. `wigo`) or blocked (e.g. `OIhw4i16o4i`); both
    // reduce to the same dense `off_v` addressing. Only the data type is pinned.
    VDISPATCH_REORDER(output_d.data_type() == s8, VERBOSE_UNSUPPORTED_DT);

    // Only scales are supported (any mask); no post-ops / zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(skip_mask_t::scales),
            VERBOSE_UNSUPPORTED_ATTR);
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
    }

    // The compensation buffer uses the dst tag trimmed to the masked logical
    // dims (a dense layout, plain or blocked; see `execute_plain_to_dense_comp`).
    // That is only well-defined when each masked dim is blocked at most once (so
    // it contributes a single outer/inner stride pair); for a plain dst no dim
    // is blocked and the layout collapses to row-major. The reduction over the
    // remaining dims is race-free because a single thread owns each comp index.
    const int comp_mask = req_s8s8 ? extra.compensation_mask
                                   : extra.asymm_compensation_mask;
    const auto &blk = output_d.blocking_desc();
    for (int d = 0; d < output_d.ndims(); ++d) {
        if (!((comp_mask >> d) & 1)) continue;
        int nblk = 0;
        for (int i = 0; i < blk.inner_nblks; ++i)
            if (blk.inner_idxs[i] == d) ++nblk;
        VDISPATCH_REORDER(nblk <= 1, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
    }

    return status::success;
}

status_t simple_reorder_t::execute(const exec_ctx_t &ctx) const {
    switch (pd()->kernel_kind_) {
        case pd_t::kernel_kind_t::direct_copy: return execute_direct_copy(ctx);
        case pd_t::kernel_kind_t::plain_to_blocked:
            return execute_plain_to_blocked(ctx);
        case pd_t::kernel_kind_t::blocked_to_blocked:
            return execute_blocked_to_blocked(ctx);
        case pd_t::kernel_kind_t::plain_to_dense_comp:
            return execute_plain_to_dense_comp(ctx);
        case pd_t::kernel_kind_t::unknown: break;
    }
    assert(!"unknown kernel kind");
    return status::runtime_error;
}

status_t simple_reorder_t::execute_direct_copy(const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    // Advance each base to its logical element 0.
    input += input_d.blk_off(0) * input_d.data_type_size();
    output += output_d.blk_off(0) * output_d.data_type_size();

    // This kernel can be used for tensors with multiple inner blocks for
    // which generic zero padding must be used.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const float beta = pd()->beta();

    // Every dimension but dim 0 is dense and shares the same layout across
    // src/dst, so each dim-0 slab is addressed by its own stride (`is`/`os`)
    // while the inner `nelems_no_d0` elements are contiguous. The fully dense
    // case falls out naturally with `is == os == nelems_no_d0`.
    const dim_t N = input_d.dims()[0];
    const dim_t is = input_d.blocking_desc().strides[0];
    const dim_t os = output_d.blocking_desc().strides[0];
    const dim_t nelems_no_d0 = input_d.nelems(false, 1);
    const dim_t work_amount = N * nelems_no_d0;
    if (work_amount == 0) return status::success;

    parallel(0, [=](const int ithr, const int nthr) {
        const float src_scale = with_src_scales
                ? io::load_float_value(src_scales_dt, src_scales, 0)
                : 1.f;
        const float dst_scale = with_dst_scales
                ? io::load_float_value(dst_scales_dt, dst_scales, 0)
                : 1.f;
        const float alpha = src_scale / dst_scale;
        // Folding `1 / dst_scale` into `alpha` requires scaling the sum term
        // the same way, since `(src_scale * src + beta * dst) / dst_scale !=
        // alpha * src + beta * dst`.
        const float beta_scaled = with_dst_scales ? beta / dst_scale : beta;

        auto convert = [&](dim_t in_off, dim_t out_off) {
            if (src_dt == data_type::e8m0) {
                // Reorder from e8m0 to f32 is used for benchdnn correctness
                // validation purpose only. A dedicated path is required to
                // preserve a minimal e8m0 value which gets converted by any
                // compiler to 0 if any floating-point operation or comparison
                // is applied around the value.
                const float s = io::load_float_value(src_dt, input, in_off);
                io::store_float_value(dst_dt, s, output, out_off);
            } else {
                float d = alpha * io::load_float_value(src_dt, input, in_off);
                if (beta_scaled != 0.f)
                    d += beta_scaled
                            * io::load_float_value(dst_dt, output, out_off);
                io::store_float_value(dst_dt, d, output, out_off);
            }
        };

        dim_t n {0}, dim1_s {0};
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
        while (start < end) {
            const dim_t work_rem = end - start;
            const dim_t dim1_e = nstl::min(dim1_s + work_rem, nelems_no_d0);
            for (dim_t e = dim1_s; e < dim1_e; ++e)
                convert(is * n + e, os * n + e);
            utils::nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
        }
    });
    return status::success;
}

status_t simple_reorder_t::execute_plain_to_blocked(
        const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    // The blocked side carries the inner blocks; the other side is plain.
    const bool to_blocked = !output_d.is_plain();
    const auto &blocked_d = to_blocked ? output_d : input_d;
    const auto &flat_d = to_blocked ? input_d : output_d;

    const auto &blk = blocked_d.blocking_desc();
    const int nblks = blk.inner_nblks;
    const int ndims = blocked_d.ndims();
    const int last = ndims - 1;

    // Collect the distinct blocked logical dims (`d0`, and `d1 >= 0` when a
    // second one exists) and their total block sizes (product of the inner
    // blocks mapped onto each dim; a dim may be blocked more than once, e.g.
    // `8a16b2a`).
    dim_t d0 = blk.inner_idxs[0], d1 = -1;
    for (int i = 1; i < nblks; ++i)
        if (blk.inner_idxs[i] != d0) d1 = blk.inner_idxs[i];
    if (d1 >= 0 && d0 > d1) nstl::swap(d0, d1);
    dim_t bs0 = 1, bs1 = 1;
    dim_t in_idxs[DNNL_MAX_NDIMS] = {0};
    dim_t in_blks[DNNL_MAX_NDIMS] = {0};
    for (int i = 0; i < nblks; ++i) {
        in_idxs[i] = blk.inner_idxs[i];
        in_blks[i] = blk.inner_blks[i];
        if (blk.inner_idxs[i] == d0)
            bs0 *= blk.inner_blks[i];
        else
            bs1 *= blk.inner_blks[i];
    }

    // Offsets below are computed from strides only, so each base is advanced to
    // its own logical element 0 here.
    input += input_d.offset0() * input_d.data_type_size();
    output += output_d.offset0() * output_d.data_type_size();

    // The blocked dst holds zero padding in the inner block tail; let the common
    // infrastructure handle it generically.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const float beta = pd()->beta();

    const dims_t &dims = blocked_d.dims();
    const dims_t &pdims = blocked_d.padded_dims();
    const dim_t *flat_strides = flat_d.blocking_desc().strides;
    const dim_t *blkd_strides = blk.strides;

    // The innermost logical dim is non-blocked and iterated as the contiguous
    // inner loop; the block positions are iterated inside it.
    const dim_t L = dims[last];
    const dim_t l_flat_stride = flat_strides[last];
    const dim_t l_blk_stride = blkd_strides[last];
    const dim_t d0_flat_stride = flat_strides[d0];
    const dim_t d1_flat_stride = d1 >= 0 ? flat_strides[d1] : 0;

    // Outer iteration space: every dim but the innermost one. Blocked dims
    // iterate over their outer blocks; the rest over their full extent.
    dims_t outer_dims = {0};
    dim_t outer_work = 1;
    for (int d = 0; d < last; ++d) {
        if (d == d0)
            outer_dims[d] = pdims[d] / bs0;
        else if (d == d1)
            outer_dims[d] = pdims[d] / bs1;
        else
            outer_dims[d] = dims[d];
        outer_work *= outer_dims[d];
    }

    parallel(0, [=](const int ithr, const int nthr) {
        const float src_scale = with_src_scales
                ? io::load_float_value(src_scales_dt, src_scales, 0)
                : 1.f;
        const float dst_scale = with_dst_scales
                ? io::load_float_value(dst_scales_dt, dst_scales, 0)
                : 1.f;
        const float alpha = src_scale / dst_scale;
        // Folding `1 / dst_scale` into `alpha` requires scaling the sum term
        // the same way, since `(src_scale * src + beta * dst) / dst_scale !=
        // alpha * src + beta * dst`.
        const float beta_scaled = with_dst_scales ? beta / dst_scale : beta;

        auto convert = [&](dim_t in_off, dim_t out_off) {
            float d = alpha * io::load_float_value(src_dt, input, in_off);
            if (beta_scaled != 0.f)
                d += beta_scaled
                        * io::load_float_value(dst_dt, output, out_off);
            io::store_float_value(dst_dt, d, output, out_off);
        };

        // With two blocked dims the within-block offset is a nested function of
        // the local positions `(p0, p1)` (it may interleave repeated blocks).
        // It is additive with the outer-block offset (both are terms of
        // `off_v`), so it is tabulated once over the whole block and reused,
        // keeping the hot loop free of the per-element modulo/division the inner
        // blocks require. The table is built per thread so its storage never
        // outlives the (possibly asynchronous) parallel region. A single
        // blocked dim is contiguous (`within(p0) == p0`), so no table is built
        // and the fast path indexes the block position directly.
        std::vector<dim_t> within_tbl;
        if (d1 >= 0) {
            within_tbl.resize((size_t)bs0 * bs1);
            for_(dim_t p0 = 0; p0 < bs0; ++p0)
            for (dim_t p1 = 0; p1 < bs1; ++p1) {
                dim_t lpos[DNNL_MAX_NDIMS] = {0};
                lpos[d0] = p0;
                lpos[d1] = p1;
                dim_t off = 0, stride = 1;
                for (int i = nblks - 1; i >= 0; --i) {
                    const dim_t d = in_idxs[i];
                    const dim_t b = in_blks[i];
                    off += (lpos[d] % b) * stride;
                    lpos[d] /= b;
                    stride *= b;
                }
                within_tbl[(size_t)p0 * bs1 + p1] = off;
            }
        }

        dim_t start {0}, end {0};
        balance211(outer_work, nthr, ithr, start, end);
        for (dim_t w = start; w < end; ++w) {
            // Decode the flat outer index into per-dimension coordinates.
            dims_t cur = {0};
            dim_t rem = w;
            for (int d = last - 1; d >= 0; --d) {
                cur[d] = rem % outer_dims[d];
                rem /= outer_dims[d];
            }

            dim_t base_flat = 0, base_blk = 0;
            for (int d = 0; d < last; ++d) {
                dim_t flat_coord;
                if (d == d0)
                    flat_coord = cur[d] * bs0;
                else if (d == d1)
                    flat_coord = cur[d] * bs1;
                else
                    flat_coord = cur[d];
                base_flat += flat_coord * flat_strides[d];
                base_blk += cur[d] * blkd_strides[d];
            }
            // Trailing blocks may be partial when a blocked dim is padded.
            const dim_t block0
                    = nstl::min<dim_t>(bs0, dims[d0] - cur[d0] * bs0);
            const dim_t block1 = d1 >= 0
                    ? nstl::min<dim_t>(bs1, dims[d1] - cur[d1] * bs1)
                    : 1;

            for (dim_t l = 0; l < L; ++l) {
                const dim_t flat_l = base_flat + l * l_flat_stride;
                const dim_t blk_l = base_blk + l * l_blk_stride;
                for_(dim_t p0 = 0; p0 < block0; ++p0)
                for (dim_t p1 = 0; p1 < block1; ++p1) {
                    const dim_t flat_off = flat_l + p0 * d0_flat_stride
                            + p1 * d1_flat_stride;
                    const dim_t blkd_off = blk_l
                            + (d1 >= 0 ? within_tbl[(size_t)p0 * bs1 + p1]
                                       : p0);
                    if (to_blocked)
                        convert(flat_off, blkd_off);
                    else
                        convert(blkd_off, flat_off);
                }
            }
        }
    });
    return status::success;
}

// TODO: remove this kernel, along with its `is_*` counterpart and the dispatch
// wiring in `init`/`execute`, once blocked activation layouts are dropped from
// the library.
status_t simple_reorder_t::execute_blocked_to_blocked(
        const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    const auto &iblk = input_d.blocking_desc();
    const auto &oblk = output_d.blocking_desc();
    // Both sides block the same single logical dim (channels) with different
    // block sizes; everything else shares the same logical iteration space.
    const dim_t cdim = iblk.inner_idxs[0];
    const dim_t bs_src = iblk.inner_blks[0];
    const dim_t bs_dst = oblk.inner_blks[0];
    const int ndims = input_d.ndims();

    // Offsets below are computed from strides only, so each base is advanced to
    // its own logical element 0 here.
    input += input_d.offset0() * input_d.data_type_size();
    output += output_d.offset0() * output_d.data_type_size();

    // The blocked dst holds zero padding in the channel block tail; let the
    // common infrastructure handle it generically.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const float beta = pd()->beta();

    const dims_t &dims = input_d.dims();
    const dim_t *src_strides = iblk.strides;
    const dim_t *dst_strides = oblk.strides;
    const dim_t C = dims[cdim];
    const dim_t src_cstride = src_strides[cdim];
    const dim_t dst_cstride = dst_strides[cdim];

    // Outer iteration space: every dim but the channel one (set to extent 1 so
    // it collapses in the mixed-radix decode). The channel dim is iterated as
    // the contiguous inner loop.
    dims_t outer_dims = {0};
    dim_t outer_work = 1;
    for (int d = 0; d < ndims; ++d) {
        outer_dims[d] = d == cdim ? 1 : dims[d];
        outer_work *= outer_dims[d];
    }

    parallel(0, [=](const int ithr, const int nthr) {
        const float src_scale = with_src_scales
                ? io::load_float_value(src_scales_dt, src_scales, 0)
                : 1.f;
        const float dst_scale = with_dst_scales
                ? io::load_float_value(dst_scales_dt, dst_scales, 0)
                : 1.f;
        const float alpha = src_scale / dst_scale;
        // Folding `1 / dst_scale` into `alpha` requires scaling the sum term
        // the same way, since `(src_scale * src + beta * dst) / dst_scale !=
        // alpha * src + beta * dst`.
        const float beta_scaled = with_dst_scales ? beta / dst_scale : beta;

        auto convert = [&](dim_t in_off, dim_t out_off) {
            float d = alpha * io::load_float_value(src_dt, input, in_off);
            if (beta_scaled != 0.f)
                d += beta_scaled
                        * io::load_float_value(dst_dt, output, out_off);
            io::store_float_value(dst_dt, d, output, out_off);
        };

        dim_t start {0}, end {0};
        balance211(outer_work, nthr, ithr, start, end);
        for (dim_t w = start; w < end; ++w) {
            // Decode the flat outer index into per-dimension coordinates.
            dims_t cur = {0};
            dim_t rem = w;
            for (int d = ndims - 1; d >= 0; --d) {
                cur[d] = rem % outer_dims[d];
                rem /= outer_dims[d];
            }

            // Both offsets decompose additively (`memory_desc_wrapper::off_v`):
            // the channel contribution `(c / bs) * cstride + (c % bs)` adds onto
            // the shared outer base.
            dim_t base_src = 0, base_dst = 0;
            for (int d = 0; d < ndims; ++d) {
                if (d == cdim) continue;
                base_src += cur[d] * src_strides[d];
                base_dst += cur[d] * dst_strides[d];
            }

            for (dim_t c = 0; c < C; ++c) {
                const dim_t src_off
                        = base_src + (c / bs_src) * src_cstride + (c % bs_src);
                const dim_t dst_off
                        = base_dst + (c / bs_dst) * dst_cstride + (c % bs_dst);
                convert(src_off, dst_off);
            }
        }
    });
    return status::success;
}

// TODO: once blocked convolution-weights layouts are reworked, revisit whether
// this compensation kernel can be folded back into `plain_to_blocked`.
status_t simple_reorder_t::execute_plain_to_dense_comp(
        const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const int ndims = output_d.ndims();

    const auto &extra = output_d.extra();
    const bool req_s8s8
            = extra.flags & memory_extra_flags::compensation_conv_s8s8;
    const bool req_asymm = extra.flags
            & memory_extra_flags::compensation_conv_asymmetric_src;
    const int comp_mask = req_s8s8 ? extra.compensation_mask
                                   : extra.asymm_compensation_mask;
    // The s8s8 compensation absorbs an extra `* 128` factor (see below); both
    // buffers share the same per-channel accumulator otherwise.
    const float adj_scale = (extra.flags & memory_extra_flags::scale_adjust)
            ? extra.scale_adjust
            : 1.f;

    // Compensation buffer(s) are appended right after the main data blob; the
    // s8s8 buffer (when present) precedes the zero-point one.
    const size_t comp_base_off
            = output_d.size() - output_d.additional_buffer_size();
    const size_t s8s8_sz = output_d.additional_buffer_size(
            memory_extra_flags::compensation_conv_s8s8);
    int32_t *cp = req_s8s8 ? reinterpret_cast<int32_t *>(output + comp_base_off)
                           : nullptr;
    int32_t *zp = req_asymm ? reinterpret_cast<int32_t *>(output + comp_base_off
                                      + (req_s8s8 ? s8s8_sz : 0))
                            : nullptr;

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const int src_scales_mask
            = with_src_scales ? scales.get_mask(DNNL_ARG_FROM) : 0;
    const int dst_scales_mask
            = with_dst_scales ? scales.get_mask(DNNL_ARG_TO) : 0;

    const dims_t &dims = output_d.dims();
    const dims_t &pdims = output_d.padded_dims();

    // Split the logical dims into the compensated (masked) set, which indexes
    // the comp buffer, and the reduction set. The outer loop enumerates every
    // padded masked coordinate (`comp_work` slots == the comp buffer size).
    int comp_idx[DNNL_MAX_NDIMS], red_idx[DNNL_MAX_NDIMS];
    dim_t comp_sz[DNNL_MAX_NDIMS], red_sz[DNNL_MAX_NDIMS];
    int n_comp = 0, n_red = 0;
    dim_t comp_work = 1, red_work = 1;
    for (int d = 0; d < ndims; ++d) {
        if ((comp_mask >> d) & 1) {
            comp_idx[n_comp] = d;
            comp_sz[n_comp] = pdims[d];
            comp_work *= pdims[d];
            ++n_comp;
        } else {
            red_idx[n_red] = d;
            red_sz[n_red] = dims[d];
            red_work *= dims[d];
            ++n_red;
        }
    }

    // The comp buffer is laid out as the dst tag trimmed to the masked dims,
    // i.e. a *blocked* layout, not a flat row-major one: a masked dim that is
    // blocked in the dst keeps its inner block in the comp buffer too.
    // Row-major indexing is only correct when the blocked masked dim is the
    // innermost masked dim; when an outer masked dim is blocked (e.g. `Abx16a`
    // with mask over dims 0 and 1) the inner block is interleaved after the
    // trailing masked dims and the offsets differ. Build per-masked-dim blocked
    // strides so the comp offset matches the library convention. The `is_`
    // guard guarantees each masked dim is blocked at most once, so a dim
    // contributes a single (outer, inner) stride pair. For a plain dst no
    // masked dim is blocked, so this degenerates to a plain row-major layout.
    dim_t comp_blk[DNNL_MAX_NDIMS], comp_inner_str[DNNL_MAX_NDIMS];
    dim_t comp_outer_str[DNNL_MAX_NDIMS];
    for (int d = 0; d < ndims; ++d) {
        comp_blk[d] = 1;
        comp_inner_str[d] = 0;
    }
    const auto &oblk = output_d.blocking_desc();
    dim_t inner_tot = 1;
    for (int i = oblk.inner_nblks - 1; i >= 0; --i) {
        const dim_t d = oblk.inner_idxs[i];
        if (!((comp_mask >> d) & 1)) continue;
        comp_inner_str[d] = inner_tot;
        comp_blk[d] = oblk.inner_blks[i];
        inner_tot *= oblk.inner_blks[i];
    }
    dim_t outer_acc = inner_tot;
    for (int i = n_comp - 1; i >= 0; --i) {
        const int d = comp_idx[i];
        comp_outer_str[d] = outer_acc;
        outer_acc *= pdims[d] / comp_blk[d];
    }

    // The blocked dst carries zero padding in the inner block tails; let the
    // common infrastructure zero it so the kernel only writes real elements.
    ctx.zero_pad_output(DNNL_ARG_TO);

    // Zero the whole comp buffer: the compute loop only writes real (non
    // padded) masked coordinates, and padded blocked slots must read back zero.
    parallel(1, [=](const int ithr, const int nthr) {
        if (cp) std::memset(cp, 0, comp_work * sizeof(int32_t));
        if (zp) std::memset(zp, 0, comp_work * sizeof(int32_t));
    });

    parallel(0, [=](const int ithr, const int nthr) {
        // Scale index for `pos` over a mask, row-major across the actual dims.
        auto scale_off = [&](const dims_t &pos, int mask) {
            dim_t off = 0;
            for (int d = 0; d < ndims; ++d)
                if ((mask >> d) & 1) off = off * dims[d] + pos[d];
            return off;
        };

        dim_t start {0}, end {0};
        balance211(comp_work, nthr, ithr, start, end);
        for (dim_t ci = start; ci < end; ++ci) {
            // Decode the masked coordinates (padded) from the comp index.
            dims_t pos = {0};
            dim_t rem = ci;
            bool is_padded = false;
            for (int i = n_comp - 1; i >= 0; --i) {
                const int d = comp_idx[i];
                const dim_t c = rem % comp_sz[i];
                rem /= comp_sz[i];
                pos[d] = c;
                if (c >= dims[d]) is_padded = true;
            }
            // A padded channel contributes nothing; its comp stays zero (the
            // buffer was pre-zeroed) and its blocked slot is never read.
            if (is_padded) continue;

            // Blocked offset into the comp buffer for these masked coords.
            dim_t comp_off = 0;
            for (int i = 0; i < n_comp; ++i) {
                const int d = comp_idx[i];
                comp_off += (pos[d] / comp_blk[d]) * comp_outer_str[d]
                        + (pos[d] % comp_blk[d]) * comp_inner_str[d];
            }

            int32_t acc = 0;
            for (dim_t ri = 0; ri < red_work; ++ri) {
                dim_t rrem = ri;
                for (int i = n_red - 1; i >= 0; --i) {
                    pos[red_idx[i]] = rrem % red_sz[i];
                    rrem /= red_sz[i];
                }
                const dim_t in_off = input_d.off_v(pos);
                const dim_t out_off = output_d.off_v(pos);
                const float src_scale = with_src_scales
                        ? io::load_float_value(src_scales_dt, src_scales,
                                  scale_off(pos, src_scales_mask))
                        : 1.f;
                const float dst_scale = with_dst_scales
                        ? io::load_float_value(dst_scales_dt, dst_scales,
                                  scale_off(pos, dst_scales_mask))
                        : 1.f;
                const float alpha = src_scale * adj_scale / dst_scale;
                const float v
                        = alpha * io::load_float_value(src_dt, input, in_off);
                // `dst_dt` is pinned to s8 (see `is_plain_to_dense_comp`), so
                // quantize once in-register, store the byte, and accumulate the
                // same value instead of storing and reloading it from memory.
                const int8_t q = q10n::saturate_and_round<int8_t>(v);
                reinterpret_cast<int8_t *>(output)[out_off] = q;
                acc += q;
            }
            // Mirror the original specializations: zp -= sum(q), s8s8 -= 128 *
            // sum(q).
            if (zp) zp[comp_off] = -acc;
            if (cp) cp[comp_off] = -128 * acc;
        }
    });
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
