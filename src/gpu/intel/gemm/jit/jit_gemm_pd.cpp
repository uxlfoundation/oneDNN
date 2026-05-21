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

#include "gpu/intel/gemm/jit/jit_gemm_pd.hpp"
#include "common/memory_desc.hpp"
#include "common/primitive_attr_quant.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/gemm/jit/problem_dump.hpp"
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

using intel::jit::eltwise_injector_f32_is_supported;
using intel::jit::get_ngen_product;

void jit_gemm_pd_t::transpose_mn_axes(memory_desc_t &md) {
    if (md.ndims < 2) return;
    const int i = md.ndims - 2, j = md.ndims - 1;
    std::swap(md.dims[i], md.dims[j]);
    std::swap(md.padded_dims[i], md.padded_dims[j]);
    std::swap(md.padded_offsets[i], md.padded_offsets[j]);
    if (md.format_kind == format_kind::blocked) {
        auto &blk = md.format_desc.blocking;
        std::swap(blk.strides[i], blk.strides[j]);
        for (int k = 0; k < blk.inner_nblks; ++k) {
            if (blk.inner_idxs[k] == i)
                blk.inner_idxs[k] = j;
            else if (blk.inner_idxs[k] == j)
                blk.inner_idxs[k] = i;
        }
    }
}

transpose_t jit_gemm_pd_t::get_trans(const memory_desc_t &md) {
    if (!md.ndims) return transpose::notrans;
    using namespace data_type;
    const bool is_4bit
            = utils::one_of(md.data_type, f4_e2m1, f4_e3m0, s4, u4);
    const dim_t inner_m = md.dims[md.ndims - 2];
    const dim_t inner_n = md.dims[md.ndims - 1];
    const auto strides = md.format_desc.blocking.strides;
    const dim_t notranspose_ld
            = inner_m > 1 ? strides[md.ndims - 2] : inner_n;
    if (is_4bit && notranspose_ld % 2 != 0) return transpose::trans;
    // When the inner (n-axis) dim is 1, the matrix has a single column and
    // the n-axis stride is unobservable — fall back to NOTRANS to match
    // base's gemm_types.hpp get_trans (`last_dim != 1 && stride != 1`). The
    // earlier worktree dropped this gate to fix RNN BWD_DW (gotcha #10) when
    // mds were mn-transposed via ld_kernel_view; that mn-transpose path was
    // removed (gotcha #11), so the gate is restored here. Without it,
    // matmul DST [M, 1] dtag=ba (strides [1, M]) is misclassified as TRANS,
    // forcing the want_un_swap path in gen_t and rejecting K==1 cases that
    // base accepts via swap_ab=true.
    return inner_n != 1 && strides[md.ndims - 1] != 1
            ? transpose::trans
            : transpose::notrans;
}

dim_t jit_gemm_pd_t::get_ld(const memory_desc_t &md) {
    auto strides = md.format_desc.blocking.strides;
    assert(md.dims[md.ndims - 1] == 1 || strides[md.ndims - 1] == 1
            || md.dims[md.ndims - 2] == 1 || strides[md.ndims - 2] == 1);
    switch (get_trans(md)) {
        case transpose::trans:
            return md.dims[md.ndims - 1] > 1 ? strides[md.ndims - 1]
                                             : md.dims[md.ndims - 2];
        default:
            return md.dims[md.ndims - 2] > 1 ? strides[md.ndims - 2]
                                             : md.dims[md.ndims - 1];
    }
}

int jit_gemm_pd_t::mask_to_gemm(int mask, int nd) {
    // Compress a per-dim mask into the 3-bit {batch, M, N} layout that
    // gemm_desc_t::bias_mask used.
    int out = 0;
    if (nd >= 2 && (mask & (1 << (nd - 2)))) out |= 0x2; // M
    if (nd >= 1 && (mask & (1 << (nd - 1)))) out |= 0x1; // N
    for (int i = 0; i < nd - 2; ++i)
        if (mask & (1 << i)) out |= 0x4; // batch
    return out;
}

int jit_gemm_pd_t::swap_mask_bits(int mask, int i, int j) {
    if (i < 0 || j < 0 || i == j) return mask;
    const int lo = (mask >> i) & 1;
    const int hi = (mask >> j) & 1;
    if (lo != hi) {
        mask ^= (1 << i);
        mask ^= (1 << j);
    }
    return mask;
}

transpose_t jit_gemm_pd_t::gemm_transa() const { return compute_gemm_transa(); }
transpose_t jit_gemm_pd_t::gemm_transb() const { return compute_gemm_transb(); }
transpose_t jit_gemm_pd_t::trans_bias() const { return compute_trans_bias(); }

int jit_gemm_pd_t::gemm_bias_mask() const {
    if (!with_bias()) return 0;
    const auto &b = kernel_input_.bias_md;
    assert(b.ndims <= 6);
    int per_dim_mask = 0;
    for (int i = 0; i < b.ndims; ++i)
        per_dim_mask |= (b.dims[i] > 1) ? 1 << i : 0;
    int m = mask_to_gemm(per_dim_mask, b.ndims);
    // Under swap, kernel M ↔ N. Flip bits 0 (N) and 1 (M) of the 3-bit mask.
    if (swap_ab_) m = swap_mask_bits(m, 0, 1);
    return m;
}

sum_ab_t jit_gemm_pd_t::gemm_sum_ab() const {
    if (!with_reduce()) return sum_ab::sum_none;
    // matmul_reduce_kind::src    → reduce along K, output [..., M, 1] (sum_a_row).
    // matmul_reduce_kind::weights → reduce along K, output [..., 1, N] (sum_b_col).
    // swap_ab_ flips kernel M ↔ N interpretation.
    const auto rk = reduce_kind();
    if (rk == matmul_reduce_kind::src)
        return swap_ab_ ? sum_ab::sum_b_col : sum_ab::sum_a_row;
    if (rk == matmul_reduce_kind::weights)
        return swap_ab_ ? sum_ab::sum_a_row : sum_ab::sum_b_col;
    return sum_ab::sum_none;
}

// Collapse a leading run of dims into the first output dim.
// Mirrors base's squash_dims (gemm.hpp:91-104).
static void squash_dims(dims_t &out_dims, const dims_t &in_dims, int in_ndims,
        int out_ndims) {
    const int diff_ndims = in_ndims - out_ndims;
    assert(diff_ndims > 0);
    for (int i = 0; i < out_ndims; ++i)
        out_dims[i] = in_dims[i + diff_ndims];
    for (int i = 0; i < diff_ndims; ++i)
        out_dims[0] *= in_dims[i];
}

// Build a reshaped quant entry per base's "option 1" rule (gemm.hpp:225-235):
// shift the mask down by the collapsed batch dims, and (when grouped) reduce
// to 2D groups derived from reshaped_md and the squashed quant dims.
static quant_entry_t reshape_quant_entry(const quant_entry_t &in_entry,
        const memory_desc_t &reshaped_md, const dims_t &qdims, int diff_dims) {
    if (in_entry.has_default_values()) return in_entry;
    int new_mask = in_entry.get_mask() >> diff_dims;
    data_type_t dt = in_entry.get_data_type();
    dims_t dims {};
    int ndims = 0;
    if (!in_entry.has_default_groups()) {
        ndims = 2;
        dims[0] = reshaped_md.dims[reshaped_md.ndims - 2]
                / qdims[reshaped_md.ndims - 2];
        dims[1] = reshaped_md.dims[reshaped_md.ndims - 1]
                / qdims[reshaped_md.ndims - 1];
    }
    quant_entry_t out_entry;
    UNUSED_STATUS(out_entry.set(new_mask, dt, ndims, dims, false,
            in_entry.get_quantization_mode()));
    return out_entry;
}

// Adjust a per-arg quant entry for the new shape: takes the original md to
// derive qdims, then writes the reshaped entry back. Mirrors base's
// adjust_quant (gemm.hpp:274-288).
static status_t adjust_quant(quant_entries_t &entries, int arg,
        const memory_desc_t &md, const memory_desc_t &reshaped_md,
        int diff_dims, int in_ndims, int out_ndims) {
    const quant_entry_t &entry = entries.get(arg);
    if (entry.is_host_scalar()) return status::success;
    if (entry.has_default_values()) return status::success;
    memory_desc_t qmd;
    CHECK(entry.get_md(qmd, md));
    dims_t qdims {};
    squash_dims(qdims, qmd.dims, in_ndims, out_ndims);
    quant_entry_t reshaped_entry
            = reshape_quant_entry(entry, reshaped_md, qdims, diff_dims);
    CHECK(entries.set(arg, reshaped_entry));
    return status::success;
}

static void copy_user_into_kernel_input(jit_gemm_pd_t::jit_gemm_input_t &ki,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr) {
    ki.src_md = src_md;
    ki.weights_md = weights_md;
    ki.dst_md = dst_md;
    ki.bias_md = bias_md;
    ki.scales = attr.scales_;
    ki.zero_points = attr.zero_points_;
    ki.precomputed_reductions = attr.precomputed_reductions_;
    ki.post_ops = attr.post_ops_;
    ki.reshape_applied = false;
}

status_t jit_gemm_pd_t::maybe_reshape_2d() {
    // Idempotency: a prior call already populated kernel_input_. Bail.
    if (kernel_input_.src_md.ndims != 0) return status::success;

    // matmul_pd_t initializes src_md_/weights_md_/dst_md_/bias_md_ from
    // desc()->X_desc in its constructor; at this point they are still the
    // user-bound mds (no impl init has run yet).
    const memory_desc_t *a_md = &src_md_;
    const memory_desc_t *b_md = &weights_md_;
    const memory_desc_t *c_md = &dst_md_;
    const memory_desc_t *bia_md = &bias_md_;

    const bool with_bia = bia_md->ndims > 0;
    const int orig_dims = a_md->ndims;

    dim_t batch_b_dims = 1;
    for (int i = 0; i < b_md->ndims - 2; ++i)
        batch_b_dims *= b_md->dims[i];
    const bool reshape_2d = (batch_b_dims == 1 && b_md->ndims > 2);
    const bool reshape_3d = (a_md->ndims > 3);
    const bool allow_reshape
            = gpu_utils::dev_getenv("GEMM_ALLOW_RESHAPE", true);

    if (!allow_reshape || !(reshape_2d || reshape_3d)) {
        copy_user_into_kernel_input(
                kernel_input_, src_md_, weights_md_, dst_md_, bias_md_, attr_);
        return status::success;
    }

    const int ndims = a_md->ndims;
    const int reshape_size = reshape_2d ? 2 : 3;
    const int diff_dims = orig_dims - reshape_size;

    // Convert raw tensors to reshaped tensors.
    dims_t a_dims, b_dims, c_dims, bia_dims;
    squash_dims(a_dims, a_md->dims, ndims, reshape_size);
    squash_dims(b_dims, b_md->dims, ndims, reshape_size);
    squash_dims(c_dims, c_md->dims, ndims, reshape_size);
    squash_dims(bia_dims, bia_md->dims, ndims, reshape_size);

    // Cannot reshape if bias is broadcast across a subset of squashed dims.
    bool bcast_ok
            = IMPLICATION(with_bia, utils::one_of(bia_dims[0], 1, c_dims[0]));

    // 3D reshaping is only possible if A and B batch sizes allow.
    bool a_broadcast = false;
    bool b_broadcast = false;
    for (int i = 0; i < ndims - 2; ++i) {
        if (a_md->dims[i] == 1 && b_md->dims[i] > 1) a_broadcast = true;
        if (b_md->dims[i] == 1 && a_md->dims[i] > 1) b_broadcast = true;
    }
    bcast_ok = bcast_ok && !(a_broadcast && b_broadcast);
    bcast_ok = bcast_ok
            && IMPLICATION(reshape_size == 3,
                    a_dims[0] == b_dims[0]
                            || utils::one_of(1, a_dims[0], b_dims[0]));

    // memory_desc_reshape can fail. If so, bcast_ok=false so we fall through
    // to the batched gemm path.
    auto safe_reshape = [](memory_desc_t &out_md, const memory_desc_t &in_md,
                                int n, const dims_t dims) -> bool {
        return memory_desc_reshape(out_md, in_md, n, dims) == status::success;
    };

    memory_desc_t a_md_reshaped {}, b_md_reshaped {}, c_md_reshaped {},
            bia_md_reshaped {};
    bcast_ok = bcast_ok
            && safe_reshape(a_md_reshaped, *a_md, reshape_size, a_dims);
    bcast_ok = bcast_ok
            && safe_reshape(b_md_reshaped, *b_md, reshape_size, b_dims);
    bcast_ok = bcast_ok
            && safe_reshape(c_md_reshaped, *c_md, reshape_size, c_dims);
    if (with_bia) {
        bcast_ok = bcast_ok
                && safe_reshape(
                        bia_md_reshaped, *bia_md, reshape_size, bia_dims);
    }

    // Stage reshaped post_ops in a local copy.
    post_ops_t reshaped_post_ops = attr_.post_ops_;
    auto fall_back_no_reshape = [&]() -> status_t {
        copy_user_into_kernel_input(
                kernel_input_, src_md_, weights_md_, dst_md_, bias_md_, attr_);
        return status::success;
    };
    for (int i = 0; i < attr_.post_ops_.len(); ++i) {
        auto &po = reshaped_post_ops.entry_[i];
        if (po.is_binary()) {
            const auto &po_desc = po.binary.src1_desc;
            dim_t a_dim = po_desc.dims[po_desc.ndims - reshape_size];
            for (int j = po_desc.ndims; j > reshape_size; --j)
                a_dim *= po_desc.dims[po_desc.ndims - j];
            // post-ops cannot be applied if applied only on a subset of
            // batch dims.
            if (a_dim != c_dims[0] && a_dim > 1)
                return fall_back_no_reshape();
            const bool has_dims = po_desc.ndims > 0;
            dims_t po_dims {};
            if (reshape_2d) {
                po_dims[0] = a_dim;
                po_dims[1]
                        = has_dims ? po_desc.dims[po_desc.ndims - 1] : 1;
            } else {
                po_dims[0] = a_dim;
                po_dims[1]
                        = has_dims ? po_desc.dims[po_desc.ndims - 2] : 1;
                po_dims[2]
                        = has_dims ? po_desc.dims[po_desc.ndims - 1] : 1;
            }
            memory_desc_t tmp_po_desc {};
            bcast_ok = bcast_ok
                    && safe_reshape(
                            tmp_po_desc, po_desc, reshape_size, po_dims);
            reshaped_post_ops.entry_[i].binary.src1_desc = tmp_po_desc;
        } else if (po.is_prelu()) {
            const int mask = po.prelu.mask;
            int new_mask = 0;
            const int batch_idx = reshape_size - 1;
            dim_t batch_dim = 1;
            dim_t mask_dim = 1;
            for (int j = 0; j < c_md->ndims - batch_idx; ++j) {
                if ((mask >> j) & 1) {
                    if (new_mask != 0) return fall_back_no_reshape();
                    new_mask |= c_md->dims[j] == 1 ? 0 : 1;
                    mask_dim *= c_md->dims[j];
                }
                batch_dim *= c_md->dims[j];
            }
            if (batch_dim != mask_dim) return fall_back_no_reshape();
            const int shift = c_md->ndims - batch_idx;
            const int non_batch_mask = mask >> shift;
            // prelu is axb-format; reshape changes which axis is innermost,
            // so a mask spanning more than one non-batch dim cannot be
            // collapsed.
            if (non_batch_mask > 2
                    || (non_batch_mask > 0 && new_mask > 0))
                return fall_back_no_reshape();
            new_mask |= non_batch_mask << 1;
            reshaped_post_ops.entry_[i].prelu.mask = new_mask;
        }
    }

    // Quantization mask/group rescale per base's "option 1" (gemm.hpp:225-).
    auto safe_bcast_quant = [&](const quant_entry_t &entry,
                                    const dims_t &orig_md_dims,
                                    const dims_t &reshape_dims) -> bool {
        const int full_tensor_mask_ = ((1 << c_md->ndims) - 1);
        const int per_oc_mask = (1 << 1);
        if (!reshape_2d || entry.has_default_values()
                || utils::one_of(entry.get_mask(), 0, per_oc_mask,
                        full_tensor_mask_))
            return true;
        if (utils::one_of(orig_md_dims[diff_dims], reshape_dims[0], 1))
            return true;
        return false;
    };

    bcast_ok = bcast_ok
            && safe_bcast_quant(
                    attr_.scales_.get(DNNL_ARG_SRC), a_md->dims, a_dims);
    bcast_ok = bcast_ok
            && safe_bcast_quant(attr_.scales_.get(DNNL_ARG_WEIGHTS),
                    b_md->dims, b_dims);
    bcast_ok = bcast_ok
            && safe_bcast_quant(attr_.zero_points_.get(DNNL_ARG_SRC),
                    a_md->dims, a_dims);
    bcast_ok = bcast_ok
            && safe_bcast_quant(attr_.zero_points_.get(DNNL_ARG_WEIGHTS),
                    b_md->dims, b_dims);
    if (!bcast_ok) return fall_back_no_reshape();

    // Stage rescaled quant entries.
    scales_t reshaped_scales = attr_.scales_;
    zero_points_t reshaped_zp = attr_.zero_points_;
    precomputed_reductions_t reshaped_pr = attr_.precomputed_reductions_;
    CHECK(adjust_quant(reshaped_scales, DNNL_ARG_SRC, *a_md, a_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_scales, DNNL_ARG_WEIGHTS, *b_md, b_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_scales, DNNL_ARG_DST, *c_md, c_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_zp, DNNL_ARG_SRC, *a_md, a_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_zp, DNNL_ARG_WEIGHTS, *b_md, b_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_zp, DNNL_ARG_DST, *c_md, c_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_pr, DNNL_ARG_SRC, *a_md, a_md_reshaped,
            diff_dims, ndims, reshape_size));
    CHECK(adjust_quant(reshaped_pr, DNNL_ARG_WEIGHTS, *b_md, b_md_reshaped,
            diff_dims, ndims, reshape_size));

    // Commit reshape into kernel_input_ only. attr_ and the user-facing
    // src_md_/weights_md_/dst_md_/bias_md_ stay untouched (their resolved
    // tags are written back in commit_resolved_tags at end of init).
    kernel_input_.src_md = a_md_reshaped;
    kernel_input_.weights_md = b_md_reshaped;
    kernel_input_.dst_md = c_md_reshaped;
    if (with_bia) kernel_input_.bias_md = bia_md_reshaped;
    kernel_input_.scales = std::move(reshaped_scales);
    kernel_input_.zero_points = std::move(reshaped_zp);
    kernel_input_.precomputed_reductions = std::move(reshaped_pr);
    kernel_input_.post_ops = std::move(reshaped_post_ops);
    kernel_input_.reshape_applied = true;

    return status::success;
}

status_t jit_gemm_pd_t::commit_resolved_tags() {
    // Project resolved format tags from kernel_input_.{src,weights,dst,bias}
    // _md back into the user-facing src_md_/weights_md_/dst_md_/bias_md_
    // (matmul_pd_t members) at the user ndims. This surfaces format::any
    // resolution to the framework (mirrors base's set_default_params at
    // ../base/src/gpu/intel/matmul/gemm.hpp:383-401).
    auto reproject = [](memory_desc_t &user,
                             const memory_desc_t &kernel) -> status_t {
        if (user.ndims == kernel.ndims) {
            user = kernel;
            return status::success;
        }
        memory_desc_t out {};
        CHECK(memory_desc_reshape(out, kernel, user.ndims, user.dims));
        user = out;
        return status::success;
    };
    CHECK(reproject(src_md_, kernel_input_.src_md));
    CHECK(reproject(weights_md_, kernel_input_.weights_md));
    CHECK(reproject(dst_md_, kernel_input_.dst_md));
    if (with_bias()) CHECK(reproject(bias_md_, kernel_input_.bias_md));

    // Project resolved binary-post-op src1_desc back to attr_.post_ops_ so
    // the framework's arg_md(ATTR_MULTIPLE_POST_OP|i|SRC_1) — which returns
    // &attr_.post_ops_.entry_[i].binary.src1_desc directly — sees the
    // concrete tag. Without this, format::any-bound user binaries fail at
    // memory creation. Bias/scales-as-binary are kernel-only (their kernel
    // entry has no matching attr_ slot), so the mapping is gated on
    // binary_srcs[k].type == binary (user-provided binary).
    const auto &binary_srcs = kernel_input_.binary_srcs;
    const auto &k_post_ops = kernel_input_.post_ops;
    for (int k = 0; k < int(binary_srcs.size()); ++k) {
        if (binary_srcs[k].type != binary_src_t::binary) continue;
        const int user_idx = binary_srcs[k].index;
        const auto &kernel_entry = k_post_ops.entry_[k];
        if (!kernel_entry.is_binary()) continue;
        auto &user_entry = attr_.post_ops_.entry_[user_idx];
        if (!user_entry.is_binary()) continue;
        CHECK(reproject(
                user_entry.binary.src1_desc, kernel_entry.binary.src1_desc));
        if (user_entry.binary.src2_desc.ndims != 0
                || kernel_entry.binary.src2_desc.ndims != 0)
            CHECK(reproject(user_entry.binary.src2_desc,
                    kernel_entry.binary.src2_desc));
    }
    return status::success;
}

bool jit_gemm_pd_t::gemm_set_default_formats() {
    const auto set_one = [](memory_desc_t *md) -> bool {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) {
            if (mdw.has_runtime_dims_or_strides()) return false;
            if (memory_desc_init_by_strides(*md, nullptr) != status::success)
                return false;
        }
        return true;
    };
    bool ok = true;
    // Resolve format_any on the KERNEL-view mds (kernel_input_.X_md). The
    // user-facing src_md_/weights_md_/dst_md_/bias_md_ inherit the resolved
    // tags later via commit_resolved_tags. set_default_formats on the
    // canonicalized kernel post_ops uses kernel_input_.dst_md so that any
    // binary src1_desc resolved against `format::any` lines up with the
    // 2D/3D-collapsed dst.
    ok = ok && set_one(&kernel_input_.src_md);
    ok = ok && set_one(&kernel_input_.weights_md);
    ok = ok && set_one(&kernel_input_.bias_md);
    ok = ok && set_one(&kernel_input_.dst_md);
    ok = ok && set_one(&reduce_md_);
    ok = ok
            && (kernel_input_.post_ops.set_default_formats(&kernel_input_.dst_md)
                    == status::success);
    if (ok) init_quant_mds();
    return ok;
}

void jit_gemm_pd_t::init_quant_mds() {
    // Build kernel-view quant mds from reshape-rescaled scales/zp/gs entries
    // and the kernel-view mds. Never re-keyed under swap; the swap moves
    // into init_GEMMProblem.
    const auto &scales = kernel_input_.scales;
    const auto &zps = kernel_input_.zero_points;
    const auto &gs = kernel_input_.precomputed_reductions;
    (void)scales.get(kA).get_md(
            kernel_input_.src_scale_md, kernel_input_.src_md);
    (void)scales.get(kB).get_md(
            kernel_input_.wei_scale_md, kernel_input_.weights_md);
    (void)scales.get(kC).get_md(
            kernel_input_.dst_scale_md, kernel_input_.dst_md);
    (void)zps.get(kA).get_md(kernel_input_.src_zp_md, kernel_input_.src_md);
    (void)zps.get(kB).get_md(
            kernel_input_.wei_zp_md, kernel_input_.weights_md);
    (void)zps.get(kC).get_md(kernel_input_.dst_zp_md, kernel_input_.dst_md);
    (void)gs.get(kA).get_md(kernel_input_.src_gs_md, kernel_input_.src_md);
    (void)gs.get(kB).get_md(
            kernel_input_.wei_gs_md, kernel_input_.weights_md);
}

status_t jit_gemm_pd_t::canonicalize_post_ops() {
    // Promote prelu to binary on the kernel-view kernel_input_.post_ops
    // copy only. attr_ stays in the framework-facing form
    // (primitive_kind::prelu) so user arg binding under
    // DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)|DNNL_ARG_WEIGHTS (the prelu slope
    // key) still validates and reaches exec — without this separation the
    // framework rejected the user's prelu-keyed bind because the post-
    // canonicalize attr advertised a binary post-op (expecting
    // DNNL_ARG_SRC_1) and exec then threw std::out_of_range looking up the
    // missing prelu key. Idempotent; safe to run whether or not
    // apply_swap_ab was invoked earlier.
    const int nd = ndims();
    auto &post_ops = kernel_input_.post_ops;
    const auto &kdst_md = kernel_input_.dst_md;
    for (int i = 0; i < post_ops.len(); ++i) {
        auto &e = post_ops.entry_[i];
        if (!e.is_prelu()) continue;
        const int mask = e.prelu.mask;
        dims_t weight_dims {};
        for (int d = 0; d < nd; ++d)
            weight_dims[d] = ((mask >> d) & 0x1) ? kdst_md.dims[d] : 1;
        memory_desc_t src1 {};
        CHECK(memory_desc_init_by_strides(
                src1, nd, weight_dims, data_type::f32, nullptr));
        auto &s = src1.format_desc.blocking.strides;
        for (int d = 2; d < nd; ++d) s[d] *= weight_dims[1];
        if (nd >= 2) s[1] = 1;
        e.kind = primitive_kind::binary;
        e.binary.alg = alg_kind::eltwise_relu;
        e.binary.src1_desc = src1;
        e.binary.user_src1_desc = src1;
        e.binary.src2_desc = memory_desc_t {};
        e.binary.user_src2_desc = memory_desc_t {};
    }
    return status::success;
}

status_t jit_gemm_pd_t::apply_swap_ab() {
    // Flag flip only. No mutation of mds, attr, quant mds, or post_ops — the
    // kernel-view transformation happens entirely inside init_GEMMProblem
    // via GEMMProblem::transpose(). canonicalize_post_ops() runs at the top
    // of init_post_ops() so it happens whether or not apply_swap_ab() was
    // called.
    swap_ab_ = !swap_ab_;
    return status::success;
}

// ----- gemmstone problem assembly + attr/post-op init. Reads layouts/types/
//       sizes from un-mutated matmul mds via the swap-aware accessors on this
//       base; writes directly into GEMMProblem and calls problem.transpose()
//       once at the end if swap_ab_. -----

int jit_gemm_pd_t::quant_entry_ndims(
        const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx) {
    if (entry.has_default_values()) return -1;
    if (qmd.ndims < 2) return 0;

    int count = 0;
    for (int i = qmd.ndims - 2; i < qmd.ndims; ++i) {
        if (qmd.dims[i] > 1) { count++; }
    }

    if (count == 0) return 0;

    // For gemmstone, 1D quantization implies a full column vector
    // (i.e. not on the K dimension). If quantization varies over K,
    // we have to send these as 2D.
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

bool jit_gemm_pd_t::valid_2d_mask(int mask, int nd, bool per_tensor_ok) {
    // The per-tensor check must use the kernel-view full mask ((1 << nd) - 1),
    // not matmul_pd_t::full_tensor_mask() which is keyed off the user ndims.
    // After maybe_reshape_2d collapses batch dims, kernel-view nd < user
    // ndims, and the scale/zp mask has been re-keyed to the kernel view too —
    // so anchoring to the user mask here misclassifies legitimate per-tensor
    // configurations (e.g. 4D matmul with MX scales collapsing to 3D, scale
    // mask 7 vs user full_mask 15).
    const int kernel_full_mask = (1 << nd) - 1;
    return (mask == kernel_full_mask && per_tensor_ok)
            || utils::one_of(mask, (1 << (nd - 1)),
                    (1 << (nd - 1)) + (1 << (nd - 2)));
}

bool jit_gemm_pd_t::dy_quant_enabled() const {
    using namespace data_type;
    // Dynamic quant is asymmetric: base's a-side checked the low-precision
    // operand and base.b-side the higher-precision one. Under base.swap_ab=0
    // base.a = matmul-WEIGHTS and base.b = matmul-SRC, so the matmul-keyed
    // analogue is WEIGHTS ∈ {u8,s8,s4,u4} AND SRC ∈ {u8,s8} (gotcha #27).
    // The previous "matmul-symmetric" form rejected s8:s4:f16 wei-zp cases
    // because matmul-SRC=s8 ∉ {s4,u4}.
    const auto src_dt = kernel_input_.src_md.data_type;
    const auto wei_dt = kernel_input_.weights_md.data_type;
    bool all_f8 = (utils::one_of(src_dt, f8_e5m2, f8_e4m3)
            && utils::one_of(wei_dt, f8_e5m2, f8_e4m3)
            && utils::one_of(c_type(), f8_e5m2, f8_e4m3, f16, bf16, f32));
    return (utils::one_of(c_type(), f32, f16, bf16, u8, s8)
                   && utils::one_of(wei_dt, u8, s8, s4, u4)
                   && utils::one_of(src_dt, u8, s8))
            || all_f8;
}

bool jit_gemm_pd_t::wei_decomp() const {
    using namespace data_type;
    // Weights-decompression: matmul-WEIGHTS is the low-precision/compressed
    // operand, matmul-SRC is the higher-precision floating operand. Mirrors
    // base's check (../base/.../pd.cpp:217-227) which keys on a_type=matmul-
    // WEIGHTS and b_type=matmul-SRC under base's BLAS view. Earlier "matmul-
    // symmetric" worktree variant broke f8_e5m2:f4_e2m1 (both operands match
    // both int_low and float_hi sets; symmetric logic picked SRC as int_t and
    // returned false).
    const auto src_dt = kernel_input_.src_md.data_type;
    const auto wei_dt = kernel_input_.weights_md.data_type;
    return (utils::one_of(c_type(), f32, f16, bf16, f8_e5m2, f8_e4m3)
                   && utils::one_of(wei_dt, u8, s8, s4, u4, f8_e4m3, f8_e5m2,
                           f4_e2m1, f4_e3m0)
                   && utils::one_of(
                           src_dt, f16, f32, bf16, f8_e5m2, f8_e4m3))
            && types::data_type_bits(wei_dt)
            < types::data_type_bits(src_dt)
            && attr()->mayiconvert(wei_dt, f32);
}

bool jit_gemm_pd_t::quant_enabled() const {
    return wei_decomp() || dy_quant_enabled();
}

bool jit_gemm_pd_t::grouped(int matmul_arg) const {
    // Matmul-natural group probe per matmul-side arg key. Read the consolidated
    // group(0)/group(1) across zp/gs/scales for the requested entry from the
    // kernel-view kernel_input_ (so reshape-rescaled groups are visible). arg
    // semantics:
    //   kA (SRC):     g0 = M-direction, g1 = K-direction.
    //   kB (WEIGHTS): g0 = K-direction, g1 = N-direction.
    //   kC (DST):     g0 = M-direction, g1 = N-direction (scales only).
    // Consumers that need a kernel-A/B view OR the two together (the typical
    // "int_acc detection" symmetric query) compose at the call site.
    auto group01 = [this](int arg, int &g0, int &g1) {
        const auto &zp = kernel_input_.zero_points.get(arg);
        const auto &gs = kernel_input_.precomputed_reductions.get(arg);
        const auto &sc = kernel_input_.scales.get(arg);
        g0 = 0;
        g1 = 0;
        auto add = [&](const quant_entry_t &e) {
            if (e.has_default_groups()) return;
            int eg0 = into<int>(e.get_group(0));
            int eg1 = into<int>(e.get_group(1));
            if (g0 == 0) g0 = eg0;
            if (g1 == 0) g1 = eg1;
        };
        add(zp);
        add(gs);
        add(sc);
    };
    int g0 = 0, g1 = 0;
    if (matmul_arg == kA) {
        group01(kA, g0, g1);
        bool mg = 1 < g0 && g0 < M();
        bool kg = 1 < g1 && g1 < K();
        return mg || kg;
    }
    if (matmul_arg == kB) {
        group01(kB, g0, g1);
        bool kg = 1 < g0 && g0 < K();
        bool ng = 1 < g1 && g1 < N();
        return kg || ng;
    }
    if (matmul_arg == kC) {
        const auto &sc = kernel_input_.scales.get(kC);
        if (sc.has_default_groups()) return false;
        g0 = into<int>(sc.get_group(0));
        g1 = into<int>(sc.get_group(1));
        bool mg = 1 < g0 && g0 < M();
        bool ng = 1 < g1 && g1 < N();
        return mg || ng;
    }
    gpu_error_not_expected();
    return false;
}

status_t jit_gemm_pd_t::init_post_ops(impl::engine_t *engine) {
    using namespace primitive_kind;
    using namespace alg_kind;
    using namespace data_type;

    // kernel_input_.post_ops is already populated by maybe_reshape_2d (from
    // attr_.post_ops_, reshape-rescaled when applicable). Canonicalize
    // prelu→binary on the kernel copy only. attr_ stays in framework form
    // for user arg binding.
    CHECK(canonicalize_post_ops());
    auto &post_ops = kernel_input_.post_ops;
    auto &binary_srcs = kernel_input_.binary_srcs;
    binary_srcs.clear();
    binary_srcs.reserve(post_ops.len() + 4);

    bool ok = true;
    int prelu_count = 0;
    const int num_orig_postops = post_ops.len();
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &e = post_ops.entry_[i];
        switch (e.kind) {
            case binary:
                if (e.binary.alg == binary_prelu) {
                    binary_srcs.push_back(
                            binary_src_t {binary_src_t::prelu, int(i)});
                    kernel_input_.prelu_wei_md = e.binary.src1_desc;
                    prelu_count++;
                    ok &= prelu_count <= 1;
                } else {
                    ok &= supported_binary_op(e.binary.alg)
                            && is_md_gemm_compatible_plain_format(
                                    &e.binary.src1_desc);
                    binary_srcs.push_back(
                            binary_src_t {binary_src_t::binary, int(i)});
                }
                break;
            case sum: {
                // Only one sum allowed; check by scanning entries seen so
                // far. Use kind-equality, not is_sum() — the latter rejects
                // sum.scale != 1.0 / sum.zero_point != 0 with default args.
                bool already_sum = false;
                for (int j = 0; j < i; ++j)
                    if (post_ops.entry_[j].kind == sum) {
                        already_sum = true;
                        break;
                    }
                ok &= !already_sum;
                binary_srcs.push_back(binary_src_t {binary_src_t::none, 0});
                break;
            }
            case eltwise:
                ok &= eltwise_injector_f32_is_supported(e.eltwise.alg);
                binary_srcs.push_back(binary_src_t {binary_src_t::none, 0});
                break;
            case prelu:
                VDISPATCH_JIT_GEMM(false,
                        "%s: prelu post-op not canonicalized",
                        VERBOSE_UNSUPPORTED_POSTOP);
                break;
            default: VDISPATCH_JIT_GEMM(false, VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    VDISPATCH_JIT_GEMM(ok, VERBOSE_UNSUPPORTED_POSTOP);

    const auto &a_scales = kernel_input_.scales.get(kA);
    const auto &b_scales = kernel_input_.scales.get(kB);
    const auto &c_scales = kernel_input_.scales.get(kC);

    // matmul-driven bias is ALWAYS routed through binary post-op. Use the
    // kernel-view bias md so the binary src1_desc carries the 2D shape.
    if (with_bias()) {
        VDISPATCH_JIT_GEMM_SC(
                post_ops.prepend_binary(binary_add, &kernel_input_.bias_md),
                "%s: bias via binary post-op", VERBOSE_UNSUPPORTED_POSTOP);
        binary_srcs.insert(
                binary_srcs.begin(), binary_src_t {binary_src_t::bias, 0});
    }

    auto maybe_convert_scales_to_postop
            = [this, engine, &post_ops, &binary_srcs](
                      const memory_desc_t &scale_md, int arg, int scale_ndims,
                      bool mx, bool &converted) -> status_t {
        const int nd = ndims();
        converted = false;
        if (scale_ndims > 1) return status::success;
        // matmul-side inner_dim: SRC scales vary per-M (inner_dim=K=ndims-1);
        // WEIGHTS scales vary per-N (inner_dim=K=ndims-2); DST has no K.
        int inner_dim = (arg == kA ? nd - 1 : nd - 2);
        bool convert = (scale_md.dims[inner_dim] <= 1) || (arg == kC);
        convert &= !mx;
        if (convert) {
            // binary_src_t::index stores the matmul attr key. At exec the
            // scales buffer is fetched via CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES
            // | src.index) directly — no swap routing.
            if (arg == kC) {
                VDISPATCH_JIT_GEMM_SC(
                        post_ops.append_binary(binary_div, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                binary_srcs.push_back(
                        binary_src_t {binary_src_t::scales, arg});
            } else {
                VDISPATCH_JIT_GEMM_SC(
                        post_ops.prepend_binary(binary_mul, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                binary_srcs.insert(binary_srcs.begin(),
                        binary_src_t {binary_src_t::scales, arg});
            }
            converted = true;
        }
        return status::success;
    };

    if (!a_scales.has_default_values() && !a_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(kernel_input_.src_scale_md, kA,
                a_scale_ndims(), a_scales.is_mx(), converted));
        if (converted) a_scale_ndims_override_ = -1;
    }

    if (!b_scales.has_default_values() && !b_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(kernel_input_.wei_scale_md, kB,
                b_scale_ndims(), b_scales.is_mx(), converted));
        if (converted) b_scale_ndims_override_ = -1;
    }

    bool try_c_scale = !c_scales.is_host_scalar() || num_orig_postops > 0
            || with_bias();
    if (!c_scales.has_default_values() && try_c_scale) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(kernel_input_.dst_scale_md, kC,
                c_scale_ndims(), c_scales.is_mx(), converted));
        gpu_assert(converted || c_scales.is_mx())
                << "Unable to convert dst scales to a post op";
    }

    return status::success;
}

status_t jit_gemm_pd_t::init_attrs(impl::engine_t *engine) {
    return status::success;
}

status_t jit_gemm_pd_t::zp_ok(impl::engine_t *engine) {
    using namespace data_type;
    auto &k_zps = kernel_input_.zero_points;
    if (k_zps.has_default_values()) return status::success;
    auto &a_zps = k_zps.get(kA);
    auto &b_zps = k_zps.get(kB);
    auto &c_zps = k_zps.get(kC);

    const int cmask_a = a_zps.get_mask();
    const int cmask_b = b_zps.get_mask();
    const int cmask_c = c_zps.get_mask();

    const int nd = ndims();
    // Type checks are matmul-symmetric; read SRC/WEIGHTS dtypes directly.
    const bool a_int4
            = utils::one_of(kernel_input_.src_md.data_type, s4, u4);
    const bool b_int4
            = utils::one_of(kernel_input_.weights_md.data_type, s4, u4);
    // weights_upconversion = "weights are upconverted (decompressed)". In
    // base this was `a_int4 && dy_quant_enabled` where base.a = matmul-
    // WEIGHTS under swap_ab=0; the matmul-keyed analogue is b_int4
    // (gotcha #27). Without this swap, s8:s4:f16 wei-zp per_tensor was
    // rejected by zp_ok because per_tensor_ok defaulted to false.
    const bool weights_upconversion
            = wei_decomp() || (b_int4 && dy_quant_enabled());

    if (!a_zps.has_default_values()) {
        // INT4 ZPs on SRC do not expand the range in a meaningful way; base's
        // analogous check sits in the kernel-B (= matmul-SRC) block. Worktree
        // is matmul-keyed so the reject lives here on a_zps (matmul-SRC).
        VDISPATCH_JIT_GEMM(!utils::one_of(a_zps.get_data_type(), s4, u4),
                VERBOSE_UNSUPPORTED_ZP_CFG);

        if (!a_zps.has_default_groups()) {
            VDISPATCH_JIT_GEMM(valid_2d_mask(cmask_a, nd, weights_upconversion),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto a_q2d_group_n = a_zps.get_group(1);
            VDISPATCH_JIT_GEMM(a_q2d_group_n == 1,
                    "%s: Grouped N dimension on A matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            bool has_prB = !kernel_input_.precomputed_reductions
                                    .has_default_values(kB);
            bool is_dequantized = !dy_quant_enabled() || !b_int4 || a_int4;
            VDISPATCH_JIT_GEMM(IMPLICATION(a_zp_2d(), is_dequantized || has_prB),
                    "%s: Nontrivial groups on A matrix, and no precomputed "
                    "reductions or dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_JIT_GEMM(utils::one_of(cmask_a, 0, mask_scalar,
                                       mask_a_m | mask_a_k),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_JIT_GEMM(IMPLICATION(a_scales_2d(),
                                       !(b_int4 && !wei_decomp() && !a_int4)),
                    "%s: 2D scales on A matrix, but no weights decompression",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!b_zps.has_default_values()) {
        if (!b_zps.has_default_groups()) {
            VDISPATCH_JIT_GEMM(valid_2d_mask(cmask_b, nd, weights_upconversion),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            // N-group on matmul-WEIGHTS. For 2D ZP with group_dims [K, N],
            // get_group(1) is the N (output-channel) group. Base's analogous
            // check on matmul-WEIGHTS zp (pd.cpp:365) strictly requires == 1.
            const auto b_q2d_group_n = b_zps.get_group(1);
            VDISPATCH_JIT_GEMM(b_q2d_group_n == 1,
                    "%s: Nontrivial N groups on B matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            bool is_dequantized = !dy_quant_enabled() || !a_int4 || b_int4;
            VDISPATCH_JIT_GEMM(IMPLICATION(b_zp_2d(), is_dequantized),
                    "%s: Grouped B zero points, and no dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_JIT_GEMM(
                    utils::one_of(cmask_b, 0, mask_b_k, mask_b_n),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!k_zps.has_default_values(kC)) {
        VDISPATCH_JIT_GEMM(
                IMPLICATION(!c_zps.is_host_scalar(),
                        utils::one_of(cmask_c, 0, mask_scalar, mask_c_m)),
                "%s: unsupported C mask", VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    return status::success;
}

status_t jit_gemm_pd_t::gs_ok(impl::engine_t *engine) {
    auto &k_gs = kernel_input_.precomputed_reductions;
    if (k_gs.has_default_values()) return status::success;

    VDISPATCH_JIT_GEMM(k_gs.has_default_values(kC),
            VERBOSE_UNSUPPORTED_PR_CFG);

    bool with_a_group_sums_ = !k_gs.has_default_values(kA);
    bool with_b_group_sums_ = !k_gs.has_default_values(kB);

    VDISPATCH_JIT_GEMM(
            IMPLICATION(with_a_group_sums_,
                    k_gs.get_data_type(kA) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_JIT_GEMM(
            IMPLICATION(with_b_group_sums_,
                    k_gs.get_data_type(kB) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);

    return status::success;
}

status_t jit_gemm_pd_t::scales_ok(impl::engine_t *engine) {
    const auto &scales = kernel_input_.scales;
    if (scales.has_default_values()) return status::success;
    const int nd = ndims();
    using namespace data_type;

    for (auto s : {kA, kB}) {
        if (scales.has_default_values(s) || scales.get(s).is_host_scalar())
            continue;
        const auto &x_scales = scales.get(s);

        auto mask = x_scales.get_mask();
        const bool scalar_or_axis = (s == kA)
                ? utils::one_of(mask, 0, mask_scalar, mask_a_m, mask_a_k)
                : utils::one_of(mask, 0, mask_scalar, mask_b_k, mask_b_n);
        bool supportedMask = scalar_or_axis
                || (!x_scales.has_default_groups()
                        && valid_2d_mask(mask, nd));
        VDISPATCH_JIT_GEMM(supportedMask, "%s: unsupported A/B mask",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    const auto &dst_scales = scales.get(kC);
    if (!dst_scales.has_default_values() && !dst_scales.is_host_scalar()) {
        auto mask = dst_scales.get_mask();
        bool supportedMask
                = utils::one_of(mask, 0, mask_scalar, mask_c_m, mask_c_n)
                || (!dst_scales.has_default_groups() && with_mx_scale()
                        && valid_2d_mask(mask, nd));
        VDISPATCH_JIT_GEMM(supportedMask, "%s: unsupported C mask",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    if (!dst_scales.has_default_values() && with_mx_scale()) {
        VDISPATCH_JIT_GEMM(dst_scales.get_group(0) == 1
                        && dst_scales.get_group(1) == 32
                        && arch_ >= compute::gpu_arch_t::xe_hpc,
                "%s: unsupported mx_scale groups",
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        // M+N dimensions must have trivial strides for Dynamic Dst Quant.
        const auto &md = kernel_input_.dst_md;
        auto strides = md.format_desc.blocking.strides;
        VDISPATCH_JIT_GEMM(strides[md.ndims - 1] == 1,
                "%s: unsupported mx_scale strides",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_JIT_GEMM(strides[md.ndims - 2] == md.dims[md.ndims - 1],
                "%s: unsupported mx_scale strides",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    return status::success;
}

status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops_in) {
    using namespace gemmstone;
    problem.postOps = std::move(post_ops_in);
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
            // relative_md_t broadcast_mask: bit 0 = last (innermost) dim,
            // bit 1 = second-to-last, etc. For 2D matmul: bit 0 = matmul-N
            // (col axis), bit 1 = matmul-M (row axis). Matmul-natural
            // pre-transpose: binaryRow = "varies along matmul-M" =
            // "M-axis not broadcast" = (mask & bit1) == 0. The final
            // problem.transpose() under swap_ab_ flips to kernel view.
            bool is_multi_row = (src_rmd.broadcast_mask & 2) == 0;
            bool is_multi_col = (src_rmd.broadcast_mask & 1) == 0;

            bool is_compatible = src_rmd.inner_layout.empty();
            if (!is_compatible) return status::unimplemented;

            bool trans = is_multi_row && !src_rmd.inner_dim.is_innermost();

            problem.Tbinary.push_back(T);
            problem.postOps.binaryRow[i] = is_multi_row;
            problem.postOps.binaryCol[i] = is_multi_col;
            problem.postOps.binaryBatch[i] = src_rmd.ndims() >= 3;
            problem.postOps.binaryTrans[i] = trans;

            MatrixAddressing atype;
            // Matmul-natural pre-transpose layout. Gated on is_multi_col (not
            // is_multi_row) so the value is the post-transpose-DUAL of base's
            // kernel-canonical layout. Base reads bit 0 (kernel-row =
            // matmul-N) for "row varies"; worktree reads bit 1 (matmul-M) for
            // matmul-natural "row varies". The two agree for "both vary" and
            // "only N varies", but diverge for "only M varies" (e.g. prelu
            // per_oc on 2D-reshaped matmul), where base.is_multi_row=false
            // while worktree.is_multi_row=true. Gating on is_multi_col keeps
            // the matmul-natural→kernel-canonical transpose round-trip aligned
            // for all four 1D/2D varying-axis cases.
            bool layout_trans
                    = is_multi_col && !src_rmd.inner_dim.is_innermost();
            atype.layout
                    = layout_trans ? MatrixLayout::N : MatrixLayout::T;
            atype.crosspack = 1;
            atype.packSize = 0;
            atype.setAlignment(T.size());

            problem.binary.push_back(atype);
        }
    }

    return status::success;
}

status_t jit_gemm_pd_t::init_GEMMProblem(
        gemmstone::GEMMProblem &problem, const intel::engine_t *engine) const {
    using namespace gemmstone;
    problem = {};

    problem.product = get_ngen_product(*engine->device_info());
    bool has_systolic
            = engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            || engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate);

    // Build the problem in MATMUL convention (A=SRC, B=WEIGHTS, C=DST). At
    // the very end, if swap_ab_ is set, the entire problem (including
    // postOps and binary[]) is GEMMProblem::transpose()'d once. This is the
    // single swap_ab boundary; nothing else in the pd is rotated under swap.

    const auto &k_src_md = kernel_input_.src_md;
    const auto &k_wei_md = kernel_input_.weights_md;
    const auto &k_dst_md = kernel_input_.dst_md;

    const auto src_t = k_src_md.data_type;
    const auto wei_t = k_wei_md.data_type;
    const auto dst_t = k_dst_md.data_type;

    const auto M_ = M();
    const auto N_ = N();
    const auto K_ = K();

    // Natural matmul leading dims. (For LDA/LDB/LDC the kernel-view values
    // come from the pd's swap-aware helpers; here we only need the matmul
    // sizes for the alignment/size estimates.)
    const auto lda_matmul = get_ld(k_src_md);
    const auto ldb_matmul = get_ld(k_wei_md);
    const auto ldc_matmul = get_ld(k_dst_md);

    // Matmul SRC is M×K row-major; the kernel views it as col-major K×M.
    // Take the natural matmul "is trans" via get_trans on the un-mutated md.
    bool tr_a_matmul = (get_trans(k_src_md) == transpose::trans);
    bool tr_b_matmul = (get_trans(k_wei_md) == transpose::trans);

    // Single-row/column padding (matmul-keyed, mirrors base's pad-to-16 in
    // jit.hpp). The padded ld is used only for alignment/size estimates;
    // the raw matmul ld is still the kernel-launch leading-dim source.
    // Conditions are matmul-axis-keyed; the final problem.transpose() under
    // swap_ab_ then maps align_a/align_b to the right kernel slots.
    dim_t pad_lda_to = lda_matmul;
    dim_t pad_ldb_to = ldb_matmul;
    if ((K_ == 1 && tr_a_matmul) || (M_ == 1 && !tr_a_matmul))
        pad_lda_to = utils::rnd_up(pad_lda_to, 16);
    if ((N_ == 1 && tr_b_matmul) || (K_ == 1 && !tr_b_matmul))
        pad_ldb_to = utils::rnd_up(pad_ldb_to, 16);

    // Pre-swap kernel-A alignment uses matmul SRC's row stride; under
    // transpose at end the kernel-A alignment field is swapped with B's.
    int align_a = utils::max_pow2_div(
            types::elements_to_bytes(src_t, pad_lda_to));
    int align_b = utils::max_pow2_div(
            types::elements_to_bytes(wei_t, pad_ldb_to));
    int align_c = utils::max_pow2_div(
            types::elements_to_bytes(dst_t, ldc_matmul));
    for (int b = 0; b < batch_dims(); b++) {
        int stride_bytes_a = utils::max_pow2_div(
                types::elements_to_bytes(src_t, stride_for(k_src_md, b)));
        int stride_bytes_b = utils::max_pow2_div(
                types::elements_to_bytes(wei_t, stride_for(k_wei_md, b)));
        int stride_bytes_c = utils::max_pow2_div(
                types::elements_to_bytes(dst_t, stride_for(k_dst_md, b)));
        if (stride_bytes_a) align_a = nstl::min(align_a, stride_bytes_a);
        if (stride_bytes_b) align_b = nstl::min(align_b, stride_bytes_b);
        if (stride_bytes_c) align_c = nstl::min(align_c, stride_bytes_c);
    }
    align_a = nstl::max(align_a, (int)types::data_type_size(src_t));
    align_b = nstl::max(align_b, (int)types::data_type_size(wei_t));
    align_c = nstl::max(align_c, (int)types::data_type_size(dst_t));

    // Total byte counts for needA64 selection. The trans flags are
    // matmul-natural (tr_a_matmul = get_trans(matmul-SRC), tr_b_matmul =
    // get_trans(matmul-WEIGHTS)). Element counts:
    //   SRC [M,K] row-major (tr_a_matmul=false): lda=K, total = M*K = M*lda.
    //   SRC [M,K] col-major (tr_a_matmul=true) : lda=M, total = K*M = K*lda.
    //   WEIGHTS [K,N] row-major (tr_b_matmul=false): ldb=N, total = K*N = K*ldb.
    //   WEIGHTS [K,N] col-major (tr_b_matmul=true) : ldb=K, total = N*K = N*ldb.
    //   DST [M,N] row-major (notrans): ldc=N, total = M*N = M*ldc.
    //   DST [M,N] col-major (trans)  : ldc=M, total = N*M = N*ldc.
    // (Base used the same formula but with KERNEL-VIEW trans, which is the
    // negated sense relative to matmul-natural — porting it verbatim under-
    // triggered needA64 for >4 GiB skinny-K / skinny-M shapes.)
    const bool tr_c_matmul = (get_trans(k_dst_md) == transpose::trans);
    auto a_size = (tr_a_matmul ? K_ : M_) * lda_matmul
            * types::data_type_size(src_t);
    auto b_size = (tr_b_matmul ? N_ : K_) * ldb_matmul
            * types::data_type_size(wei_t);
    auto c_size = (tr_c_matmul ? N_ : M_) * ldc_matmul
            * types::data_type_size(dst_t);

    bool int_acc = utils::one_of(src_t, data_type::s8, data_type::u8)
            || (types::is_integral_dt(src_t) && types::is_integral_dt(wei_t));
    int_acc &= !(grouped(kA) || grouped(kB));

    // Kernel-view group-zero detection (no kernel rotation yet).
    auto sc_src = kernel_input_.scales.get(kA);
    auto sc_wei = kernel_input_.scales.get(kB);
    auto sc_dst = kernel_input_.scales.get(kC);
    auto zp_src = kernel_input_.zero_points.get(kA);
    auto zp_wei = kernel_input_.zero_points.get(kB);

    // Bias never flows via the CO channel — it is prepended as a binary
    // post-op in init_post_ops. The CO channel carries only c-zero-points
    // (Tco=s32) or sum_ab (Tco=reduce dt).
    auto co_t = with_sum_ab() ? sum_ab_type()
            : int_acc         ? data_type::s32
                              : dst_t;

    auto acc_t = int_acc ? data_type::s32
            : (utils::one_of(data_type::f64, src_t, wei_t) ? data_type::f64
                                                           : data_type::f32);

    const auto &k_post_ops = kernel_input_.post_ops;
    bool with_binary = (k_post_ops.find(primitive_kind::binary) != -1)
            || (k_post_ops.find(primitive_kind::prelu) != -1);

    bool need_x32_acc = with_binary || !IMPLICATION(with_sum(), sum_at_begin());

    switch (attr()->acc_mode_) {
        case accumulation_mode::any:
            if (!need_x32_acc) acc_t = data_type::undef;
            break;
        case accumulation_mode::f16: acc_t = data_type::f16; break;
        case accumulation_mode::f32: acc_t = data_type::f32; break;
        case accumulation_mode::s32: acc_t = data_type::s32; break;
        default: break;
    }
    if (wei_decomp()) { acc_t = data_type::f32; }

    const bool dst_sround = with_sround();
    const bool c_offset = with_c_zero_points();

    // Matmul-side fill: A = SRC, B = WEIGHTS, C = DST. tr_a/tr_b come from
    // the natural matmul transpose; the kernel-A and kernel-B layouts will
    // be re-derived via problem.transpose() if swap_ab_ is set.
    problem.Ta = problem.Ta_ext = convert_dnnl_to_kernel_type(src_t);
    problem.Tb = problem.Tb_ext = convert_dnnl_to_kernel_type(wei_t);
    problem.Tc = convert_dnnl_to_kernel_type(acc_t);
    problem.Tc_ext = convert_dnnl_to_kernel_type(dst_t);
    problem.Ts = problem.Tc;
    problem.Tao = convert_dnnl_to_kernel_type(zp_src.get_data_type());
    problem.Tbo = convert_dnnl_to_kernel_type(zp_wei.get_data_type());
    problem.Tco = convert_dnnl_to_kernel_type(co_t);
    // MatrixLayout::T = row-major, ::N = col-major. Set A/B consistently from
    // matmul-natural storage: row-major matmul md → T, col-major → N. The
    // final problem.transpose() under swap_ab_ flips both, landing the gemm
    // col-major view (N for the row-major-matmul case) that matches base's
    // catalog hash. C.layout is hard-coded N (see post-transpose override
    // below; gotcha #25).
    problem.A.layout
            = tr_a_matmul ? MatrixLayout::N : MatrixLayout::T;
    problem.B.layout
            = tr_b_matmul ? MatrixLayout::N : MatrixLayout::T;
    problem.C.layout = MatrixLayout::N;
    problem.A.crosspack = problem.B.crosspack = problem.C.crosspack = 1;
    problem.A.packSize = problem.B.packSize = problem.C.packSize = 0;
    problem.A.setAlignment(align_a);
    problem.B.setAlignment(align_b);
    problem.C.setAlignment(align_c);

    bool needA64 = std::max({a_size, b_size, c_size})
            > std::numeric_limits<uint32_t>::max();
    problem.A.needA64 = needA64;
    problem.B.needA64 = needA64;
    problem.C.needA64 = needA64;

    if (batch_dims() > 0) {
        problem.batch = BatchMode::Strided;
        problem.batchDims = batch_dims();
    }
    // aOffset/bOffset use matmul-side zp presence; transpose() swaps them.
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

    // Pre-transpose A_scale/B_scale layout convention. Base sets A_scale.pre
    // = N (always) and B_scale.pre = T if !bScale2D else N (where bScale2D is
    // measured on base's kernel-B side). Worktree is matmul-natural pre-init
    // and applies a full transpose() at swap_ab_. To converge on base's post-
    // state for both worktree.swap_ab=1 ↔ base.swap=0 and worktree.swap_ab=0
    // ↔ base.swap=1 cases, the matmul-natural pre values are:
    //   pre.A_scale.layout = T if aScale2D (matmul-SRC 2D) else N;
    //   pre.B_scale.layout = T (always).
    // After std::swap(A_scale, B_scale) + per-field .transpose() in
    // problem.transpose(), this gives base-matching A_scale/B_scale layouts.
    if (problem.aScale2D())
        problem.A_scale.layout = MatrixLayout::T;
    problem.B_scale.layout = MatrixLayout::T;

    if (!problem.bOffset2D()) problem.BO.layout = MatrixLayout::T;
    if (b_gs_ndims() < 2) problem.Bg.layout = MatrixLayout::T;
    // Symmetric matmul-natural pre-init for Ag. Base's pre rule for Bg is
    // "if kernel-b gs_ndims < 2 → T". Under full transpose (swap A↔B + flip),
    // post.Bg = pre.Ag.flipped. To match base.post.Bg when matmul-SRC has 2D
    // GS (a_gs_ndims >= 2) and the case routes through swap_ab=1, we need
    // pre.Ag = T. Without this, post.Bg = N.flipped = T while base = N →
    // wrong layout for the matmul-SRC GS buffer.
    if (a_gs_ndims() >= 2) problem.Ag.layout = MatrixLayout::T;

    if (zp_src.get_data_type() != data_type::undef)
        problem.AO.setAlignment(
                int(types::data_type_size(zp_src.get_data_type())));
    if (zp_wei.get_data_type() != data_type::undef)
        problem.BO.setAlignment(
                int(types::data_type_size(zp_wei.get_data_type())));

    // Matmul-side group dims, consolidated per arg (zp / gs / scales must
    // agree). Read from kernel_input_ so that maybe_reshape_2d's adjusted
    // groups (post-batch-fold) are used; reading attr() here would return
    // the un-reshaped group dims and produce e.g. aqGroupM=1 instead of 2
    // for a 2x1x... → 2x... batch-into-M fold (gotcha #34).
    auto consolidate = [&](int matmul_arg, int &g0, int &g1) -> status_t {
        const auto &zp = kernel_input_.zero_points.get(matmul_arg);
        const auto &gs = kernel_input_.precomputed_reductions.get(matmul_arg);
        const auto &sc = kernel_input_.scales.get(matmul_arg);
        g0 = 0;
        g1 = 0;
        auto add = [&](const quant_entry_t &e) -> status_t {
            if (e.has_default_groups()) return status::success;
            int eg0 = into<int>(e.get_group(0));
            int eg1 = into<int>(e.get_group(1));
            if (g0 == 0)
                g0 = eg0;
            else if (g0 != eg0)
                return status::unimplemented;
            if (g1 == 0)
                g1 = eg1;
            else if (g1 != eg1)
                return status::unimplemented;
            return status::success;
        };
        CHECK(add(zp));
        CHECK(add(gs));
        CHECK(add(sc));
        return status::success;
    };

    int src_g0 = 0, src_g1 = 0; // SRC: get_group(0)=M, get_group(1)=K.
    int wei_g0 = 0, wei_g1 = 0; // WEIGHTS: get_group(0)=K, get_group(1)=N.
    int dst_g0 = 0, dst_g1 = 0; // DST scales: get_group(0)=M, get_group(1)=N.
    CHECK(consolidate(kA, src_g0, src_g1));
    CHECK(consolidate(kB, wei_g0, wei_g1));
    if (!sc_dst.has_default_groups()) {
        dst_g0 = into<int>(sc_dst.get_group(0));
        dst_g1 = into<int>(sc_dst.get_group(1));
    }

    // Write matmul-convention: aqGroupM/aqGroupK come from SRC entry;
    // bqGroupK/bqGroupN from WEIGHTS; cqGroupM/cqGroupN from DST.
    problem.aqGroupM = src_g0;
    problem.aqGroupK = src_g1;
    problem.bqGroupK = wei_g0;
    problem.bqGroupN = wei_g1;

    if (sc_src.get_data_type() != data_type::undef) {
        problem.Ta_scale = convert_dnnl_to_kernel_type(sc_src.get_data_type());
        problem.A_scale.setAlignment(
                int(types::data_type_size(sc_src.get_data_type())));
    }
    if (sc_wei.get_data_type() != data_type::undef) {
        problem.Tb_scale = convert_dnnl_to_kernel_type(sc_wei.get_data_type());
        problem.B_scale.setAlignment(
                int(types::data_type_size(sc_wei.get_data_type())));
    }

    if (sc_dst.get_data_type() != data_type::undef) {
        problem.csPtrDims = c_scale_ndims();
        problem.cMXScale = with_mx_scale();
        problem.Tc_scale = convert_dnnl_to_kernel_type(sc_dst.get_data_type());
        problem.cqGroupM = dst_g0;
        problem.cqGroupN = dst_g1;
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
            gpu_post_ops, k_post_ops, k_dst_md, get_post_op_specializations()));

    CHECK(transfer_post_ops(problem, std::move(gpu_post_ops)));

    // matmul-side sum_ab. matmul_reduce_kind::src = sum_a_row.
    sum_ab_t reduce_ab_matmul = sum_ab::sum_none;
    if (with_reduce()) {
        if (reduce_kind() == matmul_reduce_kind::src)
            reduce_ab_matmul = sum_ab::sum_a_row;
        else if (reduce_kind() == matmul_reduce_kind::weights)
            reduce_ab_matmul = sum_ab::sum_b_col;
    }

    if (c_offset || reduce_ab_matmul != sum_ab::sum_none) {
        if (c_offset) problem.cOffset = COffset::Post;
        problem.CO.crosspack = 1;
        problem.CO.alignment = problem.C.alignment;
        // CO carries c-zero-points (scalar/1D) or sum_ab (1D), both row-major.
        problem.CO.layout = MatrixLayout::T;
        problem.coPtrDims = c_zp_host_scalar() ? -1 : c_zp_ndims();
    }

    problem.sumA = (reduce_ab_matmul == sum_ab::sum_a_row);
    problem.sumB = (reduce_ab_matmul == sum_ab::sum_b_col);
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

    // Single swap boundary. The pre-init block above builds the problem in
    // matmul-natural convention (A=SRC, B=WEIGHTS, C=DST). swap_ab_ at the
    // matmul-pd boundary is semantically the math transpose of the problem
    // (kernel computes C^T = B^T A^T with kernel-M ↔ matmul-N, etc.). One
    // GEMMProblem::transpose() call performs that transformation in full.
    // NO undos, NO resets: missing field swaps must be added to
    // problem_utils.cpp::GEMMProblem::transpose() instead. See CLAUDE.md
    // "Core doctrine: init in matmul-natural + one transpose".
    if (swap_ab_) {
        problem.transpose();
        // Doctrine exemption: GEMMProblem::transpose() unconditionally flips
        // CO.layout, but base's swap_ab branch does NOT touch CO (see
        // ../base/src/gpu/intel/gemm/jit/pd.cpp:807-815). When CO is unused
        // (no c_offset / sum_ab) the field is still part of the catalog hash,
        // so leaving it flipped under swap_ab causes a kernel-selection
        // mismatch with base. Restore the default (N) here so an unused CO
        // matches base regardless of swap_ab. The c_offset/sum_ab block above
        // sets CO.layout explicitly pre-transpose so that branch still ends
        // at base's kernel-canonical value after the flip.
        if (problem.cOffset == COffset::None && !problem.sumA
                && !problem.sumB) {
            problem.CO.layout = MatrixLayout::N;
        }
        // Doctrine exemption (gotcha #26): base hard-codes AO.layout=N pre
        // and only conditionally sets BO.layout=T (../base/.../pd.cpp:698,
        // 706); under base.swap_ab=1, per-aux .transpose() at line 810-811
        // flips both. Worktree's matmul-natural pre-init has AO/BO holding
        // matmul-SRC/WEIGHTS zp slots; the full transpose() rotates them to
        // the kernel-A/B view. For wei_decomp cases (matmul-WEIGHTS zp 2D,
        // matmul-SRC zp default), worktree.swap_ab=1 corresponds to base.swap
        // _ab=0 (no per-aux flip), so the kernel-A side (matmul-WEIGHTS slot
        // post-transpose, = worktree's AO) should be N and kernel-B side (=
        // worktree's BO, holding matmul-SRC) should be T (since matmul-SRC
        // zp is not 2D). Without this, AO ends at T and the kernel reads
        // the 2D wei-zp data with wrong stride → wrong output. The condition
        // mirrors base's `if (!bOffset2D()) BO=T`, but in worktree's post-
        // transpose state bOffset2D reads matmul-SRC zp 2D-ness.
        problem.AO.layout = MatrixLayout::N;
        problem.BO.layout
                = !problem.bOffset2D() ? MatrixLayout::T : MatrixLayout::N;
    }

    // Doctrine exemption (gotcha #25): C.layout is hard-coded N in base for
    // all swap_ab paths (../base/src/gpu/intel/gemm/jit/pd.cpp:671), and the
    // kernel codegen asserts on it (gemm.cxx:517: `if (C.layout != N)
    // stub();`). The catalog selector hashes layoutChar(C.layout) (see
    // kernel_selector.cpp:392), so any divergence causes a catalog miss.
    // GEMMProblem::transpose() flips C.layout, so under swap_ab_ the pre-N
    // becomes T post-transpose; restore N here. For swap_ab_=false the pre-N
    // already matches; the assignment is a no-op.
    problem.C.layout = MatrixLayout::N;

    // DOCTRINE EXEMPTION (review.md #5). The matmul-natural skinny-N path
    // (swap_ab_=false, N==1, WEIGHTS naturally notrans) needs B.layout=N
    // and align_b sized for the K-direction stride to match base's catalog
    // hash. base's equivalent — jit.hpp:107 `if (!transa_ && m == 1) {
    // transa_=true; lda_=k(); }` under swap_ab_=true — produces the same
    // post-state via swap+negate. We cannot mirror that in the pre-transpose
    // block (Rules going forward #1: no swap_ab_ ternaries pre-transpose)
    // and GEMMProblem::transpose() doesn't run under swap_ab_=false, so this
    // post-transpose patch is the single allowed exception to the "no
    // post-transpose patches" rule above. The symmetric skinny-M case
    // (matmul M=1, swap_ab_=true) falls out through problem.transpose()
    // without explicit fixup since the default A.layout=T after the math
    // transpose maps to the same final state.
    if (!swap_ab_ && N_ == 1
            && get_trans(k_wei_md) == transpose::notrans) {
        problem.B.layout = MatrixLayout::N;
        dim_t pad_ldb_skinny = utils::rnd_up(ldb_matmul, 16);
        int align_b_skinny = utils::max_pow2_div(
                types::elements_to_bytes(wei_t, pad_ldb_skinny));
        for (int b = 0; b < batch_dims(); b++) {
            int stride_bytes_b = utils::max_pow2_div(types::elements_to_bytes(
                    wei_t, stride_for(k_wei_md, b)));
            if (stride_bytes_b)
                align_b_skinny = nstl::min(align_b_skinny, stride_bytes_b);
        }
        align_b_skinny
                = nstl::max(align_b_skinny, (int)types::data_type_size(wei_t));
        problem.B.setAlignment(align_b_skinny);
    }

    if (problem.nativeBDPAS()) {
        if (((!problem.Ta.isF4() || !problem.Tb.isF4()) || K_ % 64 == 0))
            problem.bdpasEnabled = true;
    }

    maybe_dump_gemm_problem(problem);

    return status::success;
}

dim_t jit_gemm_pd_t::ld_binary(int idx) const {
    switch (kernel_input_.binary_srcs[idx].type) {
        case binary_src_t::binary: {
            const auto &entry = kernel_input_.post_ops.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return get_ld(entry.binary.src1_desc);
        }
        case binary_src_t::bias: return ld_bias();
        case binary_src_t::prelu: return get_ld(kernel_input_.prelu_wei_md);
        default: return 1;
    }
}

dim_t jit_gemm_pd_t::stride_binary(int idx, int stride) const {
    switch (kernel_input_.binary_srcs[idx].type) {
        case binary_src_t::binary:
        case binary_src_t::scales:
        case binary_src_t::bias: {
            const auto &entry = kernel_input_.post_ops.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return get_stride(entry.binary.src1_desc, stride);
        }
        case binary_src_t::prelu:
            return get_stride(kernel_input_.prelu_wei_md, stride);
        default: return 0;
    }
}

dim_t jit_gemm_pd_t::scale_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, kA, kB));
    const memory_desc_t *md_ptr = (arg == kA) ? &kernel_input_.src_scale_md
                                              : &kernel_input_.wei_scale_md;
    gpu_assert(memory_desc_wrapper(md_ptr).is_plain())
            << "Expected plain scale_md";
    if (md_ptr->dims[idx] == 1) return 0;
    return md_ptr->format_desc.blocking.strides[idx];
}

dim_t jit_gemm_pd_t::zp_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, kA, kB));
    const memory_desc_t *md_ptr = (arg == kA) ? &kernel_input_.src_zp_md
                                              : &kernel_input_.wei_zp_md;
    gpu_assert(memory_desc_wrapper(md_ptr).is_plain())
            << "Expected plain zp_md";
    if (md_ptr->dims[idx] == 1) return 0;
    return md_ptr->format_desc.blocking.strides[idx];
}

dim_t jit_gemm_pd_t::gs_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, kA, kB));
    const memory_desc_t *md_ptr = (arg == kA) ? &kernel_input_.src_gs_md
                                              : &kernel_input_.wei_gs_md;
    gpu_assert(memory_desc_wrapper(md_ptr).is_plain())
            << "Expected plain gs_md";
    if (md_ptr->dims[idx] == 1) return 0;
    return md_ptr->format_desc.blocking.strides[idx];
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
