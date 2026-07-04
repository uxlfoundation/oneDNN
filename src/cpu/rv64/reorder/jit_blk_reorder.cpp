/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
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
#include <climits>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/reorder/jit_blk_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace tr;

namespace {

dim_t saturated_mul(dim_t a, dim_t b) {
    if (a == 0 || b == 0) return 0;
    constexpr dim_t max_dim = (dim_t)LLONG_MAX;
    if (a > max_dim / b) return max_dim;
    return a * b;
}

bool is_transpose_16c_profitable(
        const memory_desc_t &src_md, const plain_blocked_reorder_desc_t &desc) {
    if (desc.block != 16) return true;

    const memory_desc_wrapper src_d(&src_md);
    if (src_d.ndims() < 2) return true;

    const auto *dims = src_d.dims();
    const dim_t mb = dims[0];
    const dim_t channels = dims[1];
    dim_t spatial = 1;
    for (int d = 2; d < src_d.ndims(); ++d)
        spatial = saturated_mul(spatial, dims[d]);

    const dim_t padded_channels = utils::rnd_up(channels, (dim_t)desc.block);
    const dim_t logical_elems
            = saturated_mul(saturated_mul(mb, channels), spatial);
    const dim_t padded_elems
            = saturated_mul(saturated_mul(mb, padded_channels), spatial);
    const dim_t tail = channels % (dim_t)desc.block;
    const bool has_tail = tail != 0;

    // The current RVV 16c transpose is not cheap enough for tiny problems.
    // Let simple:any handle them until the micro-kernel itself is improved.
    if (padded_elems < 64 * 1024) return false;

    // Real RV data shows a middle-size band where the fixed transpose work
    // dominates: it is too large for call overhead to be the bottleneck, but
    // not large enough for the contiguous stores to win decisively.
    if (spatial <= 4096 && padded_elems >= 2 * 1024 * 1024
            && padded_elems <= 32 * 1024 * 1024)
        return false;

    // Padding-heavy tails at large batch counts write many zero lanes and are
    // better left to the generic simple implementation for now.
    const bool padding_heavy = channels < (dim_t)desc.block
            || tail <= (dim_t)desc.block / 4
            || 2 * logical_elems <= padded_elems;
    if (has_tail && padding_heavy && mb >= 64) return false;

    return true;
}

} // namespace

/* ----------------------------- transpose kernel ----------------------------- */

bool jit_single_blk_kernel_t::applicable(const prb_t &p) {
    plain_blocked_reorder_desc_t desc;
    return mayiuse(v) && prb_is_f32_default_plain_blocked_reorder(p, &desc)
            && desc.block_node_id == 0 && desc.inner_node_id == 1
            && utils::one_of(desc.block, 4ul, 8ul, 16ul, 32ul);
}

jit_single_blk_kernel_t::kernel_kind_t
jit_single_blk_kernel_t::select_kernel_kind(const prb_t &prb) {
    const auto block = prb.nodes[0].n;
    assert(utils::one_of(block, 4ul, 8ul, 16ul, 32ul));
    return block <= 8 ? kernel_kind_t::segment_4c8c
                      : kernel_kind_t::transpose_16c32c;
}

jit_single_blk_kernel_t::jit_single_blk_kernel_t(const prb_t &prb)
    : jit_generator_t("jit_single_blk_kernel")
    , prb_(prb)
    , itype_sz_((int)types::data_type_size(prb.itype))
    , otype_sz_((int)types::data_type_size(prb.otype))
    , block_sz_((int)(prb.nodes[0].n == 32 ? 16 : prb.nodes[0].n))
    , tile_cols_(get_platform_vlen() >= 256 ? 8 : 4)
    , plain_to_blocked_(false)
    , kernel_kind_(select_kernel_kind(prb)) {
    plain_blocked_reorder_desc_t desc;
    const bool ok = prb_is_f32_default_plain_blocked_reorder(prb_, &desc);
    MAYBE_UNUSED(ok);
    assert(ok && desc.block_node_id == 0 && desc.inner_node_id == 1);
    plain_to_blocked_ = desc.plain_to_blocked;
}

void jit_single_blk_kernel_t::generate() {
    if (kernel_kind_ == kernel_kind_t::segment_4c8c)
        emit_segment_kernel();
    else
        emit_transpose_kernel();
}

void jit_single_blk_kernel_t::emit_segment_kernel() {
    using namespace Xbyak_riscv;

    assert(utils::one_of(block_sz_, 4, 8));

    const uint32_t inner_is_bytes = (uint32_t)(prb_.nodes[1].is * itype_sz_);
    const uint32_t inner_os_bytes = (uint32_t)(prb_.nodes[1].os * otype_sz_);
    const uint32_t block_is_bytes = (uint32_t)(prb_.nodes[0].is * itype_sz_);
    const uint32_t block_os_bytes = (uint32_t)(prb_.nodes[0].os * otype_sz_);
    const uint32_t blocked_step_bytes = (uint32_t)(block_sz_ * sizeof(float));

    const VReg v_base(8);
    const int max_seg = 8;

    auto add_byte_offset
            = [this](const Reg &dst, const Reg &base, uint32_t byte_offset) {
        if (byte_offset == 0) {
            mv(dst, base);
        } else {
            li(dst, byte_offset);
            add(dst, base, dst);
        }
    };

    auto emit_vlseg = [this, v_base](int nf, const Reg &addr) {
        switch (nf) {
            case 4: vlseg4e32_v(v_base, addr); break;
            case 8: vlseg8e32_v(v_base, addr); break;
            default: assert(!"unsupported segment size");
        }
    };

    auto emit_vlsseg
            = [this, v_base](int nf, const Reg &addr, const Reg &stride) {
        switch (nf) {
            case 4: vlsseg4e32_v(v_base, addr, stride); break;
            case 8: vlsseg8e32_v(v_base, addr, stride); break;
            default: assert(!"unsupported segment size");
        }
    };

    auto emit_vsseg = [this, v_base](int nf, const Reg &addr) {
        switch (nf) {
            case 4: vsseg4e32_v(v_base, addr); break;
            case 8: vsseg8e32_v(v_base, addr); break;
            default: assert(!"unsupported segment size");
        }
    };

    auto emit_vssseg
            = [this, v_base](int nf, const Reg &addr, const Reg &stride) {
        switch (nf) {
            case 4: vssseg4e32_v(v_base, addr, stride); break;
            case 8: vssseg8e32_v(v_base, addr, stride); break;
            default: assert(!"unsupported segment size");
        }
    };

    auto emit_plain_to_blocked = [&](bool full_block) {
        for (int g = 0; g < block_sz_; g += max_seg) {
            const int nf = nstl::min(max_seg, block_sz_ - g);
            for (int f = 0; f < nf; ++f) {
                const int ch = g + f;
                const VReg v(v_base.getIdx() + f);
                if (full_block) {
                    add_byte_offset(t3, t1, ch * block_is_bytes);
                    vle32_v(v, t3);
                } else {
                    Label zero, done;
                    li(t4, (uint32_t)ch);
                    bge(t4, a3, zero);
                    add_byte_offset(t3, t1, ch * block_is_bytes);
                    vle32_v(v, t3);
                    j_(done);
                    L(zero);
                    vmv_v_i(v, 0);
                    L(done);
                }
            }

            add_byte_offset(t3, t2, g * (uint32_t)otype_sz_);
            if (nf == block_sz_)
                emit_vsseg(nf, t3);
            else
                emit_vssseg(nf, t3, a7);
        }
    };

    auto emit_blocked_to_plain = [&](bool full_block) {
        for (int g = 0; g < block_sz_; g += max_seg) {
            const int nf = nstl::min(max_seg, block_sz_ - g);
            add_byte_offset(t3, t1, g * (uint32_t)itype_sz_);
            if (nf == block_sz_)
                emit_vlseg(nf, t3);
            else
                emit_vlsseg(nf, t3, a7);

            for (int f = 0; f < nf; ++f) {
                const int ch = g + f;
                const VReg v(v_base.getIdx() + f);
                if (full_block) {
                    add_byte_offset(t3, t2, ch * block_os_bytes);
                    vse32_v(v, t3);
                } else {
                    Label skip;
                    li(t4, (uint32_t)ch);
                    bge(t4, a3, skip);
                    add_byte_offset(t3, t2, ch * block_os_bytes);
                    vse32_v(v, t3);
                    L(skip);
                }
            }
        }
    };

    auto emit_loop = [&](bool full_block) {
        Label col_loop, col_done;
        mv(t0, a2); // remaining columns
        L(col_loop);
        beqz(t0, col_done);
        vsetvli(t6, t0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        if (plain_to_blocked_)
            emit_plain_to_blocked(full_block);
        else
            emit_blocked_to_plain(full_block);
        mul(t3, t6, t5);
        add(t1, t1, t3);
        mul(t3, t6, a4);
        add(t2, t2, t3);
        sub(t0, t0, t6);
        j_(col_loop);
        L(col_done);
    };

    // a0=in, a1=out, a2=cols, a3=real_block.
    mv(t1, a0);
    mv(t2, a1);
    li(t5, inner_is_bytes);
    li(a4, inner_os_bytes);
    li(a7, blocked_step_bytes);

    Label tail_block, done;
    li(a0, (uint32_t)block_sz_);
    bne(a3, a0, tail_block);
    emit_loop(true);
    j_(done);
    L(tail_block);
    emit_loop(false);
    L(done);

    ret();
}

void jit_single_blk_kernel_t::emit_transpose_kernel() {
    using namespace Xbyak_riscv;

    assert(block_sz_ == 16);
    assert(utils::one_of(tile_cols_, 4, 8));

    const uint32_t inner_is_bytes = (uint32_t)(prb_.nodes[1].is * itype_sz_);
    const uint32_t inner_os_bytes = (uint32_t)(prb_.nodes[1].os * otype_sz_);
    const uint32_t block_is_bytes = (uint32_t)(prb_.nodes[0].is * itype_sz_);
    const uint32_t block_os_bytes = (uint32_t)(prb_.nodes[0].os * otype_sz_);

    const VReg v_mask(0);
    const VReg v_idx(1);
    const VReg v_tmp(2);
    const VReg v_out(3);
    const int v_base_idx = 8;

    auto v_data = [](int idx) { return VReg(v_base_idx + idx); };

    auto add_byte_offset
            = [this](const Reg &dst, const Reg &base, uint32_t byte_offset) {
        if (byte_offset == 0) {
            mv(dst, base);
        } else {
            li(dst, byte_offset);
            add(dst, base, dst);
        }
    };

    auto set_vl_cols
            = [&]() { vsetvli(x0, t6, SEW::e32, LMUL::m1, VTA::ta, VMA::ma); };

    auto set_vl_tile = [&]() {
        li(t4, (uint32_t)tile_cols_);
        vsetvli(x0, t4, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    };

    auto emit_plain_to_blocked_group = [&](int g, bool full_block) {
        set_vl_cols();

        for (int f = 0; f < tile_cols_; ++f) {
            const int ch = g + f;
            const VReg v = v_data(f);
            if (full_block) {
                add_byte_offset(t3, t1, ch * block_is_bytes);
                vle32_v(v, t3);
            } else {
                Label zero, done;
                li(t4, (uint32_t)ch);
                bge(t4, a3, zero);
                add_byte_offset(t3, t1, ch * block_is_bytes);
                vle32_v(v, t3);
                j_(done);
                L(zero);
                vmv_v_i(v, 0);
                L(done);
            }
        }

        set_vl_tile();
        vid_v(v_idx);

        for (int row = 0; row < tile_cols_; ++row) {
            Label skip_row;
            li(t4, (uint32_t)row);
            bge(t4, t6, skip_row);

            vmv_v_i(v_out, 0);
            for (int f = 0; f < tile_cols_; ++f) {
                vrgather_vi(v_tmp, v_data(f), row);
                vmseq_vi(v_mask, v_idx, f);
                vmerge_vvm(v_out, v_out, v_tmp);
            }

            add_byte_offset(
                    t3, t2, row * inner_os_bytes + g * (uint32_t)otype_sz_);
            vse32_v(v_out, t3);
            L(skip_row);
        }
    };

    auto emit_blocked_to_plain_group = [&](int g, bool full_block) {
        set_vl_tile();

        for (int row = 0; row < tile_cols_; ++row) {
            Label skip_load;
            li(t4, (uint32_t)row);
            bge(t4, t6, skip_load);
            add_byte_offset(
                    t3, t1, row * inner_is_bytes + g * (uint32_t)itype_sz_);
            vle32_v(v_data(row), t3);
            L(skip_load);
        }

        for (int f = 0; f < tile_cols_; ++f) {
            const int ch = g + f;
            Label skip_channel;
            if (!full_block) {
                li(t4, (uint32_t)ch);
                bge(t4, a3, skip_channel);
            }

            set_vl_tile();
            vid_v(v_idx);
            vmv_v_i(v_out, 0);
            for (int row = 0; row < tile_cols_; ++row) {
                Label skip_row;
                li(t4, (uint32_t)row);
                bge(t4, t6, skip_row);
                vrgather_vi(v_tmp, v_data(row), f);
                vmseq_vi(v_mask, v_idx, row);
                vmerge_vvm(v_out, v_out, v_tmp);
                L(skip_row);
            }

            set_vl_cols();
            add_byte_offset(t3, t2, ch * block_os_bytes);
            vse32_v(v_out, t3);

            if (!full_block) L(skip_channel);
        }
    };

    auto emit_loop = [&](bool full_block) {
        Label col_loop, col_done, cols_tail, cols_step_ready;
        mv(t0, a2); // remaining columns
        L(col_loop);
        beqz(t0, col_done);

        li(t6, (uint32_t)tile_cols_);
        bge(t6, t0, cols_tail);
        j_(cols_step_ready);
        L(cols_tail);
        mv(t6, t0);
        L(cols_step_ready);
        vsetvli(t6, t6, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

        for (int g = 0; g < block_sz_; g += tile_cols_) {
            if (plain_to_blocked_)
                emit_plain_to_blocked_group(g, full_block);
            else
                emit_blocked_to_plain_group(g, full_block);
        }

        mul(t3, t6, t5);
        add(t1, t1, t3);
        mul(t3, t6, a4);
        add(t2, t2, t3);
        sub(t0, t0, t6);
        j_(col_loop);
        L(col_done);
    };

    // a0=in, a1=out, a2=cols, a3=real_block.
    mv(t1, a0);
    mv(t2, a1);
    li(t5, inner_is_bytes);
    li(a4, inner_os_bytes);

    Label tail_block, done;
    li(t4, (uint32_t)block_sz_);
    bne(a3, t4, tail_block);
    emit_loop(true);
    j_(done);
    L(tail_block);
    emit_loop(false);
    L(done);

    ret();
}

/* ----------------------------- primitive ----------------------------- */

status_t jit_blk_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    VDISPATCH_REORDER_IC(impl::is_dense_format_kind({src_md, dst_md}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;

    prb_tile_normalize(prb);
    prb_node_dependency(prb);

    plain_blocked_reorder_desc_t desc;
    const bool is_plain_blocked
            = tr::prb_is_f32_default_plain_blocked_reorder(prb, &desc);

    if (!tr::jit_single_blk_kernel_t::applicable(prb))
        return status::unimplemented;

    if (!is_plain_blocked || !is_transpose_16c_profitable(*src_md, desc))
        return status::unimplemented;

    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;
    _pd->prb_ = prb;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    CHECK(_pd->init_scratchpad_md());

    return safe_ptr_assign(*reorder_pd, _pd.release());
}

void jit_blk_reorder_t::pd_t::prb_tile_normalize(tr::prb_t &p) {
    plain_blocked_reorder_desc_t desc;
    if (prb_is_f32_default_plain_blocked_reorder(p, &desc)
            && desc.block_node_id == 1) {
        nstl::swap(p.nodes[0], p.nodes[1]);
    }
}

jit_blk_reorder_t::jit_blk_reorder_t(const pd_t *apd) : primitive_t(apd) {}
jit_blk_reorder_t::~jit_blk_reorder_t() = default;

status_t jit_blk_reorder_t::init(engine_t *engine) {
    kernel_ = utils::make_unique<tr::jit_single_blk_kernel_t>(pd()->prb_);
    return kernel_->create_kernel();
}

status_t jit_blk_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    const auto &prb = pd()->prb_;
    dim_t outer_work = 1;
    for (int i = 2; i < prb.ndims; ++i)
        outer_work *= prb.nodes[i].n;

    const dim_t block_sz = (dim_t)prb.n(0);
    const dim_t cols = (dim_t)prb.n(1);
    const auto nthr = (dim_t)dnnl_get_max_threads();
    constexpr dim_t min_col_chunk = 512;
    dim_t col_chunks = 1;
    if (outer_work < nthr && cols > min_col_chunk) {
        const dim_t want_chunks
                = utils::div_up(nthr, nstl::max<dim_t>(outer_work, 1));
        const dim_t max_chunks = utils::div_up(cols, min_col_chunk);
        col_chunks = nstl::min(want_chunks, max_chunks);
    }
    const dim_t col_chunk_size = utils::div_up(cols, col_chunks);
    const int tail_parent = prb.nodes[0].parent_node_id;
    const bool has_block_tail = prb.tail(0) > 0;

    const auto itype_sz_ = types::data_type_size(prb.itype);
    const auto otype_sz_ = types::data_type_size(prb.otype);

    parallel_nd(outer_work, col_chunks,
            [= COMPAT_THIS_CAPTURE](dim_t outer_idx, dim_t col_chunk) {
        dim_t rem = outer_idx;
        ptrdiff_t i_off = 0;
        ptrdiff_t o_off = 0;
        bool is_tail_block
                = has_block_tail && tail_parent == node_t::empty_field;

        for (int d = 2; d < prb.ndims; ++d) {
            const dim_t node_n = (dim_t)prb.n(d);
            const dim_t idx = rem % node_n;
            rem /= node_n;
            i_off += idx * prb.is(d);
            o_off += idx * prb.os(d);
            if (has_block_tail && d == tail_parent && idx == node_n - 1)
                is_tail_block = true;
        }

        const auto col_start = col_chunk * col_chunk_size;
        const auto col_end = nstl::min(cols, col_start + col_chunk_size);
        if (col_start >= col_end) return;

        i_off += col_start * prb.is(1);
        o_off += col_start * prb.os(1);

        const auto real_block
                = is_tail_block ? (dim_t)prb.tail(0) : (dim_t)block_sz;
        auto *i = in + i_off * itype_sz_;
        auto *o = out + o_off * otype_sz_;

        if (block_sz == 32) {
            constexpr dim_t sub_block = 16;
            const auto lower_real = nstl::min(real_block, sub_block);
            const auto upper_real = real_block > sub_block
                    ? real_block - sub_block
                    : (dim_t)0;
            const ptrdiff_t i_sub_off = sub_block * prb.is(0) * itype_sz_;
            const ptrdiff_t o_sub_off = sub_block * prb.os(0) * otype_sz_;
            (*kernel_)(i, o, col_end - col_start, lower_real);
            (*kernel_)(i + i_sub_off, o + o_sub_off, col_end - col_start,
                    upper_real);
        } else {
            (*kernel_)(i, o, col_end - col_start, real_block);
        }
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
