/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/rv64/reorder/jit_uni_reorder.hpp"

#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) >= 5) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace dnnl::impl::types;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace tr;

namespace {

bool is_byte_data_type(data_type_t dt) {
    return utils::one_of(dt, data_type::s8, data_type::u8);
}

bool is_default_byte_reorder(const prb_t &p) {
    return is_byte_data_type(p.itype) && is_byte_data_type(p.otype)
            && p.src_scale_type == scale_type_t::NONE
            && p.dst_scale_type == scale_type_t::NONE && p.ioff == 0
            && p.ooff == 0 && p.beta == 0.f && !p.req_src_zp && !p.req_dst_zp
            && !p.req_s8s8_comp && !p.req_asymmetric_comp
            && p.scale_adjust == 1.f && prb_has_small_strides(p);
}

bool prb_is_byte_plain_blocked_16c_reorder(
        const prb_t &p, plain_blocked_reorder_desc_t *desc) {
    if (p.ndims < 2 || !is_default_byte_reorder(p)) return false;

    plain_blocked_reorder_desc_t candidates[2];
    int ncandidates = 0;

    auto try_nodes = [&](int block_node_id, int inner_node_id,
                             plain_blocked_reorder_desc_t &candidate) {
        const auto &block_node = p.nodes[block_node_id];
        const auto &inner_node = p.nodes[inner_node_id];
        if (block_node.n != 16 || block_node.tail_size > block_node.n)
            return false;

        const bool plain_to_blocked = block_node.os == 1 && inner_node.is == 1
                && block_node.is == static_cast<ptrdiff_t>(inner_node.n)
                && inner_node.os == static_cast<ptrdiff_t>(block_node.n);
        const bool blocked_to_plain = block_node.is == 1 && inner_node.os == 1
                && block_node.os == static_cast<ptrdiff_t>(inner_node.n)
                && inner_node.is == static_cast<ptrdiff_t>(block_node.n);
        if (!plain_to_blocked && !blocked_to_plain) return false;

        for (int d = 0; d < p.ndims; ++d) {
            if (p.nodes[d].tail_size > 0 && d != block_node_id) return false;
        }

        if (!block_node.is_dim_id_empty()) {
            int same_dim_nodes = 0;
            for (int d = 0; d < p.ndims; ++d)
                if (p.nodes[d].dim_id == block_node.dim_id) ++same_dim_nodes;
            if (same_dim_nodes > 2) return false;
        }

        candidate.block_node_id = block_node_id;
        candidate.inner_node_id = inner_node_id;
        candidate.block = block_node.n;
        candidate.plain_to_blocked = plain_to_blocked;
        return true;
    };

    plain_blocked_reorder_desc_t candidate;
    if (try_nodes(0, 1, candidate)) candidates[ncandidates++] = candidate;
    if (try_nodes(1, 0, candidate)) candidates[ncandidates++] = candidate;
    if (ncandidates == 0) return false;

    int selected = -1;
    for (int i = 0; i < ncandidates; ++i) {
        if (p.nodes[candidates[i].block_node_id].dim_id == 1) {
            selected = i;
            break;
        }
    }
    if (selected == -1 && ncandidates == 1
            && p.nodes[candidates[0].block_node_id].is_dim_id_empty())
        selected = 0;
    if (selected == -1) return false;

    if (desc) *desc = candidates[selected];
    return true;
}

bool is_heavy_tail_byte_plain_blocked_16c_reorder(
        const prb_t &p, const memory_desc_t *src_md) {
    plain_blocked_reorder_desc_t desc;
    if (!prb_is_byte_plain_blocked_16c_reorder(p, &desc)) return false;
    if (!desc.plain_to_blocked) return false;

    const auto &block_node = p.nodes[desc.block_node_id];
    const dim_t tail = static_cast<dim_t>(block_node.tail_size);
    if (tail == 0) return false;

    const int channel_dim = block_node.dim_id;
    if (channel_dim < 0) return tail <= static_cast<dim_t>(desc.block / 4);

    const memory_desc_wrapper src_d(src_md);
    if (channel_dim >= src_d.ndims()) return false;
    const dim_t channels = src_d.dims()[channel_dim];
    if (channels <= 0) return false;

    dim_t logical_elems = 1;
    for (int d = 0; d < src_d.ndims(); ++d)
        logical_elems *= src_d.dims()[d];

    const dim_t padded_channels
            = utils::rnd_up(channels, static_cast<dim_t>(desc.block));
    const dim_t padded_elems = logical_elems / channels * padded_channels;

    return tail <= static_cast<dim_t>(desc.block / 4)
            || 2 * logical_elems <= padded_elems;
}

} // namespace

status_t jit_uni_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    CHECK(init_scratchpad());

    return status::success;
}

status_t jit_uni_reorder_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();

    const bool compensation_needed
            = prb_.req_s8s8_comp || prb_.req_asymmetric_comp;
    if (compensation_needed) {
        const memory_desc_wrapper od(dst_md());
        const auto G = with_groups_ ? od.padded_dims()[0] : 1;
        const auto N = od.padded_dims()[with_groups_ ? 1 : 0];
        static constexpr int cache_line_size = 16;
        const auto wspace_per_thr_size
                = utils::rnd_up(G * N, cache_line_size) * sizeof(int32_t);

        const auto compensation_reduce_size = wspace_per_thr_size * nthr_;

        // Every thread gets its own scratchpad space for each N.
        scratchpad.template book<int32_t>(
                memory_tracking::names::key_reorder_space,
                compensation_reduce_size);
    }

    if (!attr()->scales_.has_default_values(DNNL_ARG_DST)) {
        const memory_desc_wrapper input_d(src_md());
        int mask = attr()->scales_.get_mask(DNNL_ARG_DST);
        get_D_values(input_d, mask, nullptr, &D_mask_, nullptr);
        // Every thread must handle scales inside a parallel task.
        const auto dst_scales_scratch_size = D_mask_ * nthr_;
        scratchpad.template book<float>(
                memory_tracking::names::key_reorder_precomputed_dst_scales,
                dst_scales_scratch_size);
    }

    return status::success;
}

status_t jit_uni_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        const engine_t *engine, const primitive_attr_t *attr,
        const engine_t *src_engine, const memory_desc_t *src_md,
        const engine_t *dst_engine, const memory_desc_t *dst_md) {
    VDISPATCH_REORDER_IC(impl::is_dense_format_kind({src_md, dst_md}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const auto &zp = attr->zero_points_;
    const auto scalar_or_default_zp = [&](int arg) {
        return zp.has_default_values(arg) || zp.get_mask(arg) == 0;
    };
    VDISPATCH_REORDER_IC(scalar_or_default_zp(DNNL_ARG_SRC)
                    && scalar_or_default_zp(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;

    if (tr::prb_is_f32_default_plain_blocked_reorder(prb))
        return status::unimplemented;

    if (is_heavy_tail_byte_plain_blocked_16c_reorder(prb, src_md))
        return status::unimplemented;

    // A huge-prime dimension cannot be split for cache/thread blocking and would
    // stall prb_thread_kernel_balance's linear factor search, so bail out to the
    // reference reorder before that runs.
    if (prb_has_huge_prime_number(prb)) return status::unimplemented;

    prb_block_for_cache(prb);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "cache: %s\n", prb_dump(prb).c_str());
    });

    int ndims_ker_max {};
    int nthr = dnnl_get_max_threads();
    prb_thread_kernel_balance(prb, ndims_ker_max, nthr);

    if (prb.is_tail_present) prb_node_dependency(prb);

    tr::kernel_t::desc_t ker_desc;
    status_t ker_init_status
            = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
    if (ker_init_status != status::success) return ker_init_status;

    const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
    VDISPATCH_REORDER_IC(ndims_driver <= jit_uni_reorder_t::ndims_driver_max,
            VERBOSE_BAD_NDIMS, "driver", ndims_driver);

    DEBUG({
        verbose_printf(verbose_t::debuginfo, "ker  : %s\n",
                prb_dump(ker_desc.prb).c_str());
    });

    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;

    _pd->nthr_ = nthr;
    _pd->prb_ = prb;
    _pd->with_groups_
            = prb.compensation_mask == tr::prb_t::comp_mask_with_groups;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    _pd->ker_desc_ = ker_desc;
    CHECK(_pd->init_scratchpad_md());

    return safe_ptr_assign(*reorder_pd, _pd.release());
}

void jit_uni_reorder_t::omp_driver_0d(int off, const char *in, char *out,
        const void *src_scales, const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;

    tr::call_param_t base_params;
    base_params.in = in;
    base_params.out = out;
    if (prb.src_scale_type != tr::scale_type_t::NONE)
        base_params.src_scales = src_scales;
    if (prb.dst_scale_type != tr::scale_type_t::NONE)
        base_params.dst_scales = dst_scales;
    base_params.src_zp = src_zp;
    base_params.dst_zp = dst_zp;
    base_params.compensation_scratch = compensation_scratch;

    if (prb.is_tail_present) {
        tr::tail_call_param_t tail_params;
        tail_params.base_params = base_params;

        static constexpr int omp_ndims = 0;
        fill_curr_data_chunks(prb, off, nullptr, omp_ndims, tail_params);

        (*kernel_)(&tail_params);
    } else {
        (*kernel_)(&base_params);
    }
}

void jit_uni_reorder_t::omp_driver_1d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
        tr::call_param_t base_params;
        base_params.in = in + d0 * ns[0].is * data_type_size(prb.itype);
        base_params.out = out + d0 * ns[0].os * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales
                    = static_cast<const float *>(src_scales) + d0 * ns[0].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales
                    = static_cast<const float *>(dst_scales) + d0 * ns[0].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 1;
            const ptrdiff_t omp_data_chunks[omp_ndims] = {d0};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_2d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d1, ptrdiff_t d0) {
        tr::call_param_t base_params;
        base_params.in = in
                + (d0 * ns[0].is + d1 * ns[1].is) * data_type_size(prb.itype);
        base_params.out = out
                + (d0 * ns[0].os + d1 * ns[1].os) * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales = static_cast<const float *>(src_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales = static_cast<const float *>(dst_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch
                = compensation_scratch + d0 * ns[0].cs + d1 * ns[1].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 2;
            const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_3d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
            (ptrdiff_t)ns[0].n, [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
        tr::call_param_t base_params;
        base_params.in = in
                + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                        * data_type_size(prb.itype);
        base_params.out = out
                + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                        * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales = static_cast<const float *>(src_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales = static_cast<const float *>(dst_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs
                + d1 * ns[1].cs + d2 * ns[2].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 3;
            const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1, d2};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_4d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
            (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
        tr::call_param_t base_params;
        base_params.in = in
                + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                          + d3 * ns[3].is)
                        * data_type_size(prb.itype);
        base_params.out = out
                + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                          + d3 * ns[3].os)
                        * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales = static_cast<const float *>(src_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                    + d3 * ns[3].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales = static_cast<const float *>(dst_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                    + d3 * ns[3].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs
                + d1 * ns[1].cs + d2 * ns[2].cs + d3 * ns[3].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 4;
            const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1, d2, d3};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::reduce_compensation(char *out,
        const int32_t *compensation_reduce_scratch, const int nthr,
        const dim_t wspace_per_thr_size) const {

    const memory_desc_wrapper od(pd()->dst_md());
    const size_t offset = od.size() - od.additional_buffer_size();

    static constexpr auto comp_dt_size = sizeof(int32_t);
    static constexpr int32_t comp_s8s8_shift = 128;

    // Note: We do not need to explicitly zero-out compensation buffer, as the
    // per_thread buffers are already zeroed out in the padded area.
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    const auto GN = G * N;
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const size_t zp_offset
            = offset + (pd()->prb_.req_s8s8_comp ? GN * comp_dt_size : 0);

    parallel_nd(GN, [=](int idx) {
        int32_t acc = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            acc -= compensation_reduce_scratch[ithr * wspace_per_thr_size
                    + idx];
        }
        if (req_s8s8_comp) {
            int32_t *out_comp = reinterpret_cast<int32_t *>(&out[offset]);
            out_comp[idx] = comp_s8s8_shift * acc;
        }
        if (req_asymmetric_comp) {
            int32_t *out_asym_comp
                    = reinterpret_cast<int32_t *>(&out[zp_offset]);
            out_asym_comp[idx] = acc;
        }
    });
}

void jit_uni_reorder_t::fill_curr_data_chunks(const tr::prb_t &prb,
        const int off, const ptrdiff_t *omp_data_chunks, const int omp_ndims,
        tr::tail_call_param_t &c) const {
    // Chunks are backwards numered i.e:
    // [0] -> [node_size]
    // [1] -> [node_size - 1]
    // ...
    // [node_size - 1] -> [1]

    // It is done like this, because it is easier to decrement counter
    // and check if it is equal to zero than increment and check
    // if it is equal to node_size in jit kernel.

    static constexpr int64_t empty_chunk_info = -1;
    static constexpr int64_t last_chunk = 1;

    for (int curr_node_id = prb.ndims - 1; curr_node_id >= 0; curr_node_id--) {
        const int parent_node_id = prb.nodes[curr_node_id].parent_node_id;
        const bool is_drv_processing_this_node
                = curr_node_id >= off && curr_node_id <= off + omp_ndims - 1;
        const bool is_tail_processing
                = prb.is_tail_in_one_of_child_nodes(curr_node_id)
                || prb.nodes[curr_node_id].tail_size > 0;

        if (is_drv_processing_this_node && is_tail_processing) {
            const int inner_idx = curr_node_id - off;
            assert(inner_idx < omp_ndims);
            const int64_t node_size = prb.nodes[curr_node_id].tail_size > 0
                    ? prb.nodes[curr_node_id].tail_size
                    : prb.nodes[curr_node_id].n;
            const int64_t data_chunk = node_size - omp_data_chunks[inner_idx];

            if (!prb.nodes[curr_node_id].is_parent_empty()) {
                const bool is_parent_chunk_last
                        = c.curr_data_chunks[parent_node_id] == last_chunk;
                c.curr_data_chunks[curr_node_id]
                        = is_parent_chunk_last ? data_chunk : empty_chunk_info;
                c.zeroing_data = static_cast<int64_t>(
                        is_parent_chunk_last && data_chunk <= 0);
            } else {
                c.curr_data_chunks[curr_node_id] = data_chunk;
                c.zeroing_data = static_cast<int64_t>(data_chunk <= 0);
            }
            c.skip_kernel_execution = static_cast<int64_t>(c.zeroing_data
                    && !prb.nodes[curr_node_id].is_zero_pad_needed);
            if (c.zeroing_data || c.skip_kernel_execution) break;
        } else
            c.curr_data_chunks[curr_node_id] = empty_chunk_info;
    }
}

status_t jit_uni_reorder_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
    return kernel_->create_kernel();
}

status_t jit_uni_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
    out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

    DEBUG({
        verbose_printf(verbose_t::debuginfo, "prb  : %s\n",
                tr::prb_dump(pd()->prb_).c_str());
    });
    DEBUG({
        verbose_printf(verbose_t::debuginfo, "ker  : %s\n",
                tr::prb_dump(pd()->ker_desc_.prb).c_str());
    });

    int ndims = pd()->prb_.ndims;
    int ndims_ker = pd()->ker_desc_.prb.ndims;
    int ndims_level = ndims - ndims_ker;

    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const bool req_compensation = req_s8s8_comp || req_asymmetric_comp;
    assert(ndims_level <= ndims_driver_max);

    int32_t *compensation_reduce_scratch = scratchpad.template get<int32_t>(
            memory_tracking::names::key_reorder_space);

    const memory_desc_wrapper od(pd()->dst_md());
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    static constexpr int cache_line_size = 16;
    const auto wspace_per_thr_size = utils::rnd_up(G * N, cache_line_size);
    const auto wspace_per_thr_bytes = wspace_per_thr_size * sizeof(int32_t);

    const int nthr_par = ndims_level == 0 ? 1 : pd()->nthr_;
    parallel(nthr_par, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        int32_t *compensation_scratch = nullptr;
        if (req_compensation) {
            if (ndims_level == 0)
                compensation_scratch = compensation_reduce_scratch;
            else
                compensation_scratch = &compensation_reduce_scratch[ithr
                        * wspace_per_thr_size];
            std::memset(compensation_scratch, 0, wspace_per_thr_bytes);
        }

        float *dst_scales_inv_ptr = nullptr;
        if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_DST)) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            const auto dst_scales_scratch_size_ithr = pd()->D_mask_;
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(memory_tracking::names::
                                      key_reorder_precomputed_dst_scales)
                    + ithr * dst_scales_scratch_size_ithr;
            for (dim_t i = 0; i < dst_scales_scratch_size_ithr; i++) {
                dst_scales_inv_ptr[i] = 1.f / dst_scales_ptr[i];
            }
        }

        auto src_zp = src_zero_points ? src_zero_points[0] : 0;
        auto dst_zp = dst_zero_points ? dst_zero_points[0] : 0;

        switch (ndims_level) {
            case 0:
                omp_driver_0d(ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 1:
                omp_driver_1d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 2:
                omp_driver_2d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 3:
                omp_driver_3d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 4:
                omp_driver_4d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            default: assert(!"unimplemented");
        }
    });

    //reduction of intermediate compensation results to the final output
    if (req_compensation) {
        reduce_compensation(out, compensation_reduce_scratch, nthr_par,
                wspace_per_thr_size);
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
