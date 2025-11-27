/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/aarch64/reorder/jit_uni_reorder.hpp"

// #define DNNL_DEV_MODE
#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace Xbyak_aarch64;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t jit_uni_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
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
        if (D_mask_ > 1) {
            scratchpad.template book<float>(
                    memory_tracking::names::key_reorder_precomputed_dst_scales,
                    D_mask_);
        }
    }

    return status::success;
}

status_t jit_uni_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    if (!impl::is_dense_format_kind({src_md, dst_md}))
        return status::unimplemented;
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;

    tr::prb_block_for_cache(prb);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "cache: %s\n", prb_dump(prb).c_str());
    });

    int ndims_ker_max {};
    int nthr = dnnl_get_max_threads();
    tr::prb_thread_kernel_balance(prb, ndims_ker_max, nthr);

    if (prb.is_tail_present) prb_node_dependency(prb);

    tr::kernel_t::desc_t ker_desc;
    status_t ker_init_status
            = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
    if (ker_init_status != status::success) return ker_init_status;

    const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
    if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
        return status::unimplemented;

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
        const float *src_scales, const float *dst_scales, int src_zp,
        int dst_zp, int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;

    tr::call_param_t base_params;
    base_params.in = in;
    base_params.out = out;
    base_params.src_scales = src_scales;
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
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
        tr::call_param_t base_params;
        base_params.in = in + d0 * ns[0].is * data_type_size(prb.itype);
        base_params.out = out + d0 * ns[0].os * data_type_size(prb.otype);
        base_params.src_scales = src_scales + d0 * ns[0].ss;
        base_params.dst_scales = dst_scales + d0 * ns[0].ss;
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
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d1, ptrdiff_t d0) {
                tr::call_param_t base_params;
                base_params.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is)
                                * data_type_size(prb.itype);
                base_params.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os)
                                * data_type_size(prb.otype);
                base_params.src_scales
                        = src_scales + d0 * ns[0].ss + d1 * ns[1].ss;
                base_params.dst_scales
                        = dst_scales + d0 * ns[0].ss + d1 * ns[1].ss;
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
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
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
                base_params.src_scales = src_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss;
                base_params.dst_scales = dst_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss;
                base_params.src_zp = src_zp;
                base_params.dst_zp = dst_zp;
                base_params.compensation_scratch = compensation_scratch
                        + d0 * ns[0].cs + d1 * ns[1].cs + d2 * ns[2].cs;

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
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
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
                base_params.src_scales = src_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss + d3 * ns[3].ss;
                base_params.dst_scales = dst_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss + d3 * ns[3].ss;
                base_params.src_zp = src_zp;
                base_params.dst_zp = dst_zp;
                base_params.compensation_scratch = compensation_scratch
                        + d0 * ns[0].cs + d1 * ns[1].cs + d2 * ns[2].cs
                        + d3 * ns[3].cs;

                if (prb.is_tail_present) {
                    tr::tail_call_param_t tail_params;
                    tail_params.base_params = base_params;

                    static constexpr int omp_ndims = 4;
                    const ptrdiff_t omp_data_chunks[omp_ndims]
                            = {d0, d1, d2, d3};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, tail_params);

                    (*kernel_)(&tail_params);
                } else {
                    (*kernel_)(&base_params);
                }
            });
}

void jit_uni_reorder_t::omp_driver(const char *in, char *out,
        const float *src_scales, const float *dst_scales,
        const int32_t *src_zero_points, const int32_t *dst_zero_points,
        const memory_tracking::grantor_t &scratchpad) const {
    in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
    out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

    DEBUG({
        verbose_printf(verbose_t::debuginfo, "prb : %s\n",
                tr::prb_dump(pd()->prb_).c_str());
    });
    DEBUG({
        verbose_printf(verbose_t::debuginfo, "ker : %s\n",
                tr::prb_dump(pd()->ker_desc_.prb).c_str());
    });

    int ndims = pd()->prb_.ndims;
    int ndims_ker = pd()->ker_desc_.prb.ndims;
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const bool req_compensation = req_s8s8_comp || req_asymmetric_comp;
    assert(ndims - ndims_ker <= ndims_driver_max);

    auto src_zp = src_zero_points ? src_zero_points[0] : 0;
    auto dst_zp = dst_zero_points ? dst_zero_points[0] : 0;
    int32_t *compensation_reduce_scratch = scratchpad.template get<int32_t>(
            memory_tracking::names::key_reorder_space);

    const memory_desc_wrapper od(pd()->dst_md());
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    static constexpr int cache_line_size = 16;
    const auto wspace_per_thr_size = utils::rnd_up(G * N, cache_line_size);
    const auto wspace_per_thr_bytes = wspace_per_thr_size * sizeof(int32_t);

    if (ndims - ndims_ker == 0) {
        if (req_compensation)
            std::memset(compensation_reduce_scratch, 0, wspace_per_thr_bytes);

        omp_driver_0d(ndims_ker, in, out, src_scales, dst_scales, src_zp,
                dst_zp, compensation_reduce_scratch);
    } else {
        parallel(pd()->nthr_, [&](const int ithr, const int nthr) {
            int32_t *compensation_scratch = nullptr;
            if (req_compensation) {
                compensation_scratch = &compensation_reduce_scratch[ithr
                        * wspace_per_thr_size];
                std::memset(compensation_scratch, 0, wspace_per_thr_bytes);
            }

            switch (ndims - ndims_ker) {
                case 1:
                    omp_driver_1d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                case 2:
                    omp_driver_2d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                case 3:
                    omp_driver_3d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                case 4:
                    omp_driver_4d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                default: assert(!"unimplemented");
            }
        });
    }

    //reduction of intermediate compensation results to the final output
    if (req_compensation) {
        const int nthr = ndims - ndims_ker == 0 ? 1 : pd()->nthr_;
        reduce_compensation(
                out, compensation_reduce_scratch, nthr, wspace_per_thr_size);
    }
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

    parallel_nd(GN, [&](int idx) {
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
    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales_, DNNL_ARG_DST);

    const float *dst_scales = pd()->precompute_scales(
            scratchpad, pd()->attr(), pd()->D_mask_, dst_scales_);
    assert(dst_scales);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    omp_driver(in, out, src_scales, dst_scales, src_zero_points,
            dst_zero_points, scratchpad);

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
