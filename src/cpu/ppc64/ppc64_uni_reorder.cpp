/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#include <assert.h>
#include <numeric>
#include <vector>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ppc64/ppc64_reorder_kernel.hpp"
#include "cpu/ppc64/ppc64_uni_reorder.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include <altivec.h>

#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) >= 5) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
constexpr static bool is_windows = true;
#else
constexpr static bool is_windows = false;
#endif

using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {

namespace tr {

static bool prb_has_small_strides(const prb_t &prb) {
    constexpr ptrdiff_t max_stride = (1LL << 31) - 1;
    for (int d = 0; d < prb.ndims; ++d) {
        const ptrdiff_t cms = max_stride / prb.nodes[d].n;
        const bool small_strides = true
                && prb.nodes[d].is < cms / (int)data_type_size(prb.itype)
                && prb.nodes[d].os < cms / (int)data_type_size(prb.otype);
        if (!small_strides) return false;
    }
    return true;
}

bool prb_has_huge_prime_number(const prb_t &prb) {
    for (int d = 0; d < prb.ndims; ++d) {
        auto n = prb.nodes[d].n;
        if (n >= INT_MAX && math::is_prime(n)) return true;
    }
    return false;
}

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims) return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0) ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;

        if (reorder_s8_s8_kernel_t::applicable(desc.prb))
            return status::success;
        //if (reorder_f32_u8_kernel_t::applicable(desc.prb))
        // return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
        case 0:
            if (desc.prb.itype == data_type::s8
                    && desc.prb.otype == data_type::s8)
                return new reorder_s8_s8_kernel_t(desc);
            //if (desc.prb.itype == data_type::f32 && desc.prb.otype == data_type::u8)
            //  return new reorder_f32_u8_kernel_t(desc);
        default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

} // namespace tr

static void prb_block_for_cache(tr::prb_t &prb) {
    /* If strides for 0th and 1st nodes are cache friendly
     * then one can altogether do away with blocking ! */
    static constexpr int num_elems_thr = 16;
    const bool stride_cache_friendly
            = ((prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > num_elems_thr)
                      || (prb.ndims > 1 && prb.nodes[1].is % num_elems_thr == 0
                              && prb.nodes[1].n > num_elems_thr))
            && !prb.is_tail_present;

    // performance improvement for shapes with large inner-most dimension
    const size_t L1_cache_sz
            = size_t(3) * platform::get_per_core_cache_size(1) / 4;
    const size_t itype_sz_ = data_type_size(prb.itype);
    const size_t inner_block_sz = prb.nodes[0].n * itype_sz_;
    const bool requires_inner_blocking = inner_block_sz > L1_cache_sz
            // 'is_tail_present' is not supported for cache_blocking when
            // asymmetric_comp is executed.
            && IMPLICATION(prb.req_asymmetric_comp, !prb.is_tail_present);

    const bool cache_blocking_needed
            = stride_cache_friendly || requires_inner_blocking;
    if (!cache_blocking_needed) return;

    int unit_input_stride_idx = -1;
    for (auto idx = 0; idx < prb.ndims; ++idx) {
        if (prb.nodes[idx].is == 1) unit_input_stride_idx = idx;
    }

    /* Re-prioritize the sequential read over sequential write:
     *                             /-> [n0:is0:1][16n1:1:osk]...
     * [n0:is0:1]...[nk:1:osk] -->     or
     *                             \-> [16n1:1:osk][n0:is0:1]... */
    if (unit_input_stride_idx != -1) {
        const auto output_stride = prb.nodes[unit_input_stride_idx].os;
        const auto num_elems = prb.nodes[unit_input_stride_idx].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed)
            prb_node_split(prb, unit_input_stride_idx, num_elems_thr);

        /* Because of cache-unfriendly nature of unit-output stride node, let
         * us move unit-input stride node on or near front! */
        if (unit_input_stride_idx != move_location)
            prb_node_move(prb, unit_input_stride_idx, move_location);
    }

    /* Potentially, split the node with os=1 in two and pull in the node with
     * is=1 between them for better cache reuse:
     * [n0:is0:1][n1:1:os1] --> [16n0:is0:1][n1:1:os1][n0/16:is0*16:16] */
    if (prb.ndims >= 2 && prb.nodes[0].os == 1 && prb.nodes[1].is == 1) {
        const auto num_elems = prb.nodes[0].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        if (split_needed) {
            prb_node_split(prb, 0, num_elems_thr);
            prb_node_move(prb, 1, 2);

            // Update node information
            prb_node_dependency(prb);

            // heuristics - looping over the unrolled dims should maximize reuse
            // of the already cached data; observation is choosing the smallest
            // dim from the remaining (from 2 up to ndims) gives good results
            constexpr int new_position = 2;
            const auto dim_beg_it = std::begin(prb.nodes);
            const auto dim_two_it = dim_beg_it + new_position;
            const auto dim_last_it = dim_beg_it + prb.ndims;
            const auto min_n_node_it = std::min_element(dim_two_it, dim_last_it,
                    [](const tr::node_t &lhs, const tr::node_t &rhs) {
                        return lhs.n < rhs.n;
                    });
            const auto min_idx = std::distance(dim_beg_it, min_n_node_it);
            // check if min_idx node is parent of node with tail processing which
            // is currently unsupported (i.e. tail processing can only be handled
            // at the inner-most dimension)
            bool inner_block_has_tail = false;
            for (int idx = min_idx - 1; idx >= new_position; idx--) {
                if (prb.nodes[idx].parent_node_id == min_idx) {
                    inner_block_has_tail = true;
                    break;
                }
            }

            if (min_idx > new_position && (!inner_block_has_tail))
                prb_node_move(prb, min_idx, new_position);
        }
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(
        tr::prb_t &prb, int &ndims_ker_max, int nthr) {
    size_t size_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        size_total *= prb.nodes[d].n;

    // The general expression for size_drv_thr can be written as
    // size_drv_min = C0 + FC * (nthr > 1 ? 1 : 0) + VC * (nthr - 1)
    // where FC and VC are fixed and variable costs respectively.
    // Though for now, the below heuristic seems to be good enough
    const size_t size_drv_thr = (nthr > 1) ? 16 * nthr : 1;

    /* size_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t size_drv_min
            = nstl::min<size_t>(size_drv_thr, utils::div_up(size_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * size_ker_cur -- product of the dimension processed by a kernel
     * size_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t size_drv_cur = 1;
    for (; kdims > 1 && size_drv_cur < size_drv_min; --kdims)
        size_drv_cur *= prb.nodes[kdims - 1].n;

    size_t size_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        size_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that size_drv_cur >= size_drv_min.
     *
     * It might happen that for chosen kdims the size_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase size_ker_cur. */
    const bool want_borrow_ker_from_drv = kdims < prb.ndims
            && size_ker_cur < tr::ker_prb_size_min
            && size_drv_cur > size_drv_min;
    if (want_borrow_ker_from_drv) {
        /* size_want_borrow is the minimal size, so that:
         *  o) size_ker_cur * size_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     size_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal size_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t size_want_borrow
                = utils::div_up(tr::ker_prb_size_min, size_ker_cur);
        for (; prb.nodes[kdims].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, size_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the size_drv_cur is too small (less than size_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * size_drv_cur. */
    const bool want_borrow_drv_from_ker = size_ker_cur > tr::ker_prb_size_min
            && size_drv_cur < size_drv_min;

    VDEBUGINFO(5, primitive, reorder,
            "size_drv_thr=%zu size_drv_min=%zu size_drv_cur=%zu "
            "tr::ker_prb_size_min=%zu want_borrow_ker_from_drv=%d "
            "want_borrow_drv_from_ker=%d",
            size_drv_thr, size_drv_min, size_drv_cur, tr::ker_prb_size_min,
            want_borrow_ker_from_drv, want_borrow_drv_from_ker);

    if (want_borrow_drv_from_ker) {
        size_t size_want_borrow = utils::div_up(size_drv_min, size_drv_cur);
        for (; prb.nodes[kdims - 1].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / size_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            verbose_printf(
                    verbose_t::debuginfo, "split: %s\n", prb_dump(prb).c_str());
            verbose_printf(verbose_t::debuginfo, "ndims_ker_max = %d\n",
                    ndims_ker_max);
        });
    }
}

status_t ppc64_uni_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    CHECK(init_scratchpad());

    return status::success;
}

status_t ppc64_uni_reorder_t::pd_t::init_scratchpad() {
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

status_t ppc64_uni_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    VDISPATCH_REORDER_IC(impl::is_dense_format_kind({src_md, dst_md}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;

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
    VDISPATCH_REORDER_IC(ndims_driver <= ppc64_uni_reorder_t::ndims_driver_max,
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

void ppc64_uni_reorder_t::omp_driver_0d(int off, const char *in, char *out,
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

void ppc64_uni_reorder_t::omp_driver_1d(int ithr, int nthr, int off,
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

void ppc64_uni_reorder_t::omp_driver_2d(int ithr, int nthr, int off,
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
void ppc64_uni_reorder_t::omp_driver_3d(int ithr, int nthr, int off,
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

void ppc64_uni_reorder_t::omp_driver_4d(int ithr, int nthr, int off,
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

void ppc64_uni_reorder_t::omp_driver(const char *in, char *out,
        const float *src_scales, const float *dst_scales, int src_zp,
        int dst_zp, const memory_tracking::grantor_t &scratchpad) const {
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
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const bool req_compensation = req_s8s8_comp || req_asymmetric_comp;
    assert(ndims - ndims_ker <= ndims_driver_max);

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

void ppc64_uni_reorder_t::reduce_compensation(char *out,
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

void ppc64_uni_reorder_t::fill_curr_data_chunks(const tr::prb_t &prb,
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

status_t ppc64_uni_reorder_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
    return kernel_->create_kernel();
}

status_t ppc64_uni_reorder_t::execute(const exec_ctx_t &ctx) const {
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

    auto src_zp = src_zero_points ? src_zero_points[0] : 0;
    auto dst_zp = dst_zero_points ? dst_zero_points[0] : 0;
    omp_driver(in, out, src_scales, dst_scales, src_zp, dst_zp, scratchpad);

    return status::success;
}

} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl
