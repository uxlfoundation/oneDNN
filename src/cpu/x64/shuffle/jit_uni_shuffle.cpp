/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/shuffle/jit_uni_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static bool impl_supports_datatype(cpu_isa_t isa, data_type_t data_type) {
    switch (data_type) {
        case data_type::bf16: return is_superset(isa, avx512_core);
        case data_type::f16: return is_superset(isa, avx512_core_fp16);
        case data_type::f32:
        case data_type::s32:
        case data_type::s8:
        case data_type::u8: return true;
        default: return false;
    }
}

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::pd_t::init(engine_t *engine) {
    using namespace format_tag;
    using namespace data_type;

    const memory_desc_wrapper src_d(is_fwd() ? src_md() : diff_src_md());
    const memory_desc_wrapper dst_d(is_fwd() ? dst_md() : diff_dst_md());

    // Disabling verbose dispatch messages for unsupported isa for better
    // readability.
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_SHUFFLE(utils::one_of(src_d.data_type(), f32, s32, bf16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_SHUFFLE(src_d.data_type() == dst_d.data_type(),
            VERBOSE_INCONSISTENT_DT, "src", "dst");
    VDISPATCH_SHUFFLE(impl_supports_datatype(isa, src_d.data_type()),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_SHUFFLE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_SHUFFLE(axis() == 1, VERBOSE_BAD_AXIS);
    VDISPATCH_SHUFFLE(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_SHUFFLE(src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
    VDISPATCH_SHUFFLE(
            impl::is_dense_format_kind({is_fwd() ? src_md() : diff_src_md(),
                    is_fwd() ? dst_md() : diff_dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    conf_.data_type = src_d.data_type();
    conf_.isa = isa;
    if (isa == avx) conf_.isa = mayiuse(avx2) ? avx2 : avx;
    if (conf_.data_type == bf16)
        conf_.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;

    const format_tag_t blocked_format
            = memory_desc_matches_one_of_tag(*src_d.md_, nCw16c, nChw16c,
                    nCdhw16c, nCw8c, nChw8c, nCdhw8c, nCw4c, nChw4c, nCdhw4c);

    VDISPATCH_SHUFFLE(
            blocked_format != format_tag::undef, VERBOSE_UNSUPPORTED_TAG);

    conf_.blk_size = src_d.blocking_desc().strides[ndims() - 1];
    conf_.simd_w = cpu_isa_traits_t<isa>::vlen / sizeof(float);
    VDISPATCH_SHUFFLE(conf_.simd_w <= conf_.blk_size, "simd_w > block_size");

    const bool has_spatial = utils::one_of(ndims(), 3, 4, 5);
    const dim_t HW = H() * W();
    conf_.sp = has_spatial ? D() * HW : HW;
    conf_.tag_kind = jit_memory_tag_kind_t::blocked;
    conf_.simd_tail = C() % conf_.simd_w;
    conf_.sp_split_size = conf_.sp;
    if (C() < std::sqrt(conf_.sp)) {
        conf_.sp_split_size = conf_.sp
                / math::gcd(
                        conf_.sp, static_cast<dim_t>(dnnl_get_max_threads()));
    }

    conf_.ndims = ndims();
    conf_.mb = MB();
    conf_.c = C();
    conf_.d = D();
    conf_.h = H();
    conf_.w = W();

    conf_.dt_size = types::data_type_size(conf_.data_type);
    conf_.stride_mb = src_d.blocking_desc().strides[0];
    conf_.group_size = group_size();
    conf_.axis = axis();
    conf_.axis_size = axis_size();
    conf_.el_size_of_indices = sizeof(unsigned);

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.template book<char>(
            memory_tracking::names::key_shuffle_precompute_transpose,
            utils::rnd_up(conf_.axis_size, conf_.blk_size) * sizeof(int));

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            kernel_, new jit_uni_shuffle_kernel_t<isa>(pd()->get_conf())));
    CHECK(kernel_->create_kernel());
    return status::success;
}

template <cpu_isa_t isa>
inline jit_uni_shuffle_t<isa>::jit_uni_shuffle_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_shuffle_t<isa>::~jit_uni_shuffle_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::execute(const exec_ctx_t &ctx) const {
    using namespace prop_kind;
    using namespace utils;

    const auto i_arg = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const auto o_arg = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;
    auto input = CTX_IN_MEM(const uint8_t *, i_arg);
    auto output = CTX_OUT_MEM(uint8_t *, o_arg);

    const auto &conf = pd()->get_conf();
    assert(conf.tag_kind == jit_memory_tag_kind_t::blocked);

    const int transpose_row = pd()->is_fwd() ? conf.group_size
                                             : conf.axis_size / conf.group_size;
    const int transpose_col = pd()->is_fwd() ? conf.axis_size / conf.group_size
                                             : conf.group_size;

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    auto scratchpad_ptr = scratchpad.template get<int>(
            memory_tracking::names::key_shuffle_precompute_transpose);

    // Precompute transposed axis helper array
    parallel_nd(transpose_col, transpose_row, [=](dim_t i, dim_t j) {
        scratchpad_ptr[j * transpose_col + i] = i * transpose_row + j;
    });

    const dim_t CB = utils::div_up(conf.c, conf.blk_size);
    const dim_t SPB = conf.sp / conf.sp_split_size;

    // Precompute input offsets using transposed axis
    parallel_nd(CB, [=](dim_t cb) {
        const int blk_end
                = nstl::min(conf.blk_size, conf.c - cb * conf.blk_size);
        PRAGMA_OMP_SIMD()
        for (int cc = 0; cc < blk_end; ++cc) {
            const int off = cb * conf.blk_size + cc;
            int input_c = scratchpad_ptr[off];
            // Re-write transposed axis data.
            scratchpad_ptr[off]
                    = (input_c / conf.blk_size * conf.sp * conf.blk_size
                              + input_c % conf.blk_size)
                    * conf.dt_size;
        }
    });

    parallel_nd(conf.mb, SPB, CB,
            [= COMPAT_THIS_CAPTURE](dim_t mb, dim_t spb, dim_t cb) {
        const dim_t c_work
                = nstl::min(conf.blk_size, conf.c - cb * conf.blk_size);
        const dim_t c_curr = cb * conf.blk_size;
        const dim_t sp_work = conf.sp_split_size;
        const dim_t sp_curr = spb * sp_work;
        const dim_t off = mb * conf.stride_mb + sp_curr * conf.blk_size;

        jit_uni_shuffle_args_t args;
        args.src = input + off * conf.dt_size;
        args.dst = output + (off + conf.sp * c_curr) * conf.dt_size;

        args.cb_loop_size = c_work;
        args.is_padded_block = cb + 1 == CB;

        args.input_off_ptr = scratchpad_ptr + c_curr;
        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_shuffle_t<sse41>;
template struct jit_uni_shuffle_t<avx>;
template struct jit_uni_shuffle_t<avx512_core>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
