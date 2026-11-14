/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <initializer_list>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/rnn.hpp"
#include "common/type_helpers.hpp"

#include "cpu/gemm/gemm_pack.hpp"

#include "cpu/rnn/ref_rnn.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace rnn_utils;
using namespace format_tag;
using namespace rnn_packed_format;
using namespace data_type;

static bool check_dims_contiguous_except_one(const memory_desc_wrapper &mdw,
        int idx_with_arbitrary_stride, std::initializer_list<int> perm) {
    if (mdw.format_kind() != format_kind::blocked) return false;
    if ((size_t)mdw.ndims() != perm.size()) return false;

    const auto &blk = mdw.blocking_desc();

    dim_t expect_stride = 1;
    for (int idx = mdw.ndims() - 1; idx >= 0; --idx) {
        const int permuted_idx = *(perm.begin() + idx);
        const bool ok = (idx == idx_with_arbitrary_stride)
                ? expect_stride <= blk.strides[permuted_idx]
                : expect_stride == blk.strides[permuted_idx];
        if (!ok) return false;
        expect_stride = mdw.dims()[permuted_idx] * blk.strides[permuted_idx];
    }

    return true;
}

bool rnn_utils::is_ldigo(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 2, {0, 1, 2, 3, 4});
}

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 3, {0, 1, 3, 4, 2});
}

bool rnn_utils::is_ldio(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 2, {0, 1, 2, 3});
}

bool rnn_utils::is_ldoi(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 2, {0, 1, 3, 2});
}

bool rnn_utils::is_ldigo_blocked(const memory_desc_wrapper &mdw) {
    format_tag_t md_format_tag = mdw.matches_one_of_tag(format_tag::ldgOi32o,
            format_tag::ldgOI32o2i, format_tag::ldgOI32o4i,
            format_tag::ldgOI16o4i, format_tag::ldgOI64o2i,
            format_tag::ldgOI64o4i, format_tag::ldgOi16o);
    return md_format_tag != format_tag::undef;
}

bool rnn_utils::is_ldgoi_blocked(const memory_desc_wrapper &mdw) {
    format_tag_t md_format_tag = mdw.matches_one_of_tag(
            format_tag::ldgIo32i, format_tag::ldgIO32i2o, format_tag::ldgIo16i);
    return md_format_tag != format_tag::undef;
}

bool rnn_utils::is_ldio_blocked(const memory_desc_wrapper &mdw) {
    format_tag_t md_format_tag = mdw.matches_one_of_tag(format_tag::ldOi32o,
            format_tag::ldOI32o4i, ldOI16o4i, format_tag::ldOi16o);
    return md_format_tag != format_tag::undef;
}

dim_t rnn_utils::get_good_ld(dim_t dim, int sizeof_dt) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    const dim_t ld = rnd_up(dim, 64 / sizeof_dt);
    return (ld % 256 == 0) ? ld + 64 / sizeof_dt : ld;
}

void rnn_utils::set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_ht_offset, size_t &ws_states_layer_offset,
        size_t &ws_states_iter_offset, size_t &ws_states_iter_c_offset,
        size_t &ws_diff_states_layer_offset, size_t &ws_diff_states_iter_offset,
        size_t &ws_diff_states_iter_c_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_ht_offset, size_t &scratch_diff_ht_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size) {

    const size_t page_size = 4096; // 2097152;
    size_t current_offset;
    /* Mandatory workspaces: go to workspace if use_workspace, scratchpad
     * otherwise */
    current_offset = 0; // assumes the workspace base pointer is page aligned

#define register_space(a) \
    do { \
        current_offset = utils::rnd_up(current_offset, page_size); \
        CONCAT2(a, _offset) = current_offset; \
        current_offset += rnn.CONCAT2(a, _size); \
    } while (false)

    register_space(ws_gates);
    register_space(ws_ht);
    register_space(ws_states_layer);

    // For all currently supported cells, ws_iter should not be used
    // at all since dst_iter == dst_layer
    assert(rnn.ws_states_layer_size == rnn.ws_states_iter_size);
    ws_states_iter_offset = ws_states_layer_offset;

    register_space(ws_states_iter_c);
    register_space(ws_diff_states_layer);
    register_space(ws_diff_states_iter);
    register_space(ws_diff_states_iter_c);
    register_space(ws_grid_comp);

    workspace_size = rnn.use_workspace ? current_offset : 0;

    /* Optional scratchpads */
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    register_space(scratch_gates);
    register_space(scratch_ht);
    register_space(scratch_diff_ht);
    register_space(scratch_cell);
    if (rnn.copy_bias)
        register_space(ws_bias);
    else
        ws_bias_offset = 0;

    scratchpad_size = current_offset;
#undef register_space
}

void rnn_utils::get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_ht_offset, ws_states_layer_offset,
            ws_states_iter_offset, ws_states_iter_c_offset,
            ws_diff_states_layer_offset, ws_diff_states_iter_offset,
            ws_diff_states_iter_c_offset, ws_grid_comp_offset,
            scratch_gates_offset, scratch_ht_offset, scratch_diff_ht_offset,
            scratch_cell_offset, ws_bias_offset;
    set_offsets(rnn, ws_gates_offset, ws_ht_offset, ws_states_layer_offset,
            ws_states_iter_offset, ws_states_iter_c_offset,
            ws_diff_states_layer_offset, ws_diff_states_iter_offset,
            ws_diff_states_iter_c_offset, ws_grid_comp_offset, ws_bias_offset,
            scratch_gates_offset, scratch_ht_offset, scratch_diff_ht_offset,
            scratch_cell_offset, scratchpad_size, workspace_size);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    const auto dims = weights_md.dims;

    int ld_dim_idx = 0;
    switch (tag) {
        case ldio:
        case ldigo:
            strides[2] = rnn_utils::get_good_ld((int)strides[2],
                    (int)types::data_type_size(weights_md.data_type));
            ld_dim_idx = 2;
            break;
        case ldoi:
        case ldgoi:
            strides[weights_md.ndims - 1]
                    = rnn_utils::get_good_ld((int)strides[weights_md.ndims - 1],
                            (int)types::data_type_size(weights_md.data_type));
            if (tag == ldgoi) strides[3] = dims[4] * strides[4];
            ld_dim_idx = 3;
            break;
        default: return status::unimplemented;
    }
    strides[1] = dims[ld_dim_idx] * strides[ld_dim_idx];
    strides[0] = dims[1] * strides[1];

    return status::success;
}

status_t rnn_utils::set_expected_desc(rnn_conf_t &rnn,
        memory_desc_t &weights_md, rnn_utils::weights_type_t weights_type) {
    using namespace rnn_utils;
    bool use_packed_gemm = false;
    switch (weights_type) {
        case weights_type_t::layer:
            use_packed_gemm = rnn.use_layer_packed_gemm;
            break;
        case weights_type_t::iter:
            use_packed_gemm = rnn.use_iter_packed_gemm;
            break;
        case weights_type_t::projection:
            use_packed_gemm = rnn.use_projection_packed_gemm;
            break;
        default: assert(!"unsupported weights type");
    }

    if (use_packed_gemm) {
        weights_md.format_kind = format_kind::rnn_packed;
        rnn_packed_desc_t &rnn_pdata = weights_md.format_desc.rnn_packed_desc;
        switch (weights_type) {
            case weights_type_t::iter:
                rnn_pdata.format = rnn.is_fwd
                        ? rnn_packed_memory_format_t::ldigo_p
                        : rnn_packed_memory_format_t::ldgoi_p;
                rnn_pdata.ldb = static_cast<int>(rnn.ws_states_iter_ld);
                rnn_pdata.n = static_cast<int>(rnn.mb);
                rnn_pdata.n_parts = rnn.n_parts_weights_iter;
                array_copy(rnn_pdata.parts, rnn.parts_weights_iter,
                        DNNL_RNN_MAX_N_PARTS);
                array_copy(rnn_pdata.part_pack_size,
                        rnn.part_weights_iter_pack_size, DNNL_RNN_MAX_N_PARTS);
                rnn_pdata.offset_compensation = rnn.weights_iter_comp_offset;
                rnn_pdata.size = rnn.weights_iter_pack_size;
                break;
            case weights_type_t::layer:
                rnn_pdata.format = rnn.is_fwd
                        ? rnn_packed_memory_format_t::ldigo_p
                        : rnn_packed_memory_format_t::ldgoi_p;
                rnn_pdata.ldb = static_cast<int>(rnn.ws_states_layer_ld);
                rnn_pdata.n = static_cast<int>(
                        rnn.merge_gemm_layer ? rnn.n_iter * rnn.mb : rnn.mb);
                rnn_pdata.n_parts = rnn.n_parts_weights_layer;
                array_copy(rnn_pdata.parts, rnn.parts_weights_layer,
                        DNNL_RNN_MAX_N_PARTS);
                array_copy(rnn_pdata.part_pack_size,
                        rnn.part_weights_layer_pack_size, DNNL_RNN_MAX_N_PARTS);
                rnn_pdata.offset_compensation = rnn.weights_layer_comp_offset;
                rnn_pdata.size = rnn.weights_layer_pack_size;
                break;
            case weights_type_t::projection:
                // TODO: add ldoi_p for bwd?
                rnn_pdata.format = rnn_packed_memory_format_t::ldio_p;
                rnn_pdata.ldb = static_cast<int>(rnn.proj_ht_ld);
                rnn_pdata.n = static_cast<int>(rnn.mb);
                rnn_pdata.n_parts = rnn.n_parts_weights_projection;
                array_copy(rnn_pdata.parts, rnn.parts_weights_projection,
                        DNNL_RNN_MAX_N_PARTS);
                array_copy(rnn_pdata.part_pack_size,
                        rnn.part_weights_projection_pack_size,
                        DNNL_RNN_MAX_N_PARTS);
                rnn_pdata.offset_compensation
                        = rnn.weights_projection_comp_offset;
                rnn_pdata.size = rnn.weights_projection_pack_size;
                break;
            default: assert(!"unsupported weights type");
        }
        if (rnn.is_signed_int8_conf()) {
            weights_md.extra.flags
                    = 0 | memory_extra_flags::rnn_s8s8_compensation;
            weights_md.extra.compensation_mask = 0;
        }
    } else {
        using namespace format_tag;
        if (rnn.is_brgemm) {
            const int n_block = static_cast<int>(rnn.n_block);
            format_tag_t tag = format_tag::undef;

            if (weights_type == weights_type_t::projection) {
                if (rnn.is_int8_conf())
                    tag = utils::map(n_block, format_tag::undef, 32,
                            format_tag::ldOI32o4i, 16, format_tag::ldOI16o4i);
                else
                    tag = utils::map(n_block, format_tag::undef, 32,
                            format_tag::ldOi32o, 16, format_tag::ldOi16o);
            } else if (rnn.is_fwd) {
                if (rnn.is_int8_conf())
                    tag = utils::map(n_block, format_tag::undef, 64,
                            format_tag::ldgOI64o4i, 32, ldgOI32o4i, 16,
                            ldgOI16o4i);
                else if (rnn.is_xf16_conf())
                    tag = utils::map(n_block, format_tag::undef, 64,
                            format_tag::ldgOI64o2i, 32, ldgOI32o2i);
                else {
                    assert(IMPLICATION(n_block == 64, rnn.is_bf32()));
                    tag = utils::map(n_block, format_tag::undef, 64,
                            format_tag::ldgOi32o /*bf32*/, 32,
                            format_tag::ldgOi32o, 16, ldgOi16o);
                }
            } else {
                if (rnn.is_xf16_conf())
                    tag = format_tag::ldgIO32i2o;
                else
                    tag = utils::map(n_block, format_tag::undef, 32,
                            format_tag::ldgIo32i, 16, format_tag::ldgIo16i);
            }

            if (tag != format_tag::undef) {
                CHECK(memory_desc_init_by_tag(weights_md, tag));
                if (rnn.is_unsigned_int8_conf()) {
                    weights_md.extra.flags
                            = 0 | memory_extra_flags::rnn_u8s8_compensation;
                    weights_md.extra.compensation_mask
                            = (weights_type == weights_type_t::projection)
                            ? 13 // 1101
                            : 27; // 11011
                } else if (rnn.is_signed_int8_conf()) {
                    weights_md.extra.flags
                            = 0 | memory_extra_flags::rnn_s8s8_compensation;
                    weights_md.extra.compensation_mask = 0;
                }
                return status::success;
            } else {
                return status::unimplemented;
            }
        } else {
            const format_tag_t tag = weights_type == weights_type_t::projection
                    ? rnn.is_fwd ? ldio : ldoi
                    : rnn.is_fwd ? ldigo
                                 : ldgoi;
            CHECK(memory_desc_init_by_tag(weights_md, tag));
            // Adjust strides for good leading dimension in GEMM
            CHECK(set_good_strides(weights_md, tag));
        }
    }
    return status::success;
}

float rnn_utils::to_float(const void *data, const data_type_t dt) {
    if (dt == data_type::f32)
        return *static_cast<const float *>(data);
    else if (dt == data_type::bf16)
        return float(*static_cast<const bfloat16_t *>(data));
    else if (dt == data_type::f16)
        return float(*static_cast<const float16_t *>(data));
    return 0.0f;
}

const void *rnn_utils::inc_ptr(
        const void *data, data_type_t data_type, dim_t offset) {
    if (data_type == data_type::f32)
        return static_cast<const float *>(data) + offset;
    else if (data_type == data_type::bf16)
        return static_cast<const bfloat16_t *>(data) + offset;
    else if (data_type == data_type::f16)
        return static_cast<const float16_t *>(data) + offset;
    else
        return data;
}

void *rnn_utils::inc_ptr(void *data, data_type_t data_type, dim_t offset) {
    return const_cast<void *>(
            inc_ptr(static_cast<const void *>(data), data_type, offset));
}

bool rnn_utils::init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const primitive_attr_t &attr, data_type_t src_type,
        data_type_t acc_type, data_type_t scratch_type,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d,
        const memory_desc_wrapper &bias_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = utils::one_of(rd.cell_kind, dnnl_lbr_gru, dnnl_lbr_augru);
    rnn.is_lstm_peephole = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_peephole_desc).is_zero();
    rnn.is_lstm_projection = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_projection_desc).is_zero();
    rnn.is_augru
            = utils::one_of(rd.cell_kind, dnnl_lbr_augru, dnnl_vanilla_augru);
    rnn.bias_dt = bias_d.is_zero() ? data_type::f32 : bias_d.data_type();
    rnn.src_iter_c_dt = src_iter_c_d.is_zero() ? data_type::f32
                                               : src_iter_c_d.data_type();
    rnn.dst_iter_c_dt = dst_iter_c_d.is_zero() ? data_type::f32
                                               : dst_iter_c_d.data_type();

    rnn.cell_dt = src_type;
    const int src_dt_size = static_cast<int>(types::data_type_size(src_type));
    const int acc_dt_size = static_cast<int>(types::data_type_size(acc_type));
    const int scratch_dt_size
            = static_cast<int>(types::data_type_size(scratch_type));
    switch (rd.direction) {
        case dnnl_unidirectional_left2right: rnn.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: rnn.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: rnn.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: rnn.exec_dir = bi_sum; break;
        default: break;
    }

    if (utils::everyone_is(data_type::f32, src_layer_d.data_type(),
                dst_layer_d.data_type(), weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (utils::everyone_is(data_type::bf16, src_layer_d.data_type(),
                     dst_layer_d.data_type(), weights_layer_d.data_type())) {
        if (!platform::has_data_type_support(data_type::bf16)) return false;
#if DNNL_X64
        if (!(x64::mayiuse(x64::avx512_core) || x64::mayiuse(x64::avx2_vnni_2)))
            return false;
#elif DNNL_AARCH64
        if (!aarch64::mayiuse_bf16()) return false;
#endif
        rnn.dt_conf = all_bf16;
    } else if (utils::everyone_is(data_type::f16, src_layer_d.data_type(),
                       dst_layer_d.data_type(), weights_layer_d.data_type())) {
        if (!platform::has_data_type_support(data_type::f16)) return false;
#if DNNL_X64
        if (!(x64::mayiuse(x64::avx512_core_fp16)
                    || x64::mayiuse(x64::avx2_vnni_2)))
            return false;
#endif
        rnn.dt_conf = all_f16;
    } else if (dst_layer_d.data_type() == data_type::u8) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else if (dst_layer_d.data_type() == data_type::s8) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::s8))
            rnn.dt_conf = s8s8s8s8;
        else
            rnn.dt_conf = f32s8f32s8;

    } else if (dst_layer_d.data_type() == data_type::f32) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::u8))
            rnn.dt_conf = u8u8u8f32;
        else if (IMPLICATION(src_iter_d.md_,
                         src_iter_d.data_type() == data_type::s8))
            rnn.dt_conf = s8s8s8f32;
        else if (IMPLICATION(src_layer_d.md_,
                         src_layer_d.data_type() == data_type::s8))
            rnn.dt_conf = f32s8f32f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }

    if (!rnn.is_fwd && !platform::has_training_support(src_layer_d.data_type()))
        return false;

    // Set problem members defining problem sizes
    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dhc = weights_layer_d.dims()[4];
    rnn.dlc = rnn.is_lstm_projection ? weights_projection_d.dims()[3] : rnn.dhc;
    // All supported cells have dic == dlc
    rnn.dic = rnn.dlc;

    // set members with user memories leading dimensions
    // Assumption: weights datatype size is the same as state datatype size
    assert(types::data_type_size(weights_layer_d.data_type())
            == types::data_type_size(src_layer_d.data_type()));

    // set workspace leading dimensions (and non leading-dimensions)

    rnn.proj_ht_nld = rnn.mb;
    rnn.proj_ht_ld = get_good_ld(rnn.dhc, src_dt_size);

    rnn.ws_gates_nld = rnn.mb;
    rnn.ws_gates_ld = get_good_ld(rnn.dhc * rnn.n_gates, src_dt_size);
    rnn.ws_ht_nld = rnn.proj_ht_nld;
    rnn.ws_ht_ld = rnn.proj_ht_ld;

    rnn.ws_states_layer_nld = rnn.mb;
    rnn.ws_states_layer_ld = get_good_ld(
            nstl::max(rnn.sic, nstl::max(rnn.slc, rnn.dlc)), src_dt_size);
    // there is no need for al separate ws_states_iter for now as all
    // supported cell have dst_iter == dst_layer
    rnn.ws_states_iter_nld = rnn.ws_states_layer_nld;
    rnn.ws_states_iter_ld = rnn.ws_states_layer_ld;

    // we do not need a good ld for iter_c as it is not involved in GEMM
    rnn.ws_states_iter_c_nld = rnn.mb;
    rnn.ws_states_iter_c_ld = rnn.dhc;

    // TODO: be more restrictive on the leading dimensions
    rnn.ws_diff_states_layer_nld = rnn.mb;
    rnn.ws_diff_states_layer_ld = get_good_ld(
            nstl::max(nstl::max(rnn.slc, rnn.dic), nstl::max(rnn.sic, rnn.dhc)),
            acc_dt_size);

    rnn.ws_diff_states_iter_nld = rnn.mb;
    rnn.ws_diff_states_iter_ld = get_good_ld(
            nstl::max(nstl::max(rnn.slc, rnn.dic), nstl::max(rnn.sic, rnn.dhc)),
            acc_dt_size);

    rnn.ws_diff_states_iter_c_nld = rnn.mb;
    rnn.ws_diff_states_iter_c_ld = rnn.dhc;

    // set scratch (not)leading dimensions
    // scratch gates is used to store intermediate gates before postgemm operation
    // temporary: we also use it in lstmp as temporary scratchpad
    // between projection and downconversion, hence the max with dlc
    rnn.scratch_gates_nld = rnn.mb;
    rnn.scratch_gates_ld = get_good_ld(
            nstl::max(rnn.dlc, rnn.n_gates * rnn.dhc), scratch_dt_size);
    rnn.scratch_ht_nld = rnn.proj_ht_nld;
    rnn.scratch_ht_ld = rnn.proj_ht_ld;

    rnn.scratch_diff_ht_nld = rnn.mb;
    rnn.scratch_diff_ht_ld = get_good_ld(rnn.dlc, acc_dt_size);

    // Assumption: {src,dst}_layer has tnc layout, {src,dst}_iter has ldnc,
    rnn.src_layer_ld_ = src_layer_d.blocking_desc().strides[1];
    rnn.dst_layer_ld_ = dst_layer_d.blocking_desc().strides[1];
    rnn.src_iter_ld_ = types::is_zero_md(src_iter_d.md_)
            ? 0
            : src_iter_d.blocking_desc().strides[2];
    rnn.dst_iter_ld_ = types::is_zero_md(dst_iter_d.md_)
            ? 0
            : dst_iter_d.blocking_desc().strides[2];
    rnn.src_iter_c_ld_ = types::is_zero_md(src_iter_c_d.md_)
            ? 0
            : src_iter_c_d.blocking_desc().strides[2];
    rnn.dst_iter_c_ld_ = types::is_zero_md(dst_iter_c_d.md_)
            ? 0
            : dst_iter_c_d.blocking_desc().strides[2];

    /* Set the correct number of weights parts */
    rnn.is_orig_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::vanilla_augru);
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = static_cast<int>(rnn.n_gates);
    rnn.parts_weights_layer[1] = 0;

    rnn.n_parts_weights_iter = rnn.is_orig_gru ? 2 : 1;
    rnn.parts_weights_iter[0]
            = rnn.is_orig_gru ? 2 : static_cast<int>(rnn.n_gates);
    rnn.parts_weights_iter[1] = rnn.is_orig_gru ? 1 : 0;

    rnn.n_parts_weights_projection = 1;
    rnn.parts_weights_projection[0] = 1;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = static_cast<int>(rnn.n_bias);
    rnn.parts_bias[1] = 0;

    rnn.use_matmul = !rnn.is_brgemm && rnn.is_fwd // TODO: Enable BWD
    // TODO: Below checks are for legacy and a performance study is
    // required to avoid regressions.
#if DNNL_X64
            && IMPLICATION(
                    rnn.is_cell_dt_bf16(), !x64::mayiuse(x64::avx512_core))
            && IMPLICATION(rnn.is_cell_dt_f32() || rnn.is_cell_dt_int8(),
                    x64::mayiuse(x64::avx2)
                            && utils::one_of(rd.cell_kind,
                                    alg_kind::vanilla_gru,
                                    alg_kind::vanilla_augru));
#else
            && !rnn.is_cell_dt_f32() && !rnn.is_cell_dt_int8();
#endif

    /* Decide which gemm implementation to use: packed/nonpacked jit/cblas
     * and if to merge gemm across iterations */
    const bool is_f32 = rnn.dt_conf == all_f32,
               is_bf16 = rnn.dt_conf == all_bf16;
    const bool is_gru = utils::one_of(rd.cell_kind, alg_kind::vanilla_gru,
            alg_kind::lbr_gru, alg_kind::vanilla_augru, alg_kind::lbr_augru);
    const bool is_inference = !rnn.is_training;

    // To be able to merge the GEMM on the layer input when not
    // copying, we need to have a trivial stride for the T dimension
    rnn.src_layer_is_trivial_stride = src_layer_d.blocking_desc().strides[0]
            == (rnn.src_layer_ld_ * rnn.mb);
    rnn.dst_layer_is_trivial_stride = dst_layer_d.blocking_desc().strides[0]
            == (rnn.dst_layer_ld_ * rnn.mb);

    rnn.merge_gemm_layer = !(rnn.is_brgemm || rnn.use_matmul)
            ? ((rnn.is_fwd && rnn.src_layer_is_trivial_stride)
                      || ((rd.prop_kind == prop_kind::backward)
                              && rnn.dst_layer_is_trivial_stride))
                    && (((rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd)
                            || rnn.is_int8_conf())
            : false;
    rnn.merge_gemm_iter = !(rnn.is_brgemm || rnn.use_matmul)
            ? rnn.dst_layer_is_trivial_stride && !(rnn.is_fwd || is_gru)
            : false;
    rnn.force_nocopy = false;
#if DNNL_X64
    rnn.force_nocopy = x64::mayiuse(x64::avx2)
            && ((is_inference && (rnn.n_layer > 1 || rnn.mb < 100))
                    || (rnn.is_training && rnn.dhc < 500));
#endif

    /* Decide to copy bias */
    rnn.copy_bias = rnn.is_int8_conf();

    rnn.use_layer_packed_gemm = !(rnn.is_brgemm || rnn.use_matmul)
            ? utils::one_of(weights_layer_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
                    && is_inference
                    && ((is_f32 && pack_sgemm_supported() && rnn.n_iter == 1)
                            || rnn.is_int8_conf() || is_bf16)
            : false;
    rnn.use_iter_packed_gemm = !(rnn.is_brgemm || rnn.use_matmul)
            ? utils::one_of(weights_iter_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
                    && is_inference
                    && ((is_f32 && pack_sgemm_supported() && rnn.mb >= 16)
                            || rnn.is_int8_conf() || is_bf16)
            : false;
    rnn.use_projection_packed_gemm = !(rnn.is_brgemm || rnn.use_matmul)
            ? utils::one_of(weights_projection_d.format_kind(),
                      format_kind::any, format_kind::rnn_packed)
                    && is_inference
                    && ((is_f32 && pack_sgemm_supported() && rnn.n_iter == 1)
                            || rnn.is_int8_conf() || is_bf16)
            : false;

    rnn.diff_weights_overwrite = rd.flags & rnn_flags::diff_weights_overwrite;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL || BUILD_GEMM_KERNELS_NONE
    // XXX: Threadpool runtime may use different number of threads at execute
    // and create stages. GEMM packed API is not aware of number of threads as
    // of now. In order to synchronize all layers, GEMM pack API should be
    // modified to accept number of threads instead of taking it from
    // `dnnl_get_max_threads()`, and rnn_packed_desc_t should be updated with
    // `nthr` member to pass this information between different parts of packed
    // API, since `get_size` call happens on RNN side, while packing happens
    // on reorder side. Consider enabling later.
    // `test_iface_runtime_attr` was disabled for RNN with threadpool due to
    // this is the only working approach for int8 computations in RNN for now.
    // Consider enabling it once resolved.
    rnn.use_layer_packed_gemm = false;
    rnn.use_iter_packed_gemm = false;
    rnn.use_projection_packed_gemm = false;
#endif

    /* Set packed gemm sizes */
    /* TODO: investigate the benefit of mixing packed and non-packed weights parts */
    const auto set_pack_sizes
            = [&](bool merge, bool &do_pack, size_t &weights_pack_size,
                      int &n_parts, int *parts, size_t *parts_pack_size,
                      size_t &comp_offset, dim_t ic, dim_t oc, dim_t weights_oc,
                      dim_t data_ld) -> bool {
        bool pack = true;
        weights_pack_size = 0;
        for (int p = 0; p < n_parts; p++) {
            const dim_t m_p = rnn.is_fwd ? (parts[p] * oc) : ic;
            const dim_t k_p = rnn.is_fwd ? ic : (parts[p] * oc);
            const dim_t n_p
                    = merge ? static_cast<dim_t>(rnn.mb) * rnn.n_iter : rnn.mb;
            bool pack_part = true;

            dnnl_status_t st = dnnl_success;
            switch (rnn.dt_conf) {
                case all_f32:
                    st = sgemm_pack_get_size("A", "N", "N", &m_p, &n_p, &k_p,
                            &m_p, &data_ld, &parts_pack_size[p], &pack_part);
                    break;
                case s8s8s8f32:
                case f32s8f32f32:
                case s8s8s8s8:
                case f32s8f32s8:
                    st = gemm_s8s8s32_pack_get_size("A", "N", "N", &m_p, &n_p,
                            &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case u8u8u8f32:
                case f32u8f32f32:
                case u8u8u8u8:
                case f32u8f32u8:
                    st = gemm_s8u8s32_pack_get_size("A", "N", "N", &m_p, &n_p,
                            &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case all_bf16:
                    st = gemm_bf16bf16f32_pack_get_size("A", "N", "N", &m_p,
                            &n_p, &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                default: assert(!"Unsupported configuration");
            }
            if (st != dnnl_success) return false;

            pack = pack && pack_part;
            weights_pack_size += rnn.n_layer * rnn.n_dir * parts_pack_size[p];
        }

        // NOTE: pack is updated only for f32. We force pack for int8
        do_pack = (rnn.dt_conf == all_f32) ? pack : true;
        comp_offset = weights_pack_size;
        const bool need_compensation = rnn.is_int8_conf();
        weights_pack_size += (need_compensation ? rnn.n_layer * rnn.n_dir : 0)
                * weights_oc * sizeof(float);

        return true;
    };
    // TODO: the activation leading dimension can vary for first layer/iteration
    if (rnn.use_layer_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_layer,
                rnn.use_layer_packed_gemm, rnn.weights_layer_pack_size,
                rnn.n_parts_weights_layer, rnn.parts_weights_layer,
                rnn.part_weights_layer_pack_size, rnn.weights_layer_comp_offset,
                rnn.slc, rnn.dhc, rnn.n_gates * rnn.dhc,
                rnn.ws_states_layer_ld);
        if (!ok) return false;
    }

    if (rnn.use_iter_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_iter, rnn.use_iter_packed_gemm,
                rnn.weights_iter_pack_size, rnn.n_parts_weights_iter,
                rnn.parts_weights_iter, rnn.part_weights_iter_pack_size,
                rnn.weights_iter_comp_offset, rnn.sic, rnn.dhc,
                rnn.n_gates * rnn.dhc, rnn.ws_states_iter_ld);
        if (!ok) return false;
    }

    if (rnn.use_projection_packed_gemm) {
        bool ok = set_pack_sizes(false, rnn.use_projection_packed_gemm,
                rnn.weights_projection_pack_size,
                rnn.n_parts_weights_projection, rnn.parts_weights_projection,
                rnn.part_weights_projection_pack_size,
                rnn.weights_projection_comp_offset, rnn.dhc, rnn.dic, rnn.dic,
                rnn.scratch_ht_ld);
        if (!ok) return false;
    }

    return true;
}

void rnn_utils::set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &diff_weights_projection_d) {

    // Set leading dimensions for input weights arrays depending on input format
    const auto set_dims
            = [&](const memory_desc_wrapper &md, dim_t &ld, dim_t &nld) {
        ld = 0;
        nld = 0;
        if (md.is_blocking_desc()) {
            if (is_ldigo(md)) {
                ld = md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else if (is_ldgoi(md)) {
                ld = md.blocking_desc().strides[4];
                nld = md.dims()[3] * md.dims()[4];
            } else if (is_ldoi(md)) {
                ld = md.blocking_desc().strides[3];
                nld = md.dims()[3];
            } else if (is_ldio(md)) {
                ld = md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else
                assert(!"unsupported weights format");
        }
    };
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    set_dims(weights_projection_d, rnn.weights_projection_ld,
            rnn.weights_projection_nld);
    if (!rnn.is_fwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
        set_dims(diff_weights_projection_d, rnn.diff_weights_projection_ld,
                rnn.diff_weights_projection_nld);
    }

    assert(weights_layer_d.data_type() == weights_iter_d.data_type());
    assert(IMPLICATION(diff_weights_layer_d.ndims() != 0,
            (diff_weights_layer_d.data_type()
                    == diff_weights_iter_d.data_type())));
}

void rnn_utils::set_workspace_sizes(rnn_conf_t &rnn, const rnn_desc_t &rd,
        data_type_t acc_type, data_type_t scratch_type) {
    const size_t src_dt_size = types::data_type_size(rnn.cell_dt);
    const size_t acc_dt_size = types::data_type_size(acc_type);
    const size_t scratch_dt_size = types::data_type_size(scratch_type);

    rnn.use_workspace = rnn.is_training;
    // TODO: for inference, we can make ws_states_* smaller, but
    // dependant of the grid execution though
    rnn.ws_states_layer_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_layer_ld * src_dt_size;
    rnn.ws_states_iter_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_iter_ld * src_dt_size;
    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;
    rnn.ws_states_iter_c_size = is_lstm ? (size_t)(rnn.n_layer + 1) * rnn.n_dir
                    * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_iter_c_ld
                    * types::data_type_size(rnn.src_iter_c_dt)
                                        : 0;

    rnn.ws_diff_states_layer_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_layer_ld * acc_dt_size
            : (size_t)0;
    rnn.ws_diff_states_iter_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_iter_ld * acc_dt_size
            : (size_t)0;
    rnn.ws_diff_states_iter_c_size = rnn.is_training && is_lstm
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_iter_c_ld * acc_dt_size
            : (size_t)0;

    rnn.ws_gates_size = rnn.is_training
            ? (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.ws_gates_nld
                    * rnn.ws_gates_ld * src_dt_size
            : (size_t)0;
    rnn.ws_ht_size = rnn.is_training ? (size_t)rnn.n_layer * rnn.n_dir
                    * rnn.n_iter * rnn.ws_ht_nld * rnn.ws_ht_ld * src_dt_size
                                     : (size_t)0;
    rnn.n_iter_scratch_gates
            = (rnn.merge_gemm_layer || rnn.merge_gemm_iter) ? rnn.n_iter : 1;
    rnn.scratch_gates_size = scratch_dt_size * rnn.n_iter_scratch_gates
            * rnn.scratch_gates_nld * rnn.scratch_gates_ld;
    rnn.scratch_ht_size = src_dt_size * rnn.scratch_ht_nld * rnn.scratch_ht_ld;
    rnn.scratch_diff_ht_size = rnn.is_training
            ? acc_dt_size * rnn.scratch_diff_ht_nld * rnn.scratch_diff_ht_ld
            : (size_t)0;

    /* set other sizes */
    /// scratchpad buffer for each cell to hold intermediate data in gru/lbr_gru
    rnn.scratch_cell_size = (utils::one_of(rd.cell_kind, alg_kind::vanilla_gru,
                                     alg_kind::vanilla_augru, alg_kind::lbr_gru,
                                     alg_kind::lbr_augru)
                    ? scratch_dt_size * rnn.scratch_gates_nld
                            * rnn.scratch_gates_ld
                    : 0);
    /// workspace needed for lbr GRU
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dhc * acc_dt_size;
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell * sizeof(float);
    /// bias ws needed to add compensation in int8
    rnn.ws_bias_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dhc
            * types::data_type_size(rnn.bias_dt);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
