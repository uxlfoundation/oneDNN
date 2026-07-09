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

#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <iomanip>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

#define VCHECK_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

int find_po_in_chain(const po_kind_t *po_chain, po_kind_t kind) {
    for (int i = 0; i < 3; ++i) {
        if (po_chain[i] == kind) return i;
    }
    return -1;
}

status_t check_post_op_chain(const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_desc, dim_t ngroups, po_kind_t *po_chain,
        data_type_t *scale_arr) {
    auto &po = attr.post_ops_;
    scale_arr[0] = data_type::undef;
    scale_arr[1] = data_type::undef;
    VCHECK_MATMUL(po.len() <= 3, VERBOSE_UNSUPPORTED_POSTOP);
    for (int i = 0; i < po.len(); ++i) {
        auto &e = po.entry_[i];
        VCHECK_MATMUL(
                e.is_eltwise() || e.is_binary(), VERBOSE_UNSUPPORTED_POSTOP);
        if (e.is_eltwise()) {
            VCHECK_MATMUL(e.eltwise.alg == alg_kind::eltwise_swish,
                    VERBOSE_UNSUPPORTED_POSTOP);
            po_chain[i] = po_kind_t::eltwise;
        } else if (e.is_binary()) {
            VCHECK_MATMUL(e.binary.alg == alg_kind::binary_mul,
                    VERBOSE_UNSUPPORTED_POSTOP);

            const memory_desc_wrapper po_mdw(po.entry_[i].binary.src1_desc);
            if (po_mdw.nelems() == ngroups
                    && po_mdw.data_type() == data_type::f32
                    && !po_mdw.is_host_scalar_desc()) {
                // [G, 1] operand: one scale per group (expert), e.g. nvfp4
                // per-expert global scale.
                po_chain[i] = po_kind_t::binary_nvfp4_scale;
            } else {
                if (po_mdw.is_grouped_desc()) {
                    // [total_tokens, N] grouped - element-wise multiply
                    VCHECK_MATMUL(find_po_in_chain(po_chain,
                                          po_kind_t::binary_grouped_scale)
                                    == -1,
                            VERBOSE_UNSUPPORTED_POSTOP);
                    po_chain[i] = po_kind_t::binary_grouped_scale;
                } else if (!po_mdw.format_any()) {
                    // Dense tensor - check dimensions
                    const auto &dims = po_mdw.dims();
                    const auto ndims = po_mdw.ndims();
                    // [total_tokens, 1] - dense scale with horizontal broadcast
                    if (ndims >= 2 && dims[ndims - 1] == 1
                            && dims[ndims - 2] > 1) {
                        VCHECK_MATMUL(find_po_in_chain(po_chain,
                                              po_kind_t::binary_dense_scale)
                                        == -1,
                                VERBOSE_UNSUPPORTED_POSTOP);
                        po_chain[i] = po_kind_t::binary_dense_scale;
                    }
                }
            }
            VCHECK_MATMUL(
                    po_chain[i] != po_kind_t::none, VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    set_binary_scales_dt(attr, po_chain, scale_arr);
    const bool has_grouped_scale
            = find_po_in_chain(po_chain, po_kind_t::binary_grouped_scale) != -1;
    const bool has_dense_scale
            = find_po_in_chain(po_chain, po_kind_t::binary_dense_scale) != -1;

    if (has_grouped_scale) {
        VCHECK_MATMUL(utils::one_of(scale_arr[0], data_type::f16,
                              data_type::bf16, data_type::f32),
                VERBOSE_UNSUPPORTED_POSTOP);
    }
    if (has_dense_scale) {
        VCHECK_MATMUL(utils::one_of(scale_arr[1], data_type::f16,
                              data_type::bf16, data_type::f32),
                VERBOSE_UNSUPPORTED_POSTOP);
    }

    return status::success;
}
void set_binary_scales_dt(const primitive_attr_t &attr,
        const po_kind_t *po_chain, data_type_t *scale_arr) {
    const auto &po = attr.post_ops_;
    for (int i = 0; i < po.len(); ++i) {
        bool set_float = false;
        if (po.entry_[i].binary.src1_desc.data_type == data_type::undef) {
            set_float = true;
        }
        if (po_chain[i] == po_kind_t::binary_grouped_scale) {
            scale_arr[0] = set_float ? data_type::f32
                                     : po.entry_[i].binary.src1_desc.data_type;
        } else if (po_chain[i] == po_kind_t::binary_dense_scale) {
            scale_arr[1] = set_float ? data_type::f32
                                     : po.entry_[i].binary.src1_desc.data_type;
        }
    }
}

static std::string to_ocl_float(float v) {
    std::ostringstream os;
    os << std::setprecision(9) << std::scientific << v << "f";
    return os.str();
}

std::string generate_post_ops_microgemm_header(
        const primitive_attr_t &attr, const po_kind_t *po_chain) {
    std::string s = R"(
inline void apply_post_ops_chain(ugemm_grouped_c_type *c_tile, long n, long m, long lddst,
    off_t sg_i0, off_t sg_j0, off_t src_offset, off_t batch,
    const global BINARY_SCALE_GROUPED_TILE_DATA_T *grouped_scale,
    const global BINARY_SCALE_DENSE_TILE_DATA_T *dense_scale,
    const global float *nvfp4_scale) {
)";
    const auto &po = attr.post_ops_;
    for (int i = 0; i < po.len(); ++i) {
        const auto &e = po.entry_[i];
        if (e.is_eltwise()) {
            s += utils::format(R"(
#define eltwise_apply_%d(v) ((%s) * ((v) / (1.0f + exp(-(%s) * (v)))))
    tile_elementwise((*c_tile), eltwise_apply_%d);
#undef eltwise_apply_%d
                     )",
                    i, to_ocl_float(e.eltwise.scale).c_str(),
                    to_ocl_float(e.eltwise.alpha).c_str(), i, i);
        } else if (e.is_binary()) {
            if (po_chain[i] == po_kind_t::binary_grouped_scale) {
                s += utils::format(R"(
#define CONCAT_I(var) var##_%d
    const global BINARY_SCALE_GROUPED_TILE_DATA_T *group_scale_ptr = grouped_scale + src_offset * lddst;

    ugemm_grouped_c_type CONCAT_I(binary_group_tile);
#if BINARY_SCALE_GROUPED_DT_F32
    tile_load(&CONCAT_I(binary_group_tile), group_scale_ptr, n, m, lddst, sg_i0, sg_j0);
#else
    binary_group_in_tile_type CONCAT_I(binary_group_in_tile);
    tile_load(&CONCAT_I(binary_group_in_tile), group_scale_ptr, n, m, lddst, sg_i0, sg_j0);
    tile_convert(CONCAT_I(binary_group_in_tile), CONCAT_I(binary_group_tile), BINARY_SCALE_GROUPED_TO_FLOAT);
#endif
#define binary_mul_%d(a, b) ((a) * (b))
    tile_binary((*c_tile), CONCAT_I(binary_group_tile), CONCAT_I(binary_mul));
#undef binary_mul_%d
#undef CONCAT_I
                         )",
                        i, i, i);
            } else if (po_chain[i] == po_kind_t::binary_dense_scale) {
                s += utils::format(
                        R"(
#define CONCAT_I(var) var##_%d
    const global BINARY_SCALE_DENSE_TILE_DATA_T *CONCAT_I(dense_scale_ptr) = dense_scale + src_offset;
    binary_dense_tile_type CONCAT_I(dense_scale_tile);
#if BINARY_SCALE_DENSE_DT_F32
    tile_load(&CONCAT_I(dense_scale_tile), CONCAT_I(dense_scale_ptr), m, 1, 0, sg_j0, 0);
#else
    binary_dense_in_tile_type CONCAT_I(dense_in_tile);
    tile_load(&CONCAT_I(dense_in_tile), CONCAT_I(dense_scale_ptr), m, 1, 0, sg_j0, 0);
    tile_convert(CONCAT_I(dense_in_tile), CONCAT_I(dense_scale_tile), BINARY_SCALE_DENSE_TO_FLOAT);
#endif
    tile_hbroadcast_mul(c_tile, CONCAT_I(dense_scale_tile));
#undef CONCAT_I
)",
                        i);
            } else if (po_chain[i] == po_kind_t::binary_nvfp4_scale) {
                s += utils::format(
                        R"(
    float gs_%d = nvfp4_scale[batch];
#define binary_mul_%d(v) ((v) * gs_%d)
    tile_elementwise((*c_tile), binary_mul_%d);
#undef binary_mul_%d
)",
                        i, i, i, i, i);
            }
        }
    }

    s += "}\n";
    return s;
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
