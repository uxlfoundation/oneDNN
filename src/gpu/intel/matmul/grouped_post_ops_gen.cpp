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
#include <sstream>

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
        const memory_desc_wrapper &dst_desc, po_kind_t *po_chain,
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
            if (po_mdw.nelems() == 1 && po_mdw.data_type() == data_type::f32
                    && !po_mdw.is_host_scalar_desc()) {
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
    std::ostringstream os;
    os << "inline void apply_post_ops_chain(ugemm_grouped_c_type *c_tile, long "
          "n, "
          "long m, long lddst, off_t sg_i0, off_t sg_j0, off_t src_offset, "
          "off_t batch,\n";
    os << "        const global BINARY_SCALE_GROUPED_TILE_DATA_T "
          "*grouped_scale, "
          "const global BINARY_SCALE_DENSE_TILE_DATA_T "
          "*dense_scale,\n";
    os << "        const global float *nvfp4_scale) {\n";

    const auto &po = attr.post_ops_;
    for (int i = 0; i < po.len(); ++i) {
        const auto &e = po.entry_[i];
        if (e.is_eltwise()) {
            os << "#define eltwise_apply_" << i << "(v) (("
               << to_ocl_float(e.eltwise.scale) << ") * ((v) / (1.0f + exp(-("
               << to_ocl_float(e.eltwise.alpha) << ") * (v)))))\n";
            os << "    tile_elementwise((*c_tile), eltwise_apply_" << i
               << ");\n";
            os << "#undef eltwise_apply_" << i << "\n";
        } else if (e.is_binary()) {
            if (po_chain[i] == po_kind_t::binary_grouped_scale) {
                os << "    const global BINARY_SCALE_GROUPED_TILE_DATA_T "
                      "*group_scale_ptr = grouped_scale + src_offset * "
                      "lddst;\n";
                os << "    ugemm_grouped_c_type binary_group_tile_" << i
                   << ";\n";
                os << "#if BINARY_SCALE_GROUPED_DT_F32\n";
                os << "    tile_load(&binary_group_tile_" << i
                   << ", group_scale_ptr, n, m, lddst, sg_i0, sg_j0);\n";
                os << "#else\n";
                os << "    binary_group_in_tile_type binary_group_in_tile_" << i
                   << ";\n";
                os << "    tile_load(&binary_group_in_tile_" << i
                   << ", group_scale_ptr, n, m, lddst, sg_i0, sg_j0);\n";
                os << "    tile_convert(binary_group_in_tile_" << i
                   << ", binary_group_tile_" << i
                   << ", BINARY_SCALE_GROUPED_TO_FLOAT);\n";
                os << "#endif\n";
                os << "#define binary_mul_" << i << "(a, b) ((a) * (b))\n";
                os << "    tile_binary((*c_tile), binary_group_tile_" << i
                   << ", binary_mul_" << i << ");\n";
                os << "#undef binary_mul_" << i << "\n";
            } else if (po_chain[i] == po_kind_t::binary_dense_scale) {
                os << "    const global BINARY_SCALE_DENSE_TILE_DATA_T "
                      "*dense_scale_ptr_"
                   << i << " = dense_scale + src_offset;\n";
                os << "    binary_dense_tile_type dense_scale_tile_" << i
                   << ";\n";
                os << "#if BINARY_SCALE_DENSE_DT_F32\n";
                os << "    tile_load(&dense_scale_tile_" << i
                   << ", dense_scale_ptr_" << i << ", m, 1, 0, sg_j0, 0);\n";
                os << "#else\n";
                os << "    binary_dense_in_tile_type dense_in_tile_" << i
                   << ";\n";
                os << "    tile_load(&dense_in_tile_" << i
                   << ", dense_scale_ptr_" << i << ", m, 1, 0, sg_j0, 0);\n";
                os << "    tile_convert(dense_in_tile_" << i
                   << ", dense_scale_tile_" << i
                   << ", BINARY_SCALE_DENSE_TO_FLOAT);\n";
                os << "#endif\n";
                os << "    tile_hbroadcast_mul(c_tile, dense_scale_tile_" << i
                   << ");\n";
            } else if (po_chain[i] == po_kind_t::binary_nvfp4_scale) {
                os << "    float gs_" << i << " = *nvfp4_scale;\n";
                os << "#define binary_mul_" << i << "(v) ((v) * gs_" << i
                   << ")\n";
                os << "    tile_elementwise((*c_tile), binary_mul_" << i
                   << ");\n";
                os << "#undef binary_mul_" << i << "\n";
            }
        }
    }

    os << "}\n";
    return os.str();
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
