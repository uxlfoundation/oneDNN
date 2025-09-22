/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CODEGEN_REDUCE_HPP
#define GPU_INTEL_JIT_CODEGEN_REDUCE_HPP

#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/codegen/reorder.hpp"
#include "gpu/intel/jit/ir/reduce.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class reduce_impl_t {
public:
    reduce_impl_t(ngen::HW hw, const reduce_t &reduce, int simd_size)
        : hw_(hw)
        , src_layout_(reduce.src_layout)
        , dst_layout_(reduce.dst_layout)
        , simd_size_(simd_size) {}

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();

        bool is_inplace = (src_rd.base() == dst_rd.base()
                && src_rd.byte_offset() == dst_rd.byte_offset());
        if (is_inplace) {
            gpu_assert(src_type == dst_type)
                    << "Inplace operation is supported for the same type only.";
        }

        std::vector<bool> seen(size_bytes(src_layout_));

        tile_t tile = find_1d_tile(src_layout_, dst_layout_);
        int tile_elems = (int)tile.elems();
        auto src_tile_layout = src_layout_.sub(tile);
        auto dst_tile_layout = dst_layout_.sub(tile);
        const auto &src_tile_blocks = src_tile_layout.blocks();
        const auto &dst_tile_blocks = dst_tile_layout.blocks();
        gpu_assert(src_tile_blocks.size() <= 1);
        gpu_assert(dst_tile_blocks.size() <= 1);
        ngen_register_scope_t block_scope(scope.register_allocator());
        int src_stride
                = src_tile_blocks.empty() ? 1 : (int)src_tile_blocks[0].stride;
        int dst_stride
                = dst_tile_blocks.empty() ? 1 : (int)dst_tile_blocks[0].stride;
        int grf_size = ngen::GRF::bytes(hw_);
        for (auto &src_start : src_layout_.iter(tile)) {
            ngen_register_scope_t tile_scope(scope.register_allocator());
            auto dst_start = src_start;
            for (dim_idx_t i = 0; i < dst_layout_.ndims(); i++) {
                if (dst_layout_.tile()[i] == 1) dst_start[i] = 0;
            }
            int src_off = src_layout_.offset<int>(src_start);
            int dst_off = dst_layout_.offset<int>(dst_start);

            if (is_inplace) {
                bool same_src_dst = (dst_off == src_off);
                if (!seen[dst_off] && !same_src_dst) {
                    gpu_error_not_expected() << "Invalid inplace reduction.";
                }
                seen[dst_off] = true;
                if (same_src_dst) return;
            }

            auto d = dst_rd.format(dst_off, tile_elems, 1, to_ngen(dst_type));
            auto s = src_rd.format(
                    src_off, tile_elems, src_stride, to_ngen(src_type));
            bool s_half_grf_aligned
                    = utils::one_of(s.byte_offset(), 0, grf_size / 2);
            bool s_is_bf = src_type.is_bf16();
            bool s_is_hf = src_type.is_f16();
            bool s_is_fp8 = src_type.is_fp8();
            bool d_is_f = dst_type.is_f32();
            bool native_bf = host->hw_info().systolic_support();
            bool sd_aligned = (tile_elems == 1
                    || (dst_stride * ngen::getBytes(d.type())
                            == src_stride * ngen::getBytes(s.type())));

            if (src_stride != 1 || !sd_aligned || s_is_hf || s_is_fp8
                    || (s_is_bf && !native_bf)
                    || (s_is_bf && !s_half_grf_aligned)) {
                auto tmp_type = src_type;
                if ((s_is_hf && d_is_f) || s_is_fp8 || (s_is_bf && !native_bf)
                        || ((d.offset() != 0 || !s_half_grf_aligned)
                                && (s_is_bf))) {
                    tmp_type = type_t::f32();
                }
                auto tmp = tile_scope.alloc_reg_data(
                        tmp_type.with_elems(tile_elems));
                emit_reorder_1d_tile(
                        host, tile_scope, tile_elems, s, src_stride, tmp, 1);
                s = tmp.format(0, tile_elems, 1, to_ngen(tmp_type));
            }
            align_src_dst_offset(host, tile_scope, tile_elems, d, s);
            host->add(tile_elems, d.reg_data(), d.reg_data(), s.reg_data());
        }
    }

private:
    tile_t find_1d_tile(layout_t a, layout_t b) const {
        align_layouts(a, b);

        gpu_assert(!a.blocks().empty());
        // Allow trivial tile for scalar dst.
        if (b.blocks().empty()) { return dst_layout_.tile(); }

        auto &a0 = a[0];
        auto &b0 = b[0];

        bool ok = (a0.dim == b0.dim && a0.block == b0.block);
        if (!ok) {
            // Try to match strided layout.
            if (a0.block == 2) {
                auto a_blocks = a.blocks();
                a_blocks.erase(a_blocks.begin());
                a = a.with(a_blocks);
                return find_1d_tile(std::move(a), std::move(b));
            }
            return tile_t(std::vector<dim_t>(b.ndims(), 1));
        }

        gpu_assert(dim_t(b0.stride) == 1)
                << "Reduction is not supported for non-unit dst stride.";

        int grf_size = ngen::GRF::bytes(hw_);
        int a_grf_elems = grf_size / a.type().size();
        int b_grf_elems = grf_size / b.type().size();

        int min_step = std::min(a_grf_elems, b_grf_elems);
        int max_step = 2 * min_step;

        min_step = std::min(
                std::min(hw_ <= ngen::HW::XeLP ? 8 : simd_size_, min_step),
                (int)a0.block);

        if (a0.block % min_step != 0) {
            // TODO: Extend implementation to support this case.
            gpu_except_not_implemented("Reduction is not supported.");
        }

        std::vector<dim_t> tile_dims(src_layout_.ndims(), 1);
        tile_dims[a0.dim]
                = ir_utils::max_divisor(int(a0.block), {min_step, max_step});

        return tile_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
    int simd_size_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
