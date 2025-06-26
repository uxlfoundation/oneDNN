/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/jit/codegen/reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

copy_operand_t::copy_operand_t(const reg_buf_data_t &rbd) : CopyOperand(rbd) {
    if (!rbd.is_empty()) {
        const auto &rd = rbd.reg_buf();
        if (!rd.with_permute() && rd.blocks() <= 1) return;
        const auto base = rbd.base();
        block_size = rd.with_permute() ? 1 : rd.block_regs();
        const auto blocks = rd.regs() / block_size;
        block_bases.reserve(blocks);
        // TODO: join contiguous registers into blocks
        for (int i = 0, j = 0; i < blocks; ++i, j += block_size) {
            auto block_base = rd.base(j);
            block_bases.push_back(block_base);
            if (block_base <= base && base < block_base + block_size)
                block_off = i;
        }
        advance(rbd.hw(), 0, 1);
    }
}

copy_operand_t &copy_operand_t::advance(
        ngen::HW hw, int elems, uint8_t stride) {
    const auto nblocks = into<int>(block_bases.size());
    const auto grf_bits = ngen::GRF::bytes(hw) << 3;
    const auto type_bit_size = ngen::getBits(type);
    const auto bit_off = (offset + elems * stride) * type_bit_size;
    const auto grf_shift = bit_off / grf_bits;
    if (temp || block_bases.empty())
        grf += grf_shift;
    else {
        const auto orig_block_base = block_bases[block_off];
        const auto grf_off = grf - orig_block_base + grf_shift;
        const auto block_shift = grf_off / block_size;
        block_off += block_shift;
        if (block_off >= nblocks)
            // If we advance past the end of the buffer, continue linearly from
            // (max_base + block_size).
            grf = [&]() {
                const auto past_end = block_size * (block_off - nblocks + 1);
                int max_base = grf;
                for (const auto &base : block_bases)
                    max_base = std::max(max_base, base);
                return (int16_t)(max_base + past_end + (grf_off % block_size));
            }();
        else
            grf = (int16_t)(block_bases[block_off] + (grf_off % block_size));
    }
    offset = (uint8_t)((bit_off % grf_bits) / type_bit_size);
    return *this;
}

void copy_plan_t::mov(int simd, ngen::InstructionModifier mod,
        const copy_operand_t &dst, const copy_operand_t &src) {
    static constexpr ngen::Opcode mov = ngen::Opcode::mov;
    const auto grf_bits = ngen::GRF::bytes(hw()) << 3;

    auto max_simd = [&](const copy_operand_t &op) {
        const auto &block_size = op.block_size;
        const auto &block_bases = op.block_bases;
        // Count contiguous registers
        int regs = block_size - (op.grf - block_bases[op.block_off]);
        for (size_t j = op.block_off; j < block_bases.size() - 1; ++j) {
            if (block_bases[j] + block_size != block_bases[j + 1]) break;
            regs += block_size;
        }
        int type_bits = ngen::getBits(op.type);
        int rem_bits = regs * grf_bits - type_bits * (1 + op.offset);
        return rem_bits / (type_bits * op.stride) + 1;
    };

    auto block_src = src, block_dst = dst;
    while (simd > 0) {
        auto block_simd = simd;
        for (auto &op : {block_dst, block_src}) {
            if (op.block_bases.empty()) continue;
            block_simd = std::min(block_simd, max_simd(op));
        }

        append(phase, mov, block_simd, mod, block_dst, block_src);
        simd -= block_simd;
        // Protection from advancing past the end of the buffer, which may
        // result in OOB accesses when registers are permuted.
        if (simd <= 0) break;
        block_dst.advance(hw(), block_simd, dst.stride);
        block_src.advance(hw(), block_simd, src.stride);
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
