/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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


#include "generator.hpp"
#include "hw_utils.hpp"

using namespace ngen;

#include "internal/namespace_start.hxx"


// Scale then add: dst <- src0 + src1 * (numerator / denominator), rounding up.
// If exact = true, ensure src1 * num / denom is integral if src1 immediate.
template <HW hw>
void BLASKernelGenerator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, int src0, const RegData &src1,
                                        int numerator, int denominator, CommonState &state, bool exact, ngen::SourceLocation loc)
{
    if (!is_zero_or_pow2(numerator)) stub();
    if (!is_zero_or_pow2(denominator)) stub();

    if (numerator == denominator) {
        (src0 != 0)   ? add(mod, dst, src1, src0, loc) :
        (src1 != dst) ? mov(mod, dst, src1, loc)
                      : noop();
    } else if (numerator > denominator) {
        (src0 == 0) ? mulConstant(mod, dst, src1, numerator / denominator, loc)
                    : mad(mod, dst, src0, src1, numerator / denominator, loc);
    } else if ((numerator * 2) == denominator)
        avg(mod, dst, src1, src0 * 2, loc);
    else {
        add(mod, dst, src1, ((src0 + 1) * denominator / numerator) - 1, loc);
        asr(mod, dst, dst, ilog2(denominator) - ilog2(numerator), loc);
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1,
                                        int numerator, int denominator, CommonState &state, bool exact, ngen::SourceLocation loc)
{
    if (!is_zero_or_pow2(numerator)) stub();
    if (!is_zero_or_pow2(denominator)) stub();

    if (numerator == denominator)
        add(mod, dst, src1, src0, loc);
    else if (numerator > denominator)
        mad(mod, dst, src0, src1, numerator / denominator, loc);
    else {
        auto temp = state.ra.alloc_sub(src1.getType());
        if (exact)
            asr(mod, temp, src1, ilog2(denominator) - ilog2(numerator), loc);
        else {
            add(mod, temp, src1, (denominator / numerator) - 1, loc);
            asr(mod, temp, temp, ilog2(denominator) - ilog2(numerator), loc);
        }
        add(mod, dst, temp, src0, loc);
        state.ra.safeRelease(temp);
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, const RegData &src0, int src1,
                                        int numerator, int denominator, CommonState &state, bool exact, ngen::SourceLocation loc)
{
    if (!is_zero_or_pow2(numerator)) stub();
    if (!is_zero_or_pow2(denominator)) stub();
    if (exact && ((numerator * src1) % denominator))
        stub("Misaligned immediate value.");
    add(mod, dst, src0, (numerator * src1) / denominator, loc);
}

template <HW hw>
template <typename S0, typename S1>
void BLASKernelGenerator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, S0 src0, S1 src1,
                                        Type T, CommonState &state, bool exact, int scale, ngen::SourceLocation loc)
{
    addScaled(mod, dst, src0, src1, T.paddedSize() * scale, T.perByte(), state, exact, loc);
}

// Multiply by a constant, optimizing for power-of-2 constants.
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::mulConstant(const InstructionModifier &mod, const RegData &dst, const RegData &src0, int32_t src1, SourceLocation loc)
{
    if (src1 == 0)
        mov<DT>(mod, dst, uint16_t(0), loc);
    else if (src1 == 1) {
        if (dst != src0) mov<DT>(mod, dst, src0, loc);
    } else if (src1 == -1)
        mov<DT>(mod, dst, -src0, loc);
    else if (is_zero_or_pow2(src1))
        shl<DT>(mod, dst, src0, uint16_t(ilog2(src1)), loc);
    else if (src1 >= 0x10000)
        mul<DT>(mod, dst, src0, uint32_t(src1), loc);
    else if (src1 < -0x8000)
        mul<DT>(mod, dst, src0, int32_t(src1), loc);
    else if (src1 > 0)
        mul<DT>(mod, dst, src0, uint16_t(src1), loc);
    else
        mul<DT>(mod, dst, src0, int16_t(src1), loc);
}

// Modulo by constant value.
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::mod(const Subregister &dst, const Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    if (is_zero_or_pow2(modulus))
        and_<DT>(1, dst, src, modulus - 1, loc);
    else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP))
        math<DT>(1, MathFunction::irem, dst, src, modulus, loc);
    else {
        auto temp = dst;
        if (src == dst)
            temp = state.ra.alloc_sub<uint32_t>();
        alignDown<DT>(temp, src, modulus, strategy, state, loc);
        add<DT>(1, dst, src, -temp, loc);
        if (src == dst)
            state.ra.safeRelease(temp);
    }
}

// Return both (a % b) and a - (a % b).
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::modExt(const Subregister &dstMod, const Subregister &dstMultiple, const Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    if (is_zero_or_pow2(modulus)) {
        and_<DT>(1, dstMultiple, src, ~uint32_t(modulus - 1), loc);
        and_<DT>(1, dstMod, src, modulus - 1, loc);
    } else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP)) {
        math<DT>(1, MathFunction::irem, dstMod, src, modulus, loc);
        add<DT>(1, dstMultiple, src, -dstMod, loc);
    } else {
        alignDown<DT>(dstMultiple, src, modulus, strategy, state, loc);
        add<DT>(1, dstMod, src, -dstMultiple, loc);
    }
}

// Divide an unsigned value by a constant, rounding down.
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::divDown(const ngen::Subregister &dst, const ngen::Subregister &src, uint16_t divisor, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    if (is_zero_or_pow2(divisor))
        shr<DT>(1, dst, src, ilog2(divisor), loc);
    else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP))
        math<DT>(1, MathFunction::iqot, dst, src, uint32_t(divisor), loc);
    else {
        // Replace integer division with multiplication by reciprocal + shift.
        // Valid for numerators <= 2^31.
        int shift = ngen::utils::bsr(divisor);
        auto recip32 = uint32_t(((uint64_t(0x100000000) << shift) + divisor - 1) / divisor);
        if (!strategy.emulate.emulate64_mul) {
            auto tmp = state.ra.alloc_sub<uint64_t>();
            mul(1, tmp, src, recip32, loc);
            shr(1, dst, tmp.ud(1), shift, loc);
            state.ra.safeRelease(tmp);
        } else {
            emul32High(1, dst, src, recip32, loc);
            shr(1, dst, dst, shift, loc);
        }
    }
}

// Align an unsigned value down to a multiple of align.
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::alignDown(const InstructionModifier &mod, const Subregister &dst, const Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    if (is_zero_or_pow2(align))
        and_<DT>(mod, dst, src, uint32_t(-align), loc);
    else {
        divDown(dst, src, align, strategy, state, loc);
        mul(mod, dst, dst, align, loc);
    }
}

template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::alignDown(const Subregister &dst, const Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    alignDown(1, dst, src, align, strategy, state, loc);
}

// Align an unsigned value up to a multiple of align.
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::alignUp(const Subregister &dst, const Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    add<DT>(1, dst, src, uint16_t(align - 1), loc);
    alignDown<DT>(dst, dst, align, strategy, state, loc);
}

// Non-constant integer division.
// Requires an auxiliary constant: ceiling(2^(32 + s) / denom), where s = floor(log2(denom)).
template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::divDown(const Subregister &dst, const Subregister &src0, const Subregister &src1, const Subregister &src1Recip, const FlagRegister &flag, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    bool emulate = strategy.emulate.emulate64_mul;
    Subregister tmp;
    auto shift = state.ra.alloc_sub<uint32_t>();
    auto pop = state.ra.alloc_sub<uint16_t>();
    cbit(1, pop, src1, loc);
    fbh(1, shift, src1, loc);
    cmp(1 | gt | flag, pop, 1, loc);
    add(1, shift, -shift, 31, loc);
    if (emulate)
        emul32High(1 | flag, dst, src0, src1Recip, loc);
    else {
        tmp = state.ra.alloc_sub<uint64_t>();
        mul(1 | flag, tmp, src0, src1Recip, loc);
    }
    shr(1 | ~flag, dst, src0, shift, loc);
    shr(1 | flag, dst, emulate ? dst : tmp.ud(1), shift, loc);
    state.ra.safeRelease(shift);
    state.ra.safeRelease(pop);
    state.ra.safeRelease(tmp);
}

template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::divUp(const Subregister &dst, const Subregister &src0, const Subregister &src1, const Subregister &src1Recip, const FlagRegister &flag, const CommonStrategy &strategy, CommonState &state, SourceLocation loc)
{
    auto adj = state.ra.alloc_sub<uint32_t>();
    eadd3(1, adj, src0, src1, -1, loc);
    divDown(dst, adj, src1, src1Recip, flag, strategy, state, loc);
    state.ra.safeRelease(adj);
}

#include "internal/namespace_end.hxx"
