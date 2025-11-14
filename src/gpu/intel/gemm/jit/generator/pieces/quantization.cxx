/*******************************************************************************
* Copyright 2019 Intel Corporation
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


#include "alloc_utils.hpp"
#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "quantization.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


// Prepare 2D dequantization layouts.
template <HW hw>
bool Generator<hw>::gemmMake2DQuantizationLayouts(bool isA, const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto lateOffset = isA ? problem.needsBGroupSums() : problem.needsAGroupSums();
    int xoPtrDims = (isA ? problem.aoPtrDims : problem.boPtrDims);
    bool xo2D = isA ? problem.aOffset2D()       : problem.bOffset2D();
    bool xs2D = isA ? problem.aScale2D()        : problem.bScale2D();
    bool xg2D = isA ? problem.needsAGroupSums() : problem.needsBGroupSums();
    bool xoTo2D = !xo2D && (isA ? problem.aOffset == ABOffset::Calc && (problem.earlyDequantizeA() || lateOffset)
                                : problem.bOffset == ABOffset::Calc && (problem.earlyDequantizeB() || lateOffset));
    bool cColMajor = isRegisterColMajor(problem.Tc_ext, problem.C, strategy.C);

    if (!xo2D && !xoTo2D && !xs2D && !xg2D) return true;

    auto &X_strategy       = isA ? strategy.A             : strategy.B;
    auto &X_offsetStrategy = isA ? strategy.AO            : strategy.BO;
    auto &X_scaleStrategy  = isA ? strategy.A_scale       : strategy.B_scale;
    auto &Xg_strategy      = isA ? strategy.Ag            : strategy.Bg;
    auto &X_offsetLayout   = isA ? state.A_offsetLayout   : state.B_offsetLayout;
    auto &X_scaleLayout    = isA ? state.A_scaleLayout    : state.B_scaleLayout;
    auto &Xg_layout        = isA ? state.Ag_layout        : state.Bg_layout;
    auto &Xr_offsetLayout  = isA ? state.Ar_offsetLayout  : state.Br_offsetLayout;
    auto &Xr_scaleLayout   = isA ? state.Ar_scaleLayout   : state.Br_scaleLayout;
    auto &Xgr_layout       = isA ? state.Agr_layout       : state.Bgr_layout;

    auto &XO         = isA ? problem.AO         : problem.BO;
    auto &XS         = isA ? problem.A_scale    : problem.B_scale;
    auto &Xg         = isA ? problem.Ag         : problem.Bg;
    auto Tx_ext      = isA ? problem.Ta_ext     : problem.Tb_ext;
    auto Tx          = isA ? problem.Ta         : problem.Tb;
    auto Txo         = isA ? problem.Tao        : problem.Tbo;
    auto Txs         = isA ? problem.Ta_scale   : problem.Tb_scale;
    auto Txg         = isA ? problem.Tag        : problem.Tbg;
    auto xqGroupK    = isA ? problem.aqGroupK   : problem.bqGroupK;
    auto xqGroupMN   = isA ? problem.aqGroupM   : problem.bqGroupN;
    auto &Txo_int    = isA ? state.Tao_int      : state.Tbo_int;
    auto &Txs_int    = isA ? state.Ta_scaleInt  : state.Tb_scaleInt;
    auto &Txg_int    = isA ? state.Tag_int      : state.Tbg_int;
    auto &lateScale  = isA ? state.lateScale2DA : state.lateScale2DB;

    Txo_int = Txo.isInteger() ? Tx.asSignedInt() : Tx;
    Txs_int = Tx;
    if (Tx == Type::bf16)
        Txs_int = Type::f32;
    Txg_int = Txg;

    int cpoDiv = 1;
    if (Txo_int.isInt8()) Txo_int = Type::s16, cpoDiv = 2;

    if (xs2D && (Txs.paddedSize() > Tx.paddedSize()) && Tx.isInteger()) {
        lateScale = true;
        Txs_int = problem.Tc;
    }

    bool int4SpecialPath = Tx_ext.isInt4() && one_of(Tx, Type::f16, Type::f32);
    if (int4SpecialPath) {
        Txo_int = Type::f16;
        Txs_int = Tx;
        if (Tx == Type::bf16) Txs_int = Type::f16;
    }

    if (lateOffset && (Txo.isInt4() || Txo.isInt8()))
        Txo_int = Type::s32;

    // Get tile sizes, depending on whether A/B are copied to SLM.
    // For late scaling (after compute), scales are always applied to the whole tile.
    int r, c, k, rNoSLM, cNoSLM;
    int tileR = 0, tileC = 0;
    bool remR = false, remC = false;
    if (isA) {
        bool slmA = strategy.slmA;
        rNoSLM = strategy.unroll[LoopM];
        cNoSLM = strategy.ka_load;
        r = slmA ? state.ma_slm : rNoSLM;
        c = slmA ? state.ka_slm : cNoSLM;
        k = slmA ? strategy.unrollKSLM : cNoSLM;
        r = std::max(1, r / xqGroupMN);
        c = state.kaq = std::max(1, c % xqGroupK == 0 ? c / xqGroupK : 1);
        state.kaqStride = std::max(1, k % xqGroupK == 0 ? k /  xqGroupK : 1);
        rNoSLM = std::max(1, rNoSLM / xqGroupMN);
        cNoSLM = state.kaqLate = std::max(1, cNoSLM % xqGroupK == 0 ? cNoSLM /  xqGroupK: 1);
        remR = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
        if (xqGroupMN <= 1 && xqGroupK > 1) tileC = 1;
        if (xqGroupMN > 1 && (xqGroupMN % strategy.unroll[LoopM] && strategy.unroll[LoopM] % xqGroupMN))
            stub("Tile size not compatible with group size in m dimension");
    } else {
        bool slmB = strategy.slmB;
        cNoSLM = strategy.unroll[LoopN];
        rNoSLM = strategy.kb_load;
        c = slmB ? state.nb_slm : cNoSLM;
        r = slmB ? state.kb_slm : rNoSLM;
        k = slmB ? strategy.unrollKSLM : rNoSLM;
        c = std::max(1, c / xqGroupMN);
        r = state.kbq = std::max(1, r % xqGroupK == 0 ? r / xqGroupK : 1);
        state.kbqStride = std::max(1, k % xqGroupK == 0 ?  k / xqGroupK : 1);
        cNoSLM = std::max(1, cNoSLM / xqGroupMN);
        rNoSLM = state.kbqLate = std::max(1, rNoSLM % xqGroupK == 0 ?  rNoSLM / xqGroupK : 1);
        remC = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
        if (xqGroupMN <= 1 && xqGroupK > 1) tileR = 1;
        if (xqGroupMN > 1 && (xqGroupMN % strategy.unroll[LoopN] && strategy.unroll[LoopN] % xqGroupMN))
            stub("Tile size not compatible with group size in n dimension");
    }

    int ro = lateOffset ? rNoSLM : r;
    int co = lateOffset ? cNoSLM : c;
    int rs = lateScale  ? rNoSLM : r;
    int cs = lateScale  ? cNoSLM : c;

    if (X_strategy.padded) {
        X_offsetStrategy.padded = X_scaleStrategy.padded = Xg_strategy.padded = true;
        remR = remC = false;
    }

    bool wantCM = isA ^ (xqGroupMN > 1);
    auto chooseAccess = [=](const MatrixAddressing &Xq) {
        return (wantCM == isColMajor(Xq.layout)) ? AccessType::Block : AccessType::Scattered;
    };

    X_offsetStrategy.accessType = chooseAccess(XO);
    X_scaleStrategy.accessType  = chooseAccess(XS);
    Xg_strategy.accessType      = chooseAccess(Xg);

    if (xo2D && !(X_offsetLayout = RegisterLayout::tryCreate(hw, Txo, ro,     co,     XO, X_offsetStrategy, remR, remC))) return false;
    if (xs2D &&  !(X_scaleLayout = RegisterLayout::tryCreate(hw, Txs, rs,     cs,     XS, X_scaleStrategy,  remR, remC))) return false;
    if (xg2D &&      !(Xg_layout = RegisterLayout::tryCreate(hw, Txg, rNoSLM, cNoSLM, Xg, Xg_strategy,      remR, remC))) return false;

    // Adjust masks for m/n grouping.
    auto adjustMask = [=](MaskInfo &mask) {
        if (!mask || xqGroupMN <= 1) return;
        if (!is_zero_or_pow2(xqGroupMN)) return;
        if (mask.fixed.isFixed) stub();
        mask.variable.rshift += ilog2(xqGroupMN);
    };

    for (auto *Xq_layout: {&X_offsetLayout, &X_scaleLayout, &Xg_layout}) {
        for (auto &block: *Xq_layout) {
            adjustMask(block.rowMask);
            adjustMask(block.colMask);
        }
    }

    // Quantization parameters will be upconverted to the size of A/B and duplicated to match crosspack.
    auto &lsrc = isA ? (strategy.slmA ? state.Ao_layout : !state.Ar_layout.empty() ? state.Ar_layout : state.A_layout)
                     : (strategy.slmB ? state.Bo_layout : !state.Br_layout.empty() ? state.Br_layout : state.B_layout);
    if (lsrc.empty()) stub();
    int crosspack = lsrc[0].crosspack;
    if (xqGroupMN > 1)
        crosspack = 1;

    int cpo = lateOffset ? 1 : div_up(crosspack, cpoDiv);
    int cps = lateScale  ? 1 : crosspack;

    auto makeQRepack = [&, tileR, tileC](Type Txq, Type Txq_int, RegisterLayout &repack, const RegisterLayout &src,
                                         int m, int n, int cp, bool forceRepack) mutable {
        if (cp > 1 || (cColMajor && (cp != src[0].crosspack)) || Txq != Txq_int || forceRepack) {
            bool allowPartialRegs = false;
            repack = RegisterLayout(hw, Txq_int, m, n, wantCM, cp, tileR, tileC, allowPartialRegs);
        }
    };

    if (xo2D) makeQRepack(Txo, Txo_int, Xr_offsetLayout, X_offsetLayout, ro,     co,     cpo, false);
    if (xs2D) makeQRepack(Txs, Txs_int, Xr_scaleLayout,  X_scaleLayout,  rs,     cs,     cps, lateScale);
    if (xg2D) makeQRepack(Txg, Txg_int, Xgr_layout,      Xg_layout,      rNoSLM, cNoSLM, 1,   true);

    if (xoTo2D) {
        if (xoPtrDims <= 0)
            Xr_offsetLayout = RegisterLayout(hw, Txo_int, 1, 1, isA);
        else if (xoPtrDims == 1)
            Xr_offsetLayout = RegisterLayout(hw, Txo_int, ro, co, isA, cpo, tileR, tileC, false);
        else stub();
    }

    return true;
}

// Convert and repack 2D grouped quantization data in preparation for dequantization.
template <HW hw>
void Generator<hw>::gemmRepack2DQuantizationData(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                                 const GRFMultirange &src, const GRFMultirange &dst,
                                                 const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (layoutDst.empty()) return;

    // Copy, broadcasting 1D to 2D data as needed.
    for (int doffR = 0; doffR < layoutDst.rows(); doffR += layoutSrc.rows())
        for (int doffC = 0; doffC < layoutDst.cols(); doffC += layoutSrc.cols())
            copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, doffR, doffC, false, strategy, state);

    // Duplicate data in padded region. TODO: do this as part of the copy.
    int cp = layoutDst[0].crosspack;
    int p0 = layoutDst[0].colMajor ? layoutDst[0].nc : layoutDst[0].nr;

    if (cp > 1) map(hw, Td, dst, layoutDst, strategy, [&](int simd, RegData r) {
        Subregister r0 = GRF(r.getBase()).sub(r.getOffset(), r.getType());
        moveToIntPipe(r0);
        auto r1 = r0;
        for (int i = p0; i < cp; i++) {
            r1.setOffset(r1.getOffset() + 1);
            mov(simd / cp, r1(cp), r0(cp));
        }
    });
}

template <HW hw>
void Generator<hw>::gemmRepack2DQuantizationData(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                                 const GRFMultirange &src, const GRFMultirange &dst,
                                                 const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    gemmRepack2DQuantizationData(layoutSrc.type(), layoutDst.type(), layoutSrc, layoutDst,
                                 src, dst, problem, strategy, state);
}

template <HW hw>
void Generator<hw>::gemmRepack2DOffsetData(Type Text, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                           const GRFMultirange &src, const GRFMultirange &dst,
                                           const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ts = layoutSrc.type(), Td = layoutDst.type();

    bool s4 = (Text == Type::s4);
    bool s8 = (Ts == Type::s8);
    bool u8 = (Ts == Type::u8);

    bool int4SpecialPath = Text.isInt4() && Td == Type::f16;
    auto tmpType = Td;

    if (int4SpecialPath) {
        if (u8) tmpType = Type::u16;
        if (s8) tmpType = Type::s16;
    }

    gemmRepack2DQuantizationData(Ts, tmpType, layoutSrc, layoutDst, src, dst, problem, strategy, state);

    if (int4SpecialPath) {
        if (s8 || u8) {
            int off = s4 ? 8 : 0;

            // Shift s8 -> u8 data.
            if (s8) {
                map(hw, Type::s16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                    add(esize, r, r, 0x80);
                });
                off -= 0x80;
            }

            // Reinterpret as f16 and undo offsets.
            if (off != 0) map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                uint16_t offF16 = std::abs(off);
                if (off < 0) offF16 |= 0x8000;
                add(esize, r, r, Immediate::hf(offF16));
            });

            // Rescale into normal range. End result is 2^(-12) * intended offset.
            map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                mul(esize, r, r, Immediate::hf(0x6C00));
            });
        } else {
            map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                s4 ? mad(esize, r, Immediate::hf(0x1800), r, Immediate::hf(0x0C00))     // 0x1800 = 8 * 2^(-12)
                   : mul(esize, r,                        r, Immediate::hf(0x0C00));    // 0x0C00 = 2^(-12)
            });
        }
    }
}

// Apply a single 2D group dequantize operation (scale/multiply).
template <HW hw>
void Generator<hw>::gemmDequantizeOperation(bool doA, Type T, Type Tq, BinaryOp op,
                                            const RegisterLayout &layout, const RegisterLayout &qlayout,
                                            const GRFMultirange &regs, const GRFMultirange &qregs,
                                            int h, int kab_load, int kq_load, const GEMMProblem &problem, CommonState &state)
{
    int xqGroupK  = doA ? problem.aqGroupK : problem.bqGroupK;
    int xqGroupMN = doA ? problem.aqGroupM : problem.bqGroupN;

    bool common = (qlayout.rows() * qlayout.cols()) == 1;

    bool bfSpecialPath = (T == Type::bf16) && (Tq == Type::f32) && (xqGroupMN > 1) && layout.hasFullCrosspack(2);
    GRFMultirange qPairs;
    int npairs = 0;

    if (bfSpecialPath) {
        for (npairs = 8; npairs > 0; npairs >>= 1) {
            qPairs = tryChunkAlloc(npairs, 1, Bundle(), BundleGroup::AllBundles(), state);
            if (!qPairs.empty()) break;
        }
        if (qPairs.empty()) throw out_of_registers_exception();
    }

    for (auto &block: layout) {
        auto crosspack = block.crosspack;
        bool colMajor = block.colMajor;
        int nx = colMajor ? block.nr : block.nc;
        int ny = colMajor ? block.nc : block.nr;

        int xqGroupX = colMajor == doA ? xqGroupMN : xqGroupK;
        int xqGroupY = colMajor == doA ? xqGroupK : xqGroupMN;

        // If crosspack spans multiple groups, use a stride to restrict to one group
        bool qbroadcastY = (xqGroupY % crosspack == 0) || common || bfSpecialPath;
        int strided = 1;
        if (!qbroadcastY) {
            strided = crosspack;
        }

        bool qbroadcastX = xqGroupX > 1 || common;
        int strideq = qbroadcastX ? 0 : 1;
        if (bfSpecialPath)
            strideq = 1;

        // Unpack iteration # and block offsets into MNK indices
        auto h_block = align_down(h, kab_load) + (doA ? block.offsetC : block.offsetR);
        auto mn_block = doA ? block.offsetR : block.offsetC;

        for(int y0 = 0; y0 < ny; y0 += qbroadcastY ? crosspack : 1) {
        for(int x0 = 0; x0 < nx; ) {
            auto ii0 = colMajor ? x0 : y0;
            auto jj0 = colMajor ? y0 : x0;

            auto h_final = h_block + (doA ? jj0 : ii0);
            auto mn_final = mn_block + (doA ? ii0 : jj0);
            auto mnq = mn_final / xqGroupMN;
            auto hq = (h_final % kq_load) / xqGroupK;

            auto io0 = doA ? mnq : hq;
            auto jo0 = doA ? hq : mnq;

            // Common scales always load the first element
            if (common) io0 = jo0 = 0;

            int ne, neq;
            const RegisterBlock *qblock;
            auto data = block.find(T, ii0, jj0, regs, &ne);
            auto qdata = qlayout.find(io0, jo0, qregs, &neq, &qblock);

            if (!qbroadcastX) ne = std::min(ne, neq);

            // If a group lies along the X-direction, limit ne to the end of the current group
            if (xqGroupX > 1) ne = std::min(ne, xqGroupX - x0 % xqGroupX);

            // Broadcast pairs of scales, in case of stride-2 bfloat16 data.
            if (bfSpecialPath) {
                int npair = (qdata.getOffset() / 2) % npairs;
                if (x0 == 0 && npair == 0)
                    for (int p = 0; p < npairs; p++)
                        mov(elementsPerGRF(hw, Tq), qPairs[p].ud(), qdata.ud(2*p)(0,2,1));
                qdata = qPairs[npair].sub(0, Tq.ngen());
            }

            int maxSIMD = (op == BinaryOp::Sub && T.isInt8()) ? 64 : 32;
            if (Tq == Type::f32) maxSIMD = elementsPerGRF(hw, Tq);
            int simd = std::min({ne * crosspack / strided, 2 * elementsPerGRF(hw, T) / strided, maxSIMD});
            switch (op) {
                case BinaryOp::Sub:
                    if (T.isInt8() && strided == 1) {
                        add(simd / 2, data(2), data(2), -qdata(strideq * 2 / Tq));
                        data.setOffset(data.getOffset() + 1);
                        qdata.setOffset(qdata.getOffset() + strideq / Tq);
                        add(simd / 2, data(2), data(2), -qdata(strideq * 2 / Tq));
                    } else
                        add(simd, data(strided), data(strided), -qdata(strideq));
                    break;
                case BinaryOp::Mul:
                    mul(simd, data(strided), data(strided), qdata(strideq));
                    break;
                case BinaryOp::ScaleSub:
                    if (T != Type::f16) stub();
                    mad(simd, data(strided), -qdata(strideq), data(strided), Immediate::hf(0x6C00));  /* 0x6C00 = 2^12 */
                    break;
                default: stub();
            }
            x0 += simd * strided / crosspack;
        }
        }
    }

    safeReleaseRanges(qPairs, state);
}

// Shift s4 data by 8 to transfrom it into u4 data.
template <HW hw>
void Generator<hw>::dequantizeInt4Shift(Type Tsrc, GRFMultirange src, const CommonStrategy &strategy)
{
    if (Tsrc != Type::s4) return;
    map(hw, Type::u16, src, src, strategy, [&](int esize, RegData r, RegData _) {
        xor_(esize, r, r, 0x8888);
    });
}

// Optimized int4 -> f16/bf16/f32 dequantization sequence.
template <HW hw>
void Generator<hw>::dequantizeInt4(bool doA, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                   const RegisterLayout &layoutOffset, const RegisterLayout &layoutScale,
                                   const GRFMultirange &src, const GRFMultirange &dst, const GRFMultirange &offset, const GRFMultirange &scale,
                                   int offR, int offC, int h, int kab_load, int kq_load,
                                   const GEMMProblem *problem, const CommonStrategy &strategy, CommonState &state, bool s4Shift)
{
    auto Tsrc = layoutSrc.type(), Tdst = layoutDst.type();
    if (!canDequantizeInt4(layoutSrc, layoutDst, layoutOffset, layoutScale))
        stub("Cannot perform dequantizeInt4");

    bool s4 = Tsrc.isSigned();
    bool f32 = (Tdst == Type::f32);
    bool bf16 = (Tdst == Type::bf16);

    RegisterLayout layoutDstF16;
    const RegisterLayout *effLayoutDst = &layoutDst;
    GRFMultirange dstF16;
    const GRFMultirange *effDst = &dst;
    if (f32 || bf16) {
        layoutDstF16 = RegisterLayout(hw, Type::f16, layoutSrc.rows(), layoutSrc.cols(), layoutDst.colMajor(), layoutDst.crosspack(), 0, 0, (hw < HW::XeHP));
        for (auto &block: layoutDstF16) {
            block.offsetR += layoutDst[0].offsetR;
            block.offsetC += layoutDst[0].offsetC;
        }
        dstF16 = chunkAlloc(layoutDstF16.regs(), 2, state);
        effLayoutDst = &layoutDstF16;
        effDst = &dstF16;
    }

    // 1) Shift s4 data to u4 data by adding 8.
    if (s4 && s4Shift)
        dequantizeInt4Shift(Tsrc, src, strategy);

    // 2) Copy u4 -> u16 data.
    copyRegisters(Type::u4, Type::u16, layoutSrc, *effLayoutDst, src, *effDst, offR, offC, false, strategy, state);

    // 3) Reinterpret u16 data as denormal f16, scale into normal range and subtract (rescaled) offsets if available.
    //     The required rescaling factor (2^24) is necessarily outside f16 range,
    //     so two multiplications are needed.
    if (!layoutOffset.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f16, Type::f16, BinaryOp::ScaleSub, *effLayoutDst, layoutOffset, *effDst, offset, h, kab_load, kq_load, *problem, state);
    } else {
        map(hw, Type::f16, *effDst, *effLayoutDst, strategy, [&](int esize, RegData r) {
            s4 ? mad(esize, r, Immediate::hf(0x9800), r, Immediate::hf(0x6C00)) /* 0x9800 = -8*2^(-12), 0x6C00 = 2^12 */
               : mul(esize, r, r, Immediate::hf(0x6C00));
        });
    }

    // 4) Finish rescaling -- remaining factor is 2^12.
    map(hw, Type::f16, *effDst, *effLayoutDst, strategy, [&](int esize, RegData r) {
        mul(esize, r, r, Immediate::hf(0x6C00));
    });

    // 5) Apply scales if present. If the scales are not too large (absolute value < 128),
    //      this could be merged into the previous multiplication.
    if (!f32 && !layoutScale.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f16, Type::f16, BinaryOp::Mul, *effLayoutDst, layoutScale, *effDst, scale, h, kab_load, kq_load, *problem, state);
    }

    // 6) Convert to dst type if needed.
    if (f32 || bf16) {
        copyRegisters(Type::f16, Tdst, layoutDstF16, layoutDst, dstF16, dst, offR, offC, false, strategy, state);
        safeReleaseRanges(dstF16, state);
    }

    // 7) Apply scales for f32 after f16->f32 upconversion.
    if (f32 && !layoutScale.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f32, Type::f32, BinaryOp::Mul, layoutDst, layoutScale, dst, scale, h, kab_load, kq_load, *problem, state);
    }
}

// Dequantize A/B, given 2D grouped quantization data.
template <HW hw>
void Generator<hw>::gemmDequantizeAB(bool doA, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst0,
                                     const GRFMultirange &src, const GRFMultirange &dst0, int h, int kab_load, int kab_repack, int kq_load,
                                     const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                     bool s4Shift)
{
    auto Tsrc = layoutSrc.type(), Tdst = layoutDst0.type();
    auto Txo_int     = doA ? state.Tao_int             : state.Tbo_int;
    auto Txs_int     = doA ? state.Ta_scaleInt         : state.Tb_scaleInt;
    auto &oiLayout   = doA ? state.A_offsetLayout      : state.B_offsetLayout;
    auto &orLayout   = doA ? state.Ar_offsetLayout     : state.Br_offsetLayout;
    auto &oiRegs     = doA ? state.A_offsetRegs        : state.B_offsetRegs;
    auto &orRegs     = doA ? state.Ar_offsetRegs       : state.Br_offsetRegs;
    auto &siLayout   = doA ? state.A_scaleLayout       : state.B_scaleLayout;
    auto &srLayout   = doA ? state.Ar_scaleLayout      : state.Br_scaleLayout;
    auto &siRegs     = doA ? state.A_scaleRegs         : state.B_scaleRegs;
    auto &srRegs     = doA ? state.Ar_scaleRegs        : state.Br_scaleRegs;
    bool lateOffset  = doA ? problem.needsBGroupSums() : problem.needsAGroupSums();
    bool lateScale   = doA ? state.lateScale2DA        : state.lateScale2DB;

    auto &oLayout = orLayout.empty() ? oiLayout : orLayout;
    auto &oRegs   = orLayout.empty() ? oiRegs   : orRegs;
    auto &sLayout = srLayout.empty() ? siLayout : srLayout;
    auto &sRegs   = srLayout.empty() ? siRegs   : srRegs;

    bool xo2D = !oLayout.empty() && !lateOffset;
    bool xs2D = !sLayout.empty() && !lateScale;

    bool copy = !layoutDst0.empty();
    auto layoutDst = copy ? layoutDst0 : layoutSrc;
    auto dst       = copy ? dst0 : src;

    auto Tx_int = xo2D ? Txo_int : Tdst;

    if (xo2D && !xs2D && (Txo_int.bits() > Tdst.bits()))
        Tx_int = Tdst;

    int offR = doA ? 0 : h % kab_repack;
    int offC = doA ? h % kab_repack : 0;
    int offR0 = offR, offC0 = offC;

    int ms = layoutSrc.rows(), ns = layoutSrc.cols();
    int md = layoutDst.rows(), nd = layoutDst.cols();

    if (ms < md || ns < nd) {
        if (!copy) stub();
        layoutDst = RegisterLayout(hw, Tdst, ms, ns, layoutDst0.colMajor(), layoutDst[0].crosspack);
        dst = chunkAlloc(layoutDst.regs(), 2, state);
        offR = offC = 0;
    }

    if (canDequantizeInt4(layoutSrc, layoutDst, oLayout, sLayout)) {
        dequantizeInt4(doA, layoutSrc, layoutDst, oLayout, sLayout,
                       src, dst, oRegs, sRegs, offR, offC, h, kab_load, kq_load, &problem,
                       strategy, state, s4Shift);
    } else {
        if (copy)
            copyRegisters(Tsrc, Tx_int, layoutSrc, layoutDst, src, dst, offR, offC, false, strategy, state);
        else if (Tsrc.asSigned() != Tx_int.asSigned())
            convert(src, Tsrc, Tx_int, strategy, state);

        if (xo2D) {
            gemmDequantizeOperation(doA, Tx_int, Txo_int, BinaryOp::Sub, layoutDst, oLayout, dst, oRegs, h, kab_load, kq_load, problem, state);
            convert(dst, Tx_int, Tdst, strategy, state);
        }

        if (xs2D)
            gemmDequantizeOperation(doA, Tdst, Txs_int, BinaryOp::Mul, layoutDst, sLayout, dst, sRegs, h, kab_load, kq_load, problem, state);
    }

    if (ms < md || ns < nd) {
        copyRegisters(Tdst, Tdst, layoutDst, layoutDst0, dst, dst0, offR0, offC0, false, strategy, state);
        safeReleaseRanges(dst, state);
    }
}

GEMMSTONE_NAMESPACE_END
