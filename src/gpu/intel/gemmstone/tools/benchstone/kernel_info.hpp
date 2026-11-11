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

#ifndef GEMMSTONE_TOOLS_BENCHSTONE_KERNEL_INFO_HPP
#define GEMMSTONE_TOOLS_BENCHSTONE_KERNEL_INFO_HPP

#include <array>
#include <cmath>

#include "gemmstone/driver_info.hpp"
#include "gemmstone/type.hpp"

using gemmstone::Type;
using gemmstone::LoopM;
using gemmstone::LoopN;
using gemmstone::LoopK;
using gemmstone::charToType;

struct Level3ProblemInfo
{
    char aOffset, bOffset;
    bool cOffset, cBias;
    bool sumA, sumB;
    bool scaleA, scaleB;
    std::array<int, 2> aqGroup, bqGroup;
    char batch;

    char precisionA, precisionB, precisionC, precisionS;
    char precisionAInt, precisionBInt, precisionCInt;
    char precisionAO, precisionBO;
    char precisionAScale, precisionBScale;
    int A_components, B_components;
    char layoutA, layoutB, layoutC;
    int packSizeA, packSizeB, packSizeC;
    int crosspackA, crosspackB, crosspackC;
    int A_tileR, A_tileC, B_tileR, B_tileC, C_tileR, C_tileC;
    int alignA, alignB, alignC;
    double alpha, beta;

    bool readCheck, strongReadCheck;

    Type typeA()                const { return charToType(precisionA);    }
    Type typeAInt()             const { return charToType(precisionAInt); }
    Type typeB()                const { return charToType(precisionB);    }
    Type typeBInt()             const { return charToType(precisionBInt); }
    Type typeC()                const { return charToType(precisionC);    }
    Type typeCInt()             const { return charToType(precisionCInt); }
    Type typeS()                const { return charToType(precisionS);    }
    Type typeAs()               const { return scaleA ? charToType(precisionAScale) : Type(Type::invalid); }
    Type typeBs()               const { return scaleB ? charToType(precisionBScale) : Type(Type::invalid); }
    Type typeAO()               const { return aOffset ? charToType(precisionAO) : Type(Type::invalid); }
    Type typeBO()               const { return bOffset ? charToType(precisionBO) : Type(Type::invalid); }
    Type typeAg()               const { return typeA().isInteger() ? Type::s32 : Type::f32; }
    Type typeBg()               const { return typeB().isInteger() ? Type::s32 : Type::f32; }
    constexpr bool copyA()      const { return (layoutA == 'A'); }
    constexpr bool copyB()      const { return (layoutB == 'B'); }
    constexpr bool copyC()      const { return (layoutC == 'A' || layoutC == 'B'); }
    bool fixedAlpha()           const { return std::isfinite(alpha); }
    bool fixedBeta()            const { return std::isfinite(beta); }

    bool aOffsetPtr()       const { return aOffset && aOffset != 's'; }
    bool bOffsetPtr()       const { return bOffset && bOffset != 's'; }
    bool canCMatrixOffset() const { return cBias; }
};

struct Level3KernelInfo : public Level3ProblemInfo, public gemmstone::CommonDriverInfo
{
    bool systolic = false;
    bool tracking = false;
    bool localA = false;
    bool localB = false;
    bool localC = false;
    bool a2D = false, b2D = false, c2D = false;
    bool relaxedAccumulation = false;

    bool needsAGroupSums() const { return needsSeparateASums() && (bOffset == 'g' || scaleB); }   // Not quite the right check but covers
    bool needsBGroupSums() const { return needsSeparateBSums() && (aOffset == 'g' || scaleA); }   //   supported use-cases
};

struct KernelInfo
{
    KernelInfo() = default;
    KernelInfo(const std::string& info_str);

    std::vector<Level3KernelInfo> l3;

    bool isL3()   const { return !l3.empty(); }

    const char *kernelName() const {
        return "gemm_kernel";
    }

    int mbAlign() const { return xbAlign(LoopM); }
    int nbAlign() const { return xbAlign(LoopN); }
    int kbAlign() const {
        auto kb = xbAlign(LoopK);

        return kb;
    }

    int xbAlign(gemmstone::LoopType loop) const {
        auto xb = 1;

        for (auto &kinfo: l3)
            xb = gemmstone::lcm(xb, kinfo.unroll[loop]);

        return xb;
    }

    bool kRemHandling() const {
        bool krh = true;
        for (auto &kinfo: l3)
            krh &= kinfo.kRemainderHandling();
        return krh;
    }

    std::array<int, 3> getAddrOffsets() const {
        auto typeSize = [](char type) -> int {
            return charToType(type).paddedSize();
        };
        auto elementsPerByte = [](char type) -> int {
            return charToType(type).perByte();
        };
        std::array<int, 3> ret = {1, 1, 1};
        if (isL3()) {
            auto &l3info = l3[0];
            ret[0] = (l3info.alignA / typeSize(l3info.precisionA)) * int(!l3info.copyA());
            ret[1] = (l3info.alignB / typeSize(l3info.precisionB)) * int(!l3info.copyB());
            ret[2] = (l3info.alignC / typeSize(l3info.precisionC)) * int(!l3info.copyC());
            ret[0] *= elementsPerByte(l3info.precisionA);
            ret[1] *= elementsPerByte(l3info.precisionB);
            ret[2] *= elementsPerByte(l3info.precisionC);
        }
        return ret;
    }

    bool canCOffset()                      const { return isL3() && (l3[0].cOffset || l3[0].cBias); }
    bool canCMatrixOffset()                const { return isL3() && l3[0].canCMatrixOffset(); }
    bool readCheck()                       const { return l3[0].readCheck; }
    bool strongReadCheck()                 const { return l3[0].strongReadCheck; }
    int  grfCount()                        const { return l3[0].grfCount; }
    bool isBatched()                       const { return isL3() && l3[0].batch; }
};

#endif
