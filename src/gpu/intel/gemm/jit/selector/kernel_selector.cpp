/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "common/verbose.hpp"

#include <cassert>
#include <cctype>
#include <cstring>
#include <vector>

GEMMSTONE_NAMESPACE_START

inline bool layoutMatch(const char *lref, const char *lpattern)
{
    return (lref[0] == lpattern[0]);        // This is a sufficient check for now.
}

inline bool precisionMatch(const char *pref, const char *ppattern, bool ignoreCase)
{
    char mask = ignoreCase ? ~0x20 : ~0;
    auto match = [mask](char c1, char c2) { return (c1 & mask) == (c2 & mask); };

    bool ok = false;
    ok = ok || (ppattern[0] == '?');
    ok = ok || match(pref[0], ppattern[0]);
    ok = ok || (ppattern[0] == '[' && match(pref[0], ppattern[1]));
    if (ok && pref[0] == '[') {
        ok = ok && match(pref[1], ppattern[1]) && match(pref[2], ppattern[2]);
        for (int i = 3; pref[i] != '\0'; i++) {
            if (pref[i] != ppattern[i]) { ok = false; break; }
        }
    }
    return ok;
}

inline bool precisionMinimumMatch(const char *pref, char pmin)
{
    uint8_t sizeTable[0x20] = {
//         A  B  C  D  E  F  G  H  I  J  K  L  M  N  O
        0, 0, 2, 8, 8, 0, 0, 0, 2, 4, 4, 4, 0, 0, 0, 1,
//      P  Q  R  S  T  U  V  W  X  Y  Z
        0, 1, 0, 4, 4, 8, 0, 2, 0, 0, 16, 0, 0, 0, 0, 0
    };

    return (sizeTable[pref[0] & 0x1F] >= sizeTable[pmin & 0x1F]);
}

inline bool alignmentMatch(int aref, int apattern)
{
    if (aref == 0) aref = 1;
    return (apattern % aref == 0);
}

inline bool tagMatch(const char *tref, const char *tpattern)
{
    for (auto c = tref; *c; c++) {
        // Lowercase tags -> must not match pattern
        // Uppercase tags -> must match pattern
        int cu = *c & ~0x20;     // toupper(c)
        bool match = (std::strchr(tpattern, cu) != nullptr);
        bool wantMatch = (*c & 0x20) == 0;
        if (match != wantMatch)
            return false;
    }
    return true;
}

inline bool strategyMatch(const CommonDriverInfo &info, const StrategyRequirement &req)
{
    int actual = 0;
    switch (req.param) {
        case StrategyRequirement::UnrollM:  actual = info.unroll[LoopM]; break;
        case StrategyRequirement::UnrollN:  actual = info.unroll[LoopN]; break;
        case StrategyRequirement::WGTileM:  actual = info.wgTile(LoopM); break;
        case StrategyRequirement::WGTileN:  actual = info.wgTile(LoopN); break;
        case StrategyRequirement::WGTileMN: actual = info.wgTile(LoopM) * info.wgTile(LoopN); break;
        case StrategyRequirement::WGM:      actual = info.wg[LoopM]; break;
        case StrategyRequirement::WGN:      actual = info.wg[LoopN]; break;
        case StrategyRequirement::WGK:      actual = info.wg[LoopK]; break;
        case StrategyRequirement::WG:       actual = info.wg[LoopM] * info.wg[LoopN] * info.wg[LoopK]; break;
        default: return false;
    }

    switch (req.relation) {
        case StrategyRequirement::Equals:  return (actual == req.value);
        case StrategyRequirement::AtLeast: return (actual >= req.value);
        case StrategyRequirement::AtMost:  return (actual <= req.value);
        default: return false;
    }
}

bool matches(const kcatalog::Entry &e, const MatchParams &p)
{
    bool ok = true;

    if (e.restrictions.steppingMin >= 0)
        ok = ok && (p.stepping >= e.restrictions.steppingMin);
    if (e.restrictions.steppingMax >= 0)
        ok = ok && (p.stepping < e.restrictions.steppingMax);
    ok = ok && layoutMatch(e.selector.layouts[0], p.selector.layouts[0]);
    ok = ok && layoutMatch(e.selector.layouts[1], p.selector.layouts[1]);
    ok = ok && layoutMatch(e.selector.layouts[2], p.selector.layouts[2]);
    ok = ok && precisionMatch(e.selector.precisions[2], p.selector.precisions[2], p.ignoreCase);
    if (p.precisionCExt)
        ok = ok && precisionMinimumMatch(e.selector.precisions[2], p.precisionCExt);
    for (int i = 0; i < 3; i++)
        ok = ok && alignmentMatch(e.restrictions.alignment[i], p.alignment[i]);
    ok = ok && tagMatch(e.restrictions.tags, p.tags);

    if (!p.ignoreSizes) {
        int64_t mnk[3] = {p.sizes.m, p.sizes.n, p.sizes.k};
        for (int i = 0; i < 3; i++) {
            if (e.restrictions.allowedSizesMin[i] >= 0)
                ok = ok && (mnk[i] >= e.restrictions.allowedSizesMin[i]);
            if (e.restrictions.allowedSizesMax[i] >= 0)
                ok = ok && (mnk[i] <= e.restrictions.allowedSizesMax[i]);
        }
    }

    if (p.reqNUnroll32){
        ok = ok && (e.driverInfo.unroll[LoopM] % 32 == 0);
    }

    for (int i = 0; i < p.nExtraReqs; i++)
        ok = ok && strategyMatch(e.driverInfo, p.extraReqs[i]);

    // Should already be matched.
    ok = ok && (e.selector.hw == p.selector.hw);
    ok = ok && precisionMatch(e.selector.precisions[0], p.selector.precisions[0], p.ignoreCase);
    ok = ok && precisionMatch(e.selector.precisions[1], p.selector.precisions[1], p.ignoreCase);

    return ok;
}

// Check whether one set of A/B alignments (alignA1, alignB1) is strictly less aligned than
//  another (alignA2, alignB2).
bool lessAligned(int alignA1, int alignB1, int alignA2, int alignB2)
{
    alignA1 = std::max(alignA1, 4);
    alignA2 = std::max(alignA2, 4);
    alignB1 = std::max(alignB1, 4);
    alignB2 = std::max(alignB2, 4);
    return (alignA1 <= alignA2) && (alignB1 <= alignB2) && (alignA1 + alignB1 < alignB1 + alignB2);
}

// Inner kernel selection logic.
// Choose the best entry, if any, matching one of the given patterns.
const std::vector<const kcatalog::Entry *> getEntries(const kcatalog::Catalog &catalog, int npatterns, const MatchParams *patterns, const EvaluateParams &eparams, EvaluateAuxOutput &aux,  SelectionObserver * observer)
{
    std::vector<const kcatalog::Entry *> entries;
    // TODO: omit evaluation if only one match, if aux output not needed.
    for (int ipattern = 0; ipattern < npatterns; ipattern++) {
        for (auto it = match(catalog, patterns[ipattern]); it; it++) {
             // Late tag checking. If late tags do not match, we skip entry.
             if (tagMatch(it->restrictions.tags, patterns[ipattern].lateTags))
                 entries.push_back(&*it);
        }
    }
    auto less = [&](const kcatalog::Entry * lhs, const kcatalog::Entry * rhs){
                      EvaluateAuxOutput thisAux;
                      bool lhsFallback = (lhs->restrictions.tags[0] == kcatalog::ReqAlignFallback);
                      int  lhsAlignA = std::max(lhs->restrictions.alignment[0], 4);
                      int  lhsAlignB = std::max(lhs->restrictions.alignment[1], 4);
                      bool rhsFallback = (rhs->restrictions.tags[0] == kcatalog::ReqAlignFallback);
                      int  rhsAlignA = std::max(rhs->restrictions.alignment[0], 4);
                      int  rhsAlignB = std::max(rhs->restrictions.alignment[1], 4);
                      if (rhsFallback && lessAligned(rhsAlignA, rhsAlignB, lhsAlignA, lhsAlignB)) return true;
                      if (lhsFallback && lessAligned(lhsAlignA, lhsAlignB, rhsAlignA, rhsAlignB)) return false;
                      double lhs_score = evaluate(*lhs, eparams, thisAux);
                      double rhs_score = evaluate(*rhs, eparams, thisAux);
                      if (lhs_score < rhs_score) return true;
                      if (lhs_score > rhs_score) return false;
                      return (lhs < rhs);
    };
    std::sort(entries.begin(), entries.end(), less);
    if (entries.size() > 0)
	    evaluate(*entries[0], eparams, aux);

    return entries;
}

// User-facing kernel selection logic.
// Includes architecture and data type fallbacks.
const std::vector<const kcatalog::Entry *> select(const kcatalog::Catalog &catalog, const MatchParams &pattern, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver *observer)
{
    return select(catalog, 1, &pattern, eparams, aux, observer);
}

const std::vector<const kcatalog::Entry *> select(const kcatalog::Catalog &catalog, int npatterns, const MatchParams *patterns, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver *observer)
{
    using namespace kcatalog;

    std::vector<const kcatalog::Entry *> entries;
    if (npatterns == 0 || !patterns)
        return entries;

    auto result = getEntries(catalog, npatterns, patterns, eparams, aux, observer);

    // Architecture fallback loop.
    bool first = true;
    auto hw = patterns[0].selector.hw;
    do {
        std::vector<MatchParams> modPatterns;
        modPatterns.reserve(npatterns);
        for (int i = 0; i < npatterns; i++) {
            modPatterns.emplace_back(patterns[i]);
            modPatterns.back().selector.hw = hw;
        }

        // Type fallback loop.
        while (true) {
            if (!first) {
                auto entries =  getEntries(catalog, npatterns, modPatterns.data(), eparams, aux, observer);
                result.insert(result.end(), entries.begin(), entries.end());
            }
            first = false;

            /* Try capital types */
            auto hasLowercase = [](kcatalog::string &str) {
                bool lc = false;
                for (auto c = str; *c; c++)
                    lc |= (*c >= 'a' && *c <= 'z');
                return lc;
            };

            bool changed = false;
            for (auto &p: modPatterns) {
                if (p.ignoreCase) continue;
                p.ignoreCase = true;
                for (int d = 0; d < 3; d++)
                    changed |= hasLowercase(p.selector.precisions[d]);
            }
            if (changed) continue;

            /* Specific pattern modifications */
            auto iequal = [](kcatalog::string &str, char c) {
                return str[0] && ((str[0] & ~0x20) == (c & ~0x20)) && (str[1] == '\0');
            };

            for (auto &p: modPatterns) {
                auto match = [&](const char *precisions) {
                    for (int i = 0; i < 3; i++)
                        if (precisions[i] && !iequal(p.selector.precisions[i], precisions[i]))
                            return false;
                    return true;
                };

                if (match("FO"))
                    p.selector.precisions[0] = "[FO]";
                else if (match("BB"))
                    p.selector.precisions[0] = p.selector.precisions[1] = "H";
                else continue;

                changed = true;
            }
            if (changed) continue;

            break;
        }

        // Architecture fallbacks, e.g. Xe2 inherits XeHPC strategies.
        switch (hw) {
            case HWTagXe2:  hw = HWTagXeHPC; break;
            case HWTagXe3:  hw = HWTagXe2; break;
            default:        hw = 0; break;
        }
    } while (hw);

    return result;
}

template <bool upper>
const kcatalog::Entry *upper_lower_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector)
{
    int n = catalog.entryCount;
    const kcatalog::Entry *cur = catalog.entries;

    while (n > 0) {
        auto half = n >> 1;
        auto mid = cur + half;
        if (upper ? (*mid <= selector) : (*mid < selector)) {
            cur = mid + 1;
            n = n - half - 1;
        } else
            n = half;
    }

    return cur;
}

const kcatalog::Entry *lower_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    return upper_lower_bound<false>(catalog, selector);
}

const kcatalog::Entry *upper_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    return upper_lower_bound<true>(catalog, selector);
}


MatchParamsBase::MatchParamsBase(ngen::HW hw, bool systolicAvailable, bool isIntegrated, const GEMMProblem &problem_)
{
    using namespace kcatalog;

    auto problem = problem_;
    reqNUnroll32 = problem.hasCMXScale();

    switch (hw) {
        default: assert(!"Unknown architecture");
        case ngen::HW::Gen12LP: selector.hw = kcatalog::HWTagGen12LP; break;
        case ngen::HW::XeHPG:   selector.hw = kcatalog::HWTagXeHPG;   break;
        case ngen::HW::XeHPC:   selector.hw = kcatalog::HWTagXeHPC;   break;
        case ngen::HW::Xe2:     selector.hw = kcatalog::HWTagXe2;     break;
        case ngen::HW::Xe3:     selector.hw = kcatalog::HWTagXe3;   break;
    }

    auto &C = problem.C;
    auto equivCLayout = C.layout;
    if (isPacked(equivCLayout)) {
        bool colMajor = (C.layout == MatrixLayout::Pc) ^ (C.crosspack * problem.Tc > 4);
        equivCLayout = (colMajor ? MatrixLayout::N : MatrixLayout::T);
    }

    auto makeABConvert = [](Type T, Type T_ext, bool mixed_fp, char *out) {
        if ((mixed_fp && T_ext.isSubsetOf(T)) || (T == T_ext))
            out[0] = precisionChar(T_ext);
        else {
            out[0] = '[';
            out[1] = precisionChar(T_ext);
            out[2] = precisionChar(T);
            out[3] = ']';
        }
    };

    selector.kernelType = "gemm";

    std::fill(temp.begin(), temp.end(), '\0');

    const bool mixed_fp = problem.Ta_ext.isFP() != problem.Tb_ext.isFP();
    makeABConvert(problem.Ta, problem.Ta_ext, mixed_fp, &temp[0]);
    makeABConvert(problem.Tb, problem.Tb_ext, mixed_fp, &temp[5]);
    temp[10] = precisionChar(problem.Tc);
    temp[12] = layoutChar(problem.A.layout);
    temp[14] = layoutChar(problem.B.layout);
    temp[16] = layoutChar(equivCLayout);

    selector.precisions[0] = &temp[0];
    selector.precisions[1] = &temp[5];
    selector.precisions[2] = &temp[10];
    selector.layouts[0] = &temp[12];
    selector.layouts[1] = &temp[14];
    selector.layouts[2] = &temp[16];

    precisionCExt = precisionChar(problem.Tc_ext);

    alignment[0] = problem.A.alignment;
    alignment[1] = problem.B.alignment;
    alignment[2] = problem.C.alignment;

    char *tagPtr = &temp[18];
    lateTags = tagPtr;

    // Late-only tags. Don't choose lower-performing kernels
    //  just to fuse reductions. Instead do reductions in a separate kernel.
    if (problem.sumA)
        *tagPtr++ = ReqSumA;
    if (problem.sumB)
        *tagPtr++ = ReqSumB;

    tags = tagPtr;

    if (systolicAvailable)
        *tagPtr++ = ReqSystolic;

    if (isIntegrated) *tagPtr++ = ReqIntegrated;

    if (problem.batch != BatchMode::None) {
        *tagPtr++ = ReqBatch;
        if (problem.batchDims > 1)
            *tagPtr++ = ReqBatchMultiDim;
    }

    if (problem.aOffset != ABOffset::None || problem.bOffset != ABOffset::None){
        VDEBUGINFO(4, primitive, kernel_selector, "MY: *tagPtr++ = ReqABOffset");
        *tagPtr++ = ReqABOffset;
    }


    // ???????????????? How it is if xxxPtrDims == -1 ???????????????
    if (problem.aoPtrDims > 0 || problem.boPtrDims > 0){
        VDEBUGINFO(4, primitive, kernel_selector, "MY: *tagPtr++ = ReqOffsetMultiDim");
        *tagPtr++ = ReqOffsetMultiDim;
    }
    // ???????????????? How it is if xxxPtrDims == -1 ???????????????

    problem.autoTypeConversions(hw, systolicAvailable);
    if (problem.needsASums() && !problem.sumA) *tagPtr++ = ReqSumA;
    if (problem.needsBSums() && !problem.sumB) *tagPtr++ = ReqSumB;

    if (hw == ngen::HW::Xe2)
        *tagPtr++ = ReqXe2Block2D;
    if (hw == ngen::HW::Xe3)
        *tagPtr++ = ReqXe2Block2D;

    sizes.batch = sizes.m = sizes.n = sizes.k = 0;
}

void StrategyRequirement::transpose()
{
    switch (param) {
        case UnrollM: param = UnrollN; break;
        case UnrollN: param = UnrollM; break;
        case WGTileM: param = WGTileN; break;
        case WGTileN: param = WGTileM; break;
        case WGM:     param = WGN;     break;
        case WGN:     param = WGM;     break;
        default:                       break;
    }
}

GEMMSTONE_NAMESPACE_END
