/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <algorithm>

#include "inject.hpp"
#include "gemmstone/microkernel/package.hpp"
#include "ngen_decoder.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

using namespace ngen;

namespace {

// Records a write to an architectural register.
void addARFWrite(CodeAnalysis &analysis, const DependencyRegion &region, HW hw)
{
    // DependencyRegion truncates ARF bases to 8 bits, which aliases the scalar
    // register onto the (unused on Xe3+) sp encoding. Undo that here.
    auto type = normalizeARFType(static_cast<ARFType>(region.base >> 4), hw);
    int idx = region.base & 0xF;
    int len = std::max<int>(region.size, 1);

    switch (type) {
        case ARFType::acc:
            if (idx + len > maxAccs) { analysis.unpreservableARF = true; break; }
            for (int i = idx; i < idx + len; i++) {
                analysis.acc |= (1u << i);
                analysis.maxAcc = std::max(analysis.maxAcc, i);
            }
            break;
        case ARFType::f:
            if (idx + len > maxFlags) { analysis.unpreservableARF = true; break; }
            for (int i = idx; i < idx + len; i++)
                analysis.flag |= (1u << i);
            break;
        default:
            // Address and scalar registers have no general save/restore
            // sequence, so microkernels writing them cannot be preserved.
            analysis.unpreservableARF = true;
            break;
    }
}

} /* anonymous namespace */

void analyzeCode(CodeAnalysis &analysis, HW hw, const void *code, size_t bytes)
{
    Decoder decoder(hw, static_cast<const uint8_t *>(code), bytes);

    for (; !decoder.done(); decoder.advance()) {
        // Check for systolic usage.
        auto op = decoder.opcode();
        analysis.systolic |= (op == Opcode::dpas || op == Opcode::dpasw);

        // Flag registers written through a conditional modifier are not
        // reported as destination operands; collect them separately.
        DependencyRegion cmodRegion;
        if (decoder.getCModRegion(cmodRegion) && cmodRegion.rf == RegFileARF)
            addARFWrite(analysis, cmodRegion, hw);

        // Get destination region and add to clobbers. This is indeterminate for
        // indirect or variable sized destinations. In this case, rely on the
        // clobbers the caller seeded.
        DependencyRegion dstRegion;
        if (!decoder.getOperandRegion(dstRegion, -1)) continue;

        if (dstRegion.rf == RegFileGRF) {
            if (dstRegion.unspecified
                && !(dstRegion.isValid() && analysis.clobbers[dstRegion.base])) {
                    analysis.uncertainClobbers = true;
            } else
                for (int j = 0; j < dstRegion.size; j++)
                    analysis.clobbers[dstRegion.base + j] = true;
        }

        if (dstRegion.rf == RegFileARF)
            addARFWrite(analysis, dstRegion, hw);
    }
}

Package::Status Package::finalize(const ClobberSet &knownClobbers) {
    using namespace ngen;

    auto &status = this->status;
    status = Status::Success;

    auto product = npack::decodeHWIPVersion(gmdidCompat);
    auto hw = getCore(product.family);

    if (hw == HW::Unknown) {
        status = Status::UnsupportedHW;
        return status;
    }

    CodeAnalysis analysis;
    analysis.clobbers = knownClobbers;
    analyzeCode(analysis, hw, binary.data(), binary.size());

    systolic |= analysis.systolic;

    // Architectural registers cannot be communicated to the host kernel as
    // clobbers; those that cannot be saved and restored by the microkernel
    // itself leave the host kernel's state uncertain.
    if (analysis.uncertainClobbers || analysis.unpreservableARF)
        status = Status::UncertainClobbers;

    // Group clobber array into consecutive ranges.
    clobbers.clear();

    int regBytes = GRF::bytes(hw);
    int base = 0, len = 0;
    for (int j = 0; j < int(analysis.clobbers.size()); j++) {
        if (analysis.clobbers[j]) {
            if (len > 0)
                len++;
            else
                base = j, len = 1;
        } else if (len > 0) {
            clobbers.emplace_back(
                    RegisterRange(base * regBytes, len * regBytes));
            len = 0;
        }
    }
    if (len > 0)
        clobbers.emplace_back(RegisterRange(base * regBytes, len * regBytes));

    // Capture GRF usage from clobbers and arguments.
    uint32_t last = 0;
    if (!clobbers.empty()) {
        auto &final = clobbers.back();
        last = final.boffset + final.blen;
    }
    for (const auto &argument : arguments)
        for (auto &range : argument.location)
            last = std::max(last, range.boffset + range.blen);

    grfMin = (last + regBytes - 1) / regBytes;

    // The host kernel's GRF mode determines how many accumulators exist; make
    // sure it provides every accumulator the microkernel uses.
    if (AccumulatorRegister::count(hw, grfMin) <= analysis.maxAcc)
        grfMin = maxGRFs;

    // Generate LUID from hash of kernel. Later, the cataloguer can update it in case of collisions.
    uint32_t luid = 0;
    uint32_t multiplier = 1357;

    auto *u32ptr = (const uint32_t *)binary.data();
    for (size_t i = 0; i < (binary.size() >> 2); i++) {
        luid ^= u32ptr[i] * multiplier;
        multiplier += 2;
        luid = (luid << 3) | (luid >> 29);
    }

    this->luid = luid;

    return status;
}

}
GEMMSTONE_NAMESPACE_END
