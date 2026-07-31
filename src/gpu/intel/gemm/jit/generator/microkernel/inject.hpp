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

#ifndef GEMMSTONE_GENERATOR_MICROKERNEL_INJECT_HPP
#define GEMMSTONE_GENERATOR_MICROKERNEL_INJECT_HPP

#include <algorithm>
#include <stdexcept>

#include "gemmstone/microkernel/package.hpp"
#include "ngen.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

// Largest GRF count addressable by a host kernel.
constexpr int maxGRFs = 256;

// Upper bounds on the architectural registers that can be preserved.
constexpr int maxAccs = 8;
constexpr int maxFlags = 4;

// Register state touched by a block of microkernel code, as determined by
// decoding its instructions.
struct CodeAnalysis {
    ClobberSet clobbers;            // GRFs written
    uint32_t acc = 0;               // Accumulator registers written
    uint32_t flag = 0;              // Flag registers written
    int maxAcc = -1;                // Highest accumulator register written
    bool systolic = false;          // Uses the systolic array
    bool unpreservableARF = false;  // Wrote an ARF with no save/restore sequence
    bool uncertainClobbers = false; // Wrote a GRF the decoder could not identify

    // Number of GRFs needed to hold saved ARF values. Accumulators are GRF
    // sized; all flag registers pack into a single GRF.
    int saveRegs() const {
        int n = (flag != 0);
        for (uint32_t m = acc; m; m &= m - 1)
            n++;
        return n;
    }

    // One past the highest GRF written.
    int grfsUsed() const {
        for (int i = int(clobbers.size()) - 1; i >= 0; i--)
            if (clobbers[i]) return i + 1;
        return 0;
    }
};

// Decodes `bytes` bytes of machine code for `hw`, accumulating register usage
// into `analysis`. GRF writes that the decoder cannot resolve are accepted if
// already marked in analysis.clobbers, and otherwise set uncertainClobbers, so
// callers should seed analysis.clobbers with any externally known clobbers.
void analyzeCode(CodeAnalysis &analysis, ngen::HW hw, const void *code, size_t bytes);


// Injects the code needed to splice a microkernel into an opaque host kernel
// around a microkernel body:
//
//  - Synchronization on entry and exit, as the host kernel's dependencies are
//    unknown to the microkernel and vice versa.
//  - Save/restore of the accumulator and flag registers the body writes.
//    Unlike GRFs, architectural registers cannot be declared as clobbers to
//    the host compiler, so a microkernel has to preserve them itself.
//
// Microkernel providers generate their body through inject() and need take no
// further action; the registers to preserve are determined by decoding the
// generated body, so nothing has to be declared by hand.
//
// The body is generated into a separate instruction stream so that its
// register usage can be decoded before the surrounding code is emitted. As the
// entire sequence is assembled before SWSB analysis runs, dependencies between
// the body and the save/restore code are resolved automatically, and no
// synchronization beyond the entry/exit pair is needed.
//
// Generator classes must grant Injector friendship, as it needs access to
// ngen's instruction stream handling.
template <ngen::HW hw, typename Generator>
class Injector {
public:
    explicit Injector(Generator &generator) : g(generator) {}

    // Generates `body`, bracketed by the preservation code it requires.
    // `knownClobbers` supplies (and receives) the GRFs clobbered by the body
    // that cannot be determined by decoding it.
    template <typename Body>
    void inject(ClobberSet &knownClobbers, Body &&body);

protected:
    Generator &g;

    void syncall();
    void preserve(const CodeAnalysis &analysis, int base, bool save);
};

template <ngen::HW hw, typename Generator>
template <typename Body>
void Injector<hw, Generator>::inject(ClobberSet &knownClobbers, Body &&body)
{
    g.pushStream();
    try {
        body();
    } catch (...) {
        g.discardStream();
        throw;
    }

    CodeAnalysis analysis;
    analysis.clobbers = knownClobbers;
    analyzeCode(analysis, hw, g.streamData(), g.streamLength());

    auto *bodyStream = g.popStream();

    // The save area lives above the microkernel's own register usage, clear of
    // the registers holding host kernel arguments on entry.
    int reserved = (ngen::GRF::bytes(hw) >= 64) ? 10 : 16;
    int base = std::max(analysis.grfsUsed(), reserved);
    int saveRegs = analysis.saveRegs();

    if (analysis.unpreservableARF)
        throw std::runtime_error("Microkernel writes architectural registers that cannot be preserved");
    if (base + saveRegs > maxGRFs)
        throw std::runtime_error("Microkernel does not leave enough registers to preserve host kernel state");

    g.setDefaultNoMask();
    g.setDefaultAutoSWSB();

    syncall();
    preserve(analysis, base, true);
    g.appendStream(bodyStream);
    delete bodyStream;
    preserve(analysis, base, false);
    syncall();

    knownClobbers.add(base, saveRegs);
}

// Synchronize on all pipes and OOO operations.
template <ngen::HW hw, typename Generator>
void Injector<hw, Generator>::syncall()
{
    using namespace ngen;
    if (hw == HW::Gen12LP)
        g.sync.allwr(SWSB(1));
    else if (hw >= HW::XeHP)
        g.sync.allwr(SWSB<AllPipes>(1));
}

// Emit the code saving (save = true) or recovering (save = false) the
// architectural registers described by `analysis`, using GRFs starting at
// `base`.
template <ngen::HW hw, typename Generator>
void Injector<hw, Generator>::preserve(const CodeAnalysis &analysis, int base, bool save)
{
    using namespace ngen;

    int simd = GRF::bytes(hw) >> 2;
    int reg = base;

    for (int i = 0; i < maxAccs; i++) {
        if (!(analysis.acc & (1u << i))) continue;
        auto mem = GRF(reg++).ud();
        auto acc = AccumulatorRegister(i).ud();
        save ? g.mov(simd, mem, acc) : g.mov(simd, acc, mem);
    }

    if (analysis.flag) {
        auto mem = GRF(reg++).ud();
        for (int i = 0; i < maxFlags; i++) {
            if (!(analysis.flag & (1u << i))) continue;
            auto flag = FlagRegister(i);
            save ? g.mov(1, mem[i], flag) : g.mov(1, flag, mem[i]);
        }
    }
}

}
GEMMSTONE_NAMESPACE_END

#endif /* header guard */
