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

/*
 * Do not #include this file directly; ngen uses it internally.
 */

#ifndef NGEN_AUTO_SWSB_HPP
#define NGEN_AUTO_SWSB_HPP

#if defined(NGEN_DEBUG) || defined(NGEN_DEBUG_PROPAGATE) || defined(NGEN_DEBUG_BB)
#include <iomanip>
#include <iostream>
#endif

#include <atomic>
#include <limits>
#include <list>
#include <map>

#include "ngen_core.hpp"

namespace NGEN_NAMESPACE {
namespace autoswsb {

/*******************/
/* Data structures */
/*******************/

typedef uint16_t PipeMask;
enum PipeMasks : PipeMask {
    PipeMaskNone = 0,
    PipeMaskA = 1,      // All in-order pipes
    PipeMaskF = 2,
    PipeMaskI = 4,
    PipeMaskL = 8,
    PipeMaskM = 0x10,
    PipeMaskS = 0x20,
    PipeMaskC = 0x40,   // All instructions (in-order/out-of-order).
    PipeMaskO = 0x80,   // All out-of-order pipes. Not a valid GeneralizedPipe.
};

enum PipeBits {
    PipeBitA = 0,
    PipeBitF = 1,
    PipeBitI = 2,
    PipeBitL = 3,
    PipeBitM = 4,
    PipeBitS = 5,
    PipeBitC = 6,
    PipeBitO = 7,
};
static constexpr int NPipes = 7;
static constexpr int NSWSBPipes = NPipes - 1;

static inline PipeMask toMask(Pipe pipe)   { return (1 << (static_cast<unsigned>(pipe) - 1)); }
static inline Pipe fromMask(PipeMask mask) { return mask ? static_cast<Pipe>(1 + utils::log2(mask)) : Pipe::Default; }

typedef uint8_t DestinationMask;
enum {
    DestNone = 0,
    DestNextIP = 1,
    DestJIP = 2,
    DestUIP = 4,
    DestUnknown = 8
};

class GeneralizedPipe {
    uint16_t v;

public:
    static constexpr uint16_t vInOrder  = 0x000;
    static constexpr uint16_t vSend     = 0x200;        // OR'ed with SFID
    static constexpr uint16_t vSystolic = 0x400;
    static constexpr uint16_t vMath     = 0x600;
    static constexpr uint16_t vTypeMask = 0x600;

    GeneralizedPipe(uint16_t v_, int dummy) : v{v_} {}

public:
    GeneralizedPipe()                    : v{uint16_t(0)} {}
    GeneralizedPipe(PipeMask pipe)       : v{uint16_t(vInOrder | pipe)} {}
    GeneralizedPipe(SharedFunction sfid) : v{uint16_t(vSend | static_cast<uint8_t>(sfid))} {}

    static GeneralizedPipe Systolic() { return GeneralizedPipe(vSystolic, 0); }
    static GeneralizedPipe Math()     { return GeneralizedPipe(vMath, 0); }

    bool operator==(GeneralizedPipe other) const { return v == other.v; }
    bool operator!=(GeneralizedPipe other) const { return v != other.v; }

    bool inOrder() const { return ((v & vTypeMask) == vInOrder) && (v != PipeMaskNone); }
    uint16_t type() const { return v & vTypeMask; }
    PipeMask inOrderPipe() const { return inOrder() ? (v & ~vTypeMask) : PipeMaskNone; }
    PipeBits inOrderPipeIdx() const { return (PipeBits) utils::log2(v & ~vTypeMask); }
    Pipe toPipe() const { return fromMask(inOrderPipe()); }
    inline PipeMask syncPipes(HW hw) const;

    inline unsigned sendClassXeHPC() const;

#ifdef NGEN_DEBUG
    inline void dump() const;
#endif
};

struct DependencyRegion {
    uint8_t base, size;
    uint8_t unspecified : 1;
    uint8_t checkWAW : 1;
    uint8_t rf : 2;
    HW hw;
    std::array<uint32_t, 32> masks;

    DependencyRegion() : DependencyRegion(HW::Unknown) {}
    explicit DependencyRegion(HW hw_) : base(0), size(0), unspecified{true}, checkWAW{false}, rf{RegFileGRF}, hw{hw_} {
        for (auto &m: masks) m = 0;
    }
    inline DependencyRegion(HW hw, RegisterRange r);
    inline DependencyRegion(HW hw, int esize, RegData rr);

    inline void intersect(const DependencyRegion &other);
    inline void subtract(const DependencyRegion &other);

    bool empty() const {
        if (unspecified) return false;
        if (size == 0) return true;
        for (auto m : masks)
            if (m != 0)
                return false;
        return true;
    }
    void clear()        { *this = DependencyRegion(hw); unspecified = false; checkWAW = false; rf = 0; }

#ifdef NGEN_DEBUG
    inline void dump() const;
#endif

protected:
    void makeFullMasks() {
        for (size_t i = 0; i < masks.size(); i++)
            masks[i] = (i < size) ? ~uint32_t(0) : uint32_t(0);
    }
};

template <bool consumer>
struct Dependency {
    int32_t label;                                      // Multipurpose label for use in algorithms

    // Source instruction information.
    GeneralizedPipe pipe;                               // Execution pipe for instruction
    uint16_t tokenTime;                                 // Estimated upper bound for token lifetime, in cycles.
    std::array<int32_t, NPipes> counters;               // Pipe counters, relative to start of BB.
    uint32_t inum;                                      // Instruction number.

    // (Mostly) dependency information.
    uint8_t rw : 1;                                     // True for writes, false for reads
    uint8_t swsb : 1;                                   // True for SWSB dependency consumers
    uint8_t active : 1;                                 // True if dependency is still alive
    uint8_t tokenTBD : 1;                               // True if token has not yet been assigned
    std::array<uint8_t, NSWSBPipes> dists;              // (consumers) Pipe distances for each pipe (0 = no dependency)
    uint32_t tokenMaskSrc, tokenMaskDst;                // Bitmasks of token src/dst dependencies
    DependencyRegion region;                            // GRF region covered

    Dependency() : label{0}, pipe{}, tokenTime{0},
        rw{false}, swsb{false}, active{true}, tokenTBD{false},
        tokenMaskSrc{0u}, tokenMaskDst{0u}, region{} { counters.fill(0); dists.fill(0); }

    bool operator==(const Dependency &other) {
        return !std::memcmp(this, &other, sizeof(Dependency));
    }
    bool operator!=(const Dependency *other) { return !(operator==(other)); }

    constexpr bool read() const          { return !rw; }
    constexpr bool write() const         { return rw; }
    constexpr bool hasToken() const      { return tokenMaskSrc || tokenMaskDst || tokenTBD; }
    constexpr bool assignedToken() const { return (tokenMaskSrc || tokenMaskDst) && !tokenTBD; }
    int getToken() const;

    Dependency<!consumer>& cast()   { return reinterpret_cast<Dependency<!consumer>&>(*this); }

    PipeMask checkPipes() const;
    PipeMask depPipes() const;

    PipeMask coalesceInOrder();

#ifdef NGEN_DEBUG
    inline void dump() const;
#endif
};

using Producer = Dependency<false>;
using Consumer = Dependency<true>;

template <bool consumer>
class DependencyTable {
    enum {
        ListTypeReg = 0,                    // Lists of DependencyFragments filtered by base register.
        ListTypeToken = 1,                  // Lists of DependencyFragments filtered by token.
        ListTypePipe = 2,                   // Lists of DependencyFragments filtered by (in-order) pipe.
                                            //   fragsByToken/fragsByPipe contain only one DependencyFragment per Dependency.
        NListTypes = 3
    };

    enum : uint32_t {
        none = ~uint32_t(0)                 // Special value indicating end of list.
    };

    enum : int {
        maxGRF = 256,
        grfListIdxUnspecified = maxGRF      // GRF list index for all unspecified regions.
    };

    struct DependencyFragment {
        uint32_t depID;                     // Index of main Dependency struct in array.
        uint8_t before, after;              // # of consecutive fragments associated with the same Dependency
                                            //  before and after this one.
        uint32_t prev[NListTypes];          // Previous pointers for doubly-linked lists.
        uint32_t next[NListTypes];          // Next pointers for doubly-linked lists.
    };

    std::vector<Dependency<consumer>> deps;         // List of all Dependencies (active or not)
    std::vector<DependencyFragment> frags;          // List of all DependencyFragments (active or not)
    std::array<uint32_t, 257> heads[NListTypes];    // Heads of doubly-linked lists.

    static bool isHeadLink(uint32_t id)         { return ((id & 0x80000000) != 0) && (id != none); }
    static uint32_t readHeadLink(uint32_t id)   { return id & 0x7FFFFFFF; }
    static uint32_t makeHeadLink(uint32_t idx)  { return idx | 0x80000000; }

    template <bool iconsumer> inline void findAndRemoveIntersections(int listType, int listIdx, const Dependency<iconsumer> &dep, std::vector<Dependency<consumer>> *out, bool doRemove = true);
    inline bool insertPrepare(int listType, int listIdx, Dependency<consumer> &dep, bool checkWeaker, bool checkStronger);
    inline void insertLinkedList(int listType, int listIdx, int32_t fragID);

public:
    DependencyTable() { clear(); }

    inline void clear();
    inline void reserve(int icount);
    inline bool insert(Dependency<consumer> &dep, bool checkWeaker = true, bool checkStronger = true);
    inline bool insertWeak(Dependency<consumer> &dep)                   { return insert(dep, true, false); }
    inline void insertStrong(const Dependency<consumer> &dep)           { (void) insert(const_cast<Dependency<consumer> &>(dep), false, true); }
    inline void remove(int fragID);
    template <bool iconsumer> inline void findIntersections(const Dependency<iconsumer> &dep, std::vector<Dependency<consumer>> &out);
    template <bool iconsumer> inline void findAndRemoveIntersections(const Dependency<iconsumer> &dep, std::vector<Dependency<consumer>> *out, bool doRemove = true);
    template <bool iconsumer> inline void removeIntersections(const Dependency<iconsumer> &dep);
    inline uint32_t removeByTokenMask(uint32_t mask, bool dst);
    inline bool containsToken(int token) { return heads[ListTypeToken][token] != none; }

    template <typename Func> inline void forEach(Func f)                { for (auto &entry : deps) if (entry.active) f(entry); }
    template <typename Func> inline void forEach(Func f) const          { for (auto &entry : deps) if (entry.active) f(entry); }

#ifdef NGEN_DEBUG
    inline void dump() const;
#endif
};

struct SyncInsertion {
    uint32_t inum;
    SWSBInfo swsb;
    SyncFunction fc;
    uint32_t mask;                                      // (allrd/allwr) 0 indicates no mask to be applied.
};

struct DummyMovInsertion {
    uint32_t inum;
    SWSBInfo swsb;
    uint16_t grf;
    bool constant;
    DataType dt;
};

struct BasicBlock;

struct BasicBlock {
    uint32_t id;                                            // Index
    int32_t label;                                          // Multipurpose flag for use in algorithms
    uint32_t istart, iend;                                  // Instruction range: [istart, iend)
    uint32_t directives;                                    // # of directives (pseudo-instructions) in this BB
    uint32_t n64;                                           // # of 64-bit instructions.
    std::array<uint32_t, NPipes> lengths;                   // # of instructions in each pipe in this BB
    std::vector<BasicBlock *> pred, succ;                   // List of predecessor/successor BBs
    DependencyTable<false> producers;                       // Table of dependencies produced and consumed by this BB.
    DependencyTable<true> consumers;                        //   Production table re-used for live incoming dependencies.
    DependencyTable<false> incoming;                        // Table of dependencies produced by prior BBs (temporary).
    std::vector<SyncInsertion> syncs;                       // List of sync instructions to generate.
    std::vector<DummyMovInsertion> movs;                    // List of mov instructions to generate.
    std::vector<std::array<DependencyRegion, 4>> opRegions; // Cache of instruction operand regions.
    bool enablePVCWARWA = false;                            // Enable workaround for PVC WAR bug.

    const DependencyRegion &getOperandRegion(int inum, int opNum) const {
        return opRegions[inum - istart][opNum + 1];
    }

    int32_t sizeAdjust(HW hw) const {
        return (int32_t(syncs.size() + movs.size()) - directives) * 16 - n64 * 8;
    }
};

using BasicBlockList = std::vector<BasicBlock>;

/*****************/
/* Pipe Handling */
/*****************/

// Get all pipes to track in-order dependencies on.
inline PipeMask allPipes(HW hw)
{
    PipeMask mask = PipeMaskA | PipeMaskO;
    if (hw >= HW::XeHP) mask |= PipeMaskF | PipeMaskI | PipeMaskL;
    if (hw >= HW::XeHPC) mask |= PipeMaskM;
    if (hw >= HW::Xe3) mask |= PipeMaskS;
    return mask;
}

// Get the execution data type for an instruction.
template <typename Instruction>
inline unsigned execType(const Instruction &insn)
{
    auto execDT = insn.dstTypecode();
    if (insn.src0Typecode() == 0b1011)
        execDT = 0b1011;
    return execDT;
}

// Get the execution pipe for an instruction.
template <typename Instruction>
inline GeneralizedPipe getPipe(HW hw, const Instruction &insn, bool checkOOO = true)
{
    auto op = insn.opcode();

    // Check jumps and no-ops
    if (isBranch(op) || op == Opcode::nop_gen12 || op == Opcode::sync || op == Opcode::illegal || op == Opcode::directive)
        return GeneralizedPipe();

    // Check OOO instructions.
    if (trackedByToken(hw, op, execType(insn))) {
        if (!checkOOO)
            return GeneralizedPipe();
        switch (op) {
            case Opcode::dpas:
            case Opcode::dpasw:
                return GeneralizedPipe::Systolic();
            case Opcode::math:
            default:
                return GeneralizedPipe::Math();
            case Opcode::send:
            case Opcode::sendc:
                return GeneralizedPipe(insn.sfid());
        }
    }

    if (hw >= HW::XeHPC && (op == Opcode::math))
        return PipeMaskM;

    PipeMask mask = PipeMaskNone;

    // For SWSB purposes, Gen12LP has a single in-order pipe.
    if (hw < HW::XeHP)
        return PipeMaskA;

    // Otherwise, in-order pipe determined by destination type.
    // Exception: if there are any long operands, it's a long pipe instruction.
    auto dt = insn.dstTypecode();
    unsigned lmask = (hw >= HW::XeHPC) ? 0b1011 : 0b0011;
    if ((dt & lmask) == lmask)
        mask = PipeMaskL;
    else if (dt & 8)
        mask = PipeMaskF;
    else
        mask = PipeMaskI;

    if ((hw < HW::XeHPC) && !(mask & PipeMaskL)) {
        if ((insn.src0Typecode() & lmask) == lmask)
            mask = PipeMaskL;
        else if ((insn.src1Typecode() & lmask) == lmask)
            mask = PipeMaskL;
    }

    if (hw >= HW::Xe3) {
        ARFType dstARF;
        if (insn.getARFType(dstARF, -1, hw) && dstARF == ARFType::s)
            mask = PipeMaskS;
    }

    return mask;
}

template <typename Instruction>
inline PipeMask getPipeMask(HW hw, const Instruction &insn)
{
    PipeMask pipe = getPipe(hw, insn, false).inOrderPipe();
    if (pipe != PipeMaskNone)
        pipe |= PipeMaskA;
    return pipe | PipeMaskC;
}

PipeMask GeneralizedPipe::syncPipes(HW hw) const
{
    if ((hw >= HW::XeHP) && (v & PipeMaskA))
        return allPipes(hw) & ~PipeMaskA & ~PipeMaskO;
    return (v == PipeMaskNone) ? allPipes(hw) : inOrderPipe();
}

unsigned GeneralizedPipe::sendClassXeHPC() const
{
    if (type() == vSend) switch (static_cast<SharedFunction>(v & 0xF)) {
        case SharedFunction::dcro:
        case SharedFunction::dc0:
        case SharedFunction::dc1:
        case SharedFunction::slm:
        case SharedFunction::ugm: return 1;
        default: return 2;
    }
    return 0;
}

static inline DataType dtForPipe(Pipe p)
{
    switch (p) {
        default:
        case Pipe::I: return DataType::ud;
        case Pipe::F: return DataType::f;
        case Pipe::L: return DataType::df;
    }
}

/**********************/
/* Dependency Regions */
/**********************/
DependencyRegion::DependencyRegion(HW hw_, RegisterRange r)
{
#ifdef NGEN_SAFE
    if (r.isInvalid() || (r.getLen() > int(masks.size())))
        throw invalid_region_exception();
#endif

    hw = hw_;
    unspecified = false;
    checkWAW = false;
    rf = RegFileGRF;
    base = r.getBase();
    size = r.getLen();
    makeFullMasks();
}

DependencyRegion::DependencyRegion(HW hw_, int esize, RegData rr)
{
    const auto mbits = GRF::bytes(hw_);
    const auto log2MBits = GRF::log2Bytes(hw_);

    hw = hw_;
    base = rr.getBase();
    unspecified = false;
    checkWAW = false;
    rf = rr.getRegFile();

    int hs = rr.getHS(), vs = rr.getVS();
    int nh = rr.getWidth();
    if (nh == 0) nh = 1;
    int nv = esize / nh;
    int bytes = rr.getBytes();
    int off = rr.getByteOffset();

    auto makeMask = [](int sz) -> uint64_t {
        if (sz == 64) return ~uint64_t(0);
        return (uint64_t(1) << sz) - 1;
    };

    auto compress = [&](uint64_t m) -> uint32_t {
        if (hw_ >= HW::XeHPC) {
            // Regions tracked at word granularity. OR and pack adjacent bits.
            // If any sub-word writes, need to track WAW dependencies.
            if ((m ^ (m >> 1)) & 0x5555555555555555)
                checkWAW = true;
            m = (m | (m >> 1)) & 0x5555555555555555;
            m = (m | (m >> 1)) & 0x3333333333333333;
            m = (m | (m >> 2)) & 0x0F0F0F0F0F0F0F0F;
            m = (m | (m >> 4)) & 0x00FF00FF00FF00FF;
            m = (m | (m >> 8)) & 0x0000FFFF0000FFFF;
            m |= (m >> 16);
        }
        return uint32_t(m);
    };

    if (hs == 0) nh = hs = 1;
    if (vs == 0) nv = 1;
    hs *= bytes;
    vs *= bytes;

    for (auto &m : masks)
        m = 0;

    uint64_t hmask = makeMask(bytes) * (makeMask(nh * hs) / makeMask(hs));
    for (int j = 0; j < nv; j++) {
        masks[off >> log2MBits] |= compress(hmask << (off & (mbits - 1)));
        off += vs;
    }

    size = ((off - vs) >> log2MBits) + 1;
}

void DependencyRegion::intersect(const DependencyRegion &other)
{
    if (rf != other.rf) {
        clear();
        return;
    }

    if (unspecified || other.unspecified)
        return;

    int i, iOther;
    for (i = 0, iOther = base - other.base; i < size; i++, iOther++) {
        if (iOther >= 0 && iOther < other.size)
            masks[i] &= other.masks[iOther];
        else
            masks[i] = 0;
    }
}

// Check whether two regions overlap.
inline bool intersects(const DependencyRegion &dep1, const DependencyRegion &dep2)
{
    // Check register file.
    if (dep1.rf != dep2.rf)
        return false;

    // Unspecified regions might always overlap.
    if (dep1.unspecified || dep2.unspecified)
        return true;

    // Quick check based on register bounds.
    int diff = dep1.base - dep2.base;
    if ((diff >= dep2.size) || (diff <= -dep1.size))
        return false;

    // Precise check.
    int i1, i2;
    for (i1 = 0, i2 = diff; i1 < dep1.size; i1++, i2++)
        if (i2 >= 0 && i2 < dep2.size)
            if (dep1.masks[i1] & dep2.masks[i2])
                return true;

    return false;
}

void DependencyRegion::subtract(const DependencyRegion &other)
{
    if (other.rf != rf)
        return;
    if (unspecified)
        return;
    if (other.unspecified)
        clear();
    else {
        int i, iOther;
        for (i = 0, iOther = base - other.base; i < size; i++, iOther++)
            if (iOther >= 0 && iOther < other.size)
                masks[i] &= ~other.masks[iOther];
    }
}

inline bool contains(const DependencyRegion &dep1, const DependencyRegion &dep2)
{
    using mtype = decltype(DependencyRegion::masks)::value_type;

    if (dep1.rf != dep2.rf) return false;
    if (dep1.unspecified) return true;
    if (dep2.unspecified) return false;

    int i1, i2;
    for (i1 = dep2.base - dep1.base, i2 = 0; i2 < dep2.size; i1++, i2++) {
        mtype mask = (i1 >= 0 && i1 < dep1.size) ? dep1.masks[i1] : 0;
        if (~mask && dep2.masks[i2])
            return false;
    }
    return true;
}

inline bool bboxContains(const DependencyRegion &dep1, const DependencyRegion &dep2)
{
    if (dep1.rf != dep2.rf) return false;
    if (dep1.unspecified || dep2.unspecified) return false;
    return (dep1.base <= dep2.base && dep1.base + dep1.size >= dep2.base + dep2.size);
}

// Check if an ARF type needs SWSB tracking.
inline bool trackableARF(ARFType type)
{
    return (type == ARFType::acc || type == ARFType::a || type == ARFType::s);
}

// Distance in an in-order pipe after which a dependency can be ignored.
inline int timeout(GeneralizedPipe pipe)
{
    switch (pipe.inOrderPipe()) {
        case PipeMaskA: return 11; // Gen12LP
        case PipeMaskI: return 11;
        case PipeMaskF: return 11;
        case PipeMaskL: return 15;
        case PipeMaskM: return 19;
        case PipeMaskS: return 11; // FIXME: use correct value when available
        default:        return std::numeric_limits<int>::max();
    }
}

// Approximate upper bound on cycle count for an OOO instruction.
template <typename Instruction>
inline int estimateLatency(HW hw, const Instruction &insn)
{
    switch (insn.opcode()) {
        default:
        case Opcode::math: return (hw == HW::Gen12LP) ? 20 : 17;
        case Opcode::dpas:
        case Opcode::dpasw: return 20;   // need correct value
        case Opcode::send:
        case Opcode::sendc: {
            switch (insn.sfid()) {
                case SharedFunction::dc0:
                case SharedFunction::dc1: {
                    MessageDescriptor desc;
                    if (insn.getSendDesc(desc))
                        if (desc.surface.index == 0xFE)
                            return (hw == HW::Gen12LP) ? 33 : 25;
                    return (hw == HW::Gen12LP) ? 106 : 150;
                }
                case SharedFunction::sampler: return (hw == HW::Gen12LP) ? 175 : 210;
                default: return 50;
            }
        }
    }
}

// Measure instruction distance between two Dependencies in a given pipe.
template <bool consumer1, bool consumer2>
inline int distance(const Dependency<consumer1> &dep1, const Dependency<consumer2> &dep2, GeneralizedPipe pipe)
{
    auto ioPipe = pipe.inOrderPipe();

    if (ioPipe == PipeMaskNone)
        return 0;

    auto pidx = utils::log2(ioPipe);
    return dep2.counters[pidx] - dep1.counters[pidx];
}

// Check whether two dependencies form a producer-consumer pair.
inline bool intersects(const Producer &dep1, const Consumer &dep2)
{
    if (!dep2.swsb) {
        // Region-based dependency. First, quick check based on dependency type:
        //   RAR:     ignore
        //   WAR/WAW: ignore if both instructions in same in-order pipe, or same out-of-order pipe (WAR only)
        // If not ignorable, check:
        //   * If consumer is in-order, is that pipe still live (unsynchronized) in the producer?
        //   * If producer is in-order, is it close enough to require tracking the dependency?
        //   * Do the producer+consumer regions overlap?
        if (dep1.read() && dep2.read())                                                             return false;
        if (!(dep1.write() && dep2.write() && (dep1.region.checkWAW || dep2.region.checkWAW)))
        if (dep1.read() || dep1.pipe.inOrder())
        if (dep2.write() && (dep1.pipe == dep2.pipe) && (dep1.pipe != GeneralizedPipe::Math()))     return false;
        if (dep1.pipe.inOrder() && (distance(dep1, dep2, dep1.pipe) >= timeout(dep1.pipe)))         return false;
        if ((dep2.region.base >> 4) != (static_cast<uint8_t>(ARFType::s) & 0xF))
        if (dep2.region.rf == RegFileARF && (dep2.read() || dep2.region.hw == HW::Gen12LP))         return false;
        return intersects(dep1.region, dep2.region);
    } else {
        // SWSB dependency.
        if (dep1.tokenTBD == dep2.tokenTBD) {
            if (dep1.tokenMaskSrc & dep2.tokenMaskSrc) return true;
            if (dep1.tokenMaskSrc & dep2.tokenMaskDst) return true;
            if (dep1.tokenMaskDst & dep2.tokenMaskDst) return true;
        }
        if (dep1.pipe.inOrder())
            for (auto pidx: {dep1.pipe.inOrderPipeIdx(), PipeBitA})
                if (dep2.dists[pidx])
                    if (distance(dep1, dep2, dep1.pipe) >= dep2.dists[pidx])
                        return true;
        return false;
    }
}

// Check whether two dependencies form a producer-consumer pair.
inline bool intersects(const Consumer &dep1, const Producer &dep2)
{
    return intersects(dep2, dep1);
}

// Check whether one producer dependency implies another, without checking regions.
inline bool impliesWithoutRegion(const Producer &dep1, const Producer &dep2)
{
    // Reads never imply writes.
    if (dep2.write() && dep1.read())
        return false;
    // Check pipes.
    if (dep2.pipe != dep1.pipe)
        return false;
    if (dep2.hasToken()) {
        // Token dependency: tokens must match. If tokens not assigned, instructions must match.
        if (!dep1.hasToken())                                             return false;
        if (dep2.tokenMaskSrc & ~(dep1.tokenMaskSrc | dep1.tokenMaskDst)) return false;
        if (dep2.tokenMaskDst & ~dep1.tokenMaskDst)                       return false;
        if (dep1.tokenTBD != dep2.tokenTBD)                               return false;
        if (dep1.tokenTBD && (dep1.inum != dep2.inum))                    return false;
    }
    if (dep2.pipe.inOrder()) {
        // Pipeline dependency: compare counters.
        if (dep1.counters[PipeBitA] < dep2.counters[PipeBitA])
            return false;
        auto pidx = dep2.pipe.inOrderPipeIdx();
        if (dep1.counters[pidx] < dep2.counters[pidx])
            return false;
    }
    return true;
}

// Check whether one consumer dependency implies another, without checking regions.
inline bool impliesWithoutRegion(const Consumer &dep1, const Consumer &dep2)
{
    // Writes never imply reads.
    if (dep2.read() && dep1.write()) return false;

    // Check pipes.
    if (dep2.pipe != dep1.pipe)
        return false;
    if (dep2.hasToken()) {
        // Token dependency.
        if (!dep1.hasToken())                                             return false;
        if (dep2.tokenMaskSrc & ~(dep1.tokenMaskSrc | dep1.tokenMaskDst)) return false;
        if (dep2.tokenMaskDst & ~dep1.tokenMaskDst)                       return false;
        if (dep1.tokenTBD != dep2.tokenTBD)                               return false;
    }
    if (dep2.pipe.inOrder()) {
        // Pipeline dependency. Consumer dependencies are only compared
        //  within BBs, so it's enough to check the A counter.
        // Note distance check not always valid for A@ consumers >= XeHP,
        //  but is never used in these cases.
        if (dep2.counters[PipeBitA] < dep1.counters[PipeBitA])
            return false;
        for (int pidx = 0; pidx < NSWSBPipes; pidx++) {
            if (dep2.dists[pidx] > 0) {
                if (dep1.dists[pidx] == 0) return false;
                if (distance(dep1, dep2, 1 << pidx) - dep2.dists[pidx] + dep1.dists[pidx] < 0)
                    return false;
            }
        }
        if (dep1.read() && dep2.write())
            return false;
    }
    return true;
}

template <bool consumer>
int Dependency<consumer>::getToken() const
{
    if (tokenTBD) return -1;
    return utils::log2(tokenMaskSrc | tokenMaskDst);
}

template <bool consumer>
PipeMask Dependency<consumer>::checkPipes() const
{
    return consumer ? depPipes() : pipe.inOrderPipe();
}

template <bool consumer>
PipeMask Dependency<consumer>::depPipes() const
{
    PipeMask out = PipeMaskNone;
    for (int pidx = 0; pidx < NSWSBPipes; pidx++)
        if (dists[pidx] > 0)
            out |= (1 << pidx);
    return out;
}

template <bool consumer>
PipeMask Dependency<consumer>::coalesceInOrder()
{
    if (!consumer) return pipe.inOrderPipe();

    PipeMask depPipe = PipeMaskNone;
    uint8_t dist = 0;
    for (int pidx = 0; pidx < NSWSBPipes; pidx++) {
        if (dists[pidx] > 0) {
            if (depPipe) {
                depPipe = PipeMaskA;
                dist = std::min(dist, dists[pidx]);
            } else {
                depPipe = 1 << pidx;
                dist = dists[pidx];
            }
        }
    }

    dists.fill(0);
    if (depPipe)
        dists[utils::log2(depPipe)] = dist;

    return depPipe;
}

template <bool consumer>
void DependencyTable<consumer>::clear()
{
    deps.clear();
    frags.clear();
    for (int l = 0; l < NListTypes; l++)
        std::fill(heads[l].begin(), heads[l].end(), none);
}

template <bool consumer>
void DependencyTable<consumer>::reserve(int icount)
{
    icount *= 4;
    deps.reserve(icount);
    frags.reserve(icount * 4);
}

template <bool consumer>
bool DependencyTable<consumer>::insertPrepare(int listType, int listIdx, Dependency<consumer> &dep, bool checkWeaker, bool checkStronger)
{
    for (auto fragID = heads[listType][listIdx]; fragID != none;) {
        auto &frag = frags[fragID];
        auto &entry = deps[frag.depID];

        bool noRegions = (dep.region.unspecified && entry.region.unspecified);

        if (checkWeaker && impliesWithoutRegion(entry, dep)) {
            if (noRegions)
                return false;
            dep.region.subtract(entry.region);
            if (dep.region.empty())
                return false;
        }

        if (checkStronger && impliesWithoutRegion(dep, entry)) {
            entry.region.subtract(dep.region);
            if (entry.region.empty() || noRegions)
                remove(fragID);
        }

        fragID = frag.next[listType];
    }

    return true;
}

template <bool consumer>
void DependencyTable<consumer>::insertLinkedList(int listType, int listIdx, int32_t fragID)
{
    auto &head = heads[listType][listIdx];
    auto &frag = frags[fragID];

    frag.next[listType] = head;
    frag.prev[listType] = makeHeadLink(listIdx);
    if (head != none)
        frags[head].prev[listType] = fragID;
    head = fragID;
}

// Insert dependency into table.
// If checkStronger set, remove any weaker existing dependencies.
// If checkWeaker set, the input dependency's region will be adjusted to remove
//   overlapping stronger dependencies. If this dependency is already implied by the
//   table, it will not be added.
// Return value indicates whether dependency added.
template <bool consumer>
bool DependencyTable<consumer>::insert(Dependency<consumer> &dep, bool checkWeaker, bool checkStronger)
{
    bool toAdd = true;

    auto checkPipes = dep.checkPipes();
    bool checkToken = dep.hasToken() && !(!consumer && dep.tokenTBD && !dep.region.unspecified);
    auto tokenMask = dep.tokenMaskSrc | dep.tokenMaskDst;

    if (checkToken) {
        if (dep.tokenTBD)
            toAdd = toAdd && insertPrepare(ListTypeToken, 0xFF, dep, checkWeaker, checkStronger);
        else for (int token = 0; token < 32; token++) if (tokenMask & (1u << token))
            toAdd = toAdd && insertPrepare(ListTypeToken, token, dep, checkWeaker, checkStronger);
    } else if (!dep.region.unspecified) {
        for (int r = dep.region.base; r < dep.region.base + dep.region.size; r++)
            toAdd = toAdd && insertPrepare(ListTypeReg, r, dep, checkWeaker, checkStronger);
    } else if (checkPipes) {
        for (int pidx = 0; pidx < NSWSBPipes; pidx++)
            if (checkPipes & (1 << pidx))
                toAdd = toAdd && insertPrepare(ListTypePipe, pidx, dep, checkWeaker, checkStronger);
    }

    if (!toAdd)
        return false;

    auto depID = int(deps.size());
    deps.push_back(dep);

    // Create fragments.
    bool hasRegion = !dep.region.unspecified && (dep.region.size > 0);
    int ridx = hasRegion ? dep.region.base : grfListIdxUnspecified;

    int nfragsRegion = hasRegion ? dep.region.size : 1;
    int nfragsPipe = utils::popcnt(checkPipes);
    int nfragsToken = dep.tokenTBD ? 1 : utils::popcnt(tokenMask);
    int nfrags = std::max({nfragsRegion, nfragsPipe, nfragsToken});

    auto fragID = int(frags.size());

    DependencyFragment frag;
    frag.before = 0;
    frag.after = nfrags - 1;
    frag.depID = depID;
    for (int l = 0; l < NListTypes; l++)
        frag.prev[l] = frag.next[l] = none;

    int token = 0, pidx = 0;

    for (int o = 0; o < nfrags; o++, fragID++, frag.before++, frag.after--) {
        frags.push_back(frag);
        if (o < nfragsRegion && (hasRegion || dep.region.unspecified))
            insertLinkedList(ListTypeReg, ridx++, fragID);
        if (o < nfragsToken) {
            if (dep.tokenTBD)
                insertLinkedList(ListTypeToken, 0xFF, fragID);
            else for (; token < 32; token++) if (tokenMask & (1u << token)) {
                insertLinkedList(ListTypeToken, token++, fragID);
                break;
            }
        }
        if (o < nfragsPipe) {
            for (; pidx < NSWSBPipes; pidx++) if (checkPipes & (1 << pidx)) {
                insertLinkedList(ListTypePipe, pidx++, fragID);
                break;
            }
        }
    }

    return true;
}

template <bool consumer>
void DependencyTable<consumer>::remove(int fragID)
{
    auto &frag0 = frags[fragID];
    deps[frag0.depID].active = false;

    fragID -= frag0.before;
    int nfrag = frag0.before + frag0.after + 1;

    for (int i = 0; i < nfrag; i++, fragID++) {
        auto &frag = frags[fragID];

        int lcount = (i == 0) ? NListTypes : 1;   // Only GRF linked lists contain multiple fragments per dependency.

        for (int l = 0; l < lcount; l++) {
            if (isHeadLink(frag.prev[l]))
                heads[l][readHeadLink(frag.prev[l])] = frag.next[l];
            else if (frag.prev[l] != none)
                frags[frag.prev[l]].next[l] = frag.next[l];
            if (frag.next[l] != none)
                frags[frag.next[l]].prev[l] = frag.prev[l];
        }
    }
}

// Find dependencies in the table intersecting the given dependency, and append them to the given list.
// NB: the resulting list may contain duplicate dependencies.
template <bool consumer>
template <bool iconsumer>
void DependencyTable<consumer>::findIntersections(const Dependency<iconsumer> &dep, std::vector<Dependency<consumer>> &out)
{
    findAndRemoveIntersections(dep, &out, false);
}

template <bool consumer>
template <bool iconsumer>
void DependencyTable<consumer>::findAndRemoveIntersections(int listType, int listIdx, const Dependency<iconsumer> &dep, std::vector<Dependency<consumer>> *out, bool doRemove)
{
    for (auto fragID = heads[listType][listIdx]; fragID != none;) {
        auto &frag = frags[fragID];
        auto &entry = deps[frag.depID];
        if (doRemove && !consumer && (distance(entry, dep, entry.pipe) >= timeout(entry.pipe)))
            remove(fragID);
        else if (intersects(dep, entry)) {
            if (out != nullptr)
                out->push_back(entry);
            if (doRemove)
                remove(fragID);
        }
        fragID = frag.next[listType];
    }
}

// Find dependencies in the table intersecting the given dependency.
// Append them to the given list, and remove from table.
// Also checks for, and removes, timed-out producer dependencies.
template <bool consumer>
template <bool iconsumer>
void DependencyTable<consumer>::findAndRemoveIntersections(const Dependency<iconsumer> &dep, std::vector<Dependency<consumer>> *out, bool doRemove)
{
    bool checkToken = false;
    bool checkRegion = !dep.region.empty();
    auto checkPipe = dep.checkPipes();

    if (iconsumer) {
        if (dep.swsb) {
            checkToken = true;
            checkRegion = false;
        }
    } else
        checkToken = true;

    // Handle token dependencies.
    if (checkToken && dep.assignedToken())
        findAndRemoveIntersections(ListTypeToken, dep.getToken(), dep, out, doRemove);

    // Handle pipeline dependencies.
    if (checkPipe & PipeMaskA)
        checkPipe = ~0;
    else if (checkPipe != PipeMaskNone)
        checkPipe |= PipeMaskA;

    for (int pidx = 0; pidx < NSWSBPipes; pidx++)
        if (checkPipe & (1 << pidx))
            findAndRemoveIntersections(ListTypePipe, pidx, dep, out, doRemove);

    // Handle GRF dependencies.
    if (checkRegion) {
        int base = dep.region.unspecified ? 0 : dep.region.base;
        int len = dep.region.unspecified ? maxGRF : dep.region.size;
        for (int r = base; r < base + len; r++)
            findAndRemoveIntersections(ListTypeReg, r, dep, out, doRemove);
        findAndRemoveIntersections(ListTypeReg, grfListIdxUnspecified, dep, out, doRemove);
    }
}

// Find dependencies in the table intersecting the given dependency, and remove them.
template <bool consumer>
template <bool iconsumer>
void DependencyTable<consumer>::removeIntersections(const Dependency<iconsumer> &dep)
{
    findAndRemoveIntersections(dep, nullptr);
}

// Remove dependencies from the table matching a token mask.
// Returns mask of unmatched tokens.
template <bool consumer>
uint32_t DependencyTable<consumer>::removeByTokenMask(uint32_t mask, bool dst)
{
    uint32_t unmatched = mask;

    while (mask) {
        uint32_t mask1 = mask & ~(mask & (mask - 1));
        mask &= ~mask1;
        int token = utils::log2(mask1);

        for (auto fragID = heads[ListTypeToken][token]; fragID != none;) {
            auto &frag = frags[fragID];
            auto &entry = deps[frag.depID];

            auto emask = entry.tokenMaskSrc;
            if (dst) emask |= entry.tokenMaskDst;

            if (emask & mask1) {
                unmatched &= ~mask1;
                remove(fragID);
            }

            fragID = frag.next[ListTypeToken];
        }
    }

    return unmatched;
}

#ifdef NGEN_DEBUG
inline void dumpPipeMask(PipeMask mask, bool spacers = true)
{
    if (spacers) {
        std::cerr << char((mask & PipeMaskA) ? 'A' : ' ');
        std::cerr << char((mask & PipeMaskF) ? 'F' : ' ');
        std::cerr << char((mask & PipeMaskI) ? 'I' : ' ');
        std::cerr << char((mask & PipeMaskL) ? 'L' : ' ');
        std::cerr << char((mask & PipeMaskM) ? 'M' : ' ');
        std::cerr << char((mask & PipeMaskS) ? 'S' : ' ');
        std::cerr << char((mask & PipeMaskO) ? 'O' : ' ');
    } else {
        if (mask & PipeMaskA) std::cerr << 'A';
        if (mask & PipeMaskF) std::cerr << 'F';
        if (mask & PipeMaskI) std::cerr << 'I';
        if (mask & PipeMaskL) std::cerr << 'L';
        if (mask & PipeMaskM) std::cerr << 'M';
        if (mask & PipeMaskS) std::cerr << 'S';
        if (mask & PipeMaskO) std::cerr << 'O';
        if (mask == PipeMaskNone) std::cerr << '-';
    }
}

void GeneralizedPipe::dump() const
{
    switch (v & vTypeMask) {
        case vInOrder:  dumpPipeMask(inOrderPipe(), false); break;
        case vSystolic: std::cerr << 'D'; break;
        case vMath:     std::cerr << 'M'; break;
        case vSend:     std::cerr << 'S' << int(v & 0xF); break;
        default:        std::cerr << '?'; break;
    }
}

void DependencyRegion::dump() const
{
    if (unspecified)
        std::cerr << "[no region]";
    else if (size == 0)
        std::cerr << "[zero size region]";
    else {
        char rfChar = 'r';
        std::cerr << rfChar << int(base);
        if (size > 1)
            std::cerr << '-' << rfChar << int(base + size - 1);

        auto fullMask = ~uint32_t(0);
        bool partial = false;
        for (int ii = 0; ii < size; ii++)
            partial |= (masks[ii] != fullMask);

        if (partial) {
            std::cerr << " (" << std::hex;
            for (int ii = 0; ii < size; ii++) {
                if (masks[ii] != fullMask)
                    std::cerr << std::setw(sizeof(masks[ii]) * 2) << masks[ii];
                else
                    std::cerr << "all";
                std::cerr << char((ii == (size - 1)) ? ')' : ' ');
            }
            std::cerr << std::dec;
        }
    }
}

template <bool consumer>
void Dependency<consumer>::dump() const
{
    if (tokenTime > 0) {
        std::cerr << '[' << counters[PipeBitA] << " + " << tokenTime;
        std::cerr << ',' << inum;
    } else {
        std::cerr << '[';
        for (auto &counter : counters)
            std::cerr << counter << ',';
        pipe.dump();
    }
    std::cerr << ']';
    if (hasToken() && tokenTBD) {
        std::cerr << " $?";
        if (!tokenMaskDst)
            std::cerr << ".src";
        else if (!tokenMaskSrc)
            std::cerr << ".dst";
        else
            std::cerr << "    ";
    } else if (hasToken()) {
        auto mask = tokenMaskSrc | tokenMaskDst;
        for (int i = 0; i < 32; i++) if (mask & (1u << i)) {
            std::cerr << " $" << std::hex << int(i) << std::dec;
            if (tokenMaskSrc & ~tokenMaskDst & (1u << i))
                std::cerr << ".src";
            else if (tokenMaskDst & ~tokenMaskSrc & (1u << i))
                std::cerr << ".dst";
            else
                std::cerr << "    ";
        }
    } else
        std::cerr << "       ";
    bool pipedep = false;
    for (int pidx = 0; pidx < NSWSBPipes; pidx++) if (dists[pidx] > 0) {
        dumpPipeMask(1 << pidx, false);
        std::cerr << '@' << int(dists[pidx]);
        pipedep = true;
    }
    if (!pipedep)
        std::cerr << "   ";

    std::cerr << (rw ? " write " : "  read ");
    if (!region.unspecified)
        region.dump();
}

template <bool consumer>
void DependencyTable<consumer>::dump() const
{
    std::cerr << (consumer ? "Consumers:\n" : "Producers:\n");
    for (size_t i = 0; i < deps.size(); i++) {
        if (!deps[i].active)
            continue;
        std::cerr << i << ":\t";
        deps[i].dump();
        std::cerr << std::endl;
    }

    for (int l = 0; l < NListTypes; l++) {
        for (size_t i = 0; i < heads[l].size(); i++) {
            auto fragID = heads[l][i], lastFragID = makeHeadLink(i);
            if (fragID != none) {
                switch (l) {
                    case ListTypeReg:
                        std::cerr << 'r';
                        if (i == grfListIdxUnspecified)
                            std::cerr << '?';
                        else
                            std::cerr << i;
                        break;
                    case ListTypeToken:
                        std::cerr << '$';
                        if (i == 0xFF)
                            std::cerr << '?';
                        else
                            std::cerr << i;
                        break;
                    case ListTypePipe:
                        if (i > NPipes)
                            std::cerr << '?';
                        else
                            std::cerr << "AFILMSCO"[i % (NPipes + 1)];
                        break;
                }
                std::cerr << ":\t";
                while (fragID != none) {
                    if (frags[fragID].prev[l] != lastFragID)
                        std::cerr << "(bad last ptr) ";
                    std::cerr << frags[fragID].depID << " -> ";
                    lastFragID = fragID;
                    fragID = frags[fragID].next[l];
                }
                std::cerr << std::endl;
            }
        }
    }
}
#endif

/***********************/
/* Instruction Helpers */
/***********************/

template <typename Program>
inline bool hasAutoSWSB(HW hw, const Program &program)
{
    if (hw < HW::Gen12LP)
        return false;
    for (uint32_t n = 0; n < program.size(); n++)
        if (program[n].autoSWSB())
            return true;
    return false;
}

template <typename Instruction>
inline Directive getDirective(const Instruction &insn)
{
    DependencyRegion region;
    if (!insn.getOperandRegion(region, -1)) {
#ifdef NGEN_SAFE
        throw std::runtime_error("nGEN internal error: invalid directive");
#else
        return static_cast<Directive>(0xFF);
#endif
    }
    return static_cast<Directive>(region.base);
}

template <typename Instruction>
inline bool canDefaultPipe(HW hw, const Instruction &insn)
{
    if (hw >= HW::XeHP && insn.opcode() == Opcode::mov_gen12 && (insn.dstTypecode() ^ insn.src0Typecode()) & 0x8)
        return false;
    if (hw >= HW::XeHPC && insn.dstTypecode() == 0xB /* :df */)
        return false;
    return true;
}

static inline bool isSync(Opcode op)
{
    return (op == Opcode::sync);
}

/*****************/
/* Main Routines */
/*****************/

// Get a list of basic blocks for this program.
template <typename Program>
inline BasicBlockList getBasicBlocks(HW hw, const Program &program)
{
    bool enablePVCWARWA = true;
    auto icount = int(program.size());

    // Create map from BB head instructions to instruction #s.
    std::map<int, int> heads;
    heads.insert({0, 0});

    // Scan through program and find all fixed jump targets. These will
    //  be the BB heads (first instruction in block).
    // Also check for instructions which end blocks.
    for (int n = 0; n < icount; n++) {
        const auto &insn = program[n];
        int jip = -1, uip = -1;
        auto dests = insn.destinations(jip, uip);

        if (dests == DestNextIP)
            continue;

#ifdef NGEN_DEBUG_BB
        std::cerr << "Instruction " << n << " ->";
        if (dests & DestNextIP) std::cerr << " " << n + 1;
        if (dests & DestJIP) std::cerr << " " << n + jip;
        if (dests & DestUIP) std::cerr << " " << n + uip;
        std::cerr << std::endl;
#endif

        heads.insert({n + 1, 0});
        if (dests & DestJIP) heads.insert({n + jip, 0});
        if (dests & DestUIP) heads.insert({n + uip, 0});
    }

    // Create basic blocks and remember mapping from instruction #s to BBs.
    auto bbCount = uint32_t(heads.size());
    BasicBlockList list{bbCount};

    int nextBB = 0;
    for (auto &head : heads) {
        auto istart = head.first;
        if (istart >= 0 && istart < icount) {
            head.second = nextBB;
            list[nextBB].id = nextBB;
            list[nextBB++].istart = istart;
        }
    }

    bbCount = nextBB;
    list.resize(bbCount);

    for (uint32_t i = 0; i < bbCount - 1; i++)
        list[i].iend = list[i + 1].istart;
    list[bbCount - 1].iend = icount;

    // Scan through basic blocks again.
    for (auto &bb : list) {
        // Count in-order instructions in each pipe, and wrdep pseudo-instructions.
        for (auto &l : bb.lengths)
            l = 0;
        bb.directives = 0;
        bb.n64 = 0;

        for (uint32_t n = bb.istart; n < bb.iend; n++) {
            const auto &insn = program[n];

            bb.n64 += insn.is64();
            if (isDirective(insn.opcode()))
                bb.directives++;
            auto pipes = getPipeMask(hw, insn);
            for (int p = 0; p < NPipes; p++)
                if (pipes & (1 << p)) bb.lengths[p]++;
        }

        // Identify successor BBs from final instruction.
        auto ntail = bb.iend - 1;
        const auto &insn = program[ntail];
        int jip = 0, uip = 0;
        auto dests = insn.destinations(jip, uip);

        auto addSuccessor = [&](int inum) {
            if ((inum >= 0) && (inum < icount)) bb.succ.push_back(&list[heads[inum]]);
        };

        if (dests & DestNextIP) addSuccessor(bb.iend);
        if (dests & DestJIP)    addSuccessor(jip + ntail);
        if (dests & DestUIP)    addSuccessor(uip + ntail);

        // Add predecessor links to every successor.
        for (auto succ : bb.succ)
            succ->pred.push_back(&bb);

        // Preallocate dependency memory.
        bb.producers.reserve(bb.iend - bb.istart);
        bb.consumers.reserve(bb.iend - bb.istart);

        // Decode and cache operand regions, handling any nodep pseudo-instructions.
        bb.opRegions.resize(bb.iend - bb.istart);
        std::array<bool, 4> ignoreDeps = {false};

        DependencyRegion subDstRegion(hw);
        subDstRegion.clear();

        for (uint32_t n = bb.istart; n < bb.iend; n++) {
            auto &regions = bb.opRegions[n - bb.istart];
            const auto &insn = program[n];

            if (isDirective(insn.opcode())) {
                switch (getDirective(insn)) {
                    case Directive::ignoredep_dst:  ignoreDeps[0] = true; break;
                    case Directive::ignoredep_src0: ignoreDeps[1] = true; break;
                    case Directive::ignoredep_src1: ignoreDeps[2] = true; break;
                    case Directive::ignoredep_src2: ignoreDeps[3] = true; break;
                    case Directive::subdep_dst:
#ifdef NGEN_SAFE
                        if (!subDstRegion.empty())
                            throw invalid_directive_exception();
#endif
                        insn.getOperandRegion(subDstRegion, 0);
                        break;
                    case Directive::wrdep:
                        regions[1].hw = hw;
                        insn.getOperandRegion(regions[1], 0);
                        break;
                    case Directive::fencedep: break;
                    case Directive::pvcwarwa:
                        enablePVCWARWA = false;
                        break;
                }
                continue;
            }

            for (int srcN = -1; srcN < 3; srcN++) {
                regions[srcN + 1].hw = hw;
                if (ignoreDeps[srcN + 1] || !insn.getOperandRegion(regions[srcN + 1], srcN))
                    regions[srcN + 1].clear();
            }

            if (!subDstRegion.empty()) {
                regions[0] = subDstRegion;
                subDstRegion.clear();
            }

            ignoreDeps.fill(false);
        }
    }

#ifndef NGEN_DISABLE_GETENV
#ifndef _WIN32
    // Check ONEAPI_PVC_SEND_WAR_WA environment variable.
    static bool checkedEnv = false;
    static bool haveEnv = false;
    static bool envEnablePWW = true;

    if (!checkedEnv) {
        if (auto e = ::getenv("ONEAPI_PVC_SEND_WAR_WA")) {
            haveEnv = true;
            if (e[0] == '0' && e[1] == '\0')
                envEnablePWW = false;
        }
        checkedEnv = true;
    }

    if (haveEnv)
        enablePVCWARWA = envEnablePWW;
#endif
#endif

    for (auto &bb: list)
        bb.enablePVCWARWA = enablePVCWARWA;

    return list;
}

// Read SWSB from instruction and output:
//  * token dependency it produces, if any
//  * dependencies it consumes
//  * whether auto SWSB requested (bool return value)
// Assumes pipe information for this instruction already set up in consume dependency.
inline bool getSWSBDependencies(HW hw, SWSBInfo swsb, PipeMask defaultPipe, Producer &produce, Consumer &consume)
{
    produce.tokenMaskSrc = 0u;
    produce.tokenMaskDst = 0u;
    consume.dists.fill(0);
    consume.tokenMaskSrc = 0u;
    consume.tokenMaskDst = 0u;
    consume.swsb = true;
    bool enableAutoSWSB = true;

    for (auto item: swsb) {
        if (item.empty() || item.isNoAccSBSet()) continue;
        if (item.isPipe()) {
            auto pipe = (hw == HW::Gen12LP) ? Pipe::A : item.getPipe();
            auto depPipe = (pipe == Pipe::Default) ? defaultPipe : toMask(pipe);
            if (depPipe) {      // if is here to ignore default pipe deps for OOO instructions.
                consume.dists[utils::log2(depPipe)] = item.pipe.dist;
                enableAutoSWSB = false;
            }
        } else {
            if (item.token.src) consume.tokenMaskSrc |= (1u << item.token.token);
            if (item.token.dst) consume.tokenMaskDst |= (1u << item.token.token);

            if (item.hasTokenSet()) {
                produce.tokenMaskSrc |= (1u << item.token.token);
                produce.tokenMaskDst |= (1u << item.token.token);
            }
        }
    }

    return enableAutoSWSB;
}

template <typename Instruction>
inline bool getSWSBDependencies(HW hw, const Instruction &insn, Producer &produce, Consumer &consume)
{
    bool autoSWSB = insn.autoSWSB();
    autoSWSB &= getSWSBDependencies(hw, insn.swsb(), getPipe(hw, insn).inOrderPipe(), produce, consume);
    return autoSWSB;
}

inline SWSBItem encodeSWSBItem(Consumer &consume, uint32_t defaultPipeMask = 0u)
{
    for (int pidx = 0; pidx < NSWSBPipes; pidx++) {
        auto &dist = consume.dists[pidx];
        if (dist > 0) {
            auto edist = std::min<int>(dist, 7);
            auto pipe = fromMask(1 << pidx);
            if (defaultPipeMask & (1 << pidx))
                pipe = Pipe::Default;
            dist = 0;
            return SWSBItem(pipe, edist);
        }
    }

    if (consume.tokenMaskSrc) {
        auto token = utils::bsr(consume.tokenMaskSrc);
        consume.tokenMaskSrc &= ~(1u << token);
        return SBID(token).src;
    }

    if (consume.tokenMaskDst) {
        auto token = utils::bsr(consume.tokenMaskDst);
        consume.tokenMaskDst &= ~(1u << token);
        return SBID(token).dst;
    }

    return SWSBItem();
}

template <size_t n>
static inline std::array<SWSBItem, n> encodeSWSBItems(Consumer &consume, uint32_t defaultPipeMask = 0u)
{
    std::array<SWSBItem, n> items;
    for (auto &item: items) item = encodeSWSBItem(consume, defaultPipeMask);
    return items;
}

// Encode SWSB information.
template <typename Instruction>
inline SWSBInfo encodeSWSB(HW hw, const Instruction *insn, const Producer &produce, Consumer consume)
{
    PipeMask defaultPipeMask = PipeMaskNone;

    if (insn && canDefaultPipe(hw, *insn)) {
        if (hw == HW::Gen12LP)
            defaultPipeMask = ~PipeMaskNone;
        else
            defaultPipeMask = consume.pipe.inOrderPipe();
    }

    consume.tokenMaskSrc &= ~produce.tokenMaskSrc;
    consume.tokenMaskDst &= ~produce.tokenMaskDst;

    auto swsb = encodeSWSBItems<2>(consume, defaultPipeMask);
    if (produce.hasToken())
        swsb[1] = SBID(produce.getToken()).set;

    return swsb;
}

// Check if ARF src/dst requires special handling.
inline bool arfNeedsSync(ARFType type)
{
    return (type == ARFType::ce || type == ARFType::cr || type == ARFType::sr);
}

// SWSBInfo utility routines.
inline bool empty(SWSBInfo info) { return !info[0] && !info[1]; }
inline bool getTokenSet(SWSBInfo info, int &token)
{
    for (auto item: info) {
        if (item.hasTokenSet()) {
            token = item.getToken();
            return true;
        }
    }
    return false;
}

// Get preferred SBID for a given GRF.
inline uint8_t preferredSBID(int tokens, uint16_t base)
{
    if (tokens >= 32)
        return (base >> 2) & 0x1F;
    else
        return (base >> 3) & 0xF;
}

// Choose SBID for an OOO instruction, based on preceding OOO instructions.
template <typename Program>
inline uint8_t chooseSBID(HW hw, int tokens, Program &program, const BasicBlock &bb, int32_t inum, int32_t counterC, const DependencyTable<false> &incoming, const DependencyTable<false> &producers, uint32_t maskDst)
{
    uint32_t unclaimed = (uint64_t(1) << tokens) - 1;
    std::array<int32_t, 32> pastExpiration;
    constexpr int32_t infinite = std::numeric_limits<int32_t>::max();

    // Priority 1: choose SBID that is an explicit dst dependency for this instruction, if any.
    if (maskDst)
        return utils::bsf(maskDst);

    // Otherwise, look through incoming OOO producers and accumulate most recent use of each token.
    for (auto &dist : pastExpiration) dist = infinite;

    auto accumulateTokens = [&](const Producer &dep) {
        if (!dep.hasToken()) return;

        auto depSWSB = program[dep.inum].swsb();
        int token;
        if (getTokenSet(depSWSB, token)) {
            unclaimed &= ~(1 << token);

            int32_t pe = counterC - (dep.counters[PipeBitC] + dep.tokenTime);
            pastExpiration[token] = std::min<int32_t>(pastExpiration[token], pe);
        }
    };

    incoming.forEach(accumulateTokens);
    producers.forEach(accumulateTokens);

    int32_t bestPE = std::numeric_limits<int32_t>::min();
    uint8_t bestPESBID = 0;
    for (int token = 0; token < tokens; token++) {
        if (pastExpiration[token] > bestPE) {
            bestPE = pastExpiration[token];
            bestPESBID = token;
        }
    }

    // Priority 2: assign SBID based on base register of dst, src1, src0 (in that order),
    //  if it's unclaimed or expired.
    for (int opNum : {-1, 1, 0}) {
        auto &region = bb.getOperandRegion(inum, opNum);
        if (region.size > 0) {
            auto sbid = preferredSBID(tokens, region.base);
            if (pastExpiration[sbid] >= 0)
                return sbid;
        }
    }

    // Priority 3: choose highest-numbered unclaimed SBID.
    if (unclaimed)
        return utils::bsr(unclaimed);

    // Priority 4: choose token that's longest expired or closest to expiring.
    return bestPESBID;
}

// Make an SWSB dependency consume a given producer.
inline void addToSWSB(Consumer &swsb, const Producer &dep, uint32_t &tokenMaskSrc, uint32_t &tokenMaskDst)
{
    if (dep.pipe.inOrder()) {
        // Accumulate in-order dependencies.
        auto thisPipe = dep.pipe.inOrderPipe();
        uint8_t thisDist = distance(dep, swsb, thisPipe);

        auto &dist = swsb.dists[utils::log2(thisPipe)];
        if (dist == 0)
            dist = thisDist;
        else
            dist = std::min(dist, thisDist);
    } else if (!dep.tokenTBD) {
        // Remember out-of-order dependencies for later.
        tokenMaskSrc |= dep.tokenMaskSrc;
        tokenMaskDst |= dep.tokenMaskDst;
    }
}

struct PVCWARWA {
    enum {
        None, Undecided, MoveDep, DummyMov, DstDep
    } strategy = None;
    uint32_t inumSrc = 0;
    uint16_t payload[2] = {0xFFFF, 0xFFFF};
    Producer dep;
    bool rs = false;

    bool operator!() const { return strategy == None; }
    operator bool()  const { return !!*this; }
};

// Detect cases that may trigger the PVC WAR-after-send bug,
//   and choose a workaround.
template <typename Program>
PVCWARWA analyzePVCWARWA(HW hw, Program &program, BasicBlock &bb, int phase,
                         Consumer &consumeOp, std::vector<Producer> &pvcWARWADeps)
{
    PVCWARWA pww;
    auto inum = consumeOp.inum;
    auto &regions = bb.opRegions[inum - bb.istart];

    // Check if the workaround is needed.
    if (phase == 0 || regions[0].empty()) return pww;

    bool pvcWARWA = (hw == HW::XeHPC) && bb.enablePVCWARWA;
    if (!pvcWARWA) return pww;

    // Look for the latest send instruction we have a WAR dependency on, if any.
    consumeOp.rw = true;
    consumeOp.region = regions[0];
    pvcWARWADeps.clear();
    bb.producers.findIntersections(consumeOp, pvcWARWADeps);
    for (auto &dep: pvcWARWADeps) {
        if (dep.write()) continue;
        if (dep.pipe.type() != GeneralizedPipe::vSend) continue;
        if ((pww.strategy == PVCWARWA::None) || dep.inum > pww.dep.inum) {
            pww.dep = dep;
            pww.strategy = PVCWARWA::Undecided;
        }
    }

    if (pww.strategy == PVCWARWA::None)
        return pww;

    // Check if send instruction is in the same BB.
    auto &dep = pww.dep;
    bool sameBB = (dep.inum >= bb.istart && dep.inum < bb.iend);
    int adjust = 0;

    if (sameBB && consumeOp.pipe.type() != GeneralizedPipe::vSystolic) {
        // Check if we have a src at least as large as our dst.
        int srcN;
        for (srcN = 0; srcN <= 2; srcN++) {
            if (regions[srcN + 1].unspecified) continue;
            if (bboxContains(regions[srcN + 1], regions[0]))
                break;
        }
        if (srcN >= 2) srcN = -1;

        // Check for potential read suppression.
        if (srcN >= 0 && consumeOp.pipe.inOrder()) {
            pww.rs = true;
            for (uint32_t iother = inum - 1; iother > dep.inum; iother--) {
                if (getPipe(hw, program[iother], false) != consumeOp.pipe)
                    continue;
                const auto &sr = bb.opRegions[iother - bb.istart][srcN + 1];
                pww.rs = bboxContains(sr, regions[srcN + 1]);
                break;
            }
        }

        // Check if we can move the dependency further down the pipe.
        if (srcN >= 0) {
            int after = std::max(0, dep.region.base + dep.region.size - regions[0].base - 1);
            bool higherPri = false;
            switch (consumeOp.pipe.type()) {
                case GeneralizedPipe::vInOrder:
                    higherPri = (program[inum].dstTypecode() == 0b1011); break;
                case GeneralizedPipe::vSend:
                case GeneralizedPipe::vSystolic:
                    higherPri = true; break;
                default: break;
            }
            adjust = (higherPri ? 2 : 1) - after;

            if (adjust <= 0) {
                pww.strategy = PVCWARWA::None;   /* no WA needed */
                return pww;
            }
        }
    }

    if (phase < 2) return pww;

    // Need to apply a WA. Decide on one, in order of priority:
    //  1) If send dst is null or in a different BB, change .src to .dst
    //  2) Move .src dependency later, if this instruction also
    //          has its dst as a non-suppressed src operand
    //  3) Add dummy mov instructions to ensure FIFO cleaned out
    //  4) Change .src to .dst
    auto sendClass = dep.pipe.sendClassXeHPC();

    // Case 1
    if (!sameBB || bb.opRegions[dep.inum - bb.istart][0].empty()) {
        pww.strategy = PVCWARWA::DstDep;
        return pww;
    }

    // Case 2: walk forward, looking for a new target send instruction.
    auto eligibleSend = [=, &program, &dep](uint32_t inum) {
        auto &insn = program[inum];
        if (inum != dep.inum && insn.predicated())
            return false;
        return (sendClass == getPipe(hw, insn).sendClassXeHPC());
    };

    if (adjust > 0) {
        for (pww.inumSrc = dep.inum + 1; pww.inumSrc < inum; pww.inumSrc++) {
            if (!eligibleSend(pww.inumSrc)) continue;
            for (int srcN = 0; srcN <= 1; srcN++) {
                auto &sr = bb.opRegions[pww.inumSrc - bb.istart][srcN + 1];
                if (!sr.unspecified)
                    adjust -= sr.size;
            }
            if (adjust <= 0) break;
        }

        if (adjust <= 0) {
            pww.strategy = PVCWARWA::MoveDep;
            return pww;
        }
    }

    // Case 3: collect 2 GRFs worth of payload from this send class, walking backward.
    int ngrf = 0;
    for (int32_t iother = dep.inum; iother >= int32_t(bb.istart) && ngrf < 2; iother--) {
        if (!eligibleSend(iother)) continue;
        for (int srcN = 0; srcN <= 1; srcN++) {
            auto &sr = bb.opRegions[iother - bb.istart][srcN + 1];
            if (sr.unspecified) continue;
            for (int i = 0; i < sr.size && ngrf < 2; i++)
                pww.payload[ngrf++] = sr.base + i;
        }
    }
    if (ngrf == 2) {
        pww.strategy = PVCWARWA::DummyMov;
        return pww;
    }

    // Case 4
    pww.strategy = PVCWARWA::DstDep;
    return pww;
}

// Main dependency analysis.
// This is run three times on every BB.
// Phase 0
//   Generate dependency tables for SBID assignment:
//      - produced OOO dependencies:  outgoing dependencies from this BB (w/o final SBIDs)
//      - consumed dependencies:  incoming dependencies this BB must synchronize on
// Phase 1
//   Input:
//      - incoming OOO dependencies, with expirations.
//   Output:
//      - produced dependencies:  outgoing dependencies this BB creates and does not synchronize on
//      - consumed dependencies:  incoming dependencies this BB must synchronize on
//      - SBIDs assigned where needed.
//   Instructions whose dependencies are all inside this BB are scoreboarded now for efficiency.
// Phase 2
//   Input: complete list of live dependencies.
//   All unscoreboarded instructions are reanalyzed and scoreboarded now.
template <typename Program>
inline void analyze(HW hw, int tokens, Program &program, BasicBlock &bb, int phase, std::atomic<bool> &cancel)
{
    if (cancel) return;
    const bool final = (phase == 2);
    const bool computeSWSB = (phase > 0);
    bool forceA1 = false;
    uint32_t forceTokenMaskDstNext = 0;
    bool forcePhase2Next = false;
    int inumChain = -1;
    uint32_t chainTokenMaskSrc = 0, chainTokenMaskDst = 0, chainTokenMaskDstX = 0;
    Consumer chainGenerated;
    std::array<int32_t, NPipes> counters;
    std::vector<Producer> depList, depListIncoming, chainProducers, pvcWARWADeps;
    std::vector<std::pair<bool, const DependencyRegion*>> depOperands;

    auto allTokens = uint32_t((uint64_t(1) << tokens) - 1);

    // Incrementing counters.
    auto incrementCounters = [&](PipeMask pipeMask) {
        for (int pidx = 0; pidx < NPipes; pidx++)
            if (pipeMask & (1 << pidx))
                counters[pidx]++;
    };

    // Initialize "preconsumes." These are region-less consumes arising from SWSB.
    int noPreconsume = std::numeric_limits<int>::min();
    std::array<std::array<int, NPipes + 1>, NPipes> preconsumeIO;
    uint32_t preconsumeTokenSrc = 0, preconsumeTokenDst = 0;

    auto recordIOPreconsumes = [&](Consumer &generated) {
        if (phase != 1) return;
        auto spipes = generated.pipe.syncPipes(hw);
        for (int pidx = 0; pidx < NSWSBPipes; pidx++) if (generated.dists[pidx] > 0) {
            auto dpipes = GeneralizedPipe(1 << pidx).syncPipes(hw);
            for (int dpidx = 0; dpidx < NPipes; dpidx++)
                if (dpipes & (1 << dpidx))
                    for (int spidx = 0; spidx <= NPipes; spidx++)
                        if (spipes & (1 << spidx))
                            preconsumeIO[dpidx][spidx] = std::max<int>(preconsumeIO[dpidx][spidx], counters[dpidx] - generated.dists[pidx]);
        }
    };

    if (phase == 1)
        for (auto &pcList : preconsumeIO)
            for (auto &pc : pcList)
                pc = noPreconsume;

    // Helper: resolve outstanding wrdep dependencies and insert syncs as needed.
    auto resolveWrdeps = [&](bool final) {
        if (!depOperands.empty()) {
            Consumer generated;
            uint32_t tokenMaskSrc = 0, tokenMaskDst = 0;

            generated.counters = counters;

            depList.clear();

            for (auto &depOp: depOperands) {
                generated.rw = depOp.first;
                generated.region = *depOp.second;
                bb.producers.findAndRemoveIntersections(generated, &depList);
            }

            if (final) {
                for (const auto &dep: depList)
                    addToSWSB(generated, dep, tokenMaskSrc, tokenMaskDst);
                if (generated.coalesceInOrder()) {
                    auto distGen = generated;
                    distGen.tokenMaskSrc = distGen.tokenMaskDst = 0u;
                    auto syncSWSB = encodeSWSB(hw, (decltype(&program[0])) nullptr, Producer(), distGen);
                    bb.syncs.push_back({uint32_t(bb.iend), syncSWSB, SyncFunction::nop, 0});
                }
                if (tokenMaskDst)
                    bb.syncs.push_back({uint32_t(bb.iend), SWSBInfo(), SyncFunction::allwr, tokenMaskDst});
            }
        }
    };

    // Initialize counters.
    for (auto &counter : counters)
        counter = 0;

    for (uint32_t inum = bb.istart; inum < bb.iend; inum++) {
        auto &insn = program[inum];
        bool forceA1Next = false;
        bool atChainStart = false;
        auto opcode = insn.opcode();

        // Ignore illegal instructions.
        if (opcode == Opcode::illegal)
            continue;

        // Process auto-SWSB directives. Only wrdep/fencedep need handling here.
        if (isDirective(opcode)) {
            switch (getDirective(insn)) {
                case Directive::wrdep: {
                    auto &region = bb.getOperandRegion(inum, 0);
                    if (!region.empty())
                        depOperands.push_back(std::make_pair(false, &region));
                    break;
                }
                case Directive::fencedep: {
                    auto swsbDep = program[inum + insn.getFencedepJIP()].swsb();
                    int token;
                    if (getTokenSet(swsbDep, token))
                        forceTokenMaskDstNext |= (1u << token);
                    else
                        forcePhase2Next = true;
                }
                default: break;
            }
            continue;
        }

        // Placeholder for dependency consumers from this instruction's operands.
        Consumer consumeOp;
        consumeOp.counters = counters;
        consumeOp.pipe = getPipe(hw, insn);
        consumeOp.inum = inum;

        // Read SWSB information for this instruction, if already present.
        Producer tokenInfo;
        Consumer generated = consumeOp;
        bool autoSWSB = getSWSBDependencies(hw, insn, tokenInfo, generated);

        // Check for beginning of {Atomic} chain.
        if (insn.atomic() && inumChain < 0) {
            inumChain = inum;
            atChainStart = true;
        }

        // Check if our token might be active.
        bool tokenMayBeActive = tokenInfo.assignedToken()
                             && bb.producers.containsToken(tokenInfo.getToken());

        // If token assigned, start by removing all live dependencies with this token.
        if (tokenInfo.hasToken()) {
            bb.producers.removeByTokenMask(tokenInfo.tokenMaskDst, true);
            preconsumeTokenSrc |= tokenInfo.tokenMaskSrc;
            preconsumeTokenDst |= tokenInfo.tokenMaskDst;
        } else if (trackedByToken(hw, insn.opcode(), execType(insn))) {
            generated.tokenTBD = tokenInfo.tokenTBD = true;
            tokenInfo.tokenMaskSrc = allTokens;
            tokenInfo.tokenMaskDst = allTokens;
        }

        // For sync.allrd/sync.allwr, consume matching dependencies and add preconsumes
        //   for unmatched tokens.
        if (isSync(opcode)) {
            auto fc = insn.syncFC();
            bool allrd = (fc == SyncFunction::allrd);
            bool allwr = (fc == SyncFunction::allwr);

            if (allrd || allwr) {
                uint32_t imm;
                if (!insn.getImm32(imm))
                    imm = ~0;

                auto unmatched = bb.producers.removeByTokenMask(imm, allwr);
                preconsumeTokenSrc |= unmatched;
                if (allwr) preconsumeTokenDst |= unmatched;
            }
        }

        // Grab pre-decoded operand regions for this instruction.
        auto &regions = bb.opRegions[inum - bb.istart];

        // Check for cr/ce/sr destination operand, and force A@1 on the next instruction.
        ARFType dstARFType;
        forceA1Next |= (insn.getARFType(dstARFType, -1, hw) && arfNeedsSync(dstARFType));

        if (autoSWSB) {
            // If auto-SWSB has been requested for this instruction, analyze its source operands.
            // Start a list of dependencies for this instruction.
            depList.clear();
            depListIncoming.clear();
            bool foundAllDeps = true;
            uint32_t tokenMaskSrc = 0, tokenMaskDst = 0, tokenMaskDstX = 0;
            SWSBInfo syncSWSB;

            if (!atChainStart && (inumChain >= 0)) {
                tokenMaskSrc = chainTokenMaskSrc;
                tokenMaskDst = chainTokenMaskDst;
                tokenMaskDstX = chainTokenMaskDstX;
                generated = chainGenerated;
            }

            tokenMaskDst |= forceTokenMaskDstNext;

            // Jumps with unknown destination: preconsume all dependencies.
            if (inum == (bb.iend - 1)) {
                int jip, uip;
                if (insn.destinations(jip, uip) & DestUnknown) {
                    tokenMaskDst = preconsumeTokenDst = allTokens;
                    for (auto &p : preconsumeIO[PipeBitA])
                        p = 0;
                    bb.producers.clear();
                    bb.consumers.clear();
                    syncSWSB[0] = (hw == HW::Gen12LP) ? SWSB(1) : SWSB<AllPipes>(1);
                }
            }

            // Check if we need to assign an SBID to this instruction.
            bool tokenInsn = trackedByToken(hw, opcode, execType(insn));
            bool assignSBID = (phase == 1) && tokenInsn && tokenInfo.tokenTBD && !insn.atomic();

            // Collect operands.
            for (int srcN = 2; srcN >= -1; srcN--) {
                // Skip non-GRF operands.
                // Special case: check for cr/sr/ce source operands and force A@1 if any.
                if (regions[srcN + 1].empty()) {
                    ARFType arfType;
                    if ((srcN >= 0) && insn.getARFType(arfType, srcN, hw) && arfNeedsSync(arfType))
                        generated.dists[PipeBitA] = 1;
                    continue;
                }

                bool rw = (srcN < 0);
                depOperands.push_back(std::make_pair(rw, &regions[srcN + 1]));
            }

            // Handle PVC HW bug with WAR dependencies on send instructions.
            auto pww = analyzePVCWARWA(hw, program, bb, phase, consumeOp, pvcWARWADeps);

            // Analyze operands.
            for (auto &depOp: depOperands) {
                // Create associated dependency consumer.
                consumeOp.rw = depOp.first;
                consumeOp.region = *depOp.second;

                // Remove all intersecting live producers from the table and save them.
                auto dStart = depList.size();
                bb.producers.findAndRemoveIntersections(consumeOp, &depList);
                auto dEnd = depList.size();

                // Do the same for the incoming producers table if we need to assign an SBID.
                size_t dStartIncoming = 0, dEndIncoming = 0;
                if (assignSBID) {
                    dStartIncoming = depListIncoming.size();
                    bb.incoming.findAndRemoveIntersections(consumeOp, &depListIncoming);
                    dEndIncoming = depListIncoming.size();
                }

                // If not final, subtract each of them from original dependency region.
                // If anything remains, add to consumer table. If it is not implied
                //   by existing consumers, we didn't find all dependencies.
                if (!final) {
                    for (auto d = dStart; d < dEnd; d++)
                        consumeOp.region.subtract(depList[d].region);
                    if (!consumeOp.region.empty())
                        foundAllDeps &= !bb.consumers.insertWeak(consumeOp);
                }

                // Add dependencies to SWSB.
                if (computeSWSB) for (auto d = dStart; d < dEnd; d++)
                    addToSWSB(generated, depList[d], tokenMaskSrc, tokenMaskDst);

                // Also collect incoming SBIDs if choosing an SBID.
                if (assignSBID) for (auto d = dStartIncoming; d < dEndIncoming; d++) {
                    auto &dep = depListIncoming[d];
                    if (!dep.tokenMaskDst) continue;
                    if (dep.tokenTBD) {
                        // Check SWSB again in case it was recently assigned.
                        auto curSWSB = program[dep.inum].swsb();
                        int token;
                        if (getTokenSet(curSWSB, token))
                            tokenMaskDstX |= (1u << token);
                    } else
                        tokenMaskDstX |= dep.tokenMaskDst;
                }
            }

            tokenMaskDstX |= tokenMaskDst;
            depOperands.clear();

            // Transfer dependencies down the {Atomic} chain (will be put later on first instruction).
            if (insn.atomic()) {
                chainTokenMaskSrc = tokenMaskSrc;
                chainTokenMaskDst = tokenMaskDst;
                chainTokenMaskDstX = tokenMaskDstX;
                chainGenerated = generated;
                tokenMaskSrc = tokenMaskDst = 0;
                generated = consumeOp;
            }

            // Always wait until phase 2 to assign SWSB to {Atomic} chains --
            //   it's not known if all dependencies for the chain have been found until the end.
            // Also delay predicated token instructions, to ensure we know all SBIDs.
            if (inumChain >= 0 || insn.atomic() || (tokenInsn && insn.predicated()) || forcePhase2Next || pww)
                foundAllDeps = false;

            // If token missing on OOO instruction, assign one during phase 1.
            if (assignSBID) {
                auto newToken = chooseSBID(hw, tokens, program, bb, inum, counters[PipeBitC], bb.incoming, bb.producers, tokenMaskDstX);
                generated.tokenTBD     = tokenInfo.tokenTBD     = false;
                generated.tokenMaskSrc = tokenInfo.tokenMaskSrc = (1u << newToken);
                generated.tokenMaskDst = tokenInfo.tokenMaskDst = (1u << newToken);

                insn.setSWSB({SBID(newToken).set});
                preconsumeTokenSrc |= generated.tokenMaskSrc;
                preconsumeTokenDst |= generated.tokenMaskDst;
                tokenMaskSrc &= ~generated.tokenMaskSrc;
                tokenMaskDst &= ~generated.tokenMaskDst;
            }

            // Finalize SWSB computation.
            if (computeSWSB) {
                bool recordSWSB = (final || foundAllDeps);
                bool tokenAssigned = tokenInfo.assignedToken();

                // If last instruction forced A@1, enforce now.
                if (forceA1) {
                    generated.dists.fill(0);
                    generated.dists[PipeBitA] = 1;
                    if (tokenMaskSrc || tokenMaskDst) {
                        bb.producers.removeIntersections(generated);
                        generated.dists[PipeBitA] = 0;
                        auto swsb = (hw == HW::Gen12LP) ? SWSB(1) : SWSB<AllPipes>(1);
                        if (recordSWSB)
                            bb.syncs.push_back({uint32_t(inum), {swsb}, SyncFunction::nop, 0});
                    }
                }

                PipeMask depPipe;

                {
                    // Coalesce multiple in-order pipe dependencies into A@n.
                    depPipe = generated.coalesceInOrder();

                    // If dual dependency (token + pipe) on OOO instruction, use A pipe for send, sync for others.
                    if (depPipe && (generated.hasToken() || tokenAssigned)) {
                        if (isSend(opcode)) {
                            if (!(hw >= HW::XeHPC && (depPipe == PipeMaskI || depPipe == PipeMaskF))) {
                                auto pidx = utils::log2(depPipe);
                                auto dist = generated.dists[pidx];
                                generated.dists[pidx] = 0;
                                generated.dists[PipeBitA] = dist;
                            }
                        } else {
                            auto distGen = generated;
                            distGen.tokenMaskSrc = distGen.tokenMaskDst = 0u;
                            syncSWSB = encodeSWSB(hw, &insn, Producer(), distGen);
                            generated.dists.fill(0);
                        }
                    }
                }

                // Handle OOO shootdown. Unless predicate is (W), it's possible that our token won't be claimed.
                //   In this case, add sync on our token as a precaution, if the token might be in use.
                // For {Atomic} chains, do the same, but for a different reason -- it's possible
                //   our token may be in use and we must clear it prior to entering the chain.
                // In all other cases, remove dependencies on our own token.
                if (tokenAssigned && inumChain < 0)
                    tokenMaskDst &= ~(1 << tokenInfo.getToken());
                if (tokenAssigned && (insn.predicated() || inumChain >= 0) && tokenMayBeActive)
                    tokenMaskDst |=  (1 << tokenInfo.getToken());

                tokenMaskSrc &= ~tokenMaskDst;

                // Clean producer list of known SWSB and sync dependencies.
                // Be careful with coalesced A@ dependencies in phase 1: if they may be
                //   reduced to a single pipe dependency later they cannot be cleared now.
                if (tokenMaskSrc) bb.producers.removeByTokenMask(tokenMaskSrc, false);
                if (tokenMaskDst) bb.producers.removeByTokenMask(tokenMaskDst, true);
                if ((depPipe == PipeMaskA) && !recordSWSB)
                    generated.dists.fill(0);
                bb.producers.removeIntersections(generated);

                if (recordSWSB) {
                    // Alter SWSB with any workarounds for PVC WAR dependencies.
                    // Note these alterations do not affect the dependency tables.
                    auto inumSync = (inumChain >= 0) ? inumChain : inum;
                    if ((tokenMaskDst & pww.dep.tokenMaskSrc) == pww.dep.tokenMaskSrc)
                        pww.strategy = PVCWARWA::None;
                    auto depTokenMask = pww.dep.tokenMaskSrc | pww.dep.tokenMaskDst;
                    switch (pww.strategy) {
                        default:
                        case PVCWARWA::None: break;
                        case PVCWARWA::MoveDep: {
                            Producer produce;
                            Consumer consume;
                            (void) getSWSBDependencies(hw, program[pww.inumSrc], produce, consume);
                            // tokenMaskSrc &= ~depTokenMask;          /* not working in certain cases */
                            tokenMaskSrc |=  produce.tokenMaskDst;
                            if (pww.rs)
                                bb.movs.push_back(DummyMovInsertion{uint32_t(inumSync), SWSBInfo{}, 0, true, dtForPipe(generated.pipe.toPipe())});
                            break;
                        }
                        case PVCWARWA::DummyMov: {
                            tokenMaskSrc &= ~depTokenMask;
                            auto pipe = (generated.pipe.inOrderPipe() == PipeMaskF) ? PipeMaskF : PipeMaskI;
                            auto dt = dtForPipe(fromMask(pipe));
                            bb.movs.push_back({uint32_t(inumSync), {SBID(pww.dep.getToken()).src}, 0, true, dt});
                            bb.movs.push_back({uint32_t(inumSync), SWSBInfo{}, pww.payload[1], false, dt});
                            bb.movs.push_back({uint32_t(inumSync), SWSBInfo{}, pww.payload[0], false, dt});
                            if (generated.pipe.inOrderPipe() != pipe) {
                                auto pidx = utils::log2(pipe);
                                Producer dep;
                                dep.pipe = pipe;
                                dep.counters[pidx] = generated.counters[pidx] - 1;
                                addToSWSB(generated, dep, tokenMaskSrc, tokenMaskDst);
                            }
                            break;
                        }
                        case PVCWARWA::DstDep:
                            tokenMaskSrc &= ~depTokenMask;
                            tokenMaskDst |=  depTokenMask;
                            break;
                    }

                    {
                        // Xe/Xe2/Xe3 SWSB finalization.
                        //    - use SWSB to mark src/dst w/o dist (in-order or no token) or dst + dist (in-order only, same pipe)
                        //    - add sync for any remaining dependencies.
                        bool defaultPipe = generated.pipe.inOrder() && (depPipe == generated.pipe.inOrderPipe())
                                                                    && canDefaultPipe(hw, insn);

                        bool acceptsSrc = false, acceptsDst = false;
                        if (generated.pipe.inOrder() || !tokenAssigned) {
                            if (hw >= HW::XeHPC) {
                                acceptsSrc = (depPipe == PipeMaskNone || defaultPipe);
                                acceptsDst = acceptsSrc || (depPipe == PipeMaskA);
                            } else {
                                acceptsSrc = (depPipe == PipeMaskNone);
                                acceptsDst = acceptsSrc || defaultPipe;
                            }
                        }

                        if (tokenMaskDst && acceptsDst) {
                            generated.tokenMaskDst = (1u << utils::bsr(tokenMaskDst));
                            tokenMaskDst &= ~generated.tokenMaskDst;
                        } else if (tokenMaskSrc && acceptsSrc) {
                            generated.tokenMaskSrc = (1u << utils::bsr(tokenMaskSrc));
                            tokenMaskSrc &= ~generated.tokenMaskSrc;
                        }

                        bool oneSrc = tokenMaskSrc && utils::is_zero_or_pow2(tokenMaskSrc);
                        bool oneDst = tokenMaskDst && utils::is_zero_or_pow2(tokenMaskDst);
                        bool oneSrcSWSB = false, oneDstSWSB = false;

                        if (empty(syncSWSB)) {
                            if (oneSrc) {
                                syncSWSB[0] = SBID(utils::bsr(tokenMaskSrc)).src;
                                oneSrcSWSB = true;
                            } else if (oneDst) {
                                syncSWSB[0] = SBID(utils::bsr(tokenMaskDst)).dst;
                                oneDstSWSB = true;
                            }
                        }
                        if (tokenMaskSrc && !oneSrcSWSB) {
                            bb.syncs.push_back({uint32_t(inumSync), syncSWSB, SyncFunction::allrd, tokenMaskSrc});
                            syncSWSB = SWSBInfo();
                        }
                        if (tokenMaskDst && !oneDstSWSB) {
                            bb.syncs.push_back({uint32_t(inumSync), syncSWSB, SyncFunction::allwr, tokenMaskDst});
                            syncSWSB = SWSBInfo();
                        }
                        if (!empty(syncSWSB))
                            bb.syncs.push_back({uint32_t(inumSync), syncSWSB, SyncFunction::nop, 0});
                    }

                    // If final, or nothing added to consumer table, assign SWSB.
                    // For {Atomic} chains, put SWSB for consumed dependencies at head of chain.
                    if (inumChain >= 0) {
                        if (!insn.atomic()) {
                            program[inumChain].setSWSB(encodeSWSB(hw, &insn, Producer(), generated));
                            insn.setSWSB(encodeSWSB(hw, &insn, tokenInfo, Consumer()));
                        }
                    } else
                        insn.setSWSB(encodeSWSB(hw, &insn, tokenInfo, generated));
                    insn.clearAutoSWSB();
                }
            }
        } else {
            // SWSB specified. Consume any dependencies associated with this SWSB.
            bb.producers.removeIntersections(generated);

            // Record token dependencies for populating the consumer table.
            if (!final) {
                preconsumeTokenSrc |= tokenInfo.tokenMaskSrc;
                preconsumeTokenDst |= tokenInfo.tokenMaskDst;
            }

            // Consume destination dependencies too.
            if (!regions[0].empty()) {
                consumeOp.region = regions[0];
                consumeOp.rw = true;
                bb.producers.removeIntersections(consumeOp);
            }

            // Emit syncs from prior wrdeps if auto-SWSB was turned off. Otherwise ignore them.
            if (phase <= 1)
                resolveWrdeps(phase == 1);
            else
                depOperands.clear();

            // Clear auto-SWSB bit if it was set.
            if (phase == 2)
                insn.clearAutoSWSB();

            // Check for prior sync insertions and update tables appropriately.
            if (phase == 2) {
                for (const auto &sync: bb.syncs) {
                    if (sync.inum != inum)
                        continue;

                    bool allrd = (sync.fc == SyncFunction::allrd);
                    bool allwr = (sync.fc == SyncFunction::allwr);

                    if (allrd || allwr) {
                        auto unmatched = bb.producers.removeByTokenMask(sync.mask, allwr);
                        preconsumeTokenSrc |= unmatched;
                        if (allwr) preconsumeTokenDst |= unmatched;
                    }

                    if (!empty(sync.swsb)) {
                        Producer produce;
                        Consumer consume;
                        (void) getSWSBDependencies(hw, sync.swsb, PipeMaskNone, produce, consume);
                        bb.producers.removeIntersections(consume);
                        preconsumeTokenSrc |= consume.tokenMaskSrc;
                        preconsumeTokenDst |= consume.tokenMaskDst;
                    }
                }
            }
        }

        forceTokenMaskDstNext = 0;
        forcePhase2Next = false;

        // First pass: record pipeline SWSB dependencies for later entry into consumer table.
        recordIOPreconsumes(generated);

        // Add producer dependencies for all operands.
        // Also record token timeout.
        // During phase 0, only do this for OOO instructions, and if dst not null, only dst.
        if ((phase > 0) || tokenInfo.hasToken()) {
            auto produceOp = consumeOp.cast();
            if (tokenInfo.hasToken()) {
                produceOp.tokenTBD = tokenInfo.tokenTBD;
                produceOp.tokenTime = estimateLatency(hw, insn);
            }

            for (int srcN = -1; srcN < 3; srcN++) {
                if (!regions[srcN + 1].empty()) {
                    produceOp.rw = (srcN < 0);
                    if (tokenInfo.hasToken()) {
                        produceOp.tokenMaskSrc = (srcN >= 0) ? tokenInfo.tokenMaskSrc : 0u;
                        produceOp.tokenMaskDst = (srcN <  0) ? tokenInfo.tokenMaskDst : 0u;
                    }
                    produceOp.region = regions[srcN + 1];
                    if (insn.atomic())
                        chainProducers.push_back(produceOp);
                    else
                        bb.producers.insertStrong(produceOp);
                    if (phase == 0 && srcN == -1) break;
                }
            }

            // Add producers for previous instructions in {Atomic} chain.
            if (!insn.atomic()) {
                for (auto &op: chainProducers) {
                    if (!op.pipe.inOrder() || op.hasToken()) {
                        if (op.tokenMaskSrc) op.tokenMaskSrc = tokenInfo.tokenMaskSrc;
                        if (op.tokenMaskDst) op.tokenMaskDst = tokenInfo.tokenMaskDst;
                        op.tokenTBD = tokenInfo.tokenTBD;
                    }
                    bb.producers.insertStrong(op);
                }
                chainProducers.clear();
            }
        }

        // Check for end of {Atomic} chain.
        if (!insn.atomic())
            inumChain = -1;

        incrementCounters(getPipeMask(hw, insn));
        forceA1 = forceA1Next;
    }

    // Create sync insertion(s) for any outstanding wrdep pseudo-instructions.
    resolveWrdeps(phase == 2);

    // Add preconsume dependencies to consume list.
    if (!final) {
        // In-order preconsumes.
        if (phase == 1) for (int pOld = 0; pOld < NSWSBPipes; pOld++) {
            for (int pNew = 0; pNew <= NSWSBPipes; pNew++) {
                auto pc = preconsumeIO[pOld][pNew];
                if (pc != noPreconsume) {
                    Consumer preconsume;
                    preconsume.swsb = true;
                    preconsume.counters[pOld] = pc + 1;
                    preconsume.pipe = (1 << pNew);
                    preconsume.dists[pOld] = 1;
                    bb.consumers.insertStrong(preconsume);
                }
            }
        }
        // Out of order preconsumes.
        auto preconsumeToken = preconsumeTokenSrc | preconsumeTokenDst;
        for (int token = 0; token < tokens; token++) {
            if (preconsumeToken & (1 << token)) {
                Dependency<true> preconsume;
                preconsume.swsb = true;
                preconsume.tokenMaskSrc = (preconsumeTokenSrc & (1 << token));
                preconsume.tokenMaskDst = (preconsumeTokenDst & (1 << token));
                bb.consumers.insertStrong(preconsume);
            }
        }
        if (preconsumeTokenSrc == allTokens || preconsumeTokenDst == allTokens) {
            Consumer preconsume;
            preconsume.swsb = true;
            preconsume.tokenTBD = true;
            preconsume.tokenMaskSrc = (preconsumeTokenSrc == allTokens) ? allTokens : 0u;
            preconsume.tokenMaskDst = (preconsumeTokenDst == allTokens) ? allTokens : 0u;
            bb.consumers.insertStrong(preconsume);
        }
    }
}

// Loop optimization. Add synchronizations before entering suspected loops to allow
//  weaker SWSB inside the loop.
inline void loopOptimize(BasicBlock &bb)
{
    // Loop through successors to this BB, looking for ones with
    //   exactly one incoming backedge, not from this BB.
    // If any found, for every dep in produce table:
    //   For each selector successor:
    //     If backedge pred's produce table doesn't imply this dep,
    //     add syncs to consume it.
}

// Propagate live dependencies forward through BB flow graph.
inline void propagate(std::vector<BasicBlock> &BBs, std::atomic<bool> &cancel)
{
    auto bbCount = int(BBs.size());
    bool done = false;
    std::vector<Consumer> consumerList;

    // Mark all incoming dependencies as new.
    for (auto &bb : BBs) {
        bb.label = 0;
        bb.producers.forEach([](Producer &dep) {
            dep.label = 0;
        });
    }

    // Main loop: propagate live dependencies until all live tables stabilize.
    // This should require no more than bbCount loops.
    for (int age = 0; (age < bbCount) && !done; age++) {
        if (cancel) return;
        done = true;
        for (auto &bb : BBs) {
            // Examine each predecessor of this BB.
            for (auto pred : bb.pred) {
                if (pred->label < age) continue;

                pred->producers.forEach([&](const Producer &dep) {
                    // New incoming dependency? If not, skip it.
                    if (dep.label != age) return;

#ifdef NGEN_DEBUG_PROPAGATE
                    std::cerr << "Prop BB " << pred->id << " -> " << bb.id << ": ";
                    dep.dump();
#endif

                    // Adjust counters.
                    // Exception for OOO tokenless dependencies: counter[0] stores instruction #; only adjust counter C.
                    auto newDep = dep;
                    if (newDep.tokenTime == 0)
                        for (int p = 0; p < NPipes; p++)
                            newDep.counters[p] -= pred->lengths[p];
                    else
                        newDep.counters[PipeBitC] -= pred->lengths[PipeBitC];

                    // If an in-order dependency, check for timeout, and skip it if so.
                    if (newDep.pipe.inOrder()) {
                        auto pidx = newDep.pipe.inOrderPipeIdx();
                        if (newDep.counters[pidx] <= -timeout(dep.pipe)) {
#ifdef NGEN_DEBUG_PROPAGATE
                            std::cerr << " timeout\n";
#endif
                            return;
                        }
                    }

                    // Intersect new dependency (producer) with killed (consumer) table.
                    // Subtract all intersections from dependency.
                    consumerList.clear();
                    bb.consumers.findIntersections(newDep, consumerList);

                    for (auto &consumer : consumerList) {
                        newDep.region.subtract(consumer.region);
                        if (newDep.region.empty()) {
#ifdef NGEN_DEBUG_PROPAGATE
                            std::cerr << " killed\n";
#endif
                            return;
                        }
                    }

#ifdef NGEN_DEBUG_PROPAGATE
                    std::cerr << " propagated\n";
#endif

                    // Dependency is new and was not consumed.
                    // Add to produce table unless it's already implied by existing producers.
                    if (&bb == pred) return;    /* pathological case, skip */
                    newDep.label = age + 1;
                    if (bb.producers.insert(newDep)) {
                        done = false;
                        bb.label = age + 1;
                    }
                });
            }
        }
    }

#ifdef NGEN_SAFE
    if (!done) throw std::runtime_error("nGEN internal error: propagation failed.");
#endif

    // Perform final half-propagation step (tail-to-head) to accumulate incoming producers
    //  for each BB.
    for (auto &bb : BBs) {
        for (auto pred : bb.pred) {
            pred->producers.forEach([&](const Producer &dep) {
                // Adjust counters, except for OOO tokenless dependencies.
                auto newDep = dep;
                if (newDep.tokenTime == 0)
                    for (int p = 0; p < NPipes; p++)
                        newDep.counters[p] -= pred->lengths[p];
                else
                    newDep.counters[PipeBitC] -= pred->lengths[PipeBitC];

                // If an in-order dependency, check for timeout, and skip it if so.
                if (newDep.pipe.inOrder()) {
                    auto pidx = newDep.pipe.inOrderPipeIdx();
                    if (newDep.counters[pidx] <= -timeout(dep.pipe))
                        return;
                }

                bb.incoming.insert(newDep);
            });
        }
    }
}

// Adjust jump targets for sync instruction insertions.
template <typename Program>
inline void adjustTargets(HW hw, Program &program, BasicBlockList &list)
{
    std::map<int32_t, int32_t> shifts;

    if (list.empty()) return;

    int32_t shift = 0;
    for (auto &bb : list) {
        shifts.insert({bb.istart, shift});
        shift += bb.sizeAdjust(hw);
    }
    shifts.insert({list.back().iend, shift});

    shift = 0;
    for (auto &bb : list) {
        shift += bb.sizeAdjust(hw);
        auto ntail = bb.iend - 1;
        auto &insn = program[ntail];
        int jip = -1, uip = -1;
        auto dests = insn.destinations(jip, uip);
        if (dests & DestJIP) insn.shiftJIP(shifts[ntail + jip] - shift);
        if (dests & DestUIP) insn.shiftUIP(shifts[ntail + uip] - shift);
    }
}

// Entrypoint for automatic software scoreboarding.
// Returns the list of basic blocks, containing information on sync instructions to insert.
template <typename Program>
inline BasicBlockList autoSWSB(HW hw, int grfCount, Program &program, std::atomic<bool> &cancel)
{
    if (!hasAutoSWSB(hw, program)) {
        return BasicBlockList();
    }

    int tokens = tokenCount(hw, grfCount);

    // Find basic blocks.
    BasicBlockList bbList = getBasicBlocks(hw, program);

#ifdef NGEN_DEBUG
    std::cerr << "BASIC BLOCKS\n";
    std::cerr << "------------\n";
    for (auto &bb : bbList) {
        std::cerr << "Basic Block " << bb.id;
        if (!bb.pred.empty()) {
            std::cerr << " <-";
            for (auto &pred : bb.pred)
                std::cerr << ' ' << pred->id;
        }
        if (!bb.succ.empty()) {
            std::cerr << " ->";
            for (auto &succ : bb.succ)
                std::cerr << ' ' << succ->id;
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
#endif

    // Analysis round 0: gather OOO instruction usage.
    for (auto &bb : bbList)
        analyze(hw, tokens, program, bb, 0, cancel);

#ifdef NGEN_DEBUG
    std::cerr << "ANALYZE PHASE 0\n";
    std::cerr << "---------------\n";
    for (auto &bb : bbList) {
        std::cerr << "Basic Block " << bb.id << std::endl;
        bb.consumers.dump();
        bb.producers.dump();
        std::cerr << std::endl;
    }
#endif

    // Propagate OOO dependency producers through BB graph.
    propagate(bbList, cancel);
    for (auto &bb : bbList) {
        bb.producers.clear();
        bb.consumers.clear();
    }

#ifdef NGEN_DEBUG
    std::cerr << "PROPAGATE\n";
    std::cerr << "---------\n";
    for (auto &bb : bbList) {
        std::cerr << "Basic Block " << bb.id << std::endl;
        bb.incoming.dump();
        std::cerr << std::endl;
    }
#endif

    // Analysis round 1: assign SBIDs and perform intra-BB analysis.
    for (auto &bb : bbList) {
        analyze(hw, tokens, program, bb, 1, cancel);
        bb.incoming.clear();
    }

#ifdef NGEN_DEBUG
    std::cerr << "ANALYZE PHASE 1\n";
    std::cerr << "---------------\n";
    for (auto &bb : bbList) {
        std::cerr << "Basic Block " << bb.id << std::endl;
        bb.consumers.dump();
        bb.producers.dump();
        std::cerr << std::endl;
    }
#endif

    // Loop optimization.
    for (auto &bb : bbList)
        loopOptimize(bb);

    // Propagate live dependency producers through BB graph.
    propagate(bbList, cancel);

    for (auto &bb : bbList) {
        std::swap(bb.incoming, bb.producers);
        bb.incoming.clear();
    }

#ifdef NGEN_DEBUG
    std::cerr << "PROPAGATE\n";
    std::cerr << "---------\n";
    for (auto &bb : bbList) {
        std::cerr << "Basic Block " << bb.id << std::endl;
        bb.producers.dump();
        std::cerr << std::endl;
    }
#endif

    // Analysis round 2: final SWSB assignment.
    for (auto &bb : bbList)
        analyze(hw, tokens, program, bb, 2, cancel);

#ifdef NGEN_DEBUG
    std::cerr << "ANALYZE PHASE 2\n";
    std::cerr << "---------------\n";
    for (auto &bb : bbList) {
        std::cerr << "Basic Block " << bb.id << std::endl;
        bb.consumers.dump();
        bb.producers.dump();
        std::cerr << std::endl;
    }
#endif

    // Adjust jump targets after sync insertions.
    adjustTargets(hw, program, bbList);

    return bbList;
}

} /* namespace autoswsb */
} /* namespace NGEN_NAMESPACE */

// Instruction interface:
// 	SWSBInfo swsb() const;
// 	void setSWSB(SWSBInfo swsb) const;
// 	Opcode opcode() const;
// 	SyncFunction syncFC() const;
//  SharedFunction sfid() const;
// 	DestinationMask destinations(int &jip, int &uip) const;
// 	bool getOperandRegion(DependencyRegion &region, int opNum) const; // returns false if no such operand.
// 	bool getImm32(uint32_t &imm) const;
//
// Program interface:
// 	Instruction operator[](int inum);
// 	size_t size() const;

#endif /* NGEN_AUTOSWSB_HPP */
