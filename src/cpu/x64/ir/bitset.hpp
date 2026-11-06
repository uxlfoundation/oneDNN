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

#ifndef CPU_X64_IR_BITSET_HPP
#define CPU_X64_IR_BITSET_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Dynamically sized bitset. Follows the `std::bitset` interface, plus
// `find_first()` and `find_next()` from `boost::dynamic_bitset`.
struct bitset_t {
    static constexpr int npos = -1;

    // Creates `size` bits, all cleared.
    explicit bitset_t(int size) : words_((size + 63) / 64, 0) {}

    // Returns bit `pos`.
    bool test(int pos) const { return (words_[pos / 64] >> (pos % 64)) & 1; }

    // Sets bit `pos`.
    bitset_t &set(int pos) {
        words_[pos / 64] |= uint64_t(1) << (pos % 64);
        return *this;
    }

    // Clears bit `pos`.
    bitset_t &reset(int pos) {
        words_[pos / 64] &= ~(uint64_t(1) << (pos % 64));
        return *this;
    }

    // Clears all bits.
    bitset_t &reset() {
        std::fill(words_.begin(), words_.end(), 0);
        return *this;
    }

    // Sets the bits that are set in `other`.
    bitset_t &operator|=(const bitset_t &other) {
        assert(words_.size() == other.words_.size());
        for (size_t w = 0; w < words_.size(); w++) {
            words_[w] |= other.words_[w];
        }
        return *this;
    }

    // True if any bit differs from `other`.
    bool operator!=(const bitset_t &other) const {
        return words_ != other.words_;
    }

    // Index of the first set bit, or `npos` if there is none.
    int find_first() const { return find_from(0); }

    // Index of the next set bit after `pos`, or `npos` if there is none.
    int find_next(int pos) const { return find_from(pos + 1); }

private:
    int find_from(int pos) const {
        int w = pos / 64;
        if (w >= (int)words_.size()) return npos;
        // Mask out the bits below `pos` and skip empty words.
        uint64_t bits = words_[w] & (~uint64_t(0) << (pos % 64));
        while (bits == 0) {
            if (++w == (int)words_.size()) return npos;
            bits = words_[w];
        }
        // Index of the lowest set bit is the number of trailing zeros.
#ifdef _MSC_VER
        unsigned long idx;
        _BitScanForward64(&idx, bits);
        return w * 64 + (int)idx;
#else
        return w * 64 + __builtin_ctzll(bits);
#endif
    }

    // Bit `pos` is bit `pos % 64` of `words_[pos / 64]`.
    std::vector<uint64_t> words_;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
