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

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Dynamically sized bitset. Follows the `std::bitset` interface, plus `-=`
// (set difference) from `boost::dynamic_bitset`.
struct bitset_t {
    // Creates `size` bits, all cleared.
    explicit bitset_t(int size) : words_((size + 63) / 64, 0) {}

    // Returns bit `pos`.
    bool test(int pos) const { return (words_[pos / 64] >> (pos % 64)) & 1; }

    // Sets bit `pos`.
    bitset_t &set(int pos) {
        words_[pos / 64] |= uint64_t(1) << (pos % 64);
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

    // Clears the bits that are set in `other`.
    bitset_t &operator-=(const bitset_t &other) {
        assert(words_.size() == other.words_.size());
        for (size_t w = 0; w < words_.size(); w++) {
            words_[w] &= ~other.words_[w];
        }
        return *this;
    }

    // True if any bit differs from `other`.
    bool operator!=(const bitset_t &other) const {
        return words_ != other.words_;
    }

private:
    // Bit `pos` is bit `pos % 64` of `words_[pos / 64]`.
    std::vector<uint64_t> words_;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
