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

#ifndef COMMON_NIBBLE_HPP
#define COMMON_NIBBLE_HPP

#include <cassert>
#include <cstdint>

namespace dnnl {
namespace impl {

// An abstraction to manipulate with bits as bytes. `2` means there are two
// elements in it.
struct nibble2_t {
    // constructs a nibble pair from a pair of uint8_t values
    nibble2_t(uint8_t low_, uint8_t high_) : low(low_), high(high_) {}

    // constructs a nibble pairs from an uin8_t, taking its low and high part
    nibble2_t(uint8_t pack_) : low(pack_ & 0xf), high((pack_ >> 4) & 0xf) {}

    // sets low (idx=0) or high (idx=1)  nibble.
    inline void set(uint8_t val, int idx) {
        switch (idx) {
            case 0: low = val; return;
            case 1: high = val; return;
            default: assert(!"Out of range index"); return;
        }
    }

    // returns low (idx = 0) or high (idx = 1) nibble in a uint8_t
    inline uint8_t get(int idx) const {
        switch (idx) {
            case 0: return low;
            case 1: return high;
            default: assert(!"out of range index"); return 0;
        }
    }

    // returns pair of nibbles as uint8_t
    inline uint8_t get() const { return static_cast<uint8_t>(high << 4 | low); }

    // Returns a size of a nibble object in bytes.
    static constexpr size_t size() { return 1; }

    // Returns the number of elements in this type of nibble.
    static constexpr int nelems() { return 2; }

private:
    uint8_t low : 4;
    uint8_t high : 4;
};
static_assert(sizeof(nibble2_t) == 1, "nibble2_t must be 1 byte");
static_assert(nibble2_t::size() == 1, "nibble2_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
