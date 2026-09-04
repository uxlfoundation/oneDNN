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

#ifndef COMMON_BIT_PAIR_HPP
#define COMMON_BIT_PAIR_HPP

#include <cassert>
#include <cstdint>

namespace dnnl {
namespace impl {

// An abstraction to manipulate with bits as bytes. `4` means there are four
// elements in it.
struct bitpair4_t {
    // constructs a nibble pair from a pair of uint8_t values
    bitpair4_t(uint8_t v0_, uint8_t v1_, uint8_t v2_, uint8_t v3_)
        : v0(v0_), v1(v1_), v2(v2_), v3(v3_) {}

    // constructs a nibble pairs from an uin8_t, taking its low and high part
    bitpair4_t(uint8_t pack_)
        : v0(pack_ & 0x3)
        , v1((pack_ >> 2) & 0x3)
        , v2((pack_ >> 4) & 0x3)
        , v3((pack_ >> 6) & 0x3) {}

    // sets low (idx=0) or high (idx=1)  nibble.
    inline void set(uint8_t val, int idx) {
        switch (idx) {
            case 0: v0 = val; return;
            case 1: v1 = val; return;
            case 2: v2 = val; return;
            case 3: v3 = val; return;
            default: assert(!"Out of range index"); return;
        }
    }

    // returns low (idx = 0) or high (idx = 1) nibble in a uint8_t
    inline uint8_t get(int idx) const {
        switch (idx) {
            case 0: return v0;
            case 1: return v1;
            case 2: return v2;
            case 3: return v3;
            default: assert(!"out of range index"); return 0;
        }
    }

    // returns pair of nibbles as uint8_t
    inline uint8_t get() const { return static_cast<uint8_t>(v3 << 6 | v2 << 4 | v1 << 2 | v0); }

    // Returns a size of a nibble object in bytes.
    static constexpr size_t size() { return 1; }

    // Returns the number of elements in this type of nibble.
    static constexpr int nelems() { return 2; }

private:
    uint8_t v0 : 2;
    uint8_t v1 : 2;
    uint8_t v2 : 2;
    uint8_t v3 : 2;
};
static_assert(sizeof(bitpair4_t) == 1, "bitpair4_t must be 1 byte");
static_assert(bitpair4_t::size() == 1, "bitpair4_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
