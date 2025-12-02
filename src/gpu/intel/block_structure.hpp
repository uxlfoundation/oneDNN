/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_INTEL_BLOCK_STRUCTURE_HPP
#define GPU_INTEL_BLOCK_STRUCTURE_HPP

#include <sstream>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"
#include "gemmstone/dsl/tensor.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

using stride_t = jit::dsl::stride_t;

namespace compute {
template <>
struct scalar_type_traits_t<stride_t> {
    static const auto type = scalar_type_t::_long;
};
} // namespace compute
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(stride_t);

static constexpr dim_idx_t undefined_dim_idx = -1;

struct block_t {
    block_t() = default;

    block_t(dim_idx_t dim_idx, dim_t block, const stride_t &stride)
        : dim_idx(dim_idx), block(block), stride(stride) {}

    bool can_merge(const block_t &other, bool same_dim_only = true) const {
        bool dim_ok = !same_dim_only || (dim_idx == other.dim_idx);
        bool is_dense = (stride * block == other.stride);
        return dim_ok && is_dense;
    }

#if __cplusplus >= 202002L
    // Enabling default operator== on C++20 for validation purposes.
    bool operator==(const block_t &) const = default;
#else
    bool operator==(const block_t &other) const {
        return (dim_idx == other.dim_idx) && (block == other.block)
                && (stride == other.stride);
    }
#endif
    bool operator!=(const block_t &other) const { return !(*this == other); }

    size_t get_hash() const { return serialization_stream_t::get_hash(*this); }

    std::string str() const {
        ostringstream_t oss;
        oss << "block_t(dim_idx = " << dim_idx;
        oss << ", block = " << block;
        oss << ", stride = " << stride.str();
        oss << ")";
        return oss.str();
    }

    bool is_empty() const { return dim_idx == undefined_dim_idx; }

    dim_idx_t dim_idx = undefined_dim_idx; // Dimension index.
    uint8_t pad[4] = {};
    dim_t block = 1; // Block size.
    stride_t stride; // Stride between elements of the block.
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(block_t);

// Static-sized layout of blocks
struct block_layout_t {
    bool operator==(const block_layout_t &other) const {
        if (num_blocks != other.num_blocks) return false;
        return blocks == other.blocks;
    }
    bool operator!=(const block_layout_t &other) const {
        return !operator==(other);
    }

    using value_type = std::array<block_t, DNNL_MAX_NDIMS>;
    using iterator = value_type::iterator;
    using reverse_iterator = value_type::reverse_iterator;
    using const_iterator = value_type::const_iterator;
    using const_reverse_iterator = value_type::const_reverse_iterator;

    block_layout_t() = default;
    block_layout_t(const memory_desc_wrapper &mdw, bool inner_only = false,
            bool do_normalize = true);

    size_t size() const { return num_blocks; }
    bool empty() const { return num_blocks == 0; }
    const block_t &front() const {
        gpu_assert(num_blocks > 0);
        return blocks[0];
    }
    block_t &front() {
        gpu_assert(num_blocks > 0);
        return blocks[0];
    }
    const block_t &back() const {
        gpu_assert(num_blocks > 0);
        return blocks[num_blocks - 1];
    }
    block_t &back() {
        gpu_assert(num_blocks > 0);
        return blocks[num_blocks - 1];
    }

    // Iterators only go up to num_blocks, not necessarily to DNNL_MAX_NDIMS
    iterator begin() { return blocks.begin(); }
    const_iterator begin() const { return blocks.begin(); }
    reverse_iterator rbegin() {
        return blocks.rbegin() + static_cast<long>(blocks.size() - num_blocks);
    }
    const_reverse_iterator rbegin() const {
        return blocks.rbegin() + static_cast<long>(blocks.size() - num_blocks);
    }
    iterator end() { return blocks.begin() + num_blocks; }
    const_iterator end() const { return blocks.begin() + num_blocks; }
    reverse_iterator rend() { return blocks.rend(); }
    const_reverse_iterator rend() const { return blocks.rend(); }

    void erase(size_t idx) {
        for (size_t i = idx + 1; i < num_blocks; i++) {
            blocks[i - 1] = blocks[i];
        }
        blocks[num_blocks] = block_t();
        num_blocks--;
    }

    void insert(size_t idx, block_t val) {
        assert(num_blocks + 1 < DNNL_MAX_NDIMS);
        for (size_t i = idx; i < num_blocks; i++) {
            std::swap(val, blocks[i]);
        }
        append(val);
    }

    const block_t &operator[](size_t idx) const { return blocks[idx]; }

    void append(const block_t &block) { blocks[num_blocks++] = block; }
    size_t get_hash() const { return serialization_stream_t::get_hash(*this); }

    block_t &operator[](size_t idx) {
        assert(idx < num_blocks);
        return blocks[idx];
    }

    std::string str() const {
        ostringstream_t ss;
        for (size_t i = 0; i < num_blocks; i++) {
            const auto &block = blocks[i];
            ss << block.str() << " ";
        }
        return ss.str();
    }

    block_layout_t normalized(bool remove_size_1_blocks = true) const;

private:
    size_t num_blocks = 0;
    value_type blocks;
};

// Alias for block_layout_t::normalized which should be removed once jit::ir
// supports block_layout_t in favor of std::vector<block_t>
std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks = true);

block_layout_t get_inner_layout(const memory_desc_wrapper &md);

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
