/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "micro_sdpa_configs.hpp"
#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// Kernel configurations:
//  h<N> -- maximum head size = N
//  s<M> -- target sequence length = M
//   2nd -- second token (thin Q)
sdpa_config_t xehpg_h32 = {32, 16, 16, 16, 2, 16, 2, 16};
sdpa_config_t xehpg_h32_s256 = {16, 16, 16, 16, 2, 8, 2, 8};
sdpa_config_t xehpg_h32_s64 = {16, 16, 16, 8, 4, 4, 2, 8};
sdpa_config_t xehpg_h32_s32 = {8, 8, 8, 8, 4, 4, 4, 4};
sdpa_config_t xehpg_h32_2nd = {8, 32, 16, 8, 8, 1, 2, 4};

sdpa_config_t xehpg_q_h32 = {32, 16, 16, 16, 2, 8, 2, 8};
sdpa_config_t xehpg_q_h32_2nd = {32, 16, 8, 8, 8, 1, 4, 2};

sdpa_config_t xehpg_h64 = {32, 16, 16, 16, 4, 8, 4, 8};
sdpa_config_t xehpg_h64_s128 = {16, 16, 16, 16, 4, 8, 4, 8};
sdpa_config_t xehpg_h64_s64 = {32, 16, 16, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_h64_2nd = {8, 16, 16, 8, 8, 1, 4, 2};

sdpa_config_t xehpg_q_h64 = {32, 16, 16, 16, 4, 8, 4, 8};
sdpa_config_t xehpg_q_h64_s128 = {16, 16, 16, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_q_h64_s64 = {32, 8, 32, 8, 2, 8, 2, 8};
sdpa_config_t xehpg_q_h64_s32 = {8, 8, 16, 8, 4, 8, 4, 8};

sdpa_config_t xehpg_q_h64_s64_2nd = {8, 8, 8, 8, 8, 2, 8, 2};
sdpa_config_t xehpg_q_h64_s128_2nd = {16, 8, 8, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h64_2nd = {16, 16, 8, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_h128 = {16, 16, 32, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_h128_s32 = {16, 16, 16, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h128_2nd = {8, 16, 16, 8, 16, 1, 8, 2};
sdpa_config_t xehpg_h128_s256_2nd = {8, 16, 32, 8, 8, 1, 4, 2};

sdpa_config_t xehpg_q_h128 = {8, 32, 16, 32, 8, 2, 8, 2};
sdpa_config_t xehpg_q_h128_s64 = {8, 8, 16, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h128_s512 = {16, 16, 16, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h128_2nd = {16, 16, 16, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_q_h128_s96_2nd = {8, 8, 8, 8, 16, 2, 16, 2};

sdpa_config_t xehpg_h256 = {16, 16, 32, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h256_s128 = {8, 16, 32, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_h256_s32 = {8, 16, 32, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_q_h256 = {16, 16, 64, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_q_h256_s512 = {16, 16, 32, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h256_s64 = {8, 8, 32, 8, 8, 4, 8, 4};

sdpa_config_t xehpg_h256_2nd = {8, 8, 16, 8, 16, 1, 16, 1};
sdpa_config_t xehpg_h256_s64_2nd = {16, 8, 16, 8, 16, 1, 16, 1};
sdpa_config_t xehpg_h256_s32_2nd = {16, 16, 32, 8, 16, 1, 8, 2};

sdpa_config_t xehpg_q_h256_2nd = {32, 8, 32, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h256_s96_2nd = {8, 8, 16, 8, 16, 2, 16, 2};

sdpa_config_t xehpg_q_h512_s64 = {8, 8, 64, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h512_s128 = {8, 16, 32, 16, 16, 2, 16, 2};
sdpa_config_t xehpg_q_h512_s256 = {16, 8, 64, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h512 = {8, 16, 64, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_q_h512_s64_2nd = {8, 16, 32, 8, 32, 1, 16, 2};
sdpa_config_t xehpg_q_h512_s256_2nd = {16, 8, 32, 8, 16, 2, 16, 2};
sdpa_config_t xehpg_q_h512_2nd = {16, 8, 16, 8, 32, 1, 32, 1};

sdpa_config_t xehpg_h512 = {8, 16, 32, 16, 16, 2, 16, 2};
sdpa_config_t xehpg_h512_2nd = {8, 8, 32, 8, 16, 1, 16, 1};

sdpa_config_t xehpc_h32 = {16, 64, 32, 16, 4, 2, 1, 8};
sdpa_config_t xehpc_h32_s32 = {16, 16, 16, 16, 2, 4, 2, 4};
sdpa_config_t xehpc_h32_2nd = {16, 64, 16, 16, 8, 1, 2, 4};

sdpa_config_t xehpc_h64 = {16, 64, 32, 16, 8, 2, 2, 8};
sdpa_config_t xehpc_h64_s64 = {32, 32, 32, 16, 4, 2, 2, 4};
sdpa_config_t xehpc_h64_s32 = {16, 16, 16, 16, 4, 2, 4, 2};
sdpa_config_t xehpc_h64_2nd = {32, 32, 32, 16, 4, 1, 2, 2};
sdpa_config_t xehpc_h64_s64_2nd = {16, 16, 16, 16, 4, 1, 4, 1};

sdpa_config_t xehpc_q_h64_s64 = {16, 16, 16, 16, 4, 4, 4, 4};
sdpa_config_t xehpc_q_h64_s384 = {16, 64, 16, 32, 8, 2, 4, 4};
sdpa_config_t xehpc_q_h64_s1024 = {16, 64, 16, 16, 16, 1, 4, 4};
sdpa_config_t xehpc_q_h64 = {16, 64, 16, 32, 8, 1, 4, 2};

sdpa_config_t xehpc_q_h64_s96_2nd = {16, 16, 16, 16, 8, 1, 4, 1};
sdpa_config_t xehpc_q_h64_s256_2nd = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h64_s1152_2nd = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h64_2nd = {64, 16, 16, 16, 16, 2, 16, 2};

sdpa_config_t xehpc_h128 = {16, 64, 32, 16, 16, 2, 4, 8};
sdpa_config_t xehpc_h128_s64 = {16, 32, 32, 32, 4, 2, 4, 2};
sdpa_config_t xehpc_h128_s32 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h128_2nd = {32, 32, 32, 16, 8, 1, 4, 2};

sdpa_config_t xehpc_q_h128 = {16, 64, 16, 32, 16, 1, 8, 2};
sdpa_config_t xehpc_q_h128_s32 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_q_h128_s128 = {16, 16, 16, 16, 8, 4, 8, 4};
sdpa_config_t xehpc_q_h128_s128_integrated = {16, 16, 16, 16, 8, 2, 8, 2};

sdpa_config_t xehpc_q_h128_2nd = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h128_2nd_integrated = {16, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_q_h128_s96_2nd = {16, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_q_h128_s512_2nd = {16, 16, 16, 16, 16, 2, 8, 2};

sdpa_config_t xehpc_h256 = {16, 32, 32, 32, 8, 4, 8, 4};
sdpa_config_t xehpc_h256_s64 = {16, 32, 32, 32, 8, 1, 8, 1};
sdpa_config_t xehpc_h256_2nd = {16, 16, 16, 16, 16, 1, 16, 1};

sdpa_config_t xehpc_h512_s32 = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h512_s128 = {16, 16, 64, 16, 8, 4, 8, 4};
sdpa_config_t xehpc_h512 = {32, 16, 64, 16, 8, 4, 8, 4};

sdpa_config_t xehpc_h512_s128_2nd = {16, 16, 64, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_h512_s512_2nd = {32, 16, 32, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_h512_s1024_2nd = {64, 16, 32, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_h512_2nd = {32, 16, 32, 16, 16, 1, 16, 1};

sdpa_config_t xehpc_h576 = {16, 32, 32, 32, 32, 1, 32, 1};
sdpa_config_t xehpc_h576_2nd = {32, 16, 32, 16, 32, 1, 31, 1};

sdpa_config_t xehpc_q_h512_s128 = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_q_h512 = {16, 32, 64, 16, 16, 2, 8, 4};

sdpa_config_t xehpc_q_h512_2nd = {16, 16, 32, 16, 16, 2, 16, 2};

sdpa_config_t xe2_q_h64 = {16, 64, 16, 32, 16, 1, 8, 2};
sdpa_config_t xe2_q_h64_s1024_integrated = {16, 64, 16, 32, 8, 4, 4, 8};
sdpa_config_t xe2_q_h64_s512 = {16, 64, 16, 32, 8, 4, 4, 8};
sdpa_config_t xe2_q_h64_s384 = {16, 64, 16, 16, 16, 1, 4, 4};
sdpa_config_t xe2_q_h64_s128 = {16, 64, 16, 32, 8, 1, 4, 2};
sdpa_config_t xe2_q_h64_s128_integrated = {16, 16, 16, 16, 4, 4, 4, 4};
sdpa_config_t xe2_q_h64_s32 = {16, 16, 16, 16, 4, 4, 4, 4};

sdpa_config_t xe2_q_h64_2nd = {16, 16, 16, 16, 16, 1, 8, 1};
sdpa_config_t xe2_q_h64_2nd_integrated = {16, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xe2_q_h64_s96_2nd_integrated = {16, 16, 16, 16, 8, 1, 4, 1};
sdpa_config_t xe2_q_h64_s384_2nd_integrated = {64, 16, 16, 16, 4, 1, 4, 1};
sdpa_config_t xe2_q_h64_s64_2nd = {16, 16, 16, 16, 4, 2, 4, 2};
sdpa_config_t xe2_q_h64_s128_2nd = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xe2_q_h64_s384_2nd = {16, 16, 16, 16, 16, 1, 4, 1};
sdpa_config_t xe2_q_h64_s512_2nd = {64, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xe2_q_h64_s768_2nd = {64, 16, 16, 16, 16, 1, 8, 1};

sdpa_config_t xe2_q_h256 = {16, 64, 16, 32, 32, 1, 16, 2};
sdpa_config_t xe2_q_h256_s384 = {16, 32, 32, 32, 8, 2, 8, 2};
sdpa_config_t xe2_q_h256_s128 = {16, 32, 32, 32, 8, 1, 8, 1};
sdpa_config_t xe2_q_h256_s128_integrated = {16, 32, 32, 32, 8, 2, 8, 2};
sdpa_config_t xe2_q_h256_s64_integrated = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xe2_q_h256_s64 = {16, 32, 64, 16, 8, 2, 4, 4};

sdpa_config_t xe2_q_h256_2nd_integrated = {32, 16, 64, 16, 4, 1, 4, 1};
sdpa_config_t xe2_q_h256_s1152_2nd_integrated = {16, 16, 64, 16, 4, 1, 4, 1};
sdpa_config_t xe2_q_h256_s768_2nd_integrated = {64, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xe2_q_h256_s512_2nd_integrated = {32, 32, 32, 16, 16, 1, 8, 2};
sdpa_config_t xe2_q_h256_s384_2nd_integrated = {16, 16, 16, 16, 16, 1, 16, 1};

sdpa_config_t xe2_h512_s64 = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xe2_h512 = {32, 16, 64, 16, 8, 4, 8, 4};

sdpa_config_t xe2_h512_s128_2nd = {16, 16, 64, 16, 8, 1, 8, 1};
sdpa_config_t xe2_h512_s512_2nd = {32, 16, 64, 16, 16, 1, 16, 1};
sdpa_config_t xe2_h512_s1024_2nd = {64, 16, 32, 16, 16, 2, 16, 2};
sdpa_config_t xe2_h512_2nd = {32, 16, 64, 16, 16, 1, 16, 1};

sdpa_config_t xe2_q_h512_s128 = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xe2_q_h512 = {16, 32, 64, 16, 16, 2, 8, 4};

sdpa_config_t xe2_q_h512_s64_2nd = {16, 16, 64, 16, 8, 1, 8, 1};
sdpa_config_t xe2_q_h512_2nd = {16, 16, 64, 16, 16, 1, 16, 1};

sdpa_config_t xe2_h512_s128_integrated = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xe2_h512_integrated = {16, 16, 32, 16, 16, 1, 16, 1};

sdpa_config_t xe2_h512_s256_2nd_integrated = {16, 16, 64, 16, 8, 1, 8, 1};
sdpa_config_t xe2_h512_s1024_2nd_integrated = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xe2_h512_2nd_integrated = {16, 16, 64, 16, 16, 2, 16, 2};

sdpa_config_t xe2_h576 = {16, 32, 32, 32, 32, 1, 32, 1};

sdpa_config_t xe2_q_h512_integrated = {16, 32, 32, 32, 16, 1, 16, 1};

sdpa_config_t xe2_q_h512_s64_2nd_integrated = {16, 32, 64, 32, 16, 2, 8, 2};
sdpa_config_t xe2_q_h512_s128_2nd_integrated = {16, 16, 64, 16, 8, 1, 32, 1};
sdpa_config_t xe2_q_h512_s256_2nd_integrated = {16, 32, 64, 32, 16, 2, 8, 2};
sdpa_config_t xe2_q_h512_s512_2nd_integrated = {16, 16, 64, 16, 4, 4, 8, 4};
sdpa_config_t xe2_q_h512_s1024_2nd_integrated = {16, 16, 64, 16, 16, 1, 16, 1};
sdpa_config_t xe2_q_h512_2nd_integrated = {32, 16, 64, 16, 8, 1, 16, 1};

sdpa_config_t *choose_config_xehpg(
        dim_t head_size, dim_t seq, bool thin_q, bool quantized) {
    if (head_size <= 32) {
        if (quantized && seq >= 128) {
            if (thin_q) return &xehpg_q_h32_2nd;
            return &xehpg_q_h32;
        }
        if (thin_q) return &xehpg_h32_2nd;
        if (seq <= 32) return &xehpg_h32_s32;
        if (seq <= 64) return &xehpg_h32_s64;
        if (seq <= 256) return &xehpg_h32_s256;
        return &xehpg_h32;
    } else if (head_size <= 64) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 64) return &xehpg_q_h64_s64_2nd;
                if (seq <= 128) return &xehpg_q_h64_s128_2nd;
                return &xehpg_q_h64_2nd;
            } else {
                if (seq <= 32) return &xehpg_q_h64_s32;
                if (seq <= 64) return &xehpg_q_h64_s64;
                if (seq <= 128) return &xehpg_q_h64_s128;
                return &xehpg_q_h64;
            }
        }
        if (thin_q) return &xehpg_h64_2nd;
        if (seq <= 64) return &xehpg_h64_s64;
        if (seq <= 128) return &xehpg_h64_s128;
        return &xehpg_h64;
    } else if (head_size <= 128) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 96) return &xehpg_q_h128_s96_2nd;
                return &xehpg_q_h128_2nd;
            }
            if (seq <= 64) return &xehpg_q_h128_s64;
            if (seq <= 512) return &xehpg_q_h128_s512;
            return &xehpg_q_h128;
        }
        if (thin_q) {
            if (seq <= 256) return &xehpg_h128_s256_2nd;
            return &xehpg_h128_2nd;
        }
        if (seq <= 32) return &xehpg_h128_s32;
        return &xehpg_h128;
    } else if (head_size <= 256) {
        if (thin_q) {
            if (quantized) {
                if (seq <= 96) return &xehpg_q_h256_s96_2nd;
                return &xehpg_q_h256_2nd;
            }
            if (seq <= 32) return &xehpg_h256_s32_2nd;
            if (seq <= 64) return &xehpg_h256_s64_2nd;
            return &xehpg_h256_2nd;
        }
        if (quantized) {
            if (seq <= 64) return &xehpg_q_h256_s64;
            if (seq <= 512) return &xehpg_q_h256_s512;
            return &xehpg_q_h256;
        }
        if (seq <= 32) return &xehpg_h256_s32;
        if (seq <= 128) return &xehpg_h256_s128;
        return &xehpg_h256;
    } else if (head_size <= 512) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 64) return &xehpg_q_h512_s64_2nd;
                if (seq <= 256) return &xehpg_q_h512_s256_2nd;
                return &xehpg_q_h512_2nd;
            }
            if (seq <= 64) return &xehpg_q_h512_s64;
            if (seq <= 128) return &xehpg_q_h512_s128;
            if (seq <= 256) return &xehpg_q_h512_s256;
            return &xehpg_q_h512;
        }
        if (thin_q) { return &xehpg_h512_2nd; }
        return &xehpg_h512;
    }
    return nullptr;
}

sdpa_config_t *choose_config_xehpc(dim_t head_size, dim_t seq, bool thin_q,
        bool quantized, bool is_integrated) {
    if (head_size <= 32) {
        if (thin_q) return &xehpc_h32_2nd;
        if (seq <= 32) return &xehpc_h32_s32;
        return &xehpc_h32;
    } else if (head_size <= 64) {
        if (thin_q) {
            if (quantized) {
                if (seq <= 96) return &xehpc_q_h64_s96_2nd;
                if (seq <= 256) return &xehpc_q_h64_s256_2nd;
                if (seq <= 1152) return &xehpc_q_h64_s1152_2nd;
                return &xehpc_q_h64_2nd;
            }

            if (seq <= 64) return &xehpc_h64_s64_2nd;
            return &xehpc_h64_2nd;
        }
        if (quantized) {
            if (seq <= 64) return &xehpc_q_h64_s64;
            if (seq <= 384) return &xehpc_q_h64_s384;
            if (seq <= 1024) return &xehpc_q_h64_s1024;
            return &xehpc_q_h64;
        }
        if (seq <= 32) return &xehpc_h64_s32;
        if (seq <= 64) return &xehpc_h64_s64;
        return &xehpc_h64;
    } else if (head_size <= 128) {
        if (quantized) {
            if (thin_q) {
                if (is_integrated) { return &xehpc_q_h128_2nd_integrated; }
                if (seq <= 96) return &xehpc_q_h128_s96_2nd;
                if (seq <= 512) return &xehpc_q_h128_s512_2nd;
                return &xehpc_q_h128_2nd;
            }
            if (is_integrated) {
                if (seq <= 128) { return &xehpc_q_h128_s128_integrated; }
            }
            if (seq <= 32) return &xehpc_q_h128_s32;
            if (seq <= 128) return &xehpc_q_h128_s128;
            return &xehpc_q_h128;
        }
        if (thin_q) return &xehpc_h128_2nd;
        if (seq <= 32) return &xehpc_h128_s32;
        if (seq <= 64) return &xehpc_h128_s64;
        return &xehpc_h128;
    } else if (head_size <= 256) {
        if (thin_q) return &xehpc_h256_2nd;
        if (seq <= 64) return &xehpc_h256_s64;
        return &xehpc_h256;
    } else if (head_size <= 512) {
        if (thin_q) {
            if (quantized) return &xehpc_q_h512_2nd;

            if (seq <= 128) return &xehpc_h512_s128_2nd;
            if (seq <= 512) return &xehpc_h512_s512_2nd;
            if (seq <= 1024) return &xehpc_h512_s1024_2nd;
            return &xehpc_h512_2nd;
        }

        if (quantized) {
            if (seq <= 128) return &xehpc_q_h512_s128;
            return &xehpc_q_h512;
        }
        if (seq <= 32) return &xehpc_h512_s32;
        if (seq <= 128) return &xehpc_h512_s128;
        return &xehpc_h512;
    } else if (head_size <= 576) {
        if (!quantized) {
            if (thin_q) return &xehpc_h576_2nd;
            return &xehpc_h576;
        }
    }
    return nullptr;
}

sdpa_config_t *choose_config_xe2(dim_t head_size, dim_t seq, bool thin_q,
        bool quantized, bool is_integrated) {
    if (head_size <= 64) {
        if (quantized) {
            if (thin_q) {
                if (is_integrated) {
                    if (seq <= 96) return &xe2_q_h64_s96_2nd_integrated;
                    if (seq <= 384) return &xe2_q_h64_s384_2nd_integrated;
                    return &xe2_q_h64_2nd_integrated;
                }
                if (seq <= 64) return &xe2_q_h64_s64_2nd;
                if (seq <= 128) return &xe2_q_h64_s128_2nd;
                if (seq <= 384) return &xe2_q_h64_s384_2nd;
                if (seq <= 512) return &xe2_q_h64_s512_2nd;
                if (seq <= 768) return &xe2_q_h64_s768_2nd;
                return &xe2_q_h64_2nd;
            }
            if (seq <= 32) return &xe2_q_h64_s32;
            if (is_integrated) {
                if (seq <= 128) return &xe2_q_h64_s128_integrated;
            }
            if (seq <= 128) return &xe2_q_h64_s128;
            if (seq <= 384) return &xe2_q_h64_s384;
            if (seq <= 512) return &xe2_q_h64_s512;
            if (is_integrated) {
                if (seq <= 1024) return &xe2_q_h64_s1024_integrated;
            }
            return &xe2_q_h64;
        }
    }

    if (head_size <= 128) {
        return choose_config_xehpc(
                head_size, seq, thin_q, quantized, is_integrated);
    }

    if (head_size <= 256) {
        if (quantized) {
            if (is_integrated) {
                if (thin_q) {
                    if (seq < 384) return &xe2_q_h256_s384_2nd_integrated;
                    if (seq < 512) return &xe2_q_h256_s512_2nd_integrated;
                    if (seq < 768) return &xe2_q_h256_s768_2nd_integrated;
                    if (seq < 1152) return &xe2_q_h256_s1152_2nd_integrated;
                    return &xe2_q_h256_2nd_integrated;
                }
                if (seq <= 64) return &xe2_q_h256_s64_integrated;
                if (seq <= 128) return &xe2_q_h256_s128_integrated;
            }
            if (!thin_q) {
                if (seq <= 64) return &xe2_q_h256_s64;
                if (seq <= 128) return &xe2_q_h256_s128;
                if (seq <= 384) return &xe2_q_h256_s384;
                return &xe2_q_h256;
            }
        }
    }

    if (head_size <= 512) {
        if (thin_q) {
            if (quantized) {
                if (is_integrated) {
                    if (seq <= 64) return &xe2_q_h512_s64_2nd_integrated;
                    if (seq <= 128) return &xe2_q_h512_s128_2nd_integrated;
                    if (seq <= 256) return &xe2_q_h512_s256_2nd_integrated;
                    if (seq <= 512) return &xe2_q_h512_s512_2nd_integrated;
                    if (seq <= 1024) return &xe2_q_h512_s1024_2nd_integrated;
                    return &xe2_q_h512_2nd_integrated;
                }
                if (seq <= 64) return &xe2_q_h512_s64_2nd;
                return &xe2_q_h512_2nd;
            }

            if (is_integrated) {
                if (seq <= 256) return &xe2_h512_s256_2nd_integrated;
                if (seq <= 1024) return &xe2_h512_s1024_2nd_integrated;
                return &xe2_h512_2nd_integrated;
            }
            if (seq <= 128) return &xe2_h512_s128_2nd;
            if (seq <= 512) return &xe2_h512_s512_2nd;
            if (seq <= 1024) return &xe2_h512_s1024_2nd;
            return &xe2_h512_2nd;
        }

        if (quantized) {
            if (is_integrated) return &xe2_q_h512_integrated;
            if (seq <= 128) return &xe2_q_h512_s128;
            return &xe2_q_h512;
        }
        if (is_integrated) {
            if (seq <= 128) return &xe2_h512_s128_integrated;
            return &xe2_h512_integrated;
        }
        if (seq <= 64) return &xe2_h512_s64;
        return &xe2_h512;
    }
    if (head_size <= 576) {
        if (!quantized) { return &xe2_h576; }
    }
    return choose_config_xehpc(
            head_size, seq, thin_q, quantized, is_integrated);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
