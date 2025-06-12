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
sdpa_config_t xehpg_fma_h32 = {16, 16, 8, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_fma_h32_2nd = {32, 16, 8, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_fma_h64 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpg_fma_h64_2nd = {16, 16, 32, 8, 32, 1, 16, 2};

sdpa_config_t xehpg_fma_h80_s634 = {8, 16, 16, 16, 8, 4, 8, 4};

sdpa_config_t xehpg_fma_h128_s2048 = {8, 16, 16, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_fma_h128_2nd = {32, 16, 32, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_fma_h256 = {8, 32, 64, 8, 16, 1, 4, 4};
sdpa_config_t xehpg_fma_h256_2nd = {16, 8, 16, 8, 32, 1, 32, 1};

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
sdpa_config_t xe2_h512_integrated = {16, 16, 16, 16, 32, 1, 32, 1};

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

sdpa_config_t *choose_config_xehpg_fma(
        dim_t head_size, dim_t seq, bool thin_q, bool quantized) {
    if (quantized) { return nullptr; }
    if (head_size <= 32) {
        if (thin_q) return &xehpg_fma_h32_2nd;
        return &xehpg_fma_h32;
    } else if (head_size <= 64) {
        if (thin_q) return &xehpg_fma_h64_2nd;
        return &xehpg_fma_h64;
    } else if (head_size <= 80) {
        return &xehpg_fma_h80_s634;
    } else if (head_size <= 128) {
        if (thin_q) { return &xehpg_fma_h128_2nd; }
        return &xehpg_fma_h128_s2048;
    } else if (head_size <= 256) {
        if (thin_q) { return &xehpg_fma_h256_2nd; }
        return &xehpg_fma_h256;
    } // no supported tile configurations for 512+
    return nullptr;
}

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
static std::vector<dim_t> seq_intervals_xehpg = {32, 64, 96, 128, 256, 512};

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
static std::vector<dim_t> seq_intervals_xehpc
        = {32, 64, 96, 128, 256, 384, 512, 768, 1024, 1152};

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
static std::vector<dim_t> seq_intervals_xe2
        = {64, 96, 128, 256, 384, 512, 768, 1024, 1152};

// adjust heuristic intervals to match the tuned intervals according
// to the sequence length and gpu architecture
// this way recompilation both matches the tuned intervals and avoids
// excessive recompilation with smaller power of 2 sizes
dim_t round_up_seq_interval(dim_t seq, compute::gpu_arch_t arch) {
    const std::vector<dim_t> *seq_intervals;
    switch (arch) {
        case compute::gpu_arch_t::xe_hpg:
            seq_intervals = &seq_intervals_xehpg;
            break;
        case compute::gpu_arch_t::xe_hpc:
            seq_intervals = &seq_intervals_xehpc;
            break;
        case compute::gpu_arch_t::xe2:
        case compute::gpu_arch_t::xe3: seq_intervals = &seq_intervals_xe2;
        default: return utils::rnd_up_pow2(seq);
    }

    for (auto seq_boundary : *seq_intervals) {
        if (seq <= seq_boundary) { return seq_boundary; }
    }
    return utils::rnd_up_pow2(seq);
}

void deserialize_config_to_gemmstone(gemmstone::HWInformation &hwInfo,
        gemmstone::GEMMProblem &problem_kq, gemmstone::GEMMProblem &problem_vs,
        micro::GEMMProtocol::Options &opts_kq,
        micro::GEMMProtocol::Options &opts_vs, gemmstone::SizeParams &sizes_kq,
        gemmstone::SizeParams &sizes_vs,
        const micro_sdpa_ukernel_params_t &ukernel_config) {

    // hardware info
    hwInfo.gmdid = ukernel_config.hwinfo.gmdid;
    hwInfo.euCount = ukernel_config.hwinfo.euCount;
    hwInfo.systolicAvailable = ukernel_config.hwinfo.systolicAvailable;

    // options kq, vs
    auto deserialize_options
            = [](micro::GEMMProtocol::Options &gemmstone_opts,
                      const ukernel_serialized_opts_t &serialized_opts) {
                  gemmstone_opts.localB = serialized_opts.localB;
                  gemmstone_opts.slmPtr = serialized_opts.slmPtr;
                  gemmstone_opts.scaleA = serialized_opts.scaleA;
                  gemmstone_opts.offsetA = serialized_opts.offsetA;
              };
    deserialize_options(opts_kq, ukernel_config.opts_kq);
    deserialize_options(opts_vs, ukernel_config.opts_vs);

    // problems kq, vs
    auto deserialize_problem = [](gemmstone::GEMMProblem &problem,
                                       const ukernel_serialized_problem_t
                                               &serialized_problem) {
        problem.Ta_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Ta_ext)};
        problem.Tb_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Tb_ext)};
        problem.Ta
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Ta)};
        problem.Tb
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tb)};
        problem.Tc_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Tc_ext)};
        problem.Tc
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tc)};
        problem.Ts
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Ts)};
        problem.A.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.A_layout);

        problem.Ta_scale = {static_cast<gemmstone::Type::_Type>(
                serialized_problem.Ta_scale)};
        problem.A_scale.setAlignment(serialized_problem.A_scale_alignment);
        problem.A_scale.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.A_scale_layout);
        problem.asPtrDims = serialized_problem.asPtrDims;
        problem.Tao
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tao)};
        problem.AO.setAlignment(serialized_problem.AO_alignment);
        problem.AO.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.AO_layout);
        problem.aoPtrDims = serialized_problem.aoPtrDims;
        problem.aOffset
                = static_cast<gemmstone::ABOffset>(serialized_problem.aOffset);
        problem.aqGroupM = serialized_problem.aqGroupM;
        problem.aqGroupK = serialized_problem.aqGroupK;

        problem.B.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.B_layout);
        problem.C.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.C_layout);
        problem.A.setAlignment(serialized_problem.A_alignment);
        problem.B.setAlignment(serialized_problem.B_alignment);
        problem.B.crosspack = serialized_problem.B_crosspack;
        problem.B.tileR = serialized_problem.B_tileR;
        problem.B.tileC = serialized_problem.B_tileC;
    };
    deserialize_problem(problem_kq, ukernel_config.problem_kq);
    deserialize_problem(problem_vs, ukernel_config.problem_vs);

    // sizes kq, vs
    auto deserialize_sizes
            = [](gemmstone::SizeParams &sizes,
                      const ukernel_serialized_sizes_t &serialized_sizes) {
                  sizes.m = serialized_sizes.m;
                  sizes.n = serialized_sizes.n;
                  sizes.k = serialized_sizes.k;
                  sizes.batch = serialized_sizes.batch;
              };
    deserialize_sizes(sizes_kq, ukernel_config.sizes_kq);
    deserialize_sizes(sizes_vs, ukernel_config.sizes_vs);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
