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

inline sdpa_property operator|(sdpa_property a, sdpa_property b) {
    return (sdpa_property)((int)a | (int)b);
}
inline sdpa_property operator&(sdpa_property a, sdpa_property b) {
    return (sdpa_property)((int)a & (int)b);
}
inline sdpa_property operator^(sdpa_property a, sdpa_property b) {
    return (sdpa_property)((int)a ^ (int)b);
}
inline sdpa_property &operator|=(sdpa_property &a, sdpa_property b) {
    return (sdpa_property &)((int &)a |= (int)b);
}
inline sdpa_property &operator&=(sdpa_property &a, sdpa_property b) {
    return (sdpa_property &)((int &)a &= (int)b);
}
inline sdpa_property &operator^=(sdpa_property &a, sdpa_property b) {
    return (sdpa_property &)((int &)a ^= (int)b);
}

std::ostream &operator<<(std::ostream &s, const config_query_t &q) {
    s << std::to_string((int)q.arch) << " " << q.head_size << " " << q.seq_len
      << " " << (int)(q.property & sdpa_property::second_token) << " "
      << (int)(q.property & sdpa_property::quantized);
    return s;
}
std::ostream &operator<<(std::ostream &s, const config_criteria_t &c) {
    s << std::to_string((int)c.arch) << " " << c.head_size << " " << c.seq_len
      << " " << (int)(c.property & sdpa_property::second_token) << " "
      << (int)(c.property & sdpa_property::quantized);
    return s;
}
std::ostream &operator<<(std::ostream &s, const sdpa_config_t &c) {
    s << c.unroll_m_kq << "," << c.unroll_n_kq << "," << c.unroll_m_vs << ","
      << c.unroll_n_vs << "," << c.wg_m_kq << "," << c.wg_n_kq << ","
      << c.wg_m_vs << "," << c.wg_n_vs;
    return s;
}

bool operator==(const config_record_t &key, const config_query_t &query) {
    bool result = ((query.arch == key.criteria.arch)
            && (key.criteria.head_size >= query.head_size)
            && ((query.seq_len == -1 && key.criteria.seq_len == -1)
                    || (query.seq_len != -1
                            && query.seq_len <= key.criteria.seq_len))
            && (((query.property & sdpa_property::second_token)
                        == sdpa_property::none)
                    || ((query.property & sdpa_property::second_token)
                            == (key.criteria.property
                                    & sdpa_property::second_token)))
            && (((query.property & sdpa_property::quantized)
                        == sdpa_property::none)
                    || ((query.property & sdpa_property::quantized)
                            == (key.criteria.property
                                    & sdpa_property::quantized))));
    return result;
}

bool operator<(const config_criteria_t &lhs, const config_criteria_t &rhs) {
    int l_set_fields = 0;
    if (lhs.arch != compute::gpu_arch_t::unknown) { l_set_fields++; }
    if (lhs.head_size != -1) { l_set_fields++; }
    if (lhs.seq_len != -1) { l_set_fields++; }
    if ((int)(lhs.property & sdpa_property::second_token)) { l_set_fields++; }
    if ((int)(lhs.property & sdpa_property::quantized)) { l_set_fields++; }
    int r_set_fields = 0;
    if (rhs.arch != compute::gpu_arch_t::unknown) { r_set_fields++; }
    if (rhs.head_size != -1) { r_set_fields++; }
    if (rhs.seq_len != -1) { r_set_fields++; }
    if ((int)(rhs.property & sdpa_property::second_token)) { r_set_fields++; }
    if ((int)(rhs.property & sdpa_property::quantized)) { r_set_fields++; }

    if (l_set_fields != r_set_fields) return !(l_set_fields < r_set_fields);
    if (lhs.arch < rhs.arch) return false;
    if (lhs.head_size < rhs.head_size) return false;
    if (lhs.seq_len != -1 && rhs.seq_len != -1 && lhs.seq_len < rhs.seq_len)
        return false;
    if ((lhs.property & sdpa_property::second_token) == sdpa_property::none
            && (rhs.property & sdpa_property::second_token)
                    == sdpa_property::none
            && (lhs.property & sdpa_property::second_token)
                    < (rhs.property & sdpa_property::second_token))
        return false;
    if ((lhs.property & sdpa_property::quantized) == sdpa_property::none
            && (rhs.property & sdpa_property::quantized) == sdpa_property::none
            && (lhs.property & sdpa_property::quantized)
                    < (rhs.property & sdpa_property::quantized))
        return false;
    return true;
}

bool operator<(const config_record_t &lhs, const config_record_t &rhs) {
    return lhs.criteria < rhs.criteria;
}

// Kernel configurations:
//  h<N> -- maximum head size = N
//  s<M> -- target sequence length = M
//   2nd -- second token (thin Q)
static std::vector<config_record_t> configs = {
        {{compute::gpu_arch_t::xe_hpg, 32}, {32, 16, 16, 16, 2, 16, 2, 16}},

        {{compute::gpu_arch_t::xe_hpg, 32, 256}, {16, 16, 16, 16, 2, 8, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32, 64}, {16, 16, 16, 8, 4, 4, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32, 32}, {8, 8, 8, 8, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe_hpg, 32, sdpa_property::second_token},
                {8, 32, 16, 8, 8, 1, 2, 4}},

        {{compute::gpu_arch_t::xe_hpg, 32, sdpa_property::quantized},
                {32, 16, 16, 16, 2, 8, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32,
                 sdpa_property::quantized | sdpa_property::second_token},
                {32, 16, 8, 8, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpg, 64}, {32, 16, 16, 16, 4, 8, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 128}, {16, 16, 16, 16, 4, 8, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 64}, {32, 16, 16, 8, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, sdpa_property::second_token},
                {8, 16, 16, 8, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpg, 64, sdpa_property::quantized},
                {32, 16, 16, 16, 4, 8, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 128, sdpa_property::quantized},
                {16, 16, 16, 8, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 64, sdpa_property::quantized},
                {32, 8, 32, 8, 2, 8, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 32, sdpa_property::quantized},
                {8, 8, 16, 8, 4, 8, 4, 8}},

        {{compute::gpu_arch_t::xe_hpg, 64, 64,
                 sdpa_property::quantized | sdpa_property::second_token},
                {8, 8, 8, 8, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpg, 64, 128,
                 sdpa_property::quantized | sdpa_property::second_token},
                {16, 8, 8, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 64,
                 sdpa_property::quantized | sdpa_property::second_token},
                {16, 16, 8, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 128}, {16, 16, 32, 8, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 128, 32}, {16, 16, 16, 8, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 256, sdpa_property::second_token},
                {8, 16, 32, 8, 8, 1, 4, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128, sdpa_property::second_token},
                {8, 16, 16, 8, 16, 1, 8, 2}},

        {{compute::gpu_arch_t::xe_hpg, 128, sdpa_property::quantized},
                {8, 32, 16, 32, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128, 64, sdpa_property::quantized},
                {8, 8, 16, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 512, sdpa_property::quantized},
                {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 96,
                 sdpa_property::quantized | sdpa_property::second_token},
                {8, 8, 8, 8, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128,
                 sdpa_property::quantized | sdpa_property::second_token},
                {16, 16, 16, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 256}, {16, 16, 32, 8, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 128}, {8, 16, 32, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 32}, {8, 16, 32, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 256, sdpa_property::quantized},
                {16, 16, 64, 8, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 256, 512, sdpa_property::quantized},
                {16, 16, 32, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 64, sdpa_property::quantized},
                {8, 8, 32, 8, 8, 4, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 256, sdpa_property::second_token},
                {8, 8, 16, 8, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpg, 256, 64, sdpa_property::second_token},
                {16, 8, 16, 8, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpg, 256, 32, sdpa_property::second_token},
                {16, 16, 32, 8, 16, 1, 8, 2}},

        {{compute::gpu_arch_t::xe_hpg, 256,
                 sdpa_property::second_token | sdpa_property::quantized},
                {32, 8, 32, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 96,
                 sdpa_property::second_token | sdpa_property::quantized},
                {8, 8, 16, 8, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe_hpg, 512, 64, sdpa_property::quantized},
                {8, 8, 64, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 512, 128, sdpa_property::quantized},
                {8, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512, 256, sdpa_property::quantized},
                {16, 8, 64, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 512, sdpa_property::quantized},
                {8, 16, 64, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 512, 64,
                 sdpa_property::second_token | sdpa_property::quantized},
                {8, 16, 32, 8, 32, 1, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512, 256,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 8, 32, 8, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 8, 16, 8, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe_hpg, 512}, {8, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512, sdpa_property::second_token},
                {8, 8, 32, 8, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe_hpc, 32}, {16, 64, 32, 16, 4, 2, 1, 8}},
        {{compute::gpu_arch_t::xe_hpc, 32, 32}, {16, 16, 16, 16, 2, 4, 2, 4}},
        {{compute::gpu_arch_t::xe_hpc, 32, sdpa_property::second_token},
                {16, 64, 16, 16, 8, 1, 2, 4}},

        {{compute::gpu_arch_t::xe_hpc, 64}, {16, 64, 32, 16, 8, 2, 2, 8}},
        {{compute::gpu_arch_t::xe_hpc, 64, 64}, {32, 32, 32, 16, 4, 2, 2, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, 32}, {16, 16, 16, 16, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe_hpc, 64, sdpa_property::second_token},
                {32, 32, 32, 16, 4, 1, 2, 2}},
        {{compute::gpu_arch_t::xe_hpc, 64, 64, sdpa_property::second_token},
                {16, 16, 16, 16, 4, 1, 4, 1}},

        {{compute::gpu_arch_t::xe_hpc, 64, 64, sdpa_property::quantized},
                {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, 384, sdpa_property::quantized},
                {16, 64, 16, 32, 8, 2, 4, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, 1024, sdpa_property::quantized},
                {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, sdpa_property::quantized},
                {16, 64, 16, 32, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpc, 64, 96,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 1, 4, 1}},
        {{compute::gpu_arch_t::xe_hpc, 64, 256,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 64, 1152,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 64,
                 sdpa_property::second_token | sdpa_property::quantized},
                {64, 16, 16, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe_hpc, 128}, {16, 64, 32, 16, 16, 2, 4, 8}},
        {{compute::gpu_arch_t::xe_hpc, 128, 64}, {16, 32, 32, 32, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 32}, {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, sdpa_property::second_token},
                {32, 32, 32, 16, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpc, 128, sdpa_property::quantized},
                {16, 64, 16, 32, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 32, sdpa_property::quantized},
                {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 128, sdpa_property::quantized},
                {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 128, 128,
                 sdpa_property::integrated | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe_hpc, 128,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 128,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe_hpc, 128, 96,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe_hpc, 128, 512,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 2, 8, 2}},

        {{compute::gpu_arch_t::xe_hpc, 256}, {16, 32, 32, 32, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 256, 64}, {16, 32, 32, 32, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe_hpc, 256, sdpa_property::second_token},
                {16, 16, 16, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe_hpc, 512, 32}, {16, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 512, 128}, {16, 16, 64, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 512}, {32, 16, 64, 16, 8, 4, 8, 4}},

        {{compute::gpu_arch_t::xe_hpc, 512, 128, sdpa_property::second_token},
                {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512, 512, sdpa_property::second_token},
                {32, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512, 1024, sdpa_property::second_token},
                {64, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512, sdpa_property::second_token},
                {32, 16, 32, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe_hpc, 576}, {16, 32, 32, 32, 32, 1, 32, 1}},
        {{compute::gpu_arch_t::xe_hpc, 576, sdpa_property::second_token},
                {32, 16, 32, 16, 32, 1, 31, 1}},

        {{compute::gpu_arch_t::xe_hpc, 512, 128, sdpa_property::quantized},
                {16, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 512, sdpa_property::quantized},
                {16, 32, 64, 16, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpc, 512,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 32, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe2, 64, sdpa_property::quantized},
                {16, 64, 16, 32, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 1024,
                 sdpa_property::integrated | sdpa_property::quantized},
                {16, 64, 16, 32, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe2, 64, 512, sdpa_property::quantized},
                {16, 64, 16, 32, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe2, 64, 384, sdpa_property::quantized},
                {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe2, 64, 128, sdpa_property::quantized},
                {16, 64, 16, 32, 8, 1, 4, 2}},
        {{compute::gpu_arch_t::xe2, 64, 128,
                 sdpa_property::integrated | sdpa_property::quantized},
                {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe2, 64, 32, sdpa_property::quantized},
                {16, 16, 16, 16, 4, 4, 4, 4}},

        {{compute::gpu_arch_t::xe2, 64,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 64,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 64, 96,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 64, 384,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {64, 16, 16, 16, 4, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 64, 64,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe2, 64, 128,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 384,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 64, 512,
                 sdpa_property::second_token | sdpa_property::quantized},
                {64, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 64, 768,
                 sdpa_property::second_token | sdpa_property::quantized},
                {64, 16, 16, 16, 16, 1, 8, 1}},

        {{compute::gpu_arch_t::xe2, 256, sdpa_property::quantized},
                {16, 64, 16, 32, 32, 1, 16, 2}},
        {{compute::gpu_arch_t::xe2, 256, 384, sdpa_property::quantized},
                {16, 32, 32, 32, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 256, 128, sdpa_property::quantized},
                {16, 32, 32, 32, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 256, 128,
                 sdpa_property::integrated | sdpa_property::quantized},
                {16, 32, 32, 32, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 256, 64,
                 sdpa_property::integrated | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 256, 64, sdpa_property::quantized},
                {16, 32, 64, 16, 8, 2, 4, 4}},

        {{compute::gpu_arch_t::xe2, 256,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {32, 16, 64, 16, 4, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 256, 1152,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 64, 16, 4, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 256, 768,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {64, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 256, 512,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {32, 32, 32, 16, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe2, 256, 384,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 16, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 512, 64}, {16, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512}, {32, 16, 64, 16, 8, 4, 8, 4}},

        {{compute::gpu_arch_t::xe2, 512, 128, sdpa_property::second_token},
                {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 512, 512, sdpa_property::second_token},
                {32, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 512, 1024, sdpa_property::second_token},
                {64, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 512, sdpa_property::second_token},
                {32, 16, 64, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 512, 128, sdpa_property::quantized},
                {16, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512, sdpa_property::quantized},
                {16, 32, 64, 16, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe2, 512, 64,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 512,
                 sdpa_property::second_token | sdpa_property::quantized},
                {16, 16, 64, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 512, 128, sdpa_property::integrated},
                {16, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512, sdpa_property::integrated},
                {16, 16, 16, 16, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe2, 512, 256,
                 sdpa_property::integrated | sdpa_property::second_token},
                {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 512, 1024,
                 sdpa_property::integrated | sdpa_property::second_token},
                {16, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512,
                 sdpa_property::integrated | sdpa_property::second_token},
                {16, 16, 64, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe2, 576}, {16, 32, 32, 32, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe2, 512,
                 sdpa_property::integrated | sdpa_property::quantized},
                {16, 32, 32, 32, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 512, 64,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 32, 64, 32, 16, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512, 128,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 64, 16, 8, 1, 32, 1}},
        {{compute::gpu_arch_t::xe2, 512, 256,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 32, 64, 32, 16, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512, 512,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 64, 16, 4, 4, 8, 4}},
        {{compute::gpu_arch_t::xe2, 512, 1024,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {16, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 512,
                 sdpa_property::integrated | sdpa_property::second_token
                         | sdpa_property::quantized},
                {32, 16, 64, 16, 8, 1, 16, 1}}};

sdpa_config_t *choose_config(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool thin_q, bool quantized, bool integrated) {
    sdpa_property query_properties = sdpa_property::none;
    if (thin_q) { query_properties |= sdpa_property::second_token; }
    if (quantized) { query_properties |= sdpa_property::quantized; }
    if (integrated) { query_properties |= sdpa_property::integrated; }

    auto query = config_query_t {arch, static_cast<int>(head_size),
            static_cast<int>(seq), query_properties};
    auto it = find(begin(configs), end(configs), query);
    if (it != end(configs)) {
        return &it->config;
    } else {
        query.seq_len = -1;
        it = find(begin(configs), end(configs), query);
        if (it != end(configs)) { return &it->config; }
    }
    return nullptr;
}

// adjust heuristic intervals to match the tuned intervals according
// to the sequence length and gpu architecture
// this way recompilation both matches the tuned intervals and avoids
// excessive recompilation with smaller power of 2 sizes
dim_t nearest_conf_seq_interval(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool thin_q, bool quantized, bool integrated) {
    sdpa_property query_properties = sdpa_property::none;
    if (thin_q) { query_properties |= sdpa_property::second_token; }
    if (quantized) { query_properties |= sdpa_property::quantized; }
    if (integrated) { query_properties |= sdpa_property::integrated; }

    auto query = config_query_t {arch, static_cast<int>(head_size),
            static_cast<int>(seq), query_properties};
    auto it = find(begin(configs), end(configs), query);
    if (it != end(configs)) { return it->criteria.seq_len; }
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
        problem.aScale2D = serialized_problem.aScale2D;
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
