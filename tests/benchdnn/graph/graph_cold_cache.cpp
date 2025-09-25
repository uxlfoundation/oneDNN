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

#include "dnnl_common.hpp"

#include "graph_cold_cache.hpp"
#include "graph_memory.hpp"

namespace graph {

graph_cold_cache_input_t graph_cold_cache_input;

graph_cold_cache_t::graph_cold_cache_t(
        const std::vector<dnnl::graph::tensor> &inputs,
        const partition_mem_map_t &partition_mem_map)
    : graph_cold_cache_input_(graph_cold_cache_input)
    , enabled_(graph_cold_cache_input_.cold_cache_mode_
              != graph_cold_cache_mode_t::none)
    , n_buffers_top_limit_(
              is_gpu() ? gpu_n_buffers_top_limit_ : cpu_n_buffers_top_limit_) {

    if (!enabled_) return;

    static cpu_cache_args_t cpu_cache_args {};
    SAFE_V(get_cpu_cache_size(cpu_cache_args));
    const auto cpu_cache_capacity = cpu_cache_args.total_socket_size;
    // `3` potentially to cover both one and two socket scenarios.
    static const size_t cpu_cache_size_upper_bound = cpu_cache_capacity * 3;

    static size_t gpu_cache_capacity = 0;
    SAFE_V(get_gpu_cache_size(gpu_cache_capacity));
    static const size_t gpu_cache_size_upper_bound = gpu_cache_capacity * 2;

    const auto cache_capacity
            = is_gpu() ? gpu_cache_capacity : cpu_cache_capacity;
    const auto cache_size_upper_bound = is_gpu() ? gpu_cache_size_upper_bound
                                                 : cpu_cache_size_upper_bound;

    size_t full_inputs_size = 0;
    for (auto &in : inputs) {
        full_inputs_size += in.get_logical_tensor().get_mem_size();
    }

    size_t hot_args_size = full_inputs_size;
    size_t cold_args_size = 0;
    std::vector<int> cc_args;
    if (graph_cold_cache_input_.cold_cache_mode_
            == graph_cold_cache_mode_t::all) {
        cc_args.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            cc_args[i] = inputs[i].get_logical_tensor().get_id();
        }
        hot_args_size = 0;
        cold_args_size = full_inputs_size;
    } else {
        assert(!"unknown cold cache mode!");
    }

    BENCHDNN_PRINT(3,
            "[COLD_CACHE]%s Size:%s; Limit:%s; Hot args:%s; Cold args:%s;\n",
            (is_gpu() ? "[GPU]" : "[CPU]"), smart_bytes(cache_capacity).c_str(),
            smart_bytes(cache_size_upper_bound).c_str(),
            smart_bytes(hot_args_size).c_str(),
            smart_bytes(cold_args_size).c_str());

    const size_t cold_mem_pool_size = cache_size_upper_bound > hot_args_size
            ? cache_size_upper_bound - hot_args_size
            : 0;

    size_t n_mem_pool_buffers = 0;
    // If `cold_args_size` are greater then allowed pool_size, it means there's
    // no sense in allocating any more buffers. Use original buffers only.
    if (cold_mem_pool_size > cold_args_size)
        n_mem_pool_buffers = div_up(cold_mem_pool_size, cold_args_size);

    n_buffers_ = MIN2(MAX2(n_mem_pool_buffers, n_buffers_bottom_limit_),
            n_buffers_top_limit_);
    override_n_buffers_ = n_mem_pool_buffers > n_buffers_top_limit_;

    BENCHDNN_PRINT(3,
            "[COLD_CACHE] n_buffer_limits: [%zu, %s]; n_mem_pool_buffers: "
            "%zu; n_buffers: %zu.\n",
            n_buffers_bottom_limit_,
            (n_buffers_top_limit_ == SIZE_MAX
                            ? "SIZE_MAX"
                            : std::to_string(n_buffers_top_limit_).c_str()),
            n_mem_pool_buffers, n_buffers_);

    if (n_buffers_ <= 0) {
        // No buffers allocation needed, return to avoid scratching `cache_`
        // object. This allows to keep rest logic intact.
        return;
    }

    for (size_t i = 0; i < cc_args.size(); i++) {
        const int idx = cc_args[i];
        auto it = partition_mem_map.find(static_cast<size_t>(idx));
        if (it == partition_mem_map.end()) continue;
        // Empty memories don't get their cold cache entry.
        const auto &orig_mem = it->second.get_mem();
        if (!orig_mem) continue;

        auto &cc_entry = cache_[idx];
        cc_entry.resize(n_buffers_);
        auto orig_cc_mem_md = query_md(orig_mem);

        // 找到对应的输入 tensor
        auto input_it = std::find_if(inputs.begin(), inputs.end(),
                [idx](const dnnl::graph::tensor &t) {
                    return t.get_logical_tensor().get_id() == idx;
                });
        if (input_it == inputs.end()) continue;
        const auto &input_tensor = *input_it;

        for (size_t j = 0; j < n_buffers_; j++) {
            const bool prefill = n_mem_pool_buffers > gpu_n_buffers_top_limit_;
            auto new_mem = dnn_mem_t(orig_cc_mem_md, get_test_engine(), true);
            void *data_handle;
            dnnl_memory_get_data_handle(new_mem.m_, &data_handle);
            cc_entry[j] = dnnl::graph::tensor(input_tensor.get_logical_tensor(),
                    input_tensor.get_engine(), data_handle);
        }
    }
}

} // namespace graph