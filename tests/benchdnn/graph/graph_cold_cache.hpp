
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

#ifndef GRAPH_COLD_CACHE_HPP
#define GRAPH_COLD_CACHE_HPP

#include <ostream>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_types.h"

#include "dnnl_memory.hpp"

namespace graph {

enum class graph_cold_cache_mode_t : unsigned {
    // Cold cache is disabled.
    none = 0x0,
    // Cold cache is enabled for all inputs and outputs.
    all = 0x1,
    // TODO: Extend to support custom cold cache for any inputs or outputs
};

// User's choices for enabling cold-cache.
struct graph_cold_cache_input_t {
    // Requested mode.
    graph_cold_cache_mode_t cold_cache_mode_ = graph_cold_cache_mode_t::none;

    bool operator==(const graph_cold_cache_input_t &other) const {
        return cold_cache_mode_ == other.cold_cache_mode_;
    }
    bool operator!=(const graph_cold_cache_input_t &other) const {
        return !operator==(other);
    }
};

struct dnn_graph_mem_t;

using partition_mem_map_t = std::unordered_map<size_t, dnn_graph_mem_t>;

// This is a global var to accept cold cache mode from user cml
extern graph_cold_cache_input_t graph_cold_cache_input;

struct graph_cold_cache_t {
    graph_cold_cache_t() = default;
    graph_cold_cache_t(const std::vector<dnnl::graph::tensor> &inputs,
            const partition_mem_map_t &partition_mem_map);
    ~graph_cold_cache_t();

    bool update_partition_inputs(std::vector<dnnl::graph::tensor> &inputs);
    bool should_stop() const;

private:
    graph_cold_cache_input_t graph_cold_cache_input_;
    bool enabled_ = false;
    size_t n_buffers_top_limit_ = 0;
    size_t n_buffers_bottom_limit_ = 0;
    // `n_buffers` is responsible for the number of allocated buffers per arg.
    size_t n_buffers_ = 0;
    bool override_n_buffers_ = false;
    std::unordered_map<int, std::vector<dnnl::graph::tensor>> cache_;

    // Memory allocations are time consuming on GPU, thus, introducing the
    // upper bound for the number of buffers in cold-cache.
    // For CPU the enormous number of buffers may lead to what looks like a
    // hang. In fact, just takes a very long time to complete.
    // Since `no_ref_memory` allocations use `memset` call to initialize the
    // data, the assumption is it makes newly created memory objects with newly
    // allocated buffer underneath get into the GPU cache. Using these memory
    // objects in cold-cache run won't be "cold" any longer.
    // Thus, introducing an extra reorder with brand new memory objects which
    // sole purpose is to reset the state of the cache by entirely thrashing it.
    static constexpr size_t gpu_n_buffers_top_limit_ = 100;
    static constexpr size_t cpu_n_buffers_top_limit_ = 10000;

    size_t cc_counter_ = 0;

    // Returns `true`, if cold-cache was requested and eligible.
    bool use_cold_cache(const std::vector<dnnl::graph::tensor> &inputs) const;

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(graph_cold_cache_t);
};

#endif
}
