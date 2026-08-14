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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_TRAINING_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_TRAINING_HPP

#include <memory>
#include <string>
#include <vector>

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/sdp_decomp_training_config.hpp"

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/subgraph.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"

#include "graph/backend/dnnl/passes/memory_planning.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct sdp_decomp_training_kernel_t : public kernel_base_t {
private:
    registry_t sdp_registry_;
    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;
    sdp_decomp_training_config_t sdp_cfg_;

public:
    sdp_decomp_training_kernel_t() {
        thread_local_cache_t<sdp_args_set_t> res_cache;
        res_cache.retain();
    }

    ~sdp_decomp_training_kernel_t() override {
        thread_local_cache_t<sdp_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
    }

    status_t compile_impl(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    void prepare_sub_args(const grantor_t &var_grantor, const int id,
            const size_t block_size,
            std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map);

    status_t execute_impl(stream_t *strm, const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf) override;

    std::string str() const override { return "sdp_decomp_training_kernel_t"; }

    size_t get_scratchpad_size() const override {
        return sdp_registry_.size() * sdp_cfg_.nthr;
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(stream_t *strm,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        UNUSED(strm);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(sycl_deps);
        UNUSED(sycl_event);
        return status::unimplemented;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(stream_t *strm,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<ocl_event_t> &ocl_deps,
            ocl_event_t &ocl_event) override {
        UNUSED(strm);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(ocl_deps);
        UNUSED(ocl_event);
        return status::unimplemented;
    }
#endif

    DNNL_DISALLOW_COPY_AND_ASSIGN(sdp_decomp_training_kernel_t)

    class sdp_args_set_t {
    public:
        sdp_args_set_t(sdp_decomp_training_kernel_t *sdp_kernel) {
            int nthr = sdp_kernel->sdp_cfg_.nthr;
            auto args_ctor
                    = [this, nthr](
                              const std::unordered_map<int, memory> &ori_args,
                              std::vector<std::unordered_map<int, memory>>
                                      &args) {
                args.resize(nthr);
                for (const auto &iter : ori_args) {
                    memory ori_mem = iter.second;
                    if (mem_map.count(ori_mem.get()) == 0) {
                        mem_map[ori_mem.get()] = std::vector<memory>(nthr);
                        for (int tid = 0; tid < nthr; tid++) {
                            mem_map[ori_mem.get()][tid]
                                    = memory(ori_mem.get_desc(),
                                            ori_mem.get_engine(), nullptr);
                            if (iter.first >= DNNL_ARG_ATTR_SCALES) {
                                mem_map[ori_mem.get()][tid].set_data_handle(
                                        ori_mem.get_data_handle());
                            }
                        }
                    }
                    for (int tid = 0; tid < nthr; tid++) {
                        args[tid].insert(
                                {iter.first, mem_map[ori_mem.get()][tid]});
                    }
                }
            };
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder0_args, sub_reorder0_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder1_args, sub_reorder1_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_mm1_args, sub_mm1_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_softmax_args, sub_softmax_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_reduce_max_P_args,
                    sub_reduce_max_P_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_reduce_max_src_args,
                    sub_reduce_max_src_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_reorder_stats_args,
                    sub_reorder_stats_args);
            if (sdp_kernel->sdp_cfg_.needs_softmax_reorder) {
                args_ctor(sdp_kernel->sdp_cfg_.sub_reorder_softmax_args,
                        sub_reorder_softmax_args);
            }
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder2_args, sub_reorder2_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_mm2_args, sub_mm2_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder3_args, sub_reorder3_args);
        }

        std::unordered_map<dnnl_memory_t, std::vector<memory>> mem_map;
        std::vector<std::unordered_map<int, memory>> sub_reorder0_args,
                sub_reorder1_args, sub_mm1_args, sub_reorder2_args,
                sub_mm2_args, sub_reorder3_args;
        std::vector<std::unordered_map<int, memory>> sub_softmax_args;
        std::vector<std::unordered_map<int, memory>> sub_reorder_softmax_args;
        std::vector<std::unordered_map<int, memory>> sub_reduce_max_P_args,
                sub_reduce_max_src_args;
        std::vector<std::unordered_map<int, memory>> sub_reorder_stats_args;
    };

    std::function<std::shared_ptr<sdp_args_set_t>()> resource_ctor_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
