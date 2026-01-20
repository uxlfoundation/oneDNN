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

#ifndef GRAPH_BACKEND_DNNL_PATTERNS_GROUPED_MATMUL_FUSION_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_GROUPED_MATMUL_FUSION_HPP

#include "graph/backend/dnnl/kernels/large_partition.hpp"

#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

// A trivial implementation to demonstrate the idea. This should be improved:
// - use the existing PM utilities to define the pattern and match.
// - support grouped matmul variants, e.g., with scale and quantization.
// - detect a grouped matmul from a graph containing multiple partitions.
class grouped_matmull_fusion_t : public pass::pass_base_t {
public:
    explicit grouped_matmull_fusion_t(std::string pbackend, std::string pname)
        : graph::pass::pass_base_t(std::move(pbackend), std::move(pname)) {
        set_priority(23.f); // preliminary priority
    }

    impl::status_t run(graph_t &agraph) override {
        const auto &ops = agraph.get_ops();
        if (ops.size() < 2) {
            // if less than 2 ops, it's not a grouped matmul.
            return status::success;
        }

        for (const auto &aop : ops) {
            // if any op is not matmul, it's not a grouped matmul.
            if (aop->get_kind() != graph::op_kind::MatMul)
                return status::success;
        }

        bool has_internal_consumer = false;
        for (const auto &aop : ops) {
            // check the output of matmul is not consumed by other ops
            const auto &outputs = aop->get_output_values();
            for (const auto &out : outputs) {
                auto consumers = out->get_consumers();
                for (const value_t::consumer_t &csm : consumers) {
                    op_t &csm_op = csm.get_op();
                    if (std::any_of(ops.begin(), ops.end(),
                                [&csm_op](const op_ptr &op) {
                        return op.get() == &csm_op;
                    })) {
                        has_internal_consumer = true;
                    }
                }
            }
        }

        if (has_internal_consumer) {
            // if any output is consumed by other ops, it's not a grouped matmul.
            return status::success;
        }

        // get a grouped matmul partition
        if (get_verbose(verbose_t::create_dispatch, component_t::graph)) {
            verbose_printf(
                    "graph,create:dispatch,pattern_matcher,found a grouped "
                    "matmul with %zu matmuls,dnnl_backend\n",
                    ops.size());
        }

        // create partition and kernel
        pattern_utils_t pu;
        const auto gmm_kernel_creater = []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        };
        std::vector<op_t *> op_vec;
        for (const auto &aop : ops) {
            op_vec.emplace_back(aop.get());
        }
        std::vector<std::vector<op_t *>> gmm_ops;
        gmm_ops.emplace_back(op_vec);
        pu.init_partition(agraph, gmm_ops, gmm_kernel_creater,
                partition_kind_t::misc_post_ops);

        return status::success;
    }
};

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
