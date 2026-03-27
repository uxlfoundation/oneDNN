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

#ifndef GRAPH_BACKEND_DNNL_PATTERNS_PATTERN_MATCHER_PASS_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_PATTERN_MATCHER_PASS_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"

#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pass_base.hpp"
#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

class pattern_utils_t {
public:
    inline void match(graph_t &backend_graph,
            std::shared_ptr<graph::utils::pm::pb_graph_t> pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops);

    inline void init_partition(graph_t &backend_graph,
            std::vector<std::vector<op_t *>> &fusion_ops,
            const FCreateKernel &kernel_creator, partition_kind_t pkind);

    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;

private:
    // Handle group pattern matching
    inline void match_group_pattern(graph_t &backend_graph,
            graph::utils::pm::pb_group_t *group_node,
            std::vector<std::vector<op_t *>> &fusion_ops) {
        // Early exit if graph doesn't have enough potential subgraphs
        size_t min_instances = group_node->get_min_instances();
        if (backend_graph.get_output_ops().size() < min_instances) return;

        auto template_graph = group_node->get_template();
        std::vector<op_t *> candidate_ops;

        // Try to match each op in the graph against the template
        topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
            // Skip if already matched or partitioned
            if (cur_op->get_partition() != nullptr) return status::success;
            if (cur_op->has_attr(op_attr::matched)
                    && cur_op->get_attr<bool>(op_attr::matched))
                return status::success;

            std::vector<op_t *> temp_fusion;
            if (graph::utils::pm::match_pattern(
                        cur_op, template_graph, temp_fusion)) {
                candidate_ops.insert(candidate_ops.end(), temp_fusion.begin(),
                        temp_fusion.end());
            }
            return status::success;
        });

        // Check instance count constraints
        size_t max_instances = group_node->get_max_instances();

        if (candidate_ops.size() == backend_graph.num_ops()
                && candidate_ops.size() >= min_instances
                && candidate_ops.size() <= max_instances) {
            // All constraints satisfied - add as a single group
            fusion_ops.emplace_back(candidate_ops);
        } else {
            // Instance count out of range - clear matched attributes
            for (auto *op : candidate_ops) {
                if (op->has_attr(op_attr::matched)) {
                    op->set_attr<bool>(op_attr::matched, false);
                }
            }
        }
    }

    // Handle regular pattern matching
    inline void match_regular_pattern(graph_t &backend_graph,
            const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops) {
        topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
            std::vector<op_t *> candidate_fusion;
            if (!graph::utils::pm::match_pattern(
                        cur_op, pgraph, candidate_fusion)) {
                return status::success;
            }
            fusion_ops.emplace_back(candidate_fusion);
            return status::success;
        });
    }
};

inline void pattern_utils_t::match(graph_t &backend_graph,
        std::shared_ptr<graph::utils::pm::pb_graph_t> pgraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    // Check if this is a group pattern
    auto nodes = pgraph->get_nodes();
    if (nodes.size() == 1
            && nodes[0]->get_node_kind()
                    == graph::utils::pm::pb_node_kind::PB_NODE_KIND_GROUP) {
        auto *group_node
                = dynamic_cast<graph::utils::pm::pb_group_t *>(nodes[0]);
        if (!group_node) return;
        match_group_pattern(backend_graph, group_node, fusion_ops);
    } else {
        match_regular_pattern(backend_graph, pgraph, fusion_ops);
    }
}

inline void pattern_utils_t::init_partition(graph_t &backend_graph,
        std::vector<std::vector<op_t *>> &fusion_ops,
        const FCreateKernel &kernel_creator, partition_kind_t pkind) {
    for (auto &pairs : fusion_ops) {
        std::shared_ptr<dnnl_partition_impl_t> pimpl
                = std::make_shared<dnnl_partition_impl_t>(
                        backend_graph.get_engine_kind(),
                        backend_graph.get_fpmath_mode(), pkind);

        // transfer the matched op's ownership from graph to partition
        for (size_t i = 0; i < pairs.size(); ++i) {
            pimpl->add_op(pairs[i]->shared_from_this());
            // claim the op belong to the partition
            pairs[i]->set_partition(pimpl.get());
        }
        pimpl->init(kernel_creator);
        backend_graph.add_partition(pimpl);
    }
}

/*!
 * \brief pattern_matcher_pass_t generates an optimized graph
 *        when a pre-defined pattern is hit.
 */
class pattern_matcher_pass_t : public graph::pass::pass_base_t {
public:
    explicit pattern_matcher_pass_t(std::string pbackend, std::string pname)
        : graph::pass::pass_base_t(std::move(pbackend), std::move(pname)) {}

    static graph::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<pattern_matcher_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    // the criteria of pass execution
    impl::status_t run(graph_t &agraph) override {
        // check if current pattern pass can be run on current graph
        engine_kind_t graph_engine_kind = agraph.get_engine_kind();
        if (get_engine_kind() != engine_kind::any_engine
                && get_engine_kind() != graph_engine_kind)
            return impl::status::success;

        // we can have multiply patterns that map to one optimized kernel
        std::vector<graph::pass::Pattern> pgraphs
                = get_attr<graph::pass::Pattern>("Pattern");

        FCreateKernel kernel_creator
                = get_attr<FCreateKernel>("FCreateKernel")[0];

        pattern_utils_t pu;
        for (const auto &pgraph : pgraphs) {
            // check if min_op_num in the pattern is larger than
            // num_unpartitioned_ops in the graph, if true,
            // no need to run this pattern any more
            if (pgraph->get_min_op_num() > agraph.num_unpartitioned_ops())
                continue;
            // for each pattern. match it
            std::vector<std::vector<op_t *>> fusion_ops;
            if (get_verbose(verbose_t::create_dispatch, component_t::graph)) {
                verbose_printf(
                        "graph,create:dispatch,pattern_matcher,%s,dnnl_"
                        "backend\n",
                        get_pass_name().c_str());
            }
            pu.match(agraph, pgraph, fusion_ops);
            if (!fusion_ops.empty()) {
                // temporary solution here for showing which pattern matched
                if (graph::utils::get_graph_dump_mode(
                            graph::graph_dump_mode_t::pattern)) {
                    verbose_printf("graph,info,pattern,hit,%s\n",
                            get_pass_name().c_str());
                }

                pu.init_partition(
                        agraph, fusion_ops, kernel_creator, get_kind());
            }
        }
        return impl::status::success;
    }
};

#define DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(backend_name, pattern_name) \
    registry.register_pass( \
            #backend_name, #pattern_name, &pattern_matcher_pass_t::create)

#define DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(pattern_class_) \
    void register_##pattern_class_(graph::pass::pass_registry_t &registry) {
#define DNNL_BACKEND_REGISTER_PATTERN_DEF_END }

#define MAX_REPETITION 20
} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
