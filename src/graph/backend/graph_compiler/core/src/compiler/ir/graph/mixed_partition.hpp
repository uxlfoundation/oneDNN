/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_MIXED_PARTITION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_MIXED_PARTITION_HPP
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "fused_op.hpp"
#include "fusion_anchor.hpp"
#include "fusion_cost_model.hpp"
#include "visitor.hpp"
#include <compiler/ir/transform/static_memory_planner.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct mixed_parti_t;

namespace mixed_partition_hint {
// pointer: partition owner
constexpr const char *parti = "partition";
// Boolean: is cut buffer hint
constexpr const char *cut_buffer = "cut_buffer";
// Boolean: dont inplace hint
constexpr const char *no_inplace = "no_inplace";
// Boolean: is optimized sub graph
constexpr const char *optimized_sub_graph = "optimized_sub_graph";
// Boolean: is single op graph
constexpr const char *single_op_graph = "single_op_graph";
// Boolean: is begining op of pre-op fuse
constexpr const char *pre_fuse_begin_op = "pre_fuse_begin_op";
// Boolean: whether directly inplaced and would not call `compute_block`
constexpr const char *inplace_optimized_op = "inplace_optimized_op";
// Boolean: is the first op for prefetching
constexpr const char *first_prefetch_op = "first_prefetch_op";
// size_t: used to judge op whether is small workload. If less than threshold,
// ignore parallelism check of cost model. (Measured by 3D fp32 InstanceNorm)
constexpr size_t small_op_workload_threshold = 1688UL;
// Boolean: is the op which could not gather input partitions
constexpr const char *no_gather_op = "no_gather_op";
// Boolean: is the op which need to split common anchor into grouped anchor
constexpr const char *split_anchor_op = "split_anchor_op";
} // namespace mixed_partition_hint

// different fusion policies prepared for dynamic shape, policies will be JIT
// and dispatched by conditions generated by dynamic cost model,/
enum class dynamic_fusion_policy_t : int {
    max_fusion = 0,
    max_loop_parallelism
};

class mxp_replacer_t : public ir_inplace_visitor_t {
private:
    node_ptr_map node_remap_;

public:
    mxp_replacer_t(node_ptr_map &node_remap) : node_remap_(node_remap) {}

    using ir_inplace_visitor_t::dispatch_impl;
    using ir_inplace_visitor_t::visit_impl;
    expr visit_impl(var v) override {
        auto itr = node_remap_.find(v.impl);
        if (itr != node_remap_.end()) {
            changed_ = true;
            return static_cast<expr_base *>(itr->second.get())
                    ->node_ptr_from_this();
        }
        return v;
    }

    expr visit_impl(tensor v) override {
        // visit and update user-defined tsr
        if (v->attr_
                && v->attr_->has_key(tensor_shrinker_attrs::should_shrink)) {
            auto &shrink_info = v->attr_->get<tensor_shrinker_t::shrink_info_t>(
                    tensor_shrinker_attrs::should_shrink);
            std::vector<expr> _new_expr;
            ir_inplace_visitor_t::dispatch_expr_vector(
                    shrink_info.base_, _new_expr);
            ir_inplace_visitor_t::dispatch_expr_vector(
                    shrink_info.shape_, _new_expr);
        }

        auto itr = node_remap_.find(v.impl);
        if (itr != node_remap_.end()) {
            changed_ = true;
            return static_cast<expr_base *>(itr->second.get())
                    ->node_ptr_from_this();
        }
        return v;
    }

    stmt visit_impl(for_loop v) override {
        // redirect reduce_root_loop if necessary
        if (v->attr_) {
            if (auto praw = v->attr_->get_or_null<std::weak_ptr<stmt_base_t>>(
                        stmt_attr_key::reduce_root_loop)) {
                auto raw = praw->lock();
                COMPILE_ASSERT(raw, "reduce_root_loop weak ptr invalidated");
                auto itr = node_remap_.find(raw);
                if (itr != node_remap_.end()) {
                    *praw = std::static_pointer_cast<stmt_base_t>(itr->second);
                }
            }
        }
        return ir_inplace_visitor_t::visit_impl(v);
    }

    void replace_func(func_t &func) {
        if (func) { dispatch_impl(func); }
    }
    void replace_anchor(const std::vector<fusion_anchor_ptr> &fanchors);
};

struct mxp_buffer_allocator {
private:
    mixed_parti_t *binded_mxp_;

public:
    gt2buf_map g2b_map_; // record graph tensor to ir tensor/tensorptr
            // mapping(maybe n-to-one)
    std::unordered_map<expr, fusion_anchor_ptr>
            tsr2anch_map_; // real tensor-to-anchor mapping
    std::unordered_map<expr, graph_tensor_ptr>
            b2g_map_; // buffer-to-gt mapping(one-to-one)

    mxp_buffer_allocator(mixed_parti_t *parti) { binded_mxp_ = parti; }

    std::vector<memory_optim::memory_alloc_trace_t>
            mem_trace_; // memory allocation trace

    memory_optim::inplace_info_map inplace_map_; // inplace map

    mixed_parti_t *const get_binded_mxp() const { return binded_mxp_; }
    // support inplace logic, allocate buffer including either tensor or
    // tensorptr
    void allocate_buffer(sc_op *op);
    // set inplace hint
    void set_buffer_inplace_hint(
            const expr &target_buf, const expr &inplace_buf);
    // get inplaced buffer
    expr get_inplaced_buffer(const expr &buf) const;
    // get allocated buffer
    std::tuple<std::vector<expr>, std::vector<expr>> get_buffer(
            sc_op *op) const;
    // update input buffer info
    void update_input_buffer_info(sc_op *op);
    // update output buffer info
    void update_output_buffer_info(sc_op *op);
    // define tensor
    void declare_tensor() const;
    // set shrink info
    void set_shrink_info() const;
    /** merge two buffer allocator
     * @param common_anchor_pair: the common anchor overlapped when two
     * partition merged. `first` comes from this partition, and `second` comes
     * from other one.
     * */
    void merge(mxp_buffer_allocator &other,
            std::unordered_map<expr, expr> &buffer_map,
            const std::pair<fusion_anchor_ptr, fusion_anchor_ptr>
                    &common_buffer_anchor_pair);
    // clear buffer allocator
    void clear();
    // check buffer_allocator whether empty
    const bool empty() const { return g2b_map_.empty(); };
    // initilize tensor
    void tensor_initialize();
    // replace the specific buffer
    void replace_buffer(const expr &old_buffer, const expr &new_buffer);
    // get real mem trace, taking consider of tensor shrink and ignore cut
    // buffer except for those in `keep_cut_set`
    std::vector<memory_optim::memory_alloc_trace_t> get_real_mem_trace(
            const std::unordered_set<graph_tensor *> &keep_cut_set = {}) const;
    // calculate real buffer usage size, taking consider of buffer schedule
    size_t get_real_buffer_usage() const;
    // get real anchor for the specfic buffer
    fusion_anchor_ptr get_real_anchor_for_buffer(const expr &buffer) const;
    // get shrinked info for buffer
    slice_range get_shrinked_info(const expr &buffer) const;
    // query buffer inplace and set hint for IR pass
    void query_inplace();
    // calibrate buffer information about inplace and shrink
    void calibrate_info();
    // validate tensor2var buffer whether meet the requirement
    bool validate_tsr2var() const;
    // count of buffer usage
    int use_count(const expr &buffer) const;
    // concat memory planning related
    void copy_concat_memory_attrs_tsr2buf();
};

struct mixed_dyn_internal_info_t {
    // The module records internal functions usually contains repeat
    // calculations which could be reused like single core brgemm. One partition
    // could hold multiple ops who have internal functions.
    ir_module_ptr mod_;
    // extra parameter for internal func dispatch, a tensor of pointer.
    expr inter_funcs_param_;
    // internal call node of internal func.
    call inter_call_;
    // internal func node.
    func_t inter_func_;
    // single core func.
    func_t single_core_func_;
    // extra args for inter call
    std::vector<expr> inter_call_extra_args_;
    // extra args for inter func
    std::vector<expr> inter_func_extra_args_;
    // extra args of single core func
    std::vector<expr> single_core_func_extra_args_;
    // number of functions in partition.
    int num_func_ = 0;
    mixed_dyn_internal_info_t(const context_ptr &ctx)
        : mod_(std::make_shared<ir_module_t>(ctx)) {}
};
using mixed_dyn_internal_info_ptr = std::shared_ptr<mixed_dyn_internal_info_t>;
struct mixed_parti_t : fusion_partition_t {
    /* related to Graph */
    // different from ops_ in base class, it records the sequence of committed
    // ops in current partition
    std::vector<sc_op_ptr> committed_ops_;
    // dep matrix
    dep_mat_ptr dep_m_;

    /* related to IR */
    context_ptr ctx_;
    // the body of func_ will be updated once the new op is committed into
    // current partition, but the name and argument maybe not confirmed until
    // final mixed_fused_op created.
    func_t func_;
    // the fanchor only manage the shared pointer of fusion_anchor struct,
    // during the whole lifetime of mixed_parti_t, it will not copy any
    // fusion_anchor struct.
    std::vector<fusion_anchor_ptr> fanchors_;
    // manage graph tensor to real tensor mapping
    mxp_buffer_allocator buf_alloc_ = mxp_buffer_allocator(this);
    // record the anchor to op mapping
    std::unordered_map<sc_op *, fusion_anchor_ptr> op_anchor_map_;

    // Cost Model
    fusion_cost_model_ptr cost_;
    // mixed fusion dyn internal info
    mixed_dyn_internal_info_ptr dyn_inter_;
    using ptr = std::shared_ptr<mixed_parti_t>;

    // append fusion anchor
    void append_fusion_anchor(const fusion_anchor_ptr &fanchor);

    void append_fusion_anchor(const std::vector<fusion_anchor_ptr> &fanchors) {
        for (auto &fanchor : fanchors) {
            append_fusion_anchor(fanchor);
        }
    }

    /**
     * The mixed partition merge will override base merge method, including
     * following several steps:
     * 1. It will firstly check two partition dependency and decide which one is
     * `to_merge` and another is `be_merged`.
     * 2. extract `outer_loops` from each one and compute greatest common outer
     * loops.
     * 3. commit inner loop body from `be_merged` to the largest used fanchor of
     * `to_merged`.
     * 4. update outer and inner fusion anchor map.
     * 5. replace expr iter/tensor/tensorptr in `func_`, `buf_alloc_` and
     * `fanchors_`.
     * 6. call base class `merge` method to do disjoint-set merge.
     * void merge(const ptr &other);
     * */

    mixed_parti_t(const context_ptr &ctx, const sc_op_ptr &op,
            const dep_mat_ptr &dep_m = nullptr);

    mixed_parti_t(const context_ptr &ctx, const func_t &func,
            const fusion_anchor_mgr_t &fmgr, const sc_graph_t &graph);

    bool is_ok_to_add(sc_op *op);

    bool add(const sc_op_ptr &op);

    void remove(const sc_op_ptr &op) {
        throw std::runtime_error("remove method is not implemented");
    }

    // if current partition contains no op or those ops generating no
    // codes(like tensorview op), return True.
    bool empty() const {
        if (merged_to) { return get_root()->empty(); }
        return (ops.empty() || !func_);
    }

    mixed_parti_t *get_root() const {
        return static_cast<mixed_parti_t *>(fusion_partition_t::get_root());
    }

    size_t get_ops_size() const { return get_root()->ops.size(); }

    sc_op_ptr get_ith_op(size_t ith) const {
        COMPILE_ASSERT(ith < get_root()->committed_ops_.size(),
                "Could not get " << ith << "-th op")
        return get_root()->committed_ops_[ith];
    }

    // get outer loops of which body(stmts) contains only one stmt or two with
    // the second one is empty fanchor
    std::vector<for_loop> get_outer_loops(
            fusion_anchor_ptr fanchor = nullptr) const;

    void try_split_outermost_loop(int64_t block) const;
    void try_split_outermost_loop_on_num_threads(int64_t num_groups);

    // query if partition can optimize its loop order
    bool can_optimize_outer_loop(bool allow_tensorview = false) const;

    // return op whether in op_anchor_map_
    bool ready_for_op(sc_op *op) const;

    // look up fanchor by op
    fusion_anchor_ptr lookup_anchor_map(
            sc_op *op, bool throw_assert = true) const;

    // look up fanchor by stmts
    fusion_anchor_ptr lookup_anchor_map(const stmts &ss) const;

    // look up sub fanchor by parent fanchor
    std::vector<fusion_anchor_ptr> lookup_sub_anchor_map(
            const fusion_anchor_ptr &parent_fanchor) const;

    // get anchor inside given loop
    fusion_anchor_ptr get_anchor_inside_loop(
            const for_loop &loop, bool input_anchor = false) const;

    /// get next inner loop including anchor
    for_loop get_next_inner_loop_with_anchor(const for_loop &cur_loop,
            const fusion_anchor_ptr &target_fanchor = nullptr) const;

    // clear all contents of given fanchor, but not erase it from
    // fanchor list
    void clear_fanchor(fusion_anchor_ptr &fanchor);

    // clear all unused fanchor, and erase them from fanchor list
    void clear_fanchors();

    // try to bind given op with given fanchor, if suitable fanchor exists, it
    // will compare two fanchor and select smaller one
    void set_anchor_for_op(sc_op *op, const fusion_anchor_ptr &fanchor_map);

    // schedule buffer
    void buffer_schedule();

    // judge whether the given graph tensor node is the input of the whole
    // partition
    bool is_parti_inp(const graph_tensor_ptr &gt) const;
    bool is_parti_inp(const graph_tensor *gt) const;

    // judge whether the given graph tensor node is the output of the whole
    // partition
    bool is_parti_out(const graph_tensor_ptr &gt) const;
    bool is_parti_out(const graph_tensor *gt) const;

    bool is_parti_cut(const graph_tensor_ptr &gt) const {
        return is_parti_inp(gt) || is_parti_out(gt);
    }
    bool is_parti_cut(const graph_tensor *gt) const {
        return is_parti_inp(gt) || is_parti_out(gt);
    }

    // count op number with given type
    template <typename T>
    size_t count_op_with_type() const {
        if (merged_to) { return get_root()->count_op_with_type<T>(); }
        size_t cnt = 0;
        for (auto &op : ops) {
            if (op->isa<T>()) cnt++;
        }
        return cnt;
    }

    // query partition whether contains input fusion anchor
    bool contain_input_anchor() const;

    // query partition whether is constant partition
    bool is_const_parti() const;

    // query partition whether contains op with given type
    template <typename T>
    bool contain_op_with_type() const {
        if (merged_to) { return get_root()->contain_op_with_type<T>(); }
        return (count_op_with_type<T>() != 0);
    }

    // query partition whether contains op with conv type
    bool contain_convolution() const;

    // query partition whether contains nested parallel for
    bool contain_nested_parallel_for() const;

    // query partition whether contains tunable op
    bool contain_tunable_op() const;

    // query partition whether contains only elementwise op
    bool contain_elemwise_op_only() const;

    // query partition whether contains only one op
    bool is_single_op_parti() const;

    // query partition whether contains op from optimized sub graph
    bool is_optimized() const;

    // check optimization whether legal or not
    bool validate_optimization() const;

    // clear all contents of partition object
    void clear();

    // call inner-build cost model to evaluate current partition, return the
    // scores
    float evaluate_perf();

    // query partition whether is small workload
    bool is_small_workload() const;

    // Get the condition expr generated by dynamic cost model.
    expr get_fusion_policy_condition() const {
        return cost_->get_fusion_policy_condition();
    }

    // get host graph
    sc_graph_t &get_host_graph() const {
        COMPILE_ASSERT(
                !committed_ops_.empty(), "No op contained in current partition")
        return committed_ops_[0]->get_owner_graph();
    }

    // transform partition to mixed fuse op
    std::shared_ptr<mixed_fuse_op_t> transform_to_mixed_op();
};

enum class parti_merge_kind : int {
    vertical = 0,
    horizontal = 1,
    parallel = 2,
};

void search_op_anchor_in_parti(sc_op *op, mixed_parti_t *parti);

using mxp_mem_info = std::pair<std::vector<memory_optim::memory_alloc_trace_t>,
        memory_optim::inplace_info_map>;

mxp_mem_info merge_real_mem_info(
        const mxp_buffer_allocator &alloc1, const mxp_buffer_allocator &alloc2);

std::vector<memory_optim::memory_alloc_trace_t> merge_mem_trace(
        const std::vector<memory_optim::memory_alloc_trace_t> &mem_trace1,
        const std::vector<memory_optim::memory_alloc_trace_t> &mem_trace2,
        const std::unordered_map<expr, expr> &buffer_map);

memory_optim::inplace_info_map merge_inplace_map(
        const memory_optim::inplace_info_map &inplace_map1,
        const memory_optim::inplace_info_map &inplace_map2,
        const std::unordered_map<expr, expr> &buffer_map);

size_t get_buffer_usage(const context_ptr &ctx,
        const std::vector<memory_optim::memory_alloc_trace_t> &mem_trace,
        const memory_optim::inplace_info_map &inplace_map);

// collect unrepeated partition set, and optionally ignore pure const partition
std::vector<mixed_parti_t::ptr> collect_parti_set(
        const std::vector<mixed_parti_t::ptr> &op_2_partition,
        bool ignore_const = true);

// do mixed partition
bool do_partition(const context_ptr &ctx, sc_graph_t &g,
        std::vector<mixed_parti_t::ptr> &op_2_partition,
        const dep_mat_ptr &dep_m);

// judge the given graph whether is second time retried graph
inline bool is_optimized_sub_graph(sc_graph_t &g) {
    return g.attrs_.get_or_else(
            mixed_partition_hint::optimized_sub_graph, false);
}

// judge the given graph whether is the graph containing only one op
inline bool is_single_op_graph(sc_graph_t &g) {
    return g.attrs_.get_or_else(mixed_partition_hint::single_op_graph, false);
}

bool concat_memory_planning_on_graph(sc_graph_t &graph);

// try optimize partition, such as reduce_op optimization
bool try_optimize_parti(mixed_parti_t *parti, sc_graph_t &sub_graph,
        const std::unordered_map<sc_op_ptr, sc_op_ptr> &graph2orig_ops = {});

// get single mixed op from the graph
mixed_fuse_op_t *get_mixed_op_from_graph(sc_graph_t &graph);

void do_mixed_partition(const context_ptr &ctx, sc_graph_t &graph);

// commit graph to TIR, usually used in UT
void commit_graph_to_func(
        sc_graph_t &g, const func_t &func, const fusion_anchor_mgr_t &fmgr);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
