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

#include "graph/backend/dnnl/kernels/sdp_decomp_training_config.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"

#define VCHECK_SDP_TRAIN(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_decomp_training_kernel_t, (cond), \
            status, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

bool sdp_decomp_training_config_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    // Record the input offset
    CHECK_BOOL(record_input_offset(sg, inputs));

    dims src1_user_dims = ltw(inputs[graph_inport[mm1_src]]).vdims();
    ndims = src1_user_dims.size();
    // TODO: support 5D GQA inputs (needs dimension extraction + offset logic)
    VCHECK_SDP_TRAIN(ndims == 4, false,
            "Training decomp only supports 4D input, but got %zu ndims",
            src1_user_dims.size());

    // Require 2 outputs: attention output + stats (logsumexp)
    VCHECK_SDP_TRAIN(outputs.size() == 2, false,
            "Training decomp requires 2 outputs (attn + stats), but got %zu",
            outputs.size());

    // Initialize SDP input dimensions
    batch_size = src1_user_dims[0];
    num_head_q = src1_user_dims[1];
    seq_len_q = src1_user_dims[2];
    head_size_qk = src1_user_dims[3];

    dims wei1_user_dims = ltw(inputs[graph_inport[mm1_wei]]).vdims();
    dims wei2_user_dims = ltw(inputs[graph_inport[mm2_wei]]).vdims();
    VCHECK_SDP_TRAIN(wei1_user_dims[1] == wei2_user_dims[1], false,
            "kv head number mismatch");
    // Without GQA, Q/K/V must have the same number of heads
    VCHECK_SDP_TRAIN(num_head_q == wei1_user_dims[1], false,
            "GQA not supported: num_head_q != num_head_kv");

    VCHECK_SDP_TRAIN(
            batch_size == wei1_user_dims[0] && batch_size == wei2_user_dims[0],
            false, "Batch size mismatch");

    seq_len_kv = wei1_user_dims[3]; // K shape: [batch, head, head_size, seq_kv]
    head_size_v = wei2_user_dims[3];

    // Check scale size
    if (graph_inport[mm1_scale] != -1) {
        auto scale_sz = ltw(inputs[graph_inport[mm1_scale]]).nelems();
        VCHECK_SDP_TRAIN(scale_sz == 1, false,
                "Only supports single scale value, but got %ld",
                static_cast<long int>(scale_sz));
    }

    // Reject quantized ops
    for (const auto &cur_op : sg->get_ops()) {
        const auto &op_kind = cur_op->get_kind();
        VCHECK_SDP_TRAIN(op_kind != graph::op_kind::Quantize
                        && op_kind != graph::op_kind::Dequantize
                        && op_kind != graph::op_kind::DynamicDequantize,
                false, "Training decomp does not support quantization");
        // Reject dropout
        VCHECK_SDP_TRAIN(op_kind != graph::op_kind::Dropout, false,
                "Training decomp does not support dropout yet");
        // Reject inf_as_zero mode softmax
        if (op_kind == graph::op_kind::SoftMax
                && cur_op->has_attr(op_attr::mode)) {
            const auto &mode = cur_op->get_attr<std::string>(op_attr::mode);
            VCHECK_SDP_TRAIN(mode != "inf_as_zero", false,
                    "Training decomp does not support inf_as_zero mode");
        }
    }

    // Check data types - only f32/bf16/f16
    auto src_dt = ltw(inputs[graph_inport[mm1_src]]).data_type();
    VCHECK_SDP_TRAIN(src_dt == graph::data_type::f32
                    || src_dt == graph::data_type::bf16
                    || src_dt == graph::data_type::f16,
            false, "Only supports f32/bf16/f16 data types");

    // K and V must have the same data type
    VCHECK_SDP_TRAIN(ltw(inputs[graph_inport[mm1_wei]]).data_type()
                    == ltw(inputs[graph_inport[mm2_wei]]).data_type(),
            false, "Key and value should have the same data type");

    // Initialize nthr with max threads num
    nthr = dnnl_get_max_threads();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    constexpr dim_t ratio = 2;
    VCHECK_SDP_TRAIN(batch_size * num_head_q > ratio * nthr, false,
            "Doesn't meet condition for decompose: batch_size * num_head_q "
            "should be larger than ratio * nthr");
#endif
    return true;
}

impl::status_t sdp_decomp_training_config_t::construct_params(
        std::shared_ptr<subgraph_t> &sg, registry_t &sdp_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    // Determine which output index is attention output vs stats.
    // Stats has last dim = 1, attention output has last dim = head_size_v.
    {
        dims out0_dims = ltw(outputs[0]).vdims();
        dims out1_dims = ltw(outputs[1]).vdims();
        if (out0_dims.back() == 1) {
            stats_out_index = 0;
            attn_out_index = 1;
        } else {
            stats_out_index = 1;
            attn_out_index = 0;
        }
    }

    CHECK(record_sdp_ops(sg));
    const dim_t last_dim = ndims - 1, second_last_dim = ndims - 2;

    // Update seq_len_kv from the actual weight shape after passes
    const auto &lt_wei = sdp_op[0]->get_input_logical_tensor(1);
    const ltw ltw_wei(lt_wei);
    seq_len_kv = ltw_wei.vdims()[last_dim];

    // Acquire data type
    memory::data_type dt_src_user = static_cast<memory::data_type>(
            ltw(inputs[graph_inport[mm1_src]]).data_type());
    // For training, intermediate computation (scores, softmax) uses f32
    // for numerical precision. User-facing I/O (Q, K, V, output, P) stays
    // in dt_src_user (e.g., bf16).
    memory::data_type dt_inter = memory::data_type::f32;

    ////////////////////////////////////////////////////////////////////////
    ////////////// Start Creating primitives ///////////////////////////////
    ////////////////////////////////////////////////////////////////////////
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    omp_set_num_threads(1);
#endif

    // Memory descriptors
    memory::desc sub_src1_md, sub_wei1_user_md, sub_wei1_md, sub_mm1_src_md,
            sub_mm1_wei_md, sub_mm1_dst_md, sub_wei2_user_md, sub_mm2_wei_md,
            sub_mm2_dst_md, sub_dst_md, sub_dst_user_md;
    std::vector<memory::desc> sub_mm1_post_md;

    // reorder0: src1 strided -> dense
    primitive_attr sub_reorder0_attr;
    sub_reorder0_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dims sub_src1_dims = {seq_len_q, head_size_qk};
    src1_strides = ltw(inputs[graph_inport[mm1_src]]).vstrides();
    sub_src1_md = memory::desc(sub_src1_dims, dt_src_user,
            {src1_strides[second_last_dim], src1_strides[last_dim]});
    auto sub_src1_d_md
            = memory::desc(sub_src1_dims, dt_src_user, format_tag::ab);
    auto sub_reorder0_pd = reorder::primitive_desc(
            p_engine, sub_src1_md, p_engine, sub_src1_d_md, sub_reorder0_attr);
    sub_reorder0.init(sub_reorder0_pd);

    // reorder1: key strided -> transposed dense
    primitive_attr sub_reorder1_attr;
    sub_reorder1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dims sub_wei1_dims = {head_size_qk, seq_len_kv};
    auto wei_md = make_dnnl_memory_desc(sdp_op[0]->get_input_logical_tensor(1));
    wei1_strides = wei_md.get_strides();
    sub_wei1_user_md = memory::desc(sub_wei1_dims, dt_src_user,
            {wei1_strides[second_last_dim], wei1_strides[last_dim]});
    sub_wei1_md = memory::desc(sub_wei1_dims, dt_src_user, format_tag::ba);
    auto sub_reorder1_pd = reorder::primitive_desc(p_engine, sub_wei1_user_md,
            p_engine, sub_wei1_md, sub_reorder1_attr);
    sub_reorder1.init(sub_reorder1_pd);

    // MatMul1: Q × K^T -> f32 scores
    dnnl::primitive_attr sub_matmul1_attr = make_primitive_attr(sdp_op[0]);
    dims sub_mm1_src_dims = {seq_len_q, head_size_qk};
    dims sub_mm1_wei_dims = {head_size_qk, seq_len_kv};
    dims sub_mm1_dst_dims = {seq_len_q, seq_len_kv};

    sub_mm1_src_md
            = memory::desc(sub_mm1_src_dims, dt_src_user, format_tag::ab);
    sub_mm1_wei_md
            = memory::desc(sub_mm1_wei_dims, dt_src_user, format_tag::ba);
    // mm1 output is f32 for numerical precision in softmax computation
    sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, format_tag::ab);

    // Handle post-ops for mm1 (scale, mask)
    dnnl::post_ops dnnl_pops;
    auto mm1_ori_dnnl_pops = sub_matmul1_attr.get_post_ops();
    for (int i = 0; i < mm1_ori_dnnl_pops.get()->len(); i++) {
        if (mm1_ori_dnnl_pops.get()->entry_[i].is_binary()) {
            auto alg = static_cast<algorithm>(
                    mm1_ori_dnnl_pops.get()->entry_[i].binary.alg);
            const dnnl::impl::memory_desc_t &ori_desc
                    = mm1_ori_dnnl_pops.get()->entry_[i].binary.user_src1_desc;
            auto post_shape = ori_desc.dims;
            auto post_stride = ori_desc.format_desc.blocking.strides;
            auto post_dt
                    = static_cast<dnnl::memory::data_type>(ori_desc.data_type);
            auto new_sub_md = memory::desc(
                    {post_shape[second_last_dim], post_shape[last_dim]},
                    post_dt,
                    {post_stride[second_last_dim], post_stride[last_dim]});
            sub_mm1_post_md.emplace_back(new_sub_md);
            dnnl_pops.append_binary(alg, new_sub_md);
        } else if (mm1_ori_dnnl_pops.get()->entry_[i].is_eltwise()) {
            auto alg = static_cast<algorithm>(
                    mm1_ori_dnnl_pops.get()->entry_[i].eltwise.alg);
            auto alpha = mm1_ori_dnnl_pops.get()->entry_[i].eltwise.alpha;
            auto beta = mm1_ori_dnnl_pops.get()->entry_[i].eltwise.beta;
            dnnl_pops.append_eltwise(alg, alpha, beta);
        }
    }
    sub_matmul1_attr.set_post_ops(dnnl_pops);
    auto sub_mm1_pd = matmul::primitive_desc(p_engine, sub_mm1_src_md,
            sub_mm1_wei_md, sub_mm1_dst_md, sub_matmul1_attr);
    sub_mm1_prim = matmul(sub_mm1_pd);

    // Softmax primitive: scores -> P
    dims scores_dims = {seq_len_q, seq_len_kv};
    dims stats_dims = {seq_len_q, 1};
    auto scores_md = memory::desc(scores_dims, dt_inter, format_tag::ab);
    auto stats_md = memory::desc(stats_dims, dt_inter, format_tag::ab);
    auto softmax_out_md = memory::desc(scores_dims, dt_inter, format_tag::ab);

    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_inference, algorithm::softmax_accurate,
            sub_mm1_dst_md, softmax_out_md,
            /*axis=*/sub_mm1_dst_md.get_ndims() - 1, softmax_attr);
    sub_softmax_prim = softmax_forward(sub_softmax_pd);

    // Stats computation: logsumexp = reduce_max(src) - log(reduce_max(P))
    // Fused into 2 primitives using post-ops:
    //   1) reduce_max(P) [log post-op] -> log_max_P
    //   2) reduce_max(scores) [binary_sub(log_max_P) post-op] -> stats
    memory::desc sub_reduce_max_P_scratchpad_md;
    memory::desc sub_reduce_max_src_scratchpad_md;
    {
        // reduce_max(P) with log post-op -> log_max_P
        primitive_attr reduce_max_P_attr;
        reduce_max_P_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        post_ops reduce_max_P_po;
        reduce_max_P_po.append_eltwise(algorithm::eltwise_log, 0.f, 0.f);
        reduce_max_P_attr.set_post_ops(reduce_max_P_po);
        auto sub_reduce_max_P_pd
                = reduction::primitive_desc(p_engine, algorithm::reduction_max,
                        softmax_out_md, stats_md, 0.f, 0.f, reduce_max_P_attr);
        sub_reduce_max_P_prim = reduction(sub_reduce_max_P_pd);
        sub_reduce_max_P_scratchpad_md = sub_reduce_max_P_pd.scratchpad_desc();

        // reduce_max(scores) with binary_sub(log_max_P) post-op -> stats
        primitive_attr reduce_max_src_attr;
        reduce_max_src_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        post_ops reduce_max_src_po;
        reduce_max_src_po.append_binary(algorithm::binary_sub, stats_md);
        reduce_max_src_attr.set_post_ops(reduce_max_src_po);
        auto sub_reduce_max_src_pd = reduction::primitive_desc(p_engine,
                algorithm::reduction_max, sub_mm1_dst_md, stats_md, 0.f, 0.f,
                reduce_max_src_attr);
        sub_reduce_max_src_prim = reduction(sub_reduce_max_src_pd);
        sub_reduce_max_src_scratchpad_md
                = sub_reduce_max_src_pd.scratchpad_desc();

        // Stats memories
        sub_log_max_P = memory(stats_md, p_engine, nullptr);
        sub_stats = memory(stats_md, p_engine, nullptr);
    }

    // reorder_stats: dense stats -> user layout
    primitive_attr sub_reorder_stats_attr;
    sub_reorder_stats_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // Extract stats strides now that shape inference has filled outputs
    stats_dst_strides = ltw(sdp_op[1]->get_output_logical_tensor(2)).vstrides();
    auto sub_stats_user_md = memory::desc(stats_dims, dt_inter,
            {stats_dst_strides[second_last_dim], stats_dst_strides[last_dim]});
    auto sub_reorder_stats_pd = reorder::primitive_desc(p_engine, stats_md,
            p_engine, sub_stats_user_md, sub_reorder_stats_attr);
    sub_reorder_stats.init(sub_reorder_stats_pd);
    sub_stats_user = memory(sub_stats_user_md, p_engine, nullptr);

    // reorder2: value strided -> dense
    primitive_attr sub_reorder2_attr;
    sub_reorder2_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dims sub_wei2_dims = {seq_len_kv, head_size_v};
    wei2_strides = ltw(inputs[graph_inport[mm2_wei]]).vstrides();
    sub_wei2_user_md = memory::desc(sub_wei2_dims, dt_src_user,
            {wei2_strides[second_last_dim], wei2_strides[last_dim]});
    auto sub_wei2_md = memory::desc(sub_wei2_dims, dt_src_user, format_tag::ab);
    auto sub_reorder2_pd = reorder::primitive_desc(p_engine, sub_wei2_user_md,
            p_engine, sub_wei2_md, sub_reorder2_attr);
    sub_reorder2.init(sub_reorder2_pd);

    // MatMul2: P x V
    // When softmax output is f32 but mm2 expects lower precision (bf16/f16),
    // add a reorder. For pure f32 SDPA, no reorder needed.
    needs_softmax_reorder = (dt_inter != dt_src_user);
    memory::desc sub_reorder_softmax_scratchpad_md;
    dnnl::primitive_attr sub_matmul2_attr = make_primitive_attr(sdp_op[2]);
    dims sub_mm2_src_dims = {seq_len_q, seq_len_kv};
    dims sub_mm2_wei_dims = {seq_len_kv, head_size_v};
    dims sub_mm2_dst_dims = {seq_len_q, head_size_v};
    auto sub_mm2_src_md
            = memory::desc(sub_mm2_src_dims, dt_src_user, format_tag::ab);

    if (needs_softmax_reorder) {
        // Reorder f32 softmax output -> bf16/f16 for mm2
        primitive_attr sub_reorder_softmax_attr;
        sub_reorder_softmax_attr.set_scratchpad_mode(
                dnnl::scratchpad_mode::user);
        auto sub_reorder_softmax_pd
                = reorder::primitive_desc(p_engine, softmax_out_md, p_engine,
                        sub_mm2_src_md, sub_reorder_softmax_attr);
        sub_reorder_softmax.init(sub_reorder_softmax_pd);
        sub_reorder_softmax_scratchpad_md
                = sub_reorder_softmax_pd.scratchpad_desc();
        sub_mm2_src = memory(sub_mm2_src_md, p_engine, nullptr);
    }

    sub_mm2_wei_md
            = memory::desc(sub_mm2_wei_dims, dt_src_user, format_tag::ab);
    sub_mm2_dst_md
            = memory::desc(sub_mm2_dst_dims, dt_src_user, format_tag::ab);
    auto sub_mm2_pd = matmul::primitive_desc(p_engine, sub_mm2_src_md,
            sub_mm2_wei_md, sub_mm2_dst_md, sub_matmul2_attr);
    sub_mm2_prim = matmul(sub_mm2_pd);

    // reorder3: output dense -> strided
    primitive_attr sub_reorder3_attr;
    sub_reorder3_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dims sub_dst_dims = {seq_len_q, head_size_v};
    auto out_lt = sdp_op[2]->get_output_logical_tensor(0);
    dst_strides = ltw(out_lt).vstrides();
    sub_dst_md = memory::desc(sub_dst_dims, dt_src_user, format_tag::ab);
    sub_dst_user_md = memory::desc(sub_dst_dims, dt_src_user,
            {dst_strides[second_last_dim], dst_strides[last_dim]});
    auto sub_reorder3_pd = reorder::primitive_desc(
            p_engine, sub_dst_md, p_engine, sub_dst_user_md, sub_reorder3_attr);
    sub_reorder3.init(sub_reorder3_pd);

    ////////////////////////////////////////////////////////////////////////
    /////////////// End Creating primitives ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    /////////////// Start Constructing exec args ///////////////////////////
    ////////////////////////////////////////////////////////////////////////

    // Find max scratchpad size
    size_t max_scratchpad_size = 0;
    memory::desc max_scratchpad_md;
    std::vector<memory::desc> scratchpads {
            sub_reorder0_pd.scratchpad_desc(),
            sub_reorder1_pd.scratchpad_desc(),
            sub_mm1_pd.scratchpad_desc(),
            sub_softmax_pd.scratchpad_desc(),
            sub_reorder2_pd.scratchpad_desc(),
            sub_mm2_pd.scratchpad_desc(),
            sub_reorder3_pd.scratchpad_desc(),
    };
    if (needs_softmax_reorder) {
        scratchpads.push_back(sub_reorder_softmax_scratchpad_md);
    }
    scratchpads.push_back(sub_reduce_max_src_scratchpad_md);
    scratchpads.push_back(sub_reduce_max_P_scratchpad_md);
    scratchpads.push_back(sub_reorder_stats_pd.scratchpad_desc());
    for (auto &sp : scratchpads) {
        const size_t size = sp.get_size();
        if (size > max_scratchpad_size) {
            max_scratchpad_size = size;
            max_scratchpad_md = sp;
        }
    }

    // Initialize memory objects
    sub_src1 = memory(sub_src1_md, p_engine, nullptr);
    sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
    sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
    sub_mm1_wei = memory(sub_wei1_md, p_engine, nullptr);
    sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);

    for (size_t i = 0; i < sub_mm1_post_md.size(); i++) {
        sub_mm1_post_mem.emplace_back(sub_mm1_post_md[i], p_engine, nullptr);
    }

    // Softmax output P is f32 (dt_inter); reorder to dt_src_user is applied before mm2 when needed.
    sub_softmax_out = memory(softmax_out_md, p_engine, nullptr);

    // mm2 memories
    sub_wei2_user = memory(sub_wei2_user_md, p_engine, nullptr);
    sub_mm2_wei = memory(sub_wei2_md, p_engine, nullptr);
    sub_mm2_dst = memory(sub_mm2_dst_md, p_engine, nullptr);
    sub_dst_user = memory(sub_dst_user_md, p_engine, nullptr);

    // scratchpad
    sub_scratchpad = memory(max_scratchpad_md, p_engine, nullptr);

    // Construct execution args
    sub_reorder0_args = {{DNNL_ARG_SRC, sub_src1}, {DNNL_ARG_DST, sub_mm1_src},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    sub_reorder1_args = {{DNNL_ARG_SRC, sub_wei1_user},
            {DNNL_ARG_DST, sub_mm1_wei}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_mm1_args = {{DNNL_ARG_SRC, sub_mm1_src},
            {DNNL_ARG_WEIGHTS, sub_mm1_wei}, {DNNL_ARG_DST, sub_mm1_dst},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    for (int i = 0; i < mm1_ori_dnnl_pops.get()->len(); i++) {
        if (mm1_ori_dnnl_pops.get()->entry_[i].is_binary()) {
            sub_mm1_args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                            sub_mm1_post_mem[i]});
        }
    }

    // Softmax args
    sub_softmax_args
            = {{DNNL_ARG_SRC, sub_mm1_dst}, {DNNL_ARG_DST, sub_softmax_out},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    // Stats args (fused: reduce_max_P[log] runs first, then reduce_max_src[sub])
    sub_reduce_max_P_args
            = {{DNNL_ARG_SRC, sub_softmax_out}, {DNNL_ARG_DST, sub_log_max_P},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    sub_reduce_max_src_args = {{DNNL_ARG_SRC, sub_mm1_dst},
            {DNNL_ARG_DST, sub_stats},
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, sub_log_max_P},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    sub_reorder_stats_args
            = {{DNNL_ARG_SRC, sub_stats}, {DNNL_ARG_DST, sub_stats_user},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    // Reorder f32 softmax_out -> bf16/f16 mm2_src.
    if (needs_softmax_reorder) {
        sub_reorder_softmax_args
                = {{DNNL_ARG_SRC, sub_softmax_out}, {DNNL_ARG_DST, sub_mm2_src},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    }

    sub_reorder2_args = {{DNNL_ARG_SRC, sub_wei2_user},
            {DNNL_ARG_DST, sub_mm2_wei}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    // mm2 src: bf16 sub_mm2_src if reordered, else f32 sub_softmax_out
    auto &mm2_src_mem = needs_softmax_reorder ? sub_mm2_src : sub_softmax_out;
    sub_mm2_args = {{DNNL_ARG_SRC, mm2_src_mem},
            {DNNL_ARG_WEIGHTS, sub_mm2_wei}, {DNNL_ARG_DST, sub_mm2_dst},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    sub_reorder3_args
            = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    ////////////////////////////////////////////////////////////////////////
    /////////////// End Constructing exec args /////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    // Memory planning
    memory_planning(sdp_registry);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    omp_set_num_threads(nthr);
#endif
    return status::success;
}

op_ptr sdp_decomp_training_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

impl::status_t sdp_decomp_training_config_t::record_input_offset(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    auto find_graph_inport = [&](std::shared_ptr<value_t> val) {
        if (val->get_consumers()[0].get_op().get_kind()
                        == graph::op_kind::MatMul
                || (val->has_producer()
                        && val->get_producer().get_kind()
                                == graph::op_kind::StaticReshape)) {
            while (val->has_producer()) {
                val = val->get_producer().get_input_value(0);
            }
        }
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        return -1;
    };

    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr;
    const std::unordered_set<graph::op_kind_t> post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::SoftMax};

    for (const auto &cur_op : sg->get_ops()) {
        if (mm1 && mm2) break;
        if (cur_op->get_kind() != graph::op_kind::MatMul) continue;

        auto post_op = get_post_op(cur_op);
        if (post_op && post_op_kind.count(post_op->get_kind())) {
            mm1 = cur_op;
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                has_scale = true;
                scale = post_op;
                post_op = get_post_op(post_op);
            }
            // Reject attention mask (Add or Select) for now
            if (post_op
                    && (post_op->get_kind() == graph::op_kind::Add
                            || post_op->get_kind() == graph::op_kind::Select)) {
                return status::unimplemented;
            }
        } else {
            mm2 = cur_op;
        }
    }

    VCHECK_SDP_TRAIN(mm1 != nullptr && mm2 != nullptr, status::invalid_graph,
            "Failed to find matmul1 or matmul2");

    graph_inport.emplace_back(find_graph_inport(mm1->get_input_value(0)));
    graph_inport.emplace_back(find_graph_inport(mm1->get_input_value(1)));
    graph_inport.emplace_back(find_graph_inport(mm2->get_input_value(1)));

    if (has_scale) {
        int scale_id = find_graph_inport(scale->get_input_value(1));
        if (scale_id == -1)
            scale_id = find_graph_inport(scale->get_input_value(0));
        graph_inport.emplace_back(scale_id);
    } else {
        graph_inport.emplace_back(-1);
    }

    return status::success;
}

impl::status_t sdp_decomp_training_config_t::record_sdp_ops(
        std::shared_ptr<subgraph_t> &sg) {
    for (const auto &cur_op : sg->get_ops()) {
        if (!cur_op || cur_op->get_kind() != op_kind::_matmul) continue;
        auto post_op = get_post_op(cur_op);
        if (!post_op || post_op->get_kind() != op_kind::_softmax) continue;
        auto ppost_op = get_post_op(post_op);
        VCHECK_SDP_TRAIN(ppost_op != nullptr, status::invalid_graph,
                "Failed to find mm2 after softmax");
        // sdp_op = [mm1, softmax, mm2]
        this->sdp_op = {cur_op, post_op, ppost_op};
        break;
    }
    VCHECK_SDP_TRAIN(
            sdp_op.size() == 3, status::invalid_graph, "Failed to record ops");
    return status::success;
}

void sdp_decomp_training_config_t::memory_planning(registry_t &sdp_registry) {
    registrar_t temporary_registrar = sdp_registry.registrar();

    // Memory reuse based on non-overlapping lifetimes:
    // key 0: mm1_src [seq_q, d_qk] then softmax_out [seq_q, seq_kv]
    // key 1: mm1_wei [d_qk, seq_kv]
    // key 2: mm1_dst [seq_q, seq_kv] then mm2_wei [seq_kv, d_v]
    // key 3: mm2_dst [seq_q, d_v]
    // key 4: scratchpad (shared across all primitives)

    // TODO(xxx): The memory planning can be further optimized if the user
    // inputs are already in dense format, then we don't need to allocate
    // separate memory for sub_mm1_src, sub_mm1_wei, and sub_mm2_wei.
    auto scores_size = sub_mm1_dst.get_desc().get_size();
    auto mm1_src_size = sub_mm1_src.get_desc().get_size();
    auto softmax_out_size = sub_softmax_out.get_desc().get_size();
    auto mm2_wei_size = sub_mm2_wei.get_desc().get_size();

    size_t key0_size = std::max(mm1_src_size, softmax_out_size);
    size_t key2_size = std::max(scores_size, mm2_wei_size);

    mem_key_map = {
            {sub_mm1_src.get(), 0},
            {sub_softmax_out.get(), 0},
            {sub_mm1_wei.get(), 1},
            {sub_mm1_dst.get(), 2},
            {sub_mm2_wei.get(), 2},
            {sub_mm2_dst.get(), 3},
            {sub_scratchpad.get(), 4},
    };

    temporary_registrar.book(0, key0_size);
    temporary_registrar.book(1, sub_mm1_wei.get_desc().get_size());
    temporary_registrar.book(2, key2_size);
    temporary_registrar.book(3, sub_mm2_dst.get_desc().get_size());
    temporary_registrar.book(4, sub_scratchpad.get_desc().get_size());

    int next_key = 5;
    if (needs_softmax_reorder) {
        mem_key_map[sub_mm2_src.get()] = next_key;
        temporary_registrar.book(next_key, sub_mm2_src.get_desc().get_size());
        next_key++;
    }

    mem_key_map[sub_log_max_P.get()] = next_key;
    temporary_registrar.book(next_key, sub_log_max_P.get_desc().get_size());
    next_key++;

    mem_key_map[sub_stats.get()] = next_key;
    temporary_registrar.book(next_key, sub_stats.get_desc().get_size());
    next_key++;
}

dnnl::primitive_attr sdp_decomp_training_config_t::make_primitive_attr(
        std::shared_ptr<op_t> &op) {
    dnnl::primitive_attr attr;
    if (op && op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    return attr;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
