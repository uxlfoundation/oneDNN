/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "graph/backend/dnnl/kernels/sdp_primitive_config.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"

#include "common/compiler_workarounds.hpp"

#define VCHECK_SDP_PRIMITIVE(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_primitive_kernel_t, (cond), status, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, handle, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void *mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

op_ptr sdp_primitive_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

op_ptr sdp_primitive_config_t::get_paged_cache_load_op(const op_ptr &op) const {
    const auto in_val = op->get_input_value(1);
    if (!in_val->has_producer()) return nullptr;
    auto &producer = in_val->get_producer();
    if (producer.get_kind() == op_kind::dnnl_permute) {
        if (!producer.get_input_value(0)->has_producer()) return nullptr;
        producer = producer.get_input_value(0)->get_producer();
    }
    if (producer.get_kind() != graph::op_kind::PagedCacheLoad
        && producer.get_kind() != op_kind::dnnl_paged_cache_load) return nullptr;
    return producer.shared_from_this();
}

status_t sdp_primitive_config_t::locate_io(std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {

    using dnnl::impl::utils::one_of;

    auto follow_back = [](std::shared_ptr<value_t> val) {
        while (val->has_producer() && val->get_producer().num_inputs() == 1)
            val = val->get_producer().get_input_value(0);
        return val;
    };

    auto in_tensor_list = [](const value_t *val,
                                  const std::vector<logical_tensor_t> &list) {
        for (auto &t : list)
            if (val->get_logical_tensor().id == t.id) return true;
        return false;
    };

    // Locate ops of interest: matmuls, scale, mask
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr, add = nullptr,
           final_op = nullptr, cache1 = nullptr, cache2 = nullptr;
    const std::unordered_set<op_kind_t> mm1_post_op_kind
            = {op_kind::dnnl_binary, op_kind::dnnl_softmax, op_kind::dnnl_mask};
    for (const auto &cur_op : sg->get_ops()) {
        if (in_tensor_list(cur_op->get_output_value(0).get(), outputs))
            final_op = cur_op;
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            // Locate mm1 and all post ops(scale and mask) here.
            // 1. locate mm1
            VCHECK_SDP_PRIMITIVE(mm1 == nullptr, status::unimplemented,
                    "Multiple mm1 found");
            mm1 = cur_op;
            // At least one of scale and mask exists
            if (post_op->get_kind() == op_kind::dnnl_binary) {
                auto binary_alg = static_cast<alg_kind_t>(
                        post_op->get_attr<int64_t>(op_attr::alg_kind));
                // 2. locate scale if have
                if (one_of(binary_alg, alg_kind::binary_mul,
                            alg_kind::binary_div)) {
                    scale = post_op;
                    invert_scale_ = (binary_alg == alg_kind::binary_div);
                    // Update `post_op` to the next op of scale
                    post_op = get_post_op(post_op);
                }

                // 3. locate mask if have
                if (post_op->get_kind() == op_kind::dnnl_binary) {
                    add = post_op;
                } else if (post_op->get_kind() == op_kind::dnnl_mask) {
                    // implicit causal mask
                    causal_mask_ = true;
                }
            } else if (post_op->get_kind() == op_kind::dnnl_mask) {
                causal_mask_ = true;
            }
        } else {
            VCHECK_SDP_PRIMITIVE(mm2 == nullptr, status::unimplemented,
                    "Multiple mm2 found");
            mm2 = cur_op;
        }
    }

    // Locate input/outputs: Q, K, V, dst, scale, mask
    mm1_ = mm1;
    mm2_ = mm2;
    VCHECK_SDP_PRIMITIVE((mm1 && mm2 && final_op), status::unimplemented,
            "Not all ops are found");

    q_ = mm1->get_input_value(0);
    k_ = mm1->get_input_value(1);
    v_ = mm2->get_input_value(1);

    // Locate paged attention input/outputs: QCache, VCache, block_table
    k_paged_cache_load_ = get_paged_cache_load_op(mm1);
    v_paged_cache_load_ = get_paged_cache_load_op(mm2);

    if (k_paged_cache_load_ != nullptr && v_paged_cache_load_ != nullptr) {
        k_cache_ = k_paged_cache_load_->get_input_value(0);
        v_cache_ = v_paged_cache_load_->get_input_value(0);
        // key and value should share the same block table
        block_table_ = k_paged_cache_load_->get_input_value(1);
        auto v_block_table = v_paged_cache_load_->get_input_value(1);
        VCHECK_SDP_PRIMITIVE(
            (k_cache_ && v_cache_ && ltw(block_table_->get_logical_tensor()).is_similar(ltw(v_block_table->get_logical_tensor()))),
            status::unimplemented,
            "Paged attention inputs/outputs not valid");
    page_attention_enabled_ = true;
    }

    if (quantized_) {
        // The input order of fused matmul is: src_0, src_1, scale, zero points
        if (mm1->num_inputs() > 2) k_scale_ = mm1->get_input_value(2);
        if (mm2->num_inputs() > 2) v_scale_ = mm2->get_input_value(2);

        // asymmetric quantization for key.
        if (4 == mm1->num_inputs()) k_zero_points_ = mm1->get_input_value(3);
        // asymmetric quantization for value.
        if (4 == mm2->num_inputs()) v_zero_points_ = mm2->get_input_value(3);
    }

    if(!page_attention_enabled_) {
    auto k_follow = follow_back(k_);
    for (auto &t : inputs)
        if (k_follow->get_logical_tensor().id == t.id) {
            kv_head_number_ = t.dims[1];
        }
    } else {
        kv_head_number_ = ltw(k_cache_->get_logical_tensor()).dims()[1];
    }
    dst_ = (final_op->get_kind() == op_kind::dnnl_transpose)
            ? final_op->get_input_value(0)
            : final_op->get_output_value(
                    0); /* for some reason final transpose is not fused into mm2 */

    if (scale) {
        auto s0 = follow_back(scale->get_input_value(0));
        auto s1 = follow_back(scale->get_input_value(1));
        scale_ = in_tensor_list(s1.get(), inputs) ? s1 : s0;
    }

    if (add) {
        auto m0 = add->get_input_value(0), m1 = add->get_input_value(1);
        if (in_tensor_list(m1.get(), inputs)) {
            attn_mask_ = m1;
        } else if (in_tensor_list(m0.get(), inputs)) {
            attn_mask_ = m0;
        } else if (m1->has_producer()
                && m1->get_producer().get_kind() == op_kind::dnnl_unsqueeze
                && in_tensor_list(
                        m1->get_producer().get_input_value(0).get(), inputs)) {
            // consider the case when mask is not 4D,
            // unsqueeze op is inserted to broadcast the mask
            attn_mask_ = m1;
        } else if (m0->has_producer()
                && m0->get_producer().get_kind() == op_kind::dnnl_unsqueeze
                && in_tensor_list(
                        m0->get_producer().get_input_value(0).get(), inputs)) {
            attn_mask_ = m0;
        } else {
            VCHECK_SDP_PRIMITIVE(
                    false, status::unimplemented, "explicit mask is not found");
        }
    }

    return status::success;
}

status_t sdp_primitive_config_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // At least 3 inputs: Q, K, V
    VCHECK_SDP_PRIMITIVE(inputs.size() >= 3, status::invalid_arguments,
            "At least 3 inputs are required");

    // step1(pattern check): Not support sdpa variants with select as mask
    // We already have a pattern matcher to ensure that the sdpa patterns
    // dispatch to here are knows ones, and we have quant check in sdpa base
    // kernel, so here we only check specific variants based on support matrix.
    const std::unordered_set<graph::op_kind_t> mm1_post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::Select,
                    graph::op_kind::SoftMax};
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr;
    for (const auto &cur_op : sg->get_ops()) {
        const auto &op_kind = cur_op->get_kind();
        if (op_kind == graph::op_kind::DynamicDequantize
                && cur_op->get_attr<std::string>(op_attr::qtype)
                        == "per_group") {
            if (!cur_op->has_attr(op_attr::group_shape))
                return status::invalid_arguments;
            const auto &group_shape = cur_op->get_attr<std::vector<int64_t>>(
                    op_attr::group_shape);
            const auto &input_lt
                    = cur_op->get_input_value(0)->get_logical_tensor();
            const auto &input_dims = ltw(input_lt).dims();
            if (static_cast<int>(group_shape.size()) != ltw(input_lt).ndims())
                return status::invalid_arguments;
            // Due to the precision issue of ukernel implementation, we only
            // support group_num=1 case for now.
            for (size_t idx = 0; idx < group_shape.size(); ++idx) {
                if (group_shape[idx] != 1
                        && group_shape[idx] != input_dims[idx])
                    return status::unimplemented;
            }
            // TODO(zhitao): execute the reorder for scale and zps mannually if the
            // transpose attribute is specified as true.
            auto post_op = get_post_op(cur_op);
            if (post_op && post_op->get_kind() == graph::op_kind::MatMul
                    && post_op->has_attr(op_attr::transpose_b)
                    && post_op->get_attr<bool>(op_attr::transpose_b))
                return status::unimplemented;
        }
        if (op_kind != graph::op_kind::MatMul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            mm1 = cur_op;
            // Not support select between mm1 and scale(optional)
            // GPT-J:[mm1] --> [select] --> [scale]* --> [mask]* --> ...
            VCHECK_SDP_PRIMITIVE(post_op->get_kind() != graph::op_kind::Select,
                    status::unimplemented,
                    "Not support select between mm1 and scale(optional)");
            // scale
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                // Scale exists, update post_op and traverse to next op
                scale = post_op;
                post_op = get_post_op(post_op);
            }
            // mask
            if (post_op) {
                if (post_op->get_kind() == graph::op_kind::Add) {
                    // Mask exists, update post_op and traverse to next op
                    post_op = get_post_op(post_op);
                }
                // Not support select after scale(optional) and mask(optional)
                // Distill-Bert:[mm1] --> [scale]* --> [mask]* --> [select] --> ...
                VCHECK_SDP_PRIMITIVE(post_op
                                && post_op->get_kind()
                                        != graph::op_kind::Select,
                        status::unimplemented,
                        "Not support select after scale(optional) and "
                        "mask(optional)");
            }
        } else {
            mm2 = cur_op;
        }
    }

    auto find_graph_inport = [&inputs](const std::shared_ptr<value_t> &val) {
        auto tmp_val = val;
        while (tmp_val->has_producer()) {
            const op_t &prod_op = tmp_val->get_producer();
            tmp_val = prod_op.get_input_value(0);
        }
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (tmp_val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        // If the corresponding input is not found, return an invalid value
        return -1;
    };

    VCHECK_SDP_PRIMITIVE(
            mm1 && mm2, status::invalid_graph, "mm1 or mm2 is not found");

    // step3(dims check): only support 4-dims now.
    int q_id = find_graph_inport(mm1->get_input_value(0));
    int k_id = find_graph_inport(mm1->get_input_value(1));
    int v_id = find_graph_inport(mm2->get_input_value(1));

    VCHECK_SDP_PRIMITIVE(q_id != -1 && k_id != -1 && v_id != -1,
            status::unimplemented, "Q, K, V are not found");
    VCHECK_SDP_PRIMITIVE(ltw(inputs[q_id]).vdims().size() == 4
                    && ltw(inputs[k_id]).vdims().size() == 4
                    && ltw(inputs[v_id]).vdims().size() == 4,
            status::unimplemented, "Q, K, V should be 4-dims");

    // sdp_primitive only supports single scale value.
    if (scale) {
        const auto &s = scale->get_input_value(1)->get_logical_tensor();
        VCHECK_SDP_PRIMITIVE(ltw(s).nelems() == 1, status::unimplemented,
                "Scale should be single value");
    }

    // Locate paged attention input/outputs: QCache, VCache, block_table
    k_paged_cache_load_ = get_paged_cache_load_op(mm1);
    v_paged_cache_load_ = get_paged_cache_load_op(mm2);

    if (k_paged_cache_load_ != nullptr && v_paged_cache_load_ != nullptr) {
        k_cache_ = k_paged_cache_load_->get_input_value(0);
        v_cache_ = v_paged_cache_load_->get_input_value(0);
        // key and value should share the same block table
        block_table_ = k_paged_cache_load_->get_input_value(1);
        auto v_block_table = v_paged_cache_load_->get_input_value(1);
        // bool same_block_table = false;
        // if (block_table_->ndims == v_block_table.ndims()) {
        //     same_block_table = block_table_ == v_block_table;
        // }
        VCHECK_SDP_PRIMITIVE(
                (k_cache_ && v_cache_ && ltw(block_table_->get_logical_tensor()).is_similar(ltw(v_block_table->get_logical_tensor()))),
                status::unimplemented,
                "Paged attention inputs/outputs not valid");
        page_attention_enabled_ = true;
    }

    return status::success;
}

status_t sdp_primitive_config_t::init(std::shared_ptr<subgraph_t> &sg,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {

    CHECK(locate_io(sg, inputs, outputs));

    // Retrieve mds and create pd, primitive
    auto md_q = make_dnnl_memory_desc(q_->get_logical_tensor());
    auto md_k = make_dnnl_memory_desc(k_->get_logical_tensor());
    auto md_v = make_dnnl_memory_desc(v_->get_logical_tensor());
    auto md_dst = make_dnnl_memory_desc(dst_->get_logical_tensor());

    dnnl::memory::desc md_q_pa, md_k_cache, md_v_cache, md_prompt_lens,
            md_subsequence_begins, md_block_indices, md_block_indices_begin;
    if (page_attention_enabled_) {
        auto q_lt = q_->get_logical_tensor();
        const auto q_strides = q_lt.layout.strides;

        // TODO: mv check to init_check
        // paged attention micro kernel only support q shape [1,1,query_num,head_num*head_size]
        // need to transpose and reshape original q[seq_num, head_num, seq_len, head_size]
        // as q_pa[seq_num * seq_len, head_num*head_size]
        if (q_strides[2] != q_lt.dims[1] * q_lt.dims[3]
                || q_strides[0] != q_lt.dims[1] * q_lt.dims[2] * q_lt.dims[3]
                || q_strides[3] != 1) {
            const auto status = status::unimplemented;
            VCONDCHECK(graph, create, dispatch, sdp, false, status,
                    "could not create sdp primitive, falling back\n");
            return status;
        }
        VCHECK_SDP_PRIMITIVE(q_lt.dims[1] == 1, status::unimplemented,
                "currently, paged attention only supports head_num 1");

        logical_tensor_t q_pa_lt = q_lt;
        q_pa_lt.dims[0] = 1;
        q_pa_lt.dims[1] = 1;
        q_pa_lt.dims[2] = q_lt.dims[0] * q_lt.dims[2];
        q_pa_lt.dims[3] = q_lt.dims[1] * q_lt.dims[3];
        q_pa_lt.layout.strides[0] = q_pa_lt.dims[2] * q_pa_lt.dims[3];
        q_pa_lt.layout.strides[1] = q_pa_lt.dims[2] * q_pa_lt.dims[3];
        q_pa_lt.layout.strides[2] = q_pa_lt.dims[3];
        q_pa_lt.layout.strides[3] = 1;
        md_q_pa = make_dnnl_memory_desc(q_pa_lt);

        auto md_k_cache_api = make_dnnl_memory_desc(k_cache_->get_logical_tensor());
        md_k_cache = transpose(md_k_cache_api, 2, 3);
        md_v_cache = make_dnnl_memory_desc(v_cache_->get_logical_tensor());
        // block_table [seq_num, max_block_num_per_seq]
        auto block_table_lt = block_table_->get_logical_tensor();
        // block_indices [1, 1, 1, seq_num * max_block_num_per_seq]
        // 1D tensor with the same number of elements as block_table;
        logical_tensor_t block_indices_lt;
        block_indices_lt.ndims = 4;
        block_indices_lt.dims[0] = 1;
        block_indices_lt.dims[1] = 1;
        block_indices_lt.dims[2] = 1;
        block_indices_lt.dims[3]
                = block_table_lt.dims[0] * block_table_lt.dims[1];
        block_indices_lt.data_type = block_table_lt.data_type;
        block_indices_lt.layout_type = layout_type::strided;
        block_indices_lt.layout.strides[0] = block_indices_lt.dims[3];
        block_indices_lt.layout.strides[1] = block_indices_lt.dims[3];
        block_indices_lt.layout.strides[2] = block_indices_lt.dims[3];
        block_indices_lt.layout.strides[3] = 1;
        block_indices_lt.property = block_table_lt.property;
        md_block_indices = make_dnnl_memory_desc(block_indices_lt);

        // block_indices_begin [1, 1, 1, seq_num + 1]
        // 1D tensor with the start position of each sequence in block_indices;
        logical_tensor_t block_indices_begin_lt;
        block_indices_begin_lt.ndims = 4;
        block_indices_begin_lt.dims[0] = 1;
        block_indices_begin_lt.dims[1] = 1;
        block_indices_begin_lt.dims[2] = 1; 
        block_indices_begin_lt.dims[3] = block_table_lt.dims[0] + 1;
        block_indices_begin_lt.data_type = block_table_lt.data_type;
        block_indices_begin_lt.layout_type = layout_type::strided;
        block_indices_begin_lt.layout.strides[0] = block_indices_begin_lt.dims[3];
        block_indices_begin_lt.layout.strides[1] = block_indices_begin_lt.dims[3];
        block_indices_begin_lt.layout.strides[2] = block_indices_begin_lt.dims[3];
        block_indices_begin_lt.layout.strides[3] = 1;
        block_indices_begin_lt.property = dnnl_graph_tensor_property_t::
                dnnl_graph_tensor_property_constant;
        md_block_indices_begin = make_dnnl_memory_desc(block_indices_begin_lt);

        // prompt_lens [1, 1, 1, query_num]
        // 1D tensor with value of seq_len of each query;
        logical_tensor_t prompt_lens_lt;
        prompt_lens_lt.ndims = 4;
        prompt_lens_lt.dims[0] = 1;
        prompt_lens_lt.dims[1] = 1;
        prompt_lens_lt.dims[2] = 1;
        prompt_lens_lt.dims[3] = q_lt.dims[0];
        prompt_lens_lt.data_type = dnnl_data_type_t::dnnl_s32;
        prompt_lens_lt.layout_type = layout_type::strided;
        prompt_lens_lt.layout.strides[0] = prompt_lens_lt.dims[3];
        prompt_lens_lt.layout.strides[1] = prompt_lens_lt.dims[3];
        prompt_lens_lt.layout.strides[2] = prompt_lens_lt.dims[3];
        prompt_lens_lt.layout.strides[3] = 1;
        prompt_lens_lt.property = dnnl_graph_tensor_property_t::
                dnnl_graph_tensor_property_constant;
        md_prompt_lens = make_dnnl_memory_desc(prompt_lens_lt);

        // subsequence_begins [1, 1, 1, seq_num + 1]
        // 1D tensor for the lenth of seq_lens of KV;
        //K&&V should have the same seq_len
        k_paged_cache_load_->has_same_attr_values(*v_paged_cache_load_);
        auto seq_lens = k_paged_cache_load_->get_attr<std::vector<int64_t>>(
                op_attr::seq_lens);
        logical_tensor_t subsequence_begins_lt;
        subsequence_begins_lt.ndims = 4;
        subsequence_begins_lt.dims[0] = 1;
        subsequence_begins_lt.dims[1] = 1;
        subsequence_begins_lt.dims[2] = 1;
        subsequence_begins_lt.dims[3] = q_lt.dims[0] + 1;
        subsequence_begins_lt.data_type = dnnl_data_type_t::dnnl_s32;
        subsequence_begins_lt.layout_type = layout_type::strided;
        subsequence_begins_lt.layout.strides[0] = subsequence_begins_lt.dims[3];
        subsequence_begins_lt.layout.strides[1] = subsequence_begins_lt.dims[3];
        subsequence_begins_lt.layout.strides[2] = subsequence_begins_lt.dims[3];
        subsequence_begins_lt.layout.strides[3] = 1;
        subsequence_begins_lt.property = dnnl_graph_tensor_property_t::
                dnnl_graph_tensor_property_constant;
        md_subsequence_begins = make_dnnl_memory_desc(subsequence_begins_lt);
    }

    dnnl::memory::desc md_mask;
    if (attn_mask_)
        md_mask = make_dnnl_memory_desc(attn_mask_->get_logical_tensor());

    auto scale_dt = impl::data_type::undef;
    if (scale_) scale_dt = scale_->get_logical_tensor().data_type;

    dnnl::primitive_attr attr, qk_attr, vs_attr;

    auto &mgr = sg->fusion_info_mgr_;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode().mode_));

    if (mm1_->has_attr(op_attr::fusion_info_key)
            && mm1_->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = mm1_->get_attr<int64_t>(op_attr::fusion_info_key);
        qk_attr = make_dnnl_primitive_attr(mm1_, mgr.get_info(key));
    }
    if (mm2_->has_attr(op_attr::fusion_info_key)
            && mm2_->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = mm2_->get_attr<int64_t>(op_attr::fusion_info_key);
        vs_attr = make_dnnl_primitive_attr(mm2_, mgr.get_info(key));
    }

    auto status = status::success;
    int context_len = 0;
    if (!page_attention_enabled_) {
        CHECK(create_sdpa_pd(sdpa_pd_, p_engine.get(), md_q.get(), md_k.get(),
                md_v.get(), md_dst.get(), md_mask.get(), scale_dt,
                invert_scale_, kv_head_number_, causal_mask_, attr.get(),
                qk_attr.get(), vs_attr.get(), md_prompt_lens.get(),
                md_subsequence_begins.get(), md_block_indices.get(),
                md_block_indices_begin.get(), context_len));
    } else {
        const auto seq_lens
                = k_paged_cache_load_->get_attr<std::vector<int64_t>>(
                        op_attr::seq_lens);
        const auto context_len
                = std::accumulate(seq_lens.begin(), seq_lens.end(), int64_t(0));
        std::cout<<"context_len: "<<context_len<<std::endl;
        CHECK(create_sdpa_pd(sdpa_pd_, p_engine.get(), md_q_pa.get(), md_k_cache.get(),
                md_v_cache.get(), md_dst.get(), md_mask.get(), scale_dt,
                invert_scale_, kv_head_number_, causal_mask_, attr.get(),
                qk_attr.get(), vs_attr.get(), md_prompt_lens.get(),
                md_subsequence_begins.get(), md_block_indices.get(),
                md_block_indices_begin.get(), context_len));
        // const auto seq_lens
        //         = k_paged_cache_load_->get_attr<std::vector<int64_t>>(
        //                 op_attr::seq_lens);
        // const auto context_len
        //         = std::accumulate(seq_lens.begin(), seq_lens.end(), int64_t(0));
        // paged_sdpa_pd_ = std::make_shared<dnnl::sdpa_micro::primitive_desc>(
        //         p_engine, context_len, md_q_pa, md_k_cache, md_v_cache, md_dst,
        //         md_mask, md_prompt_lens, md_subsequence_begins,
        //         md_block_indices, md_block_indices_begin);
    }
    status = sdpa_pd_->create_primitive(sdpa_prim_, p_engine.get());

    VCONDCHECK(graph, create, dispatch, sdp, status == status::success, status,
            "could not create sdp primitive, falling back\n");

    // prepare memory for page attention
    if (page_attention_enabled_) {
        const auto seq_num = q_->get_logical_tensor().dims[0];
        const auto q_len = q_->get_logical_tensor().dims[2];
        // query has the same length for each seq_num
        std::vector<int32_t> prompt_lens_data(seq_num, q_len);
        std::vector<int32_t> subsequence_begins_data(seq_num + 1, 0);
        // query has the same length for each subsequence
        for (auto i = 1; i < seq_num + 1; i++) {
            subsequence_begins_data[i] = i * q_len;
        }
        prompt_lens_ = dnnl::memory(md_prompt_lens, p_engine);
        write_to_dnnl_memory(prompt_lens_data.data(), prompt_lens_);
        subsequence_begins_ = dnnl::memory(md_subsequence_begins, p_engine);
        write_to_dnnl_memory(
                subsequence_begins_data.data(), subsequence_begins_);

        // block_table [seq_num, max_block_num_per_seq]
        auto block_table_lt = block_table_->get_logical_tensor();
        const auto max_block_num_per_seq = block_table_lt.dims[1];
        block_indices_begin_ = dnnl::memory(md_block_indices_begin, p_engine);
        std::vector<int32_t> block_indices_begin_data(seq_num + 1, 0);
        // block_table has the same length for each seq_num
        for (auto i = 1; i < seq_num + 1; i++) {
            block_indices_begin_data[i] = i * max_block_num_per_seq;
        }
        write_to_dnnl_memory(
                block_indices_begin_data.data(), block_indices_begin_);
    }
    return status;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
