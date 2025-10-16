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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_BATCH_NORM_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_BATCH_NORM_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct bn_folding_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER

    // bn_folding_t is a aggregated executable by using multiple primitives, so
    // we need a customized desc class to describe it.
    class desc_t {
        friend struct bn_folding_t;

        float epsilon_ = 1e-5f;
        std::string data_format_;
        std::string filter_format_;

        memory::desc epsilon_desc_;
        memory::desc new_scale_desc_;
        memory::desc new_variance_desc_;
        memory::desc scratchpad_desc_;

        dnnl::binary::primitive_desc add_pd_;
        dnnl::binary::primitive_desc mul_pd_;
        dnnl::binary::primitive_desc sub_pd_;

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
        // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
        dnnl::eltwise_forward::primitive_desc sqrt_pd_;
#endif

        bool with_bias_ {false};

    public:
        const memory::desc &scratchpad_desc() const { return scratchpad_desc_; }
    };

    static desc_t create_desc(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout);

    bn_folding_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const fpmath_t &fpmath,
            bool use_block_layout) {
        desc_ = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        add_prim_ = dnnl::binary(desc_.add_pd_);
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
        // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
        if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
            sqrt_prim_ = dnnl::eltwise_forward(desc_.sqrt_pd_);
        }
#endif
        mul_prim_ = dnnl::binary(desc_.mul_pd_);
        sub_prim_ = dnnl::binary(desc_.sub_pd_);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second
                                     : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epsilon
        memory epsilon_mem = make_dnnl_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        if (variance.get_engine().get_kind() == engine::kind::cpu) {
            float *ptr = (float *)epsilon_mem.get_data_handle();
            *ptr = desc_.epsilon_;
        } else {
            engine cpu_eng(engine::kind::cpu, 0);
            memory cpu_mem = make_dnnl_memory(
                    desc_.epsilon_desc_, cpu_eng, (void *)&desc_.epsilon_);
            dnnl::reorder(cpu_mem, epsilon_mem)
                    .execute(stream, cpu_mem, epsilon_mem);
        }

        add_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}});

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(desc_.new_scale_desc_, scale.get_engine(),
                scale.get_data_handle());
        memory new_sqrt_variance(desc_.new_variance_desc_,
                sqrt_variance.get_engine(), sqrt_variance.get_data_handle());
        mul_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
            if (mean.get_engine().get_kind() == engine::kind::cpu) {
                std::memcpy(valid_bias.get_data_handle(), zero.data(),
                        valid_bias.get_desc().get_size());
            } else {
                engine cpu_eng(engine::kind::cpu, 0);
                memory cpu_mem = make_dnnl_memory(
                        variance.get_desc(), cpu_eng, zero.data());
                dnnl::reorder(cpu_mem, valid_bias)
                        .execute(stream, cpu_mem, valid_bias);
            }
        }

        sub_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}});
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second
                                     : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epsilon
        memory epsilon_mem = make_dnnl_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), (void *)buf_start);

        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        ::sycl::event sycl_deps;

        if (scratchpad.get_engine().get_kind() == engine::kind::gpu) {
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA

            buf_start += epsilon_mem.get_desc().get_size();

            // variance + epsilon
            memory variance_epsilon = make_dnnl_memory(desc_.epsilon_desc_,
                    scratchpad.get_engine(), (void *)buf_start);

            // 1. sqrt_variance = sqrt(variance + epsilon)
            //auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
            sycl_queue
                    .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                            epsilon_mem.get_desc().get_size())
                    .wait();

            auto sycl_deps0 = dnnl::sycl_interop::execute(add_prim_, stream,
                    {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                            { DNNL_ARG_DST,
                                variance_epsilon }},
                    deps);

            sycl_deps = dnnl::sycl_interop::execute(sqrt_prim_, stream,
                    {{DNNL_ARG_SRC, variance_epsilon},
                            { DNNL_ARG_DST,
                                sqrt_variance }},
                    {sycl_deps0});
#else

            // 1. sqrt_variance = sqrt(variance + epsilon)
            //auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
            sycl_queue
                    .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                            epsilon_mem.get_desc().get_size())
                    .wait();

            sycl_deps = dnnl::sycl_interop::execute(add_prim_, stream,
                    {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                            {DNNL_ARG_DST, sqrt_variance}},
                    deps);

#endif
        } else {
            // 1. sqrt_variance = sqrt(variance + epsilon)
            sycl_queue
                    .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                            epsilon_mem.get_desc().get_size())
                    .wait();

            sycl_deps = dnnl::sycl_interop::execute(add_prim_, stream,
                    {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                            {DNNL_ARG_DST, sqrt_variance}},
                    deps);
        }
        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(desc_.new_scale_desc_, scale.get_engine(),
                scale.get_data_handle());
        memory new_sqrt_variance(desc_.new_variance_desc_,
                sqrt_variance.get_engine(), sqrt_variance.get_data_handle());

        auto sycl_deps2 = dnnl::sycl_interop::execute(mul_prim_, stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}},
                {sycl_deps});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
            sycl_queue
                    .memcpy(valid_bias.get_data_handle(), zero.data(),
                            valid_bias.get_desc().get_size())
                    .wait();
            auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                    {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                            {DNNL_ARG_DST, updated_bias},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    scale},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    sqrt_variance},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                    shift}},
                    {sycl_deps2});
            if (stream.get_engine().get_kind() == engine::kind::cpu)
                sycl_deps3.wait();
            return sycl_deps3;
        }

        auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {sycl_deps2});
        if (stream.get_engine().get_kind() == engine::kind::cpu)
            sycl_deps3.wait();
        return sycl_deps3;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second
                                     : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance

        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = dnnl::ocl_interop::make_memory(
                variance.get_desc(), scratchpad.get_engine(),
                dnnl::ocl_interop::memory_kind::usm, (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = dnnl::ocl_interop::make_memory(variance.get_desc(),
                    scratchpad.get_engine(),
                    dnnl::ocl_interop::memory_kind::usm, (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epsilon
        memory epsilon_mem = dnnl::ocl_interop::make_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), dnnl::ocl_interop::memory_kind::usm,
                (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        cl_event e;
        xpu::ocl::usm::memcpy(stream.get(), epsilon_mem.get_data_handle(),
                &desc_.epsilon_, epsilon_mem.get_desc().get_size(), 0, nullptr,
                &e);
        clWaitForEvents(1, &e);

        auto ocl_deps = dnnl::ocl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}},
                deps);

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale = dnnl::ocl_interop::make_memory(desc_.new_scale_desc_,
                scale.get_engine(), dnnl::ocl_interop::memory_kind::usm,
                scale.get_data_handle());
        memory new_sqrt_variance = dnnl::ocl_interop::make_memory(
                desc_.new_variance_desc_, sqrt_variance.get_engine(),
                dnnl::ocl_interop::memory_kind::usm,
                sqrt_variance.get_data_handle());

        auto ocl_deps2 = dnnl::ocl_interop::execute(mul_prim_, stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}},
                {ocl_deps});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
            xpu::ocl::usm::memcpy(stream.get(), valid_bias.get_data_handle(),
                    zero.data(), valid_bias.get_desc().get_size(), 0, nullptr,
                    &e);
            clWaitForEvents(1, &e);

            auto ocl_deps3 = dnnl::ocl_interop::execute(sub_prim_, stream,
                    {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                            {DNNL_ARG_DST, updated_bias},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    scale},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    sqrt_variance},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                    shift}},
                    {ocl_deps2});
            return ocl_deps3;
        }

        auto ocl_deps3 = dnnl::ocl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {ocl_deps2});
        return ocl_deps3;
    }
#endif
    status_t reset_engine(const dnnl::engine &p_engine) override {
        const auto add_desc_t = add_prim_.get_primitive_desc()->impl();
        dnnl_primitive_desc new_add_pd_t(add_desc_t, p_engine.get());
        dnnl::binary::primitive_desc new_add_pd(&new_add_pd_t);
        add_prim_ = dnnl::binary(new_add_pd);

        const auto mul_desc_t = mul_prim_.get_primitive_desc()->impl();
        dnnl_primitive_desc new_mul_pd_t(mul_desc_t, p_engine.get());
        dnnl::binary::primitive_desc new_mul_pd(&new_mul_pd_t);
        mul_prim_ = dnnl::binary(new_mul_pd);

        const auto sub_desc_t = sub_prim_.get_primitive_desc()->impl();
        dnnl_primitive_desc new_sub_pd_t(sub_desc_t, p_engine.get());
        dnnl::binary::primitive_desc new_sub_pd(&new_sub_pd_t);
        sub_prim_ = dnnl::binary(new_sub_pd);
        return status::success;
    }

private:
    desc_t desc_;
    dnnl::binary add_prim_;
    dnnl::binary mul_prim_;
    dnnl::binary sub_prim_;
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
    dnnl::eltwise_forward sqrt_prim_;
#endif
};

struct batchnorm_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::batch_normalization_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::batch_normalization_forward);

    batchnorm_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout)
        : is_training_(op->get_attr<bool>(op_attr::is_training)) {
        float momentum = 0.5;
        if (op->has_attr(op_attr::momentum))
            momentum = op->get_attr<float>(op_attr::momentum);
        scales_ = {momentum, 1 - momentum};
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::batch_normalization_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (!is_training_) {
            prim_.execute(stream, args);
            return;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        prim_.execute(stream, exe_args);

        // calculate running_mean and running_variance
        auto it_mean = args.find(DNNL_ARG_MEAN);
        auto it_var = args.find(DNNL_ARG_VARIANCE);
        auto it_src1 = args.find(DNNL_ARG_SRC_1);
        auto it_src2 = args.find(DNNL_ARG_SRC_2);
        auto it_dst1 = args.find(DNNL_ARG_DST_1);
        auto it_dst2 = args.find(DNNL_ARG_DST_2);

        if (graph::utils::one_of(args.end(), it_mean, it_var, it_src1, it_src2,
                    it_dst1, it_dst2)) {
            assert(!"cannot find one of the required memories");
            return;
        }

        auto batch_mean = it_mean->second;
        auto batch_variance = it_var->second;
        auto old_running_mean = it_src1->second;
        auto old_running_variance = it_src2->second;
        auto new_running_mean = it_dst1->second;
        auto new_running_variance = it_dst2->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        dnnl::sum({p_engine, scales_,
                          {old_running_mean.get_desc(), batch_mean.get_desc()}})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                                {DNNL_ARG_DST, new_running_mean}});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        dnnl::sum({p_engine, scales_,
                          {old_running_variance.get_desc(),
                                  batch_variance.get_desc()}})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                                {DNNL_ARG_DST, new_running_variance}});
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        if (!is_training_) {
            auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
            if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
            return e;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        auto e0 = dnnl::sycl_interop::execute(prim_, stream, exe_args, deps);

        // calculate running_mean and running_variance
        auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
        auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
        auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
        auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
        auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        auto sum_prim_0 = dnnl::sum({p_engine, scales_,
                {old_running_mean.get_desc(), batch_mean.get_desc()}});
        auto e1 = dnnl::sycl_interop::execute(sum_prim_0, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                        {DNNL_ARG_DST, new_running_mean}},
                {e0});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        auto sum_prim_1 = dnnl::sum({p_engine, scales_,
                {old_running_variance.get_desc(), batch_variance.get_desc()}});
        auto e2 = dnnl::sycl_interop::execute(sum_prim_1, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                        {DNNL_ARG_DST, new_running_variance}},
                {e1});
        if (stream.get_engine().get_kind() == engine::kind::cpu) e2.wait();
        return e2;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        if (!is_training_) {
            auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
            return e;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        auto e0 = dnnl::ocl_interop::execute(prim_, stream, exe_args, deps);

        // calculate running_mean and running_variance
        auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
        auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
        auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
        auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
        auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        auto sum_prim_0 = dnnl::sum({p_engine, scales_,
                {old_running_mean.get_desc(), batch_mean.get_desc()}});
        auto e1 = dnnl::ocl_interop::execute(sum_prim_0, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                        {DNNL_ARG_DST, new_running_mean}},
                {e0});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        auto sum_prim_1 = dnnl::sum({p_engine, scales_,
                {old_running_variance.get_desc(), batch_variance.get_desc()}});
        auto e2 = dnnl::ocl_interop::execute(sum_prim_1, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                        {DNNL_ARG_DST, new_running_variance}},
                {e1});
        return e2;
    }
#endif

private:
    dnnl::batch_normalization_forward prim_;
    bool is_training_ {false};
    std::vector<float> scales_;
};

struct batchnorm_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::batch_normalization_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::batch_normalization_backward);

    batchnorm_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::batch_normalization_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::batch_normalization_backward prim_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_BATCH_NORM_HPP
