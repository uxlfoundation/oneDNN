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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_CONV_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_CONV_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct conv_fwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::convolution_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::convolution_forward);

    conv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::convolution_forward(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    auto format_tag = get_format_tag_str(to_desc);
                    const auto &dims = to_desc.get_dims();
                    const auto &dtype = psrc_mem.get_desc().get_data_type();
                    dnnl_memory_desc_t new_to_desc_c;
                    dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                            static_cast<int>(dims.size()), dims.data(),
                            static_cast<dnnl_data_type_t>(dtype),
                            format_tag.data());
                    dnnl::memory::desc new_to_desc;
                    new_to_desc.reset(new_to_desc_c);
                    const memory to_mem
                            = dnnl::memory(new_to_desc, psrc_mem.get_engine());
                    to_mem.set_data_handle(dst_mem.get_data_handle());
                    dnnl::reorder(psrc_mem, to_mem)
                            .execute(stream, const_cast<memory &>(psrc_mem),
                                    const_cast<memory &>(to_mem));
                } else {
                    dnnl::reorder(psrc_mem, dst_mem)
                            .execute(stream, const_cast<memory &>(psrc_mem),
                                    const_cast<memory &>(dst_mem));
                }
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    auto format_tag = get_format_tag_str(to_desc);
                    const auto &dims = to_desc.get_dims();
                    const auto &dtype = psrc_mem.get_desc().get_data_type();
                    dnnl_memory_desc_t new_to_desc_c;
                    dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                            static_cast<int>(dims.size()), dims.data(),
                            static_cast<dnnl_data_type_t>(dtype),
                            format_tag.data());
                    dnnl::memory::desc new_to_desc;
                    new_to_desc.reset(new_to_desc_c);
                    const memory to_mem
                            = dnnl::memory(new_to_desc, psrc_mem.get_engine());
                    to_mem.set_data_handle(dst_mem.get_data_handle());
                    auto prim = dnnl::reorder(psrc_mem, to_mem);
                    auto e = dnnl::sycl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(to_mem)}},
                            sycl_deps);
                    sycl_deps = {e};
                    if (stream.get_engine().get_kind() == engine::kind::cpu)
                        e.wait();
                } else {
                    auto prim = dnnl::reorder(psrc_mem, dst_mem);
                    auto e = dnnl::sycl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(dst_mem)}},
                            sycl_deps);
                    sycl_deps = {e};
                }
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    auto format_tag = get_format_tag_str(to_desc);
                    const auto &dims = to_desc.get_dims();
                    const auto &dtype = psrc_mem.get_desc().get_data_type();
                    dnnl_memory_desc_t new_to_desc_c;
                    dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                            static_cast<int>(dims.size()), dims.data(),
                            static_cast<dnnl_data_type_t>(dtype),
                            format_tag.data());
                    dnnl::memory::desc new_to_desc;
                    new_to_desc.reset(new_to_desc_c);

                    const memory to_mem
                            = dnnl::ocl_interop::get_memory_kind(dst_mem)
                                    == dnnl::ocl_interop::memory_kind::usm
                            ? dnnl::ocl_interop::make_memory(new_to_desc,
                                    psrc_mem.get_engine(),
                                    dnnl::ocl_interop::memory_kind::usm,
                                    dst_mem.get_data_handle())
                            : dnnl::ocl_interop::make_memory(new_to_desc,
                                    psrc_mem.get_engine(),
                                    reinterpret_cast<cl_mem>(
                                            dst_mem.get_data_handle()));

                    auto prim = dnnl::reorder(psrc_mem, to_mem);
                    auto e = dnnl::ocl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(to_mem)}},
                            ocl_deps);
                    ocl_deps = {e};
                } else {
                    auto prim = dnnl::reorder(psrc_mem, dst_mem);
                    auto e = dnnl::ocl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(dst_mem)}},
                            ocl_deps);
                    ocl_deps = {e};
                }
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::convolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_fwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::deconvolution_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::deconvolution_forward);

    deconv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::deconvolution_forward(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::deconvolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_bwd_data_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::deconvolution_backward_data::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::deconvolution_backward_data);

    deconv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::deconvolution_backward_data(desc);
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
    dnnl::deconvolution_backward_data prim_;
};

struct deconv_bwd_weights_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::deconvolution_backward_weights::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::deconvolution_backward_weights);

    deconv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::deconvolution_backward_weights(desc);
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
    dnnl::deconvolution_backward_weights prim_;
};

struct conv_bwd_data_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::convolution_backward_data::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::convolution_backward_data);

    conv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::convolution_backward_data(desc);
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
    dnnl::convolution_backward_data prim_;
};

struct conv_bwd_weights_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::convolution_backward_weights::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::convolution_backward_weights);

    conv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::convolution_backward_weights(desc);
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
    dnnl::convolution_backward_weights prim_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
