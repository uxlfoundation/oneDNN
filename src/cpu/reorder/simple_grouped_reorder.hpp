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
#ifndef CPU_REORDER_SIMPLE_GROUPED_REORDER_HPP
#define CPU_REORDER_SIMPLE_GROUPED_REORDER_HPP

#include <cstring>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

#define SIMPLE_GROUPED_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, typename fmt_i_t, fmt_i_t fmt_i, \
            impl::data_type_t type_o, typename fmt_o_t, fmt_o_t fmt_o
#define SIMPLE_GROUPED_REORDER_TEMPL_CALL \
    type_i, fmt_i_t, fmt_i, type_o, fmt_o_t, fmt_o

template <SIMPLE_GROUPED_REORDER_TEMPL_DECL, typename spec = void>
struct simple_grouped_reorder_impl_t {};

template <SIMPLE_GROUPED_REORDER_TEMPL_DECL>
struct simple_grouped_reorder_impl_t<SIMPLE_GROUPED_REORDER_TEMPL_CALL,
        typename utils::enable_if<std::is_same<fmt_i_t, format_tag_t>::value
                && std::is_same<fmt_o_t, format_tag_t>::value
                && (fmt_i == format_tag::any)
                && (fmt_o == format_tag::any)>::type> {

    static status_t is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {

        VDISPATCH_REORDER_IC(
                input_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_FORMAT_KIND);
        VDISPATCH_REORDER_IC(
                output_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_FORMAT_KIND);

        VDISPATCH_REORDER_IC(input_d.ndims() == output_d.ndims(),
                VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
        for (int d = 0; d < input_d.ndims(); d++) {
            VDISPATCH_REORDER_IC(input_d.dims()[d] == output_d.dims()[d],
                    VERBOSE_INCONSISTENT_DIM, "src", d, "dst", d);
        }

        const auto &src_gd = input_d.sparse_desc().grouped_desc;
        const auto &dst_gd = output_d.sparse_desc().grouped_desc;
        VDISPATCH_REORDER_IC(src_gd.group_count == dst_gd.group_count,
                VERBOSE_INCONSISTENT_MDS, "src", "dst");
        VDISPATCH_REORDER_IC(src_gd.variable_dim_idx == dst_gd.variable_dim_idx,
                VERBOSE_INCONSISTENT_MDS, "src", "dst");

        VDISPATCH_REORDER_IC(
                attr->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

        return status::success;
    }

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> &reorder) {
        auto *src_offsets = CTX_IN_MEM(int32_t *, DNNL_ARG_FROM, 1);
        auto *dst_offsets = CTX_OUT_MEM(int32_t *, DNNL_ARG_TO, 1);

        const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd->src_md());
        const dim_t group_count
                = input_d.sparse_desc().grouped_desc.group_count;

        engine_t *engine = get_service_engine();

        // Wrap values (buffer 0) into dense memory objects for nested reorder
        const void *src_ptr = CTX_IN_MEM(void *, DNNL_ARG_FROM, 0);
        void *dst_ptr = CTX_OUT_MEM(void *, DNNL_ARG_TO, 0);

        std::unique_ptr<memory_t, memory_deleter_t> src_mem;
        std::unique_ptr<memory_t, memory_deleter_t> dst_mem;
        CHECK(safe_ptr_assign(src_mem,
                new memory_t(engine, reorder->pd()->src_md(),
                        memory_flags_t::use_runtime_ptr,
                        const_cast<void *>(src_ptr))));
        CHECK(safe_ptr_assign(dst_mem,
                new memory_t(engine, reorder->pd()->dst_md(),
                        memory_flags_t::use_runtime_ptr, dst_ptr)));

        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = {src_mem.get(), true};
        r_args[DNNL_ARG_DST] = {dst_mem.get(), false};
        exec_ctx_t r_ctx(ctx, std::move(r_args));

        auto *nested_grantor
                = create_nested_grantor(ctx.get_scratchpad_grantor(),
                        memory_tracking::names::key_nested,
                        reorder->pd()->scratchpad_registry());
        r_ctx.set_scratchpad_grantor(nested_grantor);

        CHECK(reorder->execute(r_ctx));

        // Copy offsets (buffer 1)
        parallel_nd(
                group_count, [&](dim_t i) { dst_offsets[i] = src_offsets[i]; });

        return status::success;
    }
};

template <SIMPLE_GROUPED_REORDER_TEMPL_DECL, typename spec = void>
struct simple_grouped_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        DECLARE_COMMON_PD_T("simple_grouped::any", simple_grouped_reorder_t);

        std::shared_ptr<primitive_desc_t> reorder_pd_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {

            const bool ok = src_md->data_type == type_i
                    && dst_md->data_type == type_o;
            if (!ok) return status::invalid_arguments;

            CHECK(simple_grouped_reorder_impl_t<
                    SIMPLE_GROUPED_REORDER_TEMPL_CALL>::is_applicable(src_md,
                    dst_md, attr));

            auto _pd = make_unique_pd<pd_t>(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            CHECK(_pd->init(engine, src_engine, dst_engine));

            CHECK(_pd->init_scratchpad_md());
            return safe_ptr_assign(*reorder_pd, _pd.release());
        }

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            const auto input_d = memory_desc_wrapper(src_md());
            const dim_t nnz = input_d.nnz();

            // Create 1D dense descriptors for the values buffer.
            memory_desc_t src_values_md, dst_values_md;
            const dims_t dims = {nnz};
            CHECK(memory_desc_init_by_tag(
                    src_values_md, 1, dims, type_i, format_tag::a));
            CHECK(memory_desc_init_by_tag(
                    dst_values_md, 1, dims, type_o, format_tag::a));

            CHECK(reorder_primitive_desc_create(
                    reorder_pd_, engine, &src_values_md, &dst_values_md));

            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    reorder_pd_->scratchpad_registry());
            return status::success;
        }

        friend dnnl::impl::impl_list_item_t;
    };

    simple_grouped_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->reorder_pd_->create_primitive(reorder_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return simple_grouped_reorder_impl_t<
                SIMPLE_GROUPED_REORDER_TEMPL_CALL>::execute(pd(), ctx,
                reorder_);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> reorder_;
};

#undef SIMPLE_GROUPED_REORDER_TEMPL_DECL
#undef SIMPLE_GROUPED_REORDER_TEMPL_CALL

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
