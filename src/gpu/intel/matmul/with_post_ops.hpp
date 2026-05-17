/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_WITH_POST_OPS_HPP
#define GPU_INTEL_MATMUL_WITH_POST_OPS_HPP

#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct with_post_ops_t : public intel::primitive_t {
    using intel::primitive_t::primitive_t;
    struct pd_t : public gpu::gpu_matmul_pd_t {
        using gpu::gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ocl:with_po:any", with_post_ops_t);

        // Matmul-native attr keys. attr_ is never re-keyed, so these resolve
        // user-bound storage at exec time.
        static constexpr int kA = DNNL_ARG_SRC;
        static constexpr int kB = DNNL_ARG_WEIGHTS;
        static constexpr int kC = DNNL_ARG_DST;

        status_t init(impl::engine_t *engine);

        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool use_scratchpad() const {
            return use_scratchpad_with_post_op_worker;
        }
        status_t query(query_t what, int idx, void *result) const override {
            // The user-facing pd is THIS (outer). The inner pd's attr has
            // post_ops/scales/zp/dropout stripped (attributes_without_po) and
            // its dst dt was forced to the intermediate accum type (f32) for
            // the gemm scratchpad. Any md query routed to the inner returns
            // the f32-intermediate view rather than the user's narrower dst
            // (s8/bf16/f16/...) — the framework then binds the user buffer
            // against a mismatched md and reads 4x the bytes. Same hazard
            // for src/weights/diff_*: those must reflect the user-facing
            // matmul md, not the inner's (potentially re-laid-out) view.
            //
            // Default: route ALL queries to THIS (outer). The outer's
            // src_md_/weights_md_/dst_md_ are the user-facing mds; the
            // outer's scratchpad_md/scratchpad_registry already books the
            // inner's registry under key_nested_multiple (see
            // init_scratchpad), so scratchpad_md from outer is correct and
            // sized to cover the inner. The outer's n_inputs/n_outputs/
            // impl_info_str ("ocl:with_po:any") are also the user-facing
            // values.
            //
            // Inner-only escape hatch — load-bearing:
            //   - preferred_gpu_threads_per_eu: the inner gemm kernel owns
            //     the GRF-derived recommendation. The outer falls through
            //     to primitive_desc_t::query which has no case for this
            //     query and returns unimplemented. ip/matmul.hpp (BWD_W)
            //     and rnn/grid.cpp call matmul_pd_->query(
            //     preferred_gpu_threads_per_eu, ...) after create_matmul_pd,
            //     and the iterator can land on with_post_ops_t::pd_t.
            //     Without the hatch, those callers silently lose the GRF
            //     hint and pick a sub-optimal threads-per-EU.
            if ((int)what == (int)query::preferred_gpu_threads_per_eu) {
                if (pd_) return pd_->query(what, idx, result);
            }
            return gpu::gpu_matmul_pd_t::query(what, idx, result);
        }

        std::shared_ptr<primitive_desc_t> pd_;
        bool use_scratchpad_with_post_op_worker = false;
        bool use_reorder = false;
        compute::dispatch_t dispatch_;
        attr_info_t attr_info_;
        bool subbyte_pack_ = false;
        bool dynamic_scales_ = false;
        bool requires_user_scales_ = false;
        bool with_dropout = false;
        bool dropout_use_host_scalars = false;
        bool dropout_use_offset = false;
        bool dropout_has_output_mask = false;
        data_type_t dst_type_ = data_type::undef;
        data_type_t acc_type_ = data_type::undef;
    };

    status_t init(impl::engine_t *engine) override {
        auto ret_status = create_nested_primitive(prim_, pd()->pd_, engine);
        CHECK(ret_status);
        primitive_attr_t attr;
        int threads_per_eu = 0;
        if (status::success
                == pd()->pd_->query(query::preferred_gpu_threads_per_eu, 0,
                        &threads_per_eu)) {
            CHECK(attr.set_gpu_attr(gpu_primitive_attr_t(threads_per_eu)));
        }
        compute::kernel_ctx_t kernel_ctx(&attr);
        CHECK(pd()->init_kernel_ctx(kernel_ctx));
        CHECK(create_kernel(engine, &kernels_[0], "gemm_post_ops", kernel_ctx));
        const bool dyn_scales = pd()->dynamic_scales_;
        if (dyn_scales) {
            compute::kernel_ctx_t alt_ctx(pd()->attr());
            const auto src_info = memory_desc_info_t::create(pd_->dst_md(0));
            dnnl_memory_desc dst_md(*(pd()->dst_md(0)));
            dst_md.data_type = pd()->dst_type_;
            memory_desc_wrapper dst_d(dst_md);
            def_memory_desc_info(alt_ctx, src_info, "SRC");
            def_memory_desc_info(
                    alt_ctx, memory_desc_info_t::create(dst_d), "DST");
            def_data_type(alt_ctx,
                    pd()->attr()->scales_.get_data_type(pd_t::kC),
                    "DST_SCALES");
            const int ndims = dst_d.ndims();
            bool runtime_dims
                    = pd()->has_runtime_dims_or_strides() || ndims > 5;
            if (!runtime_dims) {
                offsets_t off;
                set_offsets(dst_d, off.dst_off);
                def_offsets(off.dst_off, alt_ctx, "DST", ndims);
                alt_ctx.define_int("NDIMS", ndims);
            }
            CHECK(create_kernel(
                    engine, &kernels_[1], "dynamic_scale_dst", alt_ctx));
        }
        if (pd()->subbyte_pack_)
            CHECK(create_kernel(
                    engine, &kernels_[2], "subbyte_pack", kernel_ctx));
        return status::success;
    }

    status_t execute(const impl::exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return (const pd_t *)intel::primitive_t::pd().get();
    }
    std::shared_ptr<impl::primitive_t> prim_;
    std::array<compute::kernel_t, 3> kernels_ = {};
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
