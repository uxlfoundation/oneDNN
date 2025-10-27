/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_SOFTMAX_REUSABLE_HPP
#define GPU_INTEL_SOFTMAX_REUSABLE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/softmax/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace softmax {

enum algorithm_id_t {
    many_reductions_per_workgroup = 1,
    one_reduction_per_workgroup,
    one_reduction_per_subgroup,
    vectorized,
    small
};

struct reusable_params_t {
    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
        return status;
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"reusable_softmax_fwd_generic"};
        return kernel_names;
    }

#if __cplusplus >= 202002L
    bool operator==(const reusable_params_t &) const = default;
#endif
    serialization_stream_t serialize() const {
        DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(reusable_params_t);
        return serialization_stream_t(*this);
    }

    static reusable_params_t deserialize(const serialization_stream_t &s) {
        return deserializer_t(s).pop<reusable_params_t>();
    }

    compute::kernel_ctx_t get_kernel_ctx() const;

    data_type_t src_data_type;
    data_type_t dst_data_type;
    int algorithm_number;
    int subgroup_size;
    bool is_logsoftmax;
    bool is_softmax_inf_as_zero;
    bool require_large_buffers;

    uint8_t padding[5] = {0};

    compute::dispatch_compile_params_t gws_params;
};

struct reusable_runtime_params_t {
    dim_t softmax_axis_stride;
    dim_t softmax_axis_size;
    dim_t softmax_chunk_size;
    compute::dispatch_runtime_params_t gws_params;
};

struct reusable_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public fwd_pd_t {
        using fwd_pd_t::fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:reusable", reusable_fwd_t);

        status_t init(impl::engine_t *engine) {
            using arch_t = compute::gpu_arch_t;
            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
            const arch_t arch = intel_engine->device_info()->gpu_arch();

            const memory_desc_wrapper src_mdw(src_md());
            const memory_desc_wrapper dst_mdw(dst_md());
            const auto src_dt = src_mdw.data_type();
            const auto dst_dt = dst_mdw.data_type();
            const block_layout_t layout(src_mdw);

            using namespace data_type;
            VDISPATCH_SOFTMAX(is_fwd(), VERBOSE_BAD_PROPKIND);

            VDISPATCH_SOFTMAX(
                    utils::one_of(src_dt, f64, f32, f16, bf16, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(dst_dt, f64, f32, f16, bf16, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(src_dt, f16, bf16),
                                      arch == arch_t::xe_hpc),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(dst_dt, f16, bf16),
                                      arch == arch_t::xe_hpc),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(f16, src_dt, dst_dt),
                                      intel_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(
                    IMPLICATION(utils::one_of(data_type::f64, dst_dt, src_dt),
                            intel_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(attr()->has_default_values(
                                      primitive_attr_t::skip_mask_t::scales),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_SOFTMAX_SC(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // src, dst must have equal formats and dimensions
            VDISPATCH_SOFTMAX(src_mdw.ndims() == dst_mdw.ndims(),
                    VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "source",
                    "destination", src_mdw.ndims(), dst_mdw.ndims());
            const blocking_desc_t &src_blk = src_mdw.blocking_desc(),
                                  &dst_blk = dst_mdw.blocking_desc();
            for (int i = 0; i < src_mdw.ndims(); i++) {
                VDISPATCH_SOFTMAX(src_mdw.dims()[i] == dst_mdw.dims()[i],
                        VERBOSE_INCONSISTENT_DIM, "source", i, "destination",
                        i);
                VDISPATCH_SOFTMAX(src_blk.strides[i] == dst_blk.strides[i],
                        "stride source:%d is inconsistent with destination:%d",
                        i, i);
            }
            for (int i = 1; i < src_mdw.ndims(); i++) {
                VDISPATCH_SOFTMAX(src_blk.strides[i - 1] >= src_blk.strides[i],
                        "only canonical memory format tags supported");
            }

            // skip noop case
            VDISPATCH_SOFTMAX(axis_size() > 1, VERBOSE_UNSUPPORTED_TAG);

            // allow plain formats only
            bool plain_case = src_mdw.is_plain() && dst_mdw.is_plain();
            VDISPATCH_SOFTMAX(plain_case, VERBOSE_UNSUPPORTED_TAG);

            conf.require_large_buffers = std::max(src_mdw.size(0, true, true),
                                                 dst_mdw.size(0, true, true))
                    > UINT32_MAX;

            // compile-time configuration setup
            conf.is_logsoftmax = is_logsoftmax();
            conf.is_softmax_inf_as_zero
                    = alg_kind() == alg_kind::softmax_accurate_inf_as_zero;
            conf.src_data_type = src_dt;
            conf.dst_data_type = dst_dt;

            // utilize largest supported subgroup size
            conf.subgroup_size = [=] {
                for (int size : {16, 8}) {
                    if (intel_engine->mayiuse_sub_group(size)
                            && intel_engine
                                       ->mayiuse_block_reads_writes_with_sub_group(
                                               size))
                        return size;
                }
                return 0;
            }();
            if (conf.subgroup_size == 0) {
                assert(!"Device does not support a valid subgroup size");
                return status::unimplemented;
            }

            // run-time configuration setup
            rt_conf.softmax_axis_size = src_mdw.dims()[desc()->softmax_axis];
            for (const auto &block : layout) {
                if (block.dim_idx == into<dim_idx_t>(desc()->softmax_axis)) {
                    rt_conf.softmax_axis_stride = int64_t(block.stride);
                    break;
                }
            }

            // subgroup block read/write requires the tensor to be
            // 4/16-byte aligned, respectively
            const bool is_read_aligned
                    = (axis_size() * types::data_type_size(src_dt)) % 4 == 0;
            const bool is_write_aligned
                    = (axis_size() * types::data_type_size(dst_dt)) % 16 == 0;

            const auto nelems = src_mdw.nelems();

            conf.algorithm_number = [&]() { // -> int
                if (arch != arch_t::xe_hpg) {
                    if (rt_conf.softmax_axis_stride == 1
                            && rt_conf.softmax_axis_size >= 128
                            && nelems > (1 << 17)
                            && dnnl::impl::utils::div_up(
                                       rt_conf.softmax_axis_size,
                                       conf.subgroup_size)
                                    <= 1024
                            && is_read_aligned && is_write_aligned)
                        return vectorized;
                    if (rt_conf.softmax_axis_stride == 1
                            && rt_conf.softmax_axis_size <= conf.subgroup_size
                            && nelems < (1 << 15))
                        return small;
                }
                if (rt_conf.softmax_axis_size < 6 && nelems > 64000)
                    return many_reductions_per_workgroup;
                if (rt_conf.softmax_axis_size > 128)
                    return one_reduction_per_workgroup;
                if (rt_conf.softmax_axis_size <= 128)
                    return one_reduction_per_subgroup;
                return many_reductions_per_workgroup;
            }();

            const size_t max_wg_size = [&]() {
                auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
                        attr()->gpu_attr_.get());
                return intel_engine->device_info()->max_wg_size(
                        gpu_attr && gpu_attr->threads_per_eu() == 4);
            }();

            const size_t num_workers_per_workgroup = dnnl::impl::utils::div_up(
                    rt_conf.softmax_axis_size,
                    (nelems <= 64000)
                            ? 1
                            : ((rt_conf.softmax_axis_size <= 1024) ? 4 : 15));

            switch (conf.algorithm_number) {
                case many_reductions_per_workgroup:
                    CHECK(init_dispatch_default_reusable(intel_engine));
                    break;
                case one_reduction_per_workgroup:
                    // do not solve problems beyond hardware workgroup limit
                    VDISPATCH_SOFTMAX(num_workers_per_workgroup <= max_wg_size,
                            "softmax axis size too large");

                    CHECK(init_dispatch_workgroup_per_reduction(
                            intel_engine, num_workers_per_workgroup));
                    break;
                case one_reduction_per_subgroup:
                    CHECK(init_dispatch_workgroup_per_reduction(
                            intel_engine, conf.subgroup_size));
                    break;
                case vectorized:
                case small:
                    CHECK(init_dispatch_subgroup_per_reduction(intel_engine));
                    break;
            }

            return status::success;
        }

        status_t init_dispatch_default_reusable(gpu::engine_t *engine);
        status_t init_dispatch_workgroup_per_reduction(
                gpu::engine_t *engine, const size_t num_workers_per_workgroup);
        status_t init_dispatch_subgroup_per_reduction(gpu::engine_t *engine);

        reusable_params_t conf;
        reusable_runtime_params_t rt_conf;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(
                engine, kernels, pd()->conf.get_kernel_names(), pd()->conf));
        kernel_ = kernels[0];

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace softmax
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
