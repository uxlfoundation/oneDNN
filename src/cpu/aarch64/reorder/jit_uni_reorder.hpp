/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022, 2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_REORDER_JIT_UNI_REORDER_HPP
#define CPU_AARCH64_REORDER_JIT_UNI_REORDER_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "cpu/aarch64/reorder/jit_uni_reorder_kernel.hpp"
#include "cpu/aarch64/reorder/jit_uni_reorder_utils.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct jit_uni_reorder_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
        int nthr_;
        bool with_groups_ = false;
        dim_t D_mask_ = 0;

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

    private:
        status_t init_scratchpad();
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        friend dnnl::impl::impl_list_item_t;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    enum { ndims_driver_max = 4 };

private:
    void omp_driver_0d(int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;

    void omp_driver(const char *in, char *out, const float *src_scales,
            const float *dst_scales, const int32_t *src_zero_points,
            const int32_t *dst_zero_points,
            const memory_tracking::grantor_t &scratchpad) const;

    void fill_curr_data_chunks(const tr::prb_t &prb, const int off,
            const ptrdiff_t *omp_data_chunks, const int omp_ndims,
            tr::tail_call_param_t &c) const;

    void reduce_compensation(char *out,
            const int32_t *compensation_reduce_scratch, const int nthr,
            const dim_t wspace_per_thr_size) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::kernel_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
