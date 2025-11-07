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

#ifndef CPU_REF_GROUPED_GEMM_HPP
#define CPU_REF_GROUPED_GEMM_HPP

#include "common/grouped_gemm_pd.hpp"
#include "common/primitive.hpp"

#define VDISPATCH_GROUPED_GEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, grouped_gemm, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_grouped_gemm_t : public primitive_t {
    struct pd_t : public grouped_gemm_pd_t {
        using grouped_gemm_pd_t::grouped_gemm_pd_t;

        DECLARE_GROUPED_GEMM_PD_t("ref:any", ref_grouped_gemm_t);

        status_t init(engine_t *engine) {
            bool ok = grouped_gemm_pd_t::init(engine) == status::success;
            VDISPATCH_GROUPED_GEMM(ok, VERBOSE_BAD_ENGINE_KIND);

            // TODO: figure out the implementation
            // init_scratchpad();
            return status::success;
        }
    };

    ref_grouped_gemm_t(const pd_t *apd) : primitive_t(apd) {}

    // NOTE: Only supports float data type and 2D matrices for now
    status_t execute(const exec_ctx_t &ctx) const override {
        // Get the group size from the primitive descriptor
        // TODO: possibly change from n_outputs() since it could be confusing
        const int num_groups = pd()->n_outputs();

        for (int group_id = 0; group_id < num_groups; ++group_id) {
            // Extract dimensions (resolves DNNL_RUNTIME_DIM_VAL)
            const dim_t M = ctx.memory_mdw(DNNL_ARG_MULTIPLE_SRC + group_id)
                                    .dims()[0];
            const dim_t K = ctx.memory_mdw(DNNL_ARG_MULTIPLE_SRC + group_id)
                                    .dims()[1];
            const dim_t N = ctx.memory_mdw(DNNL_ARG_MULTIPLE_WEIGHTS + group_id)
                                    .dims()[1];

            const float *group_input = CTX_IN_MEM(
                    const float *, DNNL_ARG_MULTIPLE_SRC + group_id);
            float *group_output
                    = CTX_OUT_MEM(float *, DNNL_ARG_MULTIPLE_DST + group_id);
            const float *W = CTX_IN_MEM(
                    const float *, DNNL_ARG_MULTIPLE_WEIGHTS + group_id);
            const float *b = CTX_IN_MEM(
                    const float *, DNNL_ARG_MULTIPLE_BIAS + group_id);

            // TODO: see if needs better handling
            if (!group_input || !group_output || !W) {
                return status::invalid_arguments;
            }

            // Extract source (matrix A) scales
            const float *scale_src = nullptr;
            int src_scale_mask = 0;
            const auto &src_scale_arg = ctx.args().find(
                    DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_SRC + group_id));
            if (src_scale_arg != ctx.args().end()) {
                scale_src = CTX_IN_MEM(const float *,
                        DNNL_ARG_ATTR_SCALES
                                | (DNNL_ARG_MULTIPLE_SRC + group_id));
                src_scale_mask = pd()->attr()->scales_.get_mask(
                        DNNL_ARG_MULTIPLE_SRC + group_id);
            }

            // Extract weights (matrix B) scales
            const float *scale_wei = nullptr;
            int wei_scale_mask = 0;
            const auto &wei_scale_arg = ctx.args().find(DNNL_ARG_ATTR_SCALES
                    | (DNNL_ARG_MULTIPLE_WEIGHTS + group_id));
            if (wei_scale_arg != ctx.args().end()) {
                scale_wei = CTX_IN_MEM(const float *,
                        DNNL_ARG_ATTR_SCALES
                                | (DNNL_ARG_MULTIPLE_WEIGHTS + group_id));
                wei_scale_mask = pd()->attr()->scales_.get_mask(
                        DNNL_ARG_MULTIPLE_WEIGHTS + group_id);
            }

            // Extract destination (output) scales
            // TODO: for now it is always per-tensor (i.e. mask == 0)
            const float *scale_dst = nullptr;
            //int dst_scale_mask = 0;
            const auto &dst_scale_arg = ctx.args().find(
                    DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_DST + group_id));
            if (dst_scale_arg != ctx.args().end()) {
                scale_dst = CTX_IN_MEM(const float *,
                        DNNL_ARG_ATTR_SCALES
                                | (DNNL_ARG_MULTIPLE_DST + group_id));
                //dst_scale_mask = pd()->attr()->scales_.get_mask(
                //DNNL_ARG_MULTIPLE_DST + group_id);
            }

            // GEMM with scaling
            // Output = ((A * scale_A) * (B * scale_B) + bias) / scale_dst
            // TODO: only supports mask == 0 or row-wise for src and col-wise for wei
            for (dim_t m = 0; m < M; ++m) {
                for (dim_t n = 0; n < N; ++n) {
                    float sum = 0.0f;

                    for (dim_t k = 0; k < K; ++k) {
                        float a_val = group_input[m * K + k];
                        float b_val = W[k * N + n];

                        if (scale_src) {
                            float s = (src_scale_mask == 1) ? scale_src[m]
                                                            : scale_src[0];
                            a_val *= s;
                        }

                        if (scale_wei) {
                            float s = (wei_scale_mask == 2) ? scale_wei[n]
                                                            : scale_wei[0];
                            b_val *= s;
                        }

                        sum += a_val * b_val;
                    }

                    if (b) { sum += b[n]; }

                    if (scale_dst) { sum /= scale_dst[0]; }

                    group_output[m * N + n] = sum;
                }
            }
        }

        return status::success;
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif