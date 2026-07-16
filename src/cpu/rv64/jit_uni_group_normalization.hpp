/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#ifndef CPU_RV64_JIT_UNI_GROUP_NORMALIZATION_HPP
#define CPU_RV64_JIT_UNI_GROUP_NORMALIZATION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_group_normalization_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Pure-JIT forward group normalization for RV64 (RVV / Zvfh), structured after
// the x64 jit_uni_group_normalization: a single non-templated primitive whose
// internal kernel factories pick the templated kernel at runtime via mayiuse.
// Two plain layouts are supported in one primitive (pooling-style tag_kind):
//   * ncsp (channel-first, nchw...): the group is one contiguous C_PER_G*SP run;
//     stats are a flat f64 reduction; normalize vectorizes over spatial.
//   * nspc (channel-last, nhwc...): channels are contiguous per spatial position;
//     stats accumulate channel chunks across spatial; normalize vectorizes over
//     channels with per-lane scale/shift.
// dtypes: f32 (isa v) and f16 (isa zvfh, computed at f32 via widen/narrow).
// post-ops: any injector-supported eltwise chain plus a per-tensor (scalar)
// broadcast binary. The per-tensor-only restriction matches x64's group-norm
// binary, but the binary capability is narrower than x64 on dtype: the rv64
// binary injector reads the rhs as f32 only (flw/vle32), so the pd rejects a
// non-f32 binary src1, whereas x64's binary injector is dtype-aware. sum/prelu,
// per-element binary, and non-f32 binary src1 all fall back to ncsp/ref.
struct jit_uni_group_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_group_normalization_fwd_pd_t {
        using cpu_group_normalization_fwd_pd_t::
                cpu_group_normalization_fwd_pd_t;

        // isa_ (set in init from the data type) drives the impl name, mirroring
        // the rv64 pool pd: "jit:rvv" for f32 (v) / "jit:rvv_zvfh" for f16.
        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa_, ""),
                jit_uni_group_normalization_fwd_t);

        status_t init(const engine_t *engine);

        bool is_ncsp_ = true; // tag_kind: channel-first vs channel-last
        cpu_isa_t isa_ = v; // kernel isa; selects the impl-name suffix
    };

    jit_uni_group_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    // Per-group stats call: the kernel reduces the group and writes the raw
    // sum and sum-of-squares (as doubles); the host turns those into mean/var,
    // matching the previous intrinsic path's numerics exactly. Shape counts are
    // passed (not baked) so the kernel stays shape-agnostic and large extents
    // never hit the 32-bit immediate limit.
    struct stat_call_params_t {
        const void *src;
        double *sum;
        double *sumsq;
        dim_t c_per_g; // channels per group
        dim_t sp; // spatial extent (D*H*W)
        dim_t c; // total channels (nspc channel-block stride)
    };

    // Per-group normalize call. mean/inv_std are precomputed by the host. scale
    // and shift already point at this group's first channel (base + g*C_PER_G).
    struct norm_call_params_t {
        const void *src;
        void *dst;
        float mean;
        float inv_std;
        const float *scale; // nullptr when !use_scale
        const float *shift; // nullptr when !use_shift
        const void *const *post_ops_binary_rhs; // rhs base array, or nullptr
        dim_t c_per_g; // channels per group
        dim_t sp; // spatial extent (D*H*W)
        dim_t c; // total channels (nspc position stride)
    };

    struct kernel_stat_base_t {
        virtual void operator()(const stat_call_params_t *p) const = 0;
        virtual status_t create_kernel() = 0;
        static kernel_stat_base_t *create(const pd_t *pd);
        virtual ~kernel_stat_base_t() = default;
    };

    struct kernel_norm_base_t {
        virtual void operator()(const norm_call_params_t *p) const = 0;
        virtual status_t create_kernel() = 0;
        static kernel_norm_base_t *create(const pd_t *pd);
        virtual ~kernel_norm_base_t() = default;
    };

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<kernel_stat_base_t> kernel_stat_;
    std::unique_ptr<kernel_norm_base_t> kernel_norm_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_GROUP_NORMALIZATION_HPP
