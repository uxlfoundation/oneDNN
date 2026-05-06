/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_RVV_WINOGRAD_CONVOLUTION_HPP
#define CPU_RV64_RVV_WINOGRAD_CONVOLUTION_HPP

#include <memory>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/rv64/brgemm/brgemm.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Winograd domain specification for GEMM-based Winograd convolution.
// Single-core execution: processes batches sequentially with
// local input/output buffers to keep working set in cache.
struct WinogradDomainSpec_t {
    // Matrix dimensions for brgemm: C[OC×tiles] = A[OC×IC] × B[IC×tiles]
    dim_t M; // Total tiles per batch = ceil(oh/2) * ceil(ow/2)
    dim_t K; // Input channels (IC)
    dim_t N; // Output channels (OC)

    dim_t n_gemms; // = 16 (Winograd F(2×2, 3×3) has 16 transformed elements)
    dim_t n_batches; // = mb (batch size)

    // Weight layout: [16][IC_rounded × OC_rounded] col-major per element
    dim_t weight_ld_row; // LDA for brgemm = oc_rounded
    dim_t weight_ld_matrix; // Per-element size = oc_rounded × ic_rounded
    dim_t weight_ic_rounded; // Round up IC for cache-line alignment
    dim_t weight_oc_rounded; // Round up OC for cache-line alignment

    // Input buffer: [16][tiles × IC_rounded] per element
    dim_t input_ld_row; // LDB for brgemm = ic_rounded
    dim_t input_ld_batch; // Per-element stride = tiles × ic_rounded

    // Output buffer: [16][tiles × OC] per element
    dim_t output_ld_row; // LDC for brgemm = OC
    dim_t output_ld_batch; // Per-element stride = tiles × OC

    // Buffer sizes in floats
    size_t weight_matrix_size; // Total weight buffer = 16 × weight_ld_matrix
    size_t V_buffer_size; // Input buffer size
    size_t M_buffer_size; // Output buffer size
};

struct rvv_winograd_conf_t {
    // Convolution parameters
    bool with_bias;
    dim_t ih, iw; // Input spatial dimensions
    dim_t ic, oc; // Input/output channels
    dim_t kh, kw; // Kernel size (should be 3x3)
    dim_t stride_h, stride_w; // Stride (should be 1x1)
    dim_t pad_t, pad_b; // Top/bottom padding
    dim_t pad_l, pad_r; // Left/right padding

    // Output dimensions
    dim_t oh, ow;

    // Winograd transform parameters
    dim_t mb; // Batch size

    // Winograd domain specification for GEMM-based execution
    WinogradDomainSpec_t wspec;
};

status_t rvv_winograd_init_conf(rvv_winograd_conf_t &conf,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr);

// JIT transform kernels for Winograd convolution
struct jit_wino_input_transform_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_wino_input_transform_t)

    struct call_params_t {
        const float *src_batch;
        float *V;
        dim_t ic_spatial_stride;
        dim_t nb_oh;
        dim_t nb_ow;
    };

    dim_t ic_, ih_, iw_, pad_t_, pad_l_;
    dim_t input_ld_row_, V_elem_stride_;

    jit_wino_input_transform_t(const rvv_winograd_conf_t &conf);

    void generate() override;
};

struct jit_wino_output_transform_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_wino_output_transform_t)

    struct call_params_t {
        const float *M;
        const float *bias;
        float *dst_batch;
        dim_t nb_oh;
        dim_t nb_ow;
    };

    dim_t oc_, oh_, ow_, N_;
    dim_t M_elem_stride_, oc_spatial_stride_;
    bool with_bias_;

    jit_wino_output_transform_t(const rvv_winograd_conf_t &conf);

    void generate() override;
};

struct rvv_wino_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                "wino:rvv", rvv_wino_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            VDISPATCH_CONV(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);

            // Check data types: f32 only
            VDISPATCH_CONV(with_bias()
                            ? expect_data_types(f32, f32, f32, f32, f32)
                            : expect_data_types(
                                      f32, f32, data_type::undef, f32, f32),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_CONV(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_CONV(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            // Set default formats if format_kind == any
            // IMPORTANT: Must do this BEFORE creating memory_desc_wrapper!
            if (src_md_.format_kind == format_kind::any) set_default_formats();

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper weights_d(&weights_md_);
            const memory_desc_wrapper dst_d(&dst_md_);

            // Check kernel size: 3x3 only
            VDISPATCH_CONV(weights_d.dims()[2] == 3 && weights_d.dims()[3] == 3,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "only 3x3 kernel is supported for winograd");

            // Check stride: must be 1x1
            VDISPATCH_CONV(desc()->strides[0] == 1 && desc()->strides[1] == 1,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "only stride 1x1 is supported for winograd");

            // Check padding: <= 1
            VDISPATCH_CONV(desc()->padding[0][0] <= 1
                            && desc()->padding[0][1] <= 1
                            && desc()->padding[1][0] <= 1
                            && desc()->padding[1][1] <= 1,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "padding must be <= 1 for winograd");

            // Check dilation: no dilation
            VDISPATCH_CONV(desc()->dilates[0] == 0 && desc()->dilates[1] == 0,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "dilation is not supported for winograd");

            // Check spatial dims: <= 112
            VDISPATCH_CONV(src_d.dims()[2] <= 112 && src_d.dims()[3] <= 112,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "input spatial dimensions must be <= 112 for winograd");

            // Check channels: IC >= 96 && OC >= 96
            VDISPATCH_CONV(src_d.dims()[1] >= 96 && dst_d.dims()[1] >= 96,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "ic and oc must be >= 96 for winograd");

            // Winograd dispatch is restricted to single-core execution.
            // On multi-core, brgemm's spatial parallelism is superior.
            VDISPATCH_CONV(dnnl_get_max_threads() <= 1,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "winograd only beneficial for single-thread execution");

            // Minimum output spatial dimensions for Winograd benefit
            VDISPATCH_CONV(dst_d.dims()[2] >= 7 && dst_d.dims()[3] >= 7,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "output spatial dimensions too small for winograd");
            auto scratchpad = scratchpad_registry().registrar();
            CHECK(rvv_winograd_init_conf(conf_, scratchpad, *desc(), src_md_,
                    weights_md_, dst_md_, bias_md_, attr_));

            // Create brgemm kernel for Winograd GEMM.
            // col-major: C[OC×tiles] = A[OC×IC] × B[IC×tiles]
            {
                brgemm_desc_t brg_desc;
                CHECK(brgemm_desc_init(&brg_desc, v, brgemm_strd,
                        data_type::f32, data_type::f32, brgemm_col_major, 1.0f,
                        0.0f, conf_.wspec.weight_ld_row,
                        conf_.wspec.input_ld_row, conf_.wspec.N, conf_.wspec.N,
                        conf_.wspec.M, conf_.wspec.K));
                brgemm_kernel_t *kernel = nullptr;
                CHECK(brgemm_kernel_create(&kernel, brg_desc));
                brg_kernel_.reset(kernel);
            }

            if (desc()->alg_kind == alg_kind::convolution_auto) {
                set_default_alg_kind(alg_kind::convolution_winograd);
            }

            return status::success;
        }

        rvv_winograd_conf_t conf_ = {};
        std::shared_ptr<brgemm_kernel_t> brg_kernel_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            const int n = ndims();
            const bool g = with_groups();
            const auto dat_tag = utils::pick(n - 3, nwc, nchw, ncdhw);
            const auto wei_tag = utils::pick(2 * n - 6 + (g ? 1 : 0), oiw, goiw,
                    oihw, goihw, oidhw, goidhw);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool post_ops_ok() const {
            // Winograd doesn't support post-ops currently
            return attr()->post_ops_.len() == 0;
        }
    };

    rvv_wino_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    using data_t = typename prec_traits_t<data_type::f32>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_wino_input_transform_t> input_xform_;
    std::unique_ptr<jit_wino_output_transform_t> output_xform_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
