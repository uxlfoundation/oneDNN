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

#include <cstring>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/rvv_winograd_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::memory_tracking::names;

namespace {
// Winograd F(2x2, 3x3) transform matrices
// Input transform matrix B^T (4x4)
constexpr float BT[4][4] = {{1.0f, 0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 1.0f}};

// Output transform matrix A^T (2x4)
constexpr float AT[2][4]
        = {{1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, -1.0f, 1.0f}};

// Filter transform matrix G (4x3)
constexpr float G[4][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};

// Pre-compute filter transform with GEMM-layout: 3x3 -> 4x4
// Transform = G * filter * G^T
// Output layout: [16][ic_rounded][oc_rounded] for brgemm col-major access
// Computes directly without intermediate allocation.
void compute_filter_transform_3x3_to_4x4_gemm_layout(const float *filter,
        float *transformed, int oc, int ic, int ic_rounded, int oc_rounded) {

    std::memset(transformed, 0, 16 * ic_rounded * oc_rounded * sizeof(float));

    for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
        for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
            const float *f = &filter[(oc_idx * ic + ic_idx) * 9];

            // Step 1: temp = G * filter (4x3)
            float temp[4][3];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += G[i][k] * f[k * 3 + j];
                    }
                    temp[i][j] = sum;
                }
            }

            // Step 2: result = temp * G^T, store directly to GEMM layout
            // Layout: [elem][ic_idx * oc_rounded + oc_idx]
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += temp[i][k] * G[j][k];
                    }
                    int elem = i * 4 + j;
                    transformed[elem * oc_rounded * ic_rounded
                            + ic_idx * oc_rounded + oc_idx]
                            = sum;
                }
            }
        }
    }
}

} // namespace

status_t rvv_winograd_init_conf(rvv_winograd_conf_t &conf,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const dim_t mb = src_d.dims()[0];
    conf.mb = mb;

    conf.ih = src_d.dims()[2];
    conf.iw = src_d.dims()[3];
    conf.oh = dst_d.dims()[2];
    conf.ow = dst_d.dims()[3];

    conf.ic = src_d.dims()[1];
    conf.oc = dst_d.dims()[1];

    conf.kh = weights_d.dims()[2];
    conf.kw = weights_d.dims()[3];

    conf.stride_h = cd.strides[0];
    conf.stride_w = cd.strides[1];

    conf.pad_t = cd.padding[0][0];
    conf.pad_l = cd.padding[0][1];
    conf.pad_b = cd.padding[1][0];
    conf.pad_r = cd.padding[1][1];

    conf.with_bias = cd.bias_desc.data_type != data_type::undef;

    // Compute Winograd domain specification for GEMM-based execution
    constexpr dim_t CACHE_LINE_SIZE = platform::get_cache_line_size();
    constexpr dim_t CACHE_LINE_FLOATS = CACHE_LINE_SIZE / sizeof(float); // 16

    // Matrix dimensions: C[M][N] = A[M][K] * B[K][N]
    conf.wspec.M = ((conf.oh + 1) / 2) * ((conf.ow + 1) / 2); // Total 2x2 tiles
    conf.wspec.K = conf.ic; // Input channels
    conf.wspec.N = conf.oc; // Output channels
    conf.wspec.n_gemms = 16; // Winograd F(2×2, 3×3)
    conf.wspec.n_batches = conf.mb;

    // 64-byte aligned leading dimensions (for efficient vectorization)
    // Weight matrix: B[N][K] where N=OC, K=IC (row-major for GEMM)
    // For weight transform, we need separate rounded dimensions
    conf.wspec.weight_oc_rounded = rnd_up(conf.wspec.N, CACHE_LINE_FLOATS);
    conf.wspec.weight_ic_rounded = rnd_up(conf.wspec.K, CACHE_LINE_FLOATS);

    // weight_ld_row is the leading dimension for column-major OC x IC matrix
    // A[oc_idx + ic_idx * lda], so lda = oc_rounded
    conf.wspec.weight_ld_row = conf.wspec.weight_oc_rounded;
    conf.wspec.weight_ld_matrix
            = conf.wspec.weight_oc_rounded * conf.wspec.weight_ic_rounded;

    // Input matrix: A[K][M] column-major where K=IC, M=tiles
    // Input buffer per thread: [16][tile_chunk × IC_rounded] per element
    conf.wspec.input_ld_row
            = rnd_up(conf.wspec.K, CACHE_LINE_FLOATS); // LDB = IC_rounded
    conf.wspec.input_ld_batch
            = conf.wspec.input_ld_row * conf.wspec.M; // per-elem stride

    // Output buffer per thread: [16][tiles x OC] per element
    conf.wspec.output_ld_row = conf.wspec.N; // LDC = OC
    conf.wspec.output_ld_batch
            = conf.wspec.output_ld_row * conf.wspec.M; // per-elem stride

    // Buffer sizes in floats
    conf.wspec.weight_matrix_size
            = conf.wspec.n_gemms * conf.wspec.weight_ld_matrix;
    conf.wspec.V_buffer_size = conf.wspec.n_gemms * conf.wspec.input_ld_batch;
    conf.wspec.M_buffer_size = conf.wspec.n_gemms * conf.wspec.output_ld_batch;

    // Scratchpad: V and M buffers for single-thread execution
    // Weight buffer is in persistent resource_t (not scratchpad)
    using namespace memory_tracking::names;

    scratchpad.book<float>(key_wino_V, conf.wspec.V_buffer_size);
    scratchpad.book<float>(key_wino_M, conf.wspec.M_buffer_size);

    return status::success;
}

status_t rvv_wino_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const auto &conf = pd()->conf_;
    const auto scratchpad = ctx.get_scratchpad_grantor();
    const auto *brg_kernel = pd()->brg_kernel_.get();

    // Get persistent weight buffer from resource (cached across execute calls)
    auto *wino_resource
            = ctx.get_resource_mapper()->get<rvv_wino_resource_t>(this);
    float *transformed_weights = wino_resource->get_weight_buffer();

    // Transform weights on first execute, cache for subsequent calls
    if (!wino_resource->weights_valid()) {
        compute_filter_transform_3x3_to_4x4_gemm_layout(weights,
                transformed_weights, conf.wspec.N, conf.wspec.K,
                conf.wspec.weight_ic_rounded, conf.wspec.weight_oc_rounded);
        wino_resource->set_weights_valid();
    }

    using namespace memory_tracking::names;
    float *V = scratchpad.template get<float>(key_wino_V);
    float *M = scratchpad.template get<float>(key_wino_M);

    // Sequential batch processing: input transform -> GEMM -> output transform
    // per batch, keeping working set in L2 cache.
    const dim_t nb_oh = (conf.oh + 1) / 2;
    const dim_t nb_ow = (conf.ow + 1) / 2;
    const dim_t total_tiles = nb_oh * nb_ow;
    const dim_t input_ld_row = conf.wspec.input_ld_row;
    const dim_t V_elem_stride = conf.wspec.input_ld_batch;
    const dim_t M_elem_stride = conf.wspec.output_ld_batch;
    const dim_t ic_spatial_stride = conf.ih * conf.iw;
    const dim_t oc_spatial_stride = conf.oh * conf.ow;

    for (dim_t mb_idx = 0; mb_idx < conf.mb; mb_idx++) {
        const float *src_batch = src + mb_idx * conf.ic * ic_spatial_stride;

        // Step 1: Input transform - vectorized across IC channels
        // Row-by-row approach: for each output row out_i of B^T * tile * B,
        // compute fused result using at most 9 vector registers.
        // result[out_i][j] = Σ_k BT[out_i][k] * Σ_m tile[k][m]*B[m][j]
        for (dim_t tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
            const dim_t oh_tile = tile_idx / nb_ow;
            const dim_t ow_tile = tile_idx % nb_ow;
            const dim_t oh_s = oh_tile * 2;
            const dim_t ow_s = ow_tile * 2;

            dim_t vl = __riscv_vsetvl_e32m1(conf.ic);
            for (dim_t ic_base = 0; ic_base < conf.ic; ic_base += vl) {
                vl = __riscv_vsetvl_e32m1(conf.ic - ic_base);

                // Load 4x4 input tile: 4 rows of 4 strided loads
                // tile[i][j] = src at (oh_s+i, ow_s+j) for each IC channel
                // Using 4 variables for the 4 values in current row i.
                // We process one output row at a time: for each out_i,
                // we load row k=0..3 on the fly and accumulate.
                for (int out_i = 0; out_i < 4; out_i++) {
                    // Accumulators for 4 columns of result[out_i][*]
                    vfloat32m1_t res0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t res1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t res2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t res3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                    for (int k = 0; k < 4; k++) {
                        // Exploit BT sparsity: skip when BT[out_i][k] == 0
                        // BT = [1,0,-1,0; 0,1,1,0; 0,-1,1,0; 0,-1,0,1]
                        float bt_val = BT[out_i][k];
                        if (bt_val == 0.0f) continue;

                        dim_t ih_idx = oh_s + k - conf.pad_t;
                        // Load tile[k][0..3] for this IC chunk
                        vfloat32m1_t t0, t1, t2, t3;
                        if (ih_idx >= 0 && ih_idx < conf.ih) {
                            dim_t iw0 = ow_s + 0 - conf.pad_l;
                            dim_t iw1 = ow_s + 1 - conf.pad_l;
                            dim_t iw2 = ow_s + 2 - conf.pad_l;
                            dim_t iw3 = ow_s + 3 - conf.pad_l;
                            t0 = (iw0 >= 0 && iw0 < conf.iw)
                                    ? __riscv_vlse32_v_f32m1(src_batch
                                                      + ic_base
                                                              * ic_spatial_stride
                                                      + ih_idx * conf.iw + iw0,
                                              ic_spatial_stride * sizeof(float),
                                              vl)
                                    : __riscv_vfmv_v_f_f32m1(0.0f, vl);
                            t1 = (iw1 >= 0 && iw1 < conf.iw)
                                    ? __riscv_vlse32_v_f32m1(src_batch
                                                      + ic_base
                                                              * ic_spatial_stride
                                                      + ih_idx * conf.iw + iw1,
                                              ic_spatial_stride * sizeof(float),
                                              vl)
                                    : __riscv_vfmv_v_f_f32m1(0.0f, vl);
                            t2 = (iw2 >= 0 && iw2 < conf.iw)
                                    ? __riscv_vlse32_v_f32m1(src_batch
                                                      + ic_base
                                                              * ic_spatial_stride
                                                      + ih_idx * conf.iw + iw2,
                                              ic_spatial_stride * sizeof(float),
                                              vl)
                                    : __riscv_vfmv_v_f_f32m1(0.0f, vl);
                            t3 = (iw3 >= 0 && iw3 < conf.iw)
                                    ? __riscv_vlse32_v_f32m1(src_batch
                                                      + ic_base
                                                              * ic_spatial_stride
                                                      + ih_idx * conf.iw + iw3,
                                              ic_spatial_stride * sizeof(float),
                                              vl)
                                    : __riscv_vfmv_v_f_f32m1(0.0f, vl);
                        } else {
                            t0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                            t1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                            t2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                            t3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                        }

                        // Exploit B sparsity: B*[t0,t1,t2,t3]^T =
                        // [t0-t2, t1+t2, t2-t1, t3-t1]
                        // (B = [1,0,0,0; 0,1,-1,-1; -1,1,1,0; 0,0,0,1])
                        vfloat32m1_t tmp0 = __riscv_vfsub_vv_f32m1(t0, t2, vl);
                        vfloat32m1_t tmp1 = __riscv_vfadd_vv_f32m1(t1, t2, vl);
                        vfloat32m1_t tmp2 = __riscv_vfsub_vv_f32m1(t2, t1, vl);
                        vfloat32m1_t tmp3 = __riscv_vfsub_vv_f32m1(t3, t1, vl);

                        // Accumulate: res += bt_val * tmp
                        // bt_val is always 1.0f or -1.0f here
                        if (bt_val > 0.0f) {
                            res0 = __riscv_vfadd_vv_f32m1(res0, tmp0, vl);
                            res1 = __riscv_vfadd_vv_f32m1(res1, tmp1, vl);
                            res2 = __riscv_vfadd_vv_f32m1(res2, tmp2, vl);
                            res3 = __riscv_vfadd_vv_f32m1(res3, tmp3, vl);
                        } else {
                            res0 = __riscv_vfsub_vv_f32m1(res0, tmp0, vl);
                            res1 = __riscv_vfsub_vv_f32m1(res1, tmp1, vl);
                            res2 = __riscv_vfsub_vv_f32m1(res2, tmp2, vl);
                            res3 = __riscv_vfsub_vv_f32m1(res3, tmp3, vl);
                        }
                    }

                    // Store result[out_i][0..3] to V buffer
                    int base_elem = out_i * 4;
                    __riscv_vse32_v_f32m1(V + (base_elem + 0) * V_elem_stride
                                    + tile_idx * input_ld_row + ic_base,
                            res0, vl);
                    __riscv_vse32_v_f32m1(V + (base_elem + 1) * V_elem_stride
                                    + tile_idx * input_ld_row + ic_base,
                            res1, vl);
                    __riscv_vse32_v_f32m1(V + (base_elem + 2) * V_elem_stride
                                    + tile_idx * input_ld_row + ic_base,
                            res2, vl);
                    __riscv_vse32_v_f32m1(V + (base_elem + 3) * V_elem_stride
                                    + tile_idx * input_ld_row + ic_base,
                            res3, vl);
                }

                vl = __riscv_vsetvl_e32m1(conf.ic);
            }
        }

        // Step 2: GEMM using brgemm kernel (16 elements)
        for (int elem = 0; elem < 16; elem++) {
            const float *A_weights
                    = transformed_weights + elem * conf.wspec.weight_ld_matrix;
            const float *B_input = V + elem * V_elem_stride;
            float *C_output = M + elem * M_elem_stride;

            brgemm_kernel_execute(brg_kernel, A_weights, B_input, C_output,
                    total_tiles, 0.0f);
        }

        // Step 3: Output transform - vectorized across OC channels
        // Row-by-row approach: for each output row out_i of A^T * wino * A,
        // compute fused result using at most 7 vector registers.
        // output[out_i][j] = Σ_k AT[out_i][k] * Σ_m wino[k][m]*A[m][j]
        data_t *dst_batch = dst + mb_idx * conf.oc * oc_spatial_stride;
        for (dim_t tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
            const dim_t oh_tile = tile_idx / nb_ow;
            const dim_t ow_tile = tile_idx % nb_ow;
            const dim_t oh_s = oh_tile * 2;
            const dim_t ow_s = ow_tile * 2;

            dim_t vl = __riscv_vsetvl_e32m1(conf.oc);
            for (dim_t oc_base = 0; oc_base < conf.oc; oc_base += vl) {
                vl = __riscv_vsetvl_e32m1(conf.oc - oc_base);

                for (int out_i = 0; out_i < 2; out_i++) {
                    // Accumulators for 2 columns of output[out_i][*]
                    vfloat32m1_t res0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t res1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                    for (int k = 0; k < 4; k++) {
                        // Exploit AT sparsity: skip when AT[out_i][k] == 0
                        // AT = [1,1,1,0; 0,1,-1,1]
                        float at_val = AT[out_i][k];
                        if (at_val == 0.0f) continue;

                        // Load wino[k][0..3]
                        vfloat32m1_t w0 = __riscv_vle32_v_f32m1(M
                                        + (k * 4 + 0) * M_elem_stride
                                        + tile_idx * conf.wspec.N + oc_base,
                                vl);
                        vfloat32m1_t w1 = __riscv_vle32_v_f32m1(M
                                        + (k * 4 + 1) * M_elem_stride
                                        + tile_idx * conf.wspec.N + oc_base,
                                vl);
                        vfloat32m1_t w2 = __riscv_vle32_v_f32m1(M
                                        + (k * 4 + 2) * M_elem_stride
                                        + tile_idx * conf.wspec.N + oc_base,
                                vl);
                        vfloat32m1_t w3 = __riscv_vle32_v_f32m1(M
                                        + (k * 4 + 3) * M_elem_stride
                                        + tile_idx * conf.wspec.N + oc_base,
                                vl);

                        // Exploit A sparsity: A*[w0,w1,w2,w3]^T =
                        // [w0+w1+w2, w1-w2+w3]
                        // (A = [1,0; 1,1; 1,-1; 0,1])
                        vfloat32m1_t tmp0 = __riscv_vfadd_vv_f32m1(
                                __riscv_vfadd_vv_f32m1(w0, w1, vl), w2, vl);
                        vfloat32m1_t tmp1 = __riscv_vfadd_vv_f32m1(
                                __riscv_vfsub_vv_f32m1(w1, w2, vl), w3, vl);

                        // Accumulate: res += at_val * tmp
                        // at_val is 1.0f or -1.0f here
                        if (at_val > 0.0f) {
                            res0 = __riscv_vfadd_vv_f32m1(res0, tmp0, vl);
                            res1 = __riscv_vfadd_vv_f32m1(res1, tmp1, vl);
                        } else {
                            res0 = __riscv_vfsub_vv_f32m1(res0, tmp0, vl);
                            res1 = __riscv_vfsub_vv_f32m1(res1, tmp1, vl);
                        }
                    }

                    // Add bias
                    vfloat32m1_t v_bias = (conf.with_bias && bias)
                            ? __riscv_vle32_v_f32m1(bias + oc_base, vl)
                            : __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    res0 = __riscv_vfadd_vv_f32m1(res0, v_bias, vl);
                    res1 = __riscv_vfadd_vv_f32m1(res1, v_bias, vl);

                    // Store to dst using strided stores
                    // dst layout: [OC][OH][OW], stride between OC = OH*OW
                    {
                        dim_t oh = oh_s + out_i;
                        if (oh < conf.oh) {
                            dim_t ow = ow_s + 0;
                            if (ow < conf.ow) {
                                __riscv_vsse32_v_f32m1(dst_batch
                                                + oc_base * oc_spatial_stride
                                                + oh * conf.ow + ow,
                                        oc_spatial_stride * sizeof(float), res0,
                                        vl);
                            }
                            ow = ow_s + 1;
                            if (ow < conf.ow) {
                                __riscv_vsse32_v_f32m1(dst_batch
                                                + oc_base * oc_spatial_stride
                                                + oh * conf.ow + ow,
                                        oc_spatial_stride * sizeof(float), res1,
                                        vl);
                            }
                        }
                    }
                }

                vl = __riscv_vsetvl_e32m1(conf.oc);
            }
        }
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
