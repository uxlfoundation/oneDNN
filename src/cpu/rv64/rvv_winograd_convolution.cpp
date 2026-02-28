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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
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

// B matrix (transpose of B^T)
constexpr float B[4][4] = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, -1.0f, -1.0f},
        {-1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}};

// Output transform matrix A^T (2x4)
constexpr float AT[2][4]
        = {{1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, -1.0f, 1.0f}};

// A matrix (transpose of A^T, which is 4x2)
constexpr float A[4][2]
        = {{1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, -1.0f}, {0.0f, 1.0f}};

// Filter transform matrix G (4x3)
constexpr float G[4][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};

// Maximum supported vector length in floats (for VLEN=1024: 1024/32=32)
constexpr dim_t MAX_VL_FLOATS = 32;

// Pre-compute filter transform: 3x3 -> 4x4
// Transform = G * filter * G^T
// Output layout: [oc][ic][16]
__attribute__((unused)) static void compute_filter_transform_3x3_to_4x4(
        const float *filter, float *transformed, int oc, int ic) {
    for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
        for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
            const float *f = &filter[(oc_idx * ic + ic_idx) * 9];
            float *out = &transformed[(oc_idx * ic + ic_idx) * 16];

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

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += temp[i][k] * G[j][k];
                    }
                    out[i * 4 + j] = sum;
                }
            }
        }
    }
}

// Pre-compute filter transform with GEMM-layout: 3x3 -> 4x4
// Transform = G * filter * G^T
// Output layout: [16][ic_rounded][oc_rounded] for efficient aligned memory access
// Pads with zeros for non-multiple-of-4 dimensions
void compute_filter_transform_3x3_to_4x4_gemm_layout(const float *filter,
        float *transformed, int oc, int ic, int ic_rounded, int oc_rounded) {

    for (size_t i = 0; i < static_cast<size_t>(16 * ic_rounded * oc_rounded);
            i++) {
        transformed[i] = 0.0f;
    }

    float *temp_std = new float[oc * ic * 16];

    // Step 1: Compute Winograd transform into standard [oc][ic][16] layout
    for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
        for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
            const float *f = &filter[(oc_idx * ic + ic_idx) * 9];
            float *out = &temp_std[(oc_idx * ic + ic_idx) * 16];

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

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += temp[i][k] * G[j][k];
                    }
                    out[i * 4 + j] = sum;
                }
            }
        }
    }

    // B[oc_idx * ldb + ic_idx] where ldb = number of columns = ic_rounded
    for (int elem = 0; elem < 16; elem++) {
        for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
            for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
                // Store as [elem][oc][ic] row-major: B[oc_idx][ic_idx]
                transformed[elem * oc_rounded * ic_rounded + oc_idx * ic_rounded
                        + ic_idx]
                        = temp_std[oc_idx * ic * 16 + ic_idx * 16 + elem];
            }
        }
    }

    delete[] temp_std;
}

// RVV-optimized Winograd input transform: d = B^T * input_tile * B (4x4 -> 4x4)
inline void winograd_input_transform_4x4(
        const float *input_tile, float *output_tile) {
    const dim_t vl_max = static_cast<dim_t>(__riscv_vsetvlmax_e32m1());
    float temp[16];

    // Step 1: input_tile * B
    for (int i = 0; i < 4; i++) {
        vfloat32m1_t v_in_row
                = __riscv_vle32_v_f32m1(input_tile + i * 4, vl_max);
        for (int j = 0; j < 4; j++) {
            float B_col[4] = {B[0][j], B[1][j], B[2][j], B[3][j]};
            vfloat32m1_t v_b_col = __riscv_vle32_v_f32m1(B_col, vl_max);
            vfloat32m1_t v_prod
                    = __riscv_vfmul_vv_f32m1(v_in_row, v_b_col, vl_max);
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);
            vfloat32m1_t v_sum
                    = __riscv_vfredusum_vs_f32m1_f32m1(v_prod, v_zero, vl_max);
            temp[i * 4 + j] = __riscv_vfmv_f_s_f32m1_f32(v_sum);
        }
    }

    // Step 2: B^T * temp
    for (int i = 0; i < 4; i++) {
        float BT_row[4] = {BT[i][0], BT[i][1], BT[i][2], BT[i][3]};
        vfloat32m1_t v_bt_row = __riscv_vle32_v_f32m1(BT_row, vl_max);
        for (int j = 0; j < 4; j++) {
            float temp_col[4] = {temp[0 * 4 + j], temp[1 * 4 + j],
                    temp[2 * 4 + j], temp[3 * 4 + j]};
            vfloat32m1_t v_temp_col = __riscv_vle32_v_f32m1(temp_col, vl_max);
            vfloat32m1_t v_prod
                    = __riscv_vfmul_vv_f32m1(v_bt_row, v_temp_col, vl_max);
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);
            vfloat32m1_t v_sum
                    = __riscv_vfredusum_vs_f32m1_f32m1(v_prod, v_zero, vl_max);
            output_tile[i * 4 + j] = __riscv_vfmv_f_s_f32m1_f32(v_sum);
        }
    }
}

// RVV-optimized Winograd output transform: m = A^T * winograd_domain * A (4x4 -> 2x2)
inline void winograd_output_transform_2x2(
        const float *winograd_domain, float *output_tile) {
    const dim_t vl_max = static_cast<dim_t>(__riscv_vsetvlmax_e32m1());
    float temp[8];

    // Step 1: winograd_domain * A
    for (int i = 0; i < 4; i++) {
        vfloat32m1_t v_wino_row
                = __riscv_vle32_v_f32m1(winograd_domain + i * 4, vl_max);
        for (int j = 0; j < 2; j++) {
            float A_col[4] = {A[0][j], A[1][j], A[2][j], A[3][j]};
            vfloat32m1_t v_a_col = __riscv_vle32_v_f32m1(A_col, vl_max);
            vfloat32m1_t v_prod
                    = __riscv_vfmul_vv_f32m1(v_wino_row, v_a_col, vl_max);
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);
            vfloat32m1_t v_sum
                    = __riscv_vfredusum_vs_f32m1_f32m1(v_prod, v_zero, vl_max);
            temp[i * 2 + j] = __riscv_vfmv_f_s_f32m1_f32(v_sum);
        }
    }

    // Step 2: A^T * temp
    for (int i = 0; i < 2; i++) {
        float AT_row[4] = {AT[i][0], AT[i][1], AT[i][2], AT[i][3]};
        vfloat32m1_t v_at_row = __riscv_vle32_v_f32m1(AT_row, vl_max);
        for (int j = 0; j < 2; j++) {
            float temp_col[4] = {temp[0 * 2 + j], temp[1 * 2 + j],
                    temp[2 * 2 + j], temp[3 * 2 + j]};
            vfloat32m1_t v_temp_col = __riscv_vle32_v_f32m1(temp_col, vl_max);
            vfloat32m1_t v_prod
                    = __riscv_vfmul_vv_f32m1(v_at_row, v_temp_col, vl_max);
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);
            vfloat32m1_t v_sum
                    = __riscv_vfredusum_vs_f32m1_f32m1(v_prod, v_zero, vl_max);
            output_tile[i * 2 + j] = __riscv_vfmv_f_s_f32m1_f32(v_sum);
        }
    }
}

} // namespace

// Helper: round up to multiple of n (for alignment)
static inline dim_t round_up(dim_t x, dim_t n) {
    return ((x + n - 1) / n) * n;
}

status_t rvv_winograd_init_conf(rvv_winograd_conf_t &conf,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr, int max_threads) {
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

    conf.nthr = max_threads;

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
    conf.wspec.weight_oc_rounded = round_up(conf.wspec.N, CACHE_LINE_FLOATS);
    conf.wspec.weight_ic_rounded = round_up(conf.wspec.K, CACHE_LINE_FLOATS);

    // weight_ld_row is the leading dimension (stride) for B[N][K] row-major
    // B[oc_idx][ic_idx] = B[oc_idx * ldb + ic_idx], so ldb = number of columns = ic_rounded
    conf.wspec.weight_ld_row = conf.wspec.weight_ic_rounded;
    conf.wspec.weight_ld_matrix
            = conf.wspec.weight_oc_rounded * conf.wspec.weight_ic_rounded;

    // Input matrix: A[K][M] column-major where K=IC, M=tiles
    // A[k][m] stored as A[m * lda + k] where lda = round_up(K, CACHE_LINE_FLOATS)
    conf.wspec.input_ld_row
            = round_up(conf.wspec.K, CACHE_LINE_FLOATS); // lda = K rounded
    conf.wspec.input_ld_batch
            = conf.wspec.input_ld_row * conf.wspec.M; // M columns per batch
    conf.wspec.input_ld_matrix
            = conf.wspec.input_ld_batch * conf.wspec.n_batches;

    // Output matrix: C^T[M][N] row-major where M=tiles, N=OC
    // GEMM computes C^T where C^T[m][n] = C[n][m]
    // C^T[m][n] stored as C^T[m * N + n]
    conf.wspec.output_ld_row = conf.wspec.N; // N columns per row of C^T
    conf.wspec.output_ld_batch
            = conf.wspec.output_ld_row * conf.wspec.M; // M rows per batch
    conf.wspec.output_ld_matrix
            = conf.wspec.output_ld_batch * conf.wspec.n_batches;

    // Matrix sizes in floats
    conf.wspec.weight_matrix_size
            = conf.wspec.n_gemms * conf.wspec.weight_ld_matrix;
    conf.wspec.input_matrix_size
            = conf.wspec.n_gemms * conf.wspec.input_ld_matrix;
    conf.wspec.output_matrix_size
            = conf.wspec.n_gemms * conf.wspec.output_ld_matrix;

    // GEMM-layout scratchpad allocation
    using namespace memory_tracking::names;

    scratchpad.book<float>(key_wino_U, conf.wspec.weight_matrix_size);
    scratchpad.book<float>(key_wino_V, conf.wspec.input_matrix_size);
    scratchpad.book<float>(key_wino_M, conf.wspec.output_matrix_size);

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

    using namespace memory_tracking::names;
    float *transformed_weights = scratchpad.template get<float>(key_wino_U);
    float *transformed_input = scratchpad.template get<float>(key_wino_V);
    float *winograd_output = scratchpad.template get<float>(key_wino_M);

    // Phase 1: Weight transform
    compute_filter_transform_3x3_to_4x4_gemm_layout(weights,
            transformed_weights, conf.wspec.N, conf.wspec.K,
            conf.wspec.weight_ic_rounded, conf.wspec.weight_oc_rounded);

    // Phase 2: Input transform
    CHECK(execute_input_transform(src, transformed_input));

    // Phase 3: Batched GEMM (16 independent matrix multiplications)
    CHECK(execute_gemm_batched(
            transformed_input, transformed_weights, winograd_output));

    // Phase 4: Output transform
    CHECK(execute_output_transform(winograd_output, bias, dst));

    return status::success;
}

status_t rvv_wino_convolution_fwd_t::execute_input_transform(
        const data_t *src, float *transformed_input) const {

    const auto &conf = pd()->conf_;
    const dim_t nb_oh = (conf.oh + 1) / 2;
    const dim_t nb_ow = (conf.ow + 1) / 2;
    const dim_t total_tiles = nb_oh * nb_ow;

    // Parallelize over tiles
    parallel(conf.nthr, [&](const int ithr, const int nthr) {
        dim_t start = 0, end = 0;
        balance211(conf.mb * total_tiles, nthr, ithr, start, end);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const dim_t mb_idx = iwork / total_tiles;
            const dim_t tile_idx = iwork % total_tiles;
            const dim_t oh_tile = tile_idx / nb_ow;
            const dim_t ow_tile = tile_idx % nb_ow;
            const dim_t oh_start = oh_tile * 2;
            const dim_t ow_start = ow_tile * 2;

            const float *src_base = src + mb_idx * conf.ic * conf.ih * conf.iw;

            // Transform each input channel's 4x4 tile
            // Output layout: transformed_input[elem][batch][k][m]
            for (dim_t ic_idx = 0; ic_idx < conf.ic; ic_idx++) {
                float input_tile[16];
                for (dim_t i = 0; i < 4; i++) {
                    dim_t ih_idx = oh_start + i - conf.pad_t;
                    for (dim_t j = 0; j < 4; j++) {
                        dim_t iw_idx = ow_start + j - conf.pad_l;
                        if (ih_idx >= 0 && ih_idx < conf.ih && iw_idx >= 0
                                && iw_idx < conf.iw) {
                            size_t offset = ic_idx * conf.ih * conf.iw
                                    + ih_idx * conf.iw + iw_idx;
                            input_tile[i * 4 + j] = src_base[offset];
                        } else {
                            input_tile[i * 4 + j] = 0.0f;
                        }
                    }
                }

                // Winograd transform this 4x4 tile
                float transformed[16];
                winograd_input_transform_4x4(input_tile, transformed);

                // Store in GEMM layout: transformed_input[elem][batch][m][k]
                // where m=tile_idx, k=ic_idx
                const dim_t vl_max
                        = static_cast<dim_t>(__riscv_vsetvlmax_e32m1());
                for (int elem = 0; elem < 16; elem += vl_max) {
                    const dim_t vl
                            = (elem + vl_max <= 16) ? vl_max : (16 - elem);

                    float *dst_base = transformed_input
                            + elem * conf.wspec.input_ld_matrix
                            + mb_idx * conf.wspec.input_ld_batch
                            + tile_idx * conf.wspec.input_ld_row + ic_idx;

                    for (dim_t i = 0; i < vl; i++) {
                        dst_base[i * conf.wspec.input_ld_matrix]
                                = transformed[elem + i];
                    }
                }
            }
        }
    });

    return status::success;
}

status_t rvv_wino_convolution_fwd_t::execute_gemm_batched(
        const float *transformed_input, const float *transformed_weights,
        float *winograd_output) const {

    const auto &conf = pd()->conf_;

    // Execute 16 GEMMs in parallel (each Winograd element is independent)
    const int n_gemms_threads
            = static_cast<int>(std::min(static_cast<dim_t>(16), conf.nthr));

    parallel(n_gemms_threads, [&](const int ithr, const int nthr) {
        for (int elem = ithr; elem < 16; elem += nthr) {
            for (dim_t batch = 0; batch < conf.mb; batch++) {
                const float *A = transformed_input
                        + elem * conf.wspec.input_ld_matrix
                        + batch * conf.wspec.input_ld_batch;
                const dim_t lda = conf.wspec.input_ld_row;

                const float *B = transformed_weights
                        + elem * conf.wspec.weight_ld_matrix;
                const dim_t ldb = conf.wspec.weight_ld_row;

                float *C = winograd_output + elem * conf.wspec.output_ld_matrix
                        + batch * conf.wspec.output_ld_batch;

                // RVV GEMM: C^T[M×N] where C^T[m][n] = sum_k A[m][k] * B[n][k]
                const dim_t M = conf.wspec.M;
                const dim_t N = conf.wspec.N;
                const dim_t K = conf.wspec.K;
                const dim_t vl_max
                        = static_cast<dim_t>(__riscv_vsetvlmax_e32m1());

                // Initialize C to zero
                for (dim_t m = 0; m < M; m++) {
                    float *C_row = C + m * N;
                    for (dim_t n = 0; n < N; n++) {
                        C_row[n] = 0.0f;
                    }
                }

                // RVV vectorized GEMM: for each m, compute all n in parallel
                for (dim_t m = 0; m < M; m++) {
                    const float *A_row = A + m * lda;
                    float *C_row = C + m * N;

                    dim_t n_vec_base = 0;
                    for (; n_vec_base + vl_max <= N; n_vec_base += vl_max) {
                        vfloat32m1_t vacc
                                = __riscv_vfmv_v_f_f32m1(0.0f, vl_max);

                        for (dim_t k = 0; k < K; k++) {
                            float a_scalar = A_row[k];
                            float B_local[MAX_VL_FLOATS];
                            for (dim_t i = 0; i < vl_max; i++) {
                                B_local[i] = B[(n_vec_base + i) * ldb + k];
                            }
                            vfloat32m1_t vb
                                    = __riscv_vle32_v_f32m1(B_local, vl_max);
                            vacc = __riscv_vfmacc_vf_f32m1(
                                    vacc, a_scalar, vb, vl_max);
                        }

                        __riscv_vse32_v_f32m1(C_row + n_vec_base, vacc, vl_max);
                    }

                    // Handle remaining channels
                    if (n_vec_base < N) {
                        const dim_t n_remain = N - n_vec_base;
                        vfloat32m1_t vacc
                                = __riscv_vfmv_v_f_f32m1(0.0f, n_remain);

                        for (dim_t k = 0; k < K; k++) {
                            float a_scalar = A_row[k];
                            float B_local[16];
                            for (dim_t i = 0; i < n_remain; i++) {
                                B_local[i] = B[(n_vec_base + i) * ldb + k];
                            }
                            vfloat32m1_t vb
                                    = __riscv_vle32_v_f32m1(B_local, n_remain);
                            vacc = __riscv_vfmacc_vf_f32m1(
                                    vacc, a_scalar, vb, n_remain);
                        }

                        __riscv_vse32_v_f32m1(
                                C_row + n_vec_base, vacc, n_remain);
                    }
                }
            }
        }
    });

    return status::success;
}

status_t rvv_wino_convolution_fwd_t::execute_output_transform(
        const float *winograd_output, const data_t *bias, data_t *dst) const {

    const auto &conf = pd()->conf_;
    const dim_t nb_oh = (conf.oh + 1) / 2;
    const dim_t nb_ow = (conf.ow + 1) / 2;
    const dim_t total_tiles = nb_oh * nb_ow;

    // Zero initialize dst
    const dim_t dst_size = conf.mb * conf.oc * conf.oh * conf.ow;
    for (dim_t i = 0; i < dst_size; i++) {
        dst[i] = 0.0f;
    }

    // Parallelize over tiles
    parallel(conf.nthr, [&](const int ithr, const int nthr) {
        dim_t start = 0, end = 0;
        balance211(conf.mb * total_tiles, nthr, ithr, start, end);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const dim_t mb_idx = iwork / total_tiles;
            const dim_t tile_idx = iwork % total_tiles;
            const dim_t oh_tile = tile_idx / nb_ow;
            const dim_t ow_tile = tile_idx % nb_ow;
            const dim_t oh_start = oh_tile * 2;
            const dim_t ow_start = ow_tile * 2;

            // For each output channel, extract 16 Winograd elements and inverse transform
            const dim_t vl_max = static_cast<dim_t>(__riscv_vsetvlmax_e32m1());

            for (dim_t oc_idx = 0; oc_idx < conf.oc; oc_idx++) {
                float winograd_domain_tile[16];

                for (int elem = 0; elem < 16; elem += vl_max) {
                    const dim_t vl
                            = (elem + vl_max <= 16) ? vl_max : (16 - elem);
                    const float *src_base = winograd_output
                            + elem * conf.wspec.output_ld_matrix
                            + mb_idx * conf.wspec.output_ld_batch
                            + tile_idx * conf.wspec.output_ld_row + oc_idx;

                    for (dim_t i = 0; i < vl; i++) {
                        winograd_domain_tile[elem + i]
                                = src_base[i * conf.wspec.output_ld_matrix];
                    }
                }

                float output_tile_2x2[4];
                winograd_output_transform_2x2(
                        winograd_domain_tile, output_tile_2x2);

                data_t *dst_base = dst + mb_idx * conf.oc * conf.oh * conf.ow
                        + oc_idx * conf.oh * conf.ow;
                for (dim_t i = 0; i < 2; i++) {
                    dim_t oh = oh_start + i;
                    if (oh >= conf.oh) continue;
                    dim_t row_start = oh * conf.ow + ow_start;
                    for (dim_t j = 0; j < 2; j++) {
                        dim_t ow = ow_start + j;
                        if (ow >= conf.ow) continue;
                        dst_base[row_start + j] = output_tile_2x2[i * 2 + j];
                    }
                }
            }

            // Add bias if present (once per tile, for all OC)
            if (conf.with_bias && bias) {
                for (dim_t oc_idx = 0; oc_idx < conf.oc; oc_idx++) {
                    float bias_val = bias[oc_idx];
                    data_t *dst_base = dst
                            + mb_idx * conf.oc * conf.oh * conf.ow
                            + oc_idx * conf.oh * conf.ow;

                    vfloat32m1_t v_bias
                            = __riscv_vfmv_v_f_f32m1(bias_val, vl_max);

                    for (dim_t i = 0; i < 2; i++) {
                        dim_t oh = oh_start + i;
                        if (oh >= conf.oh) continue;
                        dim_t row_start = oh * conf.ow + ow_start;

                        if (ow_start + 2 <= conf.ow) {
                            vfloat32m1_t v_data = __riscv_vle32_v_f32m1(
                                    dst_base + row_start, vl_max);
                            v_data = __riscv_vfadd_vv_f32m1(
                                    v_data, v_bias, vl_max);
                            __riscv_vse32_v_f32m1(
                                    dst_base + row_start, v_data, vl_max);
                        } else {
                            // Scalar fallback for partial row
                            for (dim_t j = 0; j < 2; j++) {
                                dim_t ow = ow_start + j;
                                if (ow >= conf.ow) continue;
                                dst_base[row_start + j] += bias_val;
                            }
                        }
                    }
                }
            }
        }
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
