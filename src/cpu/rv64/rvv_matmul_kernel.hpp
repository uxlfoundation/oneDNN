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
#ifndef CPU_RV64_RVV_MATMUL_KERNEL_HPP
#define CPU_RV64_RVV_MATMUL_KERNEL_HPP

#include <cstdint>
#include <vector>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

static inline void rvv_matmul_colmajor_kernel_f32_f32_f32(
        const void *src_base_void_ptr, const void *weights_base_void_ptr,
        void *dst_base_void_ptr, dim_t K, dim_t N, const void *bias_void_ptr,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    const float *src_base_ptr = static_cast<const float *>(src_base_void_ptr);
    const float *weights_base_ptr
            = static_cast<const float *>(weights_base_void_ptr);
    float *dst_base_ptr = static_cast<float *>(dst_base_void_ptr);
    const float *bias = static_cast<const float *>(bias_void_ptr);

    for (dim_t n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m1(N - n0);
        std::vector<float> out_vals(vl, 0.0f);

        for (dim_t k0 = 0; k0 < K;) {
            size_t k_vl = __riscv_vsetvl_e32m1(K - k0);

            vfloat32m1_t src_vec
                    = __riscv_vle32_v_f32m1(src_base_ptr + k0, k_vl);

            for (size_t ni = 0; ni < vl; ++ni) {
                const float *weight_col_ptr
                        = weights_base_ptr + (size_t)(n0 + ni) * (size_t)K;
                vfloat32m1_t wei_vec
                        = __riscv_vle32_v_f32m1(weight_col_ptr + k0, k_vl);

                vfloat32m1_t prod
                        = __riscv_vfmul_vv_f32m1(src_vec, wei_vec, k_vl);
                vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(
                        prod, __riscv_vfmv_v_f_f32m1(0.0f, k_vl), k_vl);
                float partial = __riscv_vfmv_f_s_f32m1_f32(reduced);

                out_vals[ni] += partial;
            }

            k0 += k_vl;
        }

        vfloat32m1_t acc = __riscv_vle32_v_f32m1(out_vals.data(), vl);

        if (bias) {
            if (bias_d.nelems() == 1) {
                acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
            } else {
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                std::vector<size_t> bias_strides(bias_ndims);
                bias_strides[bias_ndims - 1] = 1;
                for (int d = bias_ndims - 2; d >= 0; --d)
                    bias_strides[d]
                            = bias_strides[d + 1] * (size_t)bias_dims[d + 1];

                size_t base_bias_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx = (bias_dims[d] == 1)
                            ? 0
                            : dst_idx_prefix[dst_dim_idx];
                    base_bias_off += idx * bias_strides[d];
                }

                if (bias_dims[bias_ndims - 1] == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[base_bias_off], vl);
                } else {
                    const float *bias_ptr = bias + base_bias_off + n0;
                    vfloat32m1_t bias_vec = __riscv_vle32_v_f32m1(bias_ptr, vl);
                    acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                }
            }
        }

        acc = postops_handler.apply(acc, vl);
        __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
        n0 += vl;
    }
}

static inline void rvv_matmul_colmajor_kernel_f16_f16_f16(
        const void *src_base_void_ptr, const void *weights_base_void_ptr,
        void *dst_base_void_ptr, dim_t K, dim_t N, const void *bias_void_ptr,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    const _Float16 *src_base_ptr
            = static_cast<const _Float16 *>(src_base_void_ptr);
    const _Float16 *weights_base_ptr
            = static_cast<const _Float16 *>(weights_base_void_ptr);
    _Float16 *dst_base_ptr = static_cast<_Float16 *>(dst_base_void_ptr);
    const _Float16 *bias = static_cast<const _Float16 *>(bias_void_ptr);

    for (dim_t n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m1(N - n0);
        std::vector<float> out_vals(vl, 0.0f);

        for (dim_t k0 = 0; k0 < K;) {
            size_t k_vl = __riscv_vsetvl_e32m1(K - k0);

            vfloat16m1_t src_vec_f16
                    = __riscv_vle16_v_f16m1(src_base_ptr + k0, k_vl);
            vfloat32m2_t src_vec
                    = __riscv_vfwcvt_f_f_v_f32m2(src_vec_f16, k_vl);

            for (size_t ni = 0; ni < vl; ++ni) {
                const _Float16 *weight_col_ptr
                        = weights_base_ptr + (size_t)(n0 + ni) * (size_t)K;
                vfloat16m1_t wei_vec_f16
                        = __riscv_vle16_v_f16m1(weight_col_ptr + k0, k_vl);
                vfloat32m2_t wei_vec
                        = __riscv_vfwcvt_f_f_v_f32m2(wei_vec_f16, k_vl);

                vfloat32m2_t prod
                        = __riscv_vfmul_vv_f32m2(src_vec, wei_vec, k_vl);
                vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m2_f32m1(
                        prod, __riscv_vfmv_v_f_f32m1(0.0f, k_vl), k_vl);
                float partial = __riscv_vfmv_f_s_f32m1_f32(reduced);

                out_vals[ni] += partial;
            }

            k0 += k_vl;
        }

        vfloat32m2_t acc = __riscv_vle32_v_f32m2(out_vals.data(), vl);

        if (bias) {
            if (bias_d.nelems() == 1) {
                float b = (float)bias[0];
                acc = __riscv_vfadd_vf_f32m2(acc, b, vl);
            } else {
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                std::vector<size_t> bias_strides(bias_ndims);
                bias_strides[bias_ndims - 1] = 1;
                for (int d = bias_ndims - 2; d >= 0; --d)
                    bias_strides[d]
                            = bias_strides[d + 1] * (size_t)bias_dims[d + 1];

                size_t base_bias_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx = (bias_dims[d] == 1)
                            ? 0
                            : dst_idx_prefix[dst_dim_idx];
                    base_bias_off += idx * bias_strides[d];
                }

                if (bias_dims[bias_ndims - 1] == 1) {
                    float b = (float)bias[base_bias_off];
                    acc = __riscv_vfadd_vf_f32m2(acc, b, vl);
                } else {
                    const _Float16 *bias_ptr = bias + base_bias_off + n0;
                    vfloat16m1_t bias_vec_f16
                            = __riscv_vle16_v_f16m1(bias_ptr, vl);
                    vfloat32m2_t bias_vec
                            = __riscv_vfwcvt_f_f_v_f32m2(bias_vec_f16, vl);
                    acc = __riscv_vfadd_vv_f32m2(acc, bias_vec, vl);
                }
            }
        }

        acc = postops_handler.apply(acc, vl);
        vfloat16m1_t acc_f16 = __riscv_vfncvt_f_f_w_f16m1(acc, vl);
        __riscv_vse16_v_f16m1(&dst_base_ptr[n0], acc_f16, vl);
        n0 += vl;
    }
}

static inline void rvv_matmul_colmajor_kernel_f16_f16_f32(
        const void *src_base_void_ptr, const void *weights_base_void_ptr,
        void *dst_base_void_ptr, dim_t K, dim_t N, const void *bias_void_ptr,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    const _Float16 *src_base_ptr
            = static_cast<const _Float16 *>(src_base_void_ptr);
    const _Float16 *weights_base_ptr
            = static_cast<const _Float16 *>(weights_base_void_ptr);
    float *dst_base_ptr = static_cast<float *>(dst_base_void_ptr);
    const float *bias = static_cast<const float *>(bias_void_ptr);

    for (dim_t n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m1(N - n0);
        std::vector<float> out_vals(vl, 0.0f);

        for (dim_t k0 = 0; k0 < K;) {
            size_t k_vl = __riscv_vsetvl_e32m1(K - k0);

            vfloat16m1_t src_vec_f16
                    = __riscv_vle16_v_f16m1(src_base_ptr + k0, k_vl);
            vfloat32m2_t src_vec
                    = __riscv_vfwcvt_f_f_v_f32m2(src_vec_f16, k_vl);

            for (size_t ni = 0; ni < vl; ++ni) {
                const _Float16 *weight_col_ptr
                        = weights_base_ptr + (size_t)(n0 + ni) * (size_t)K;
                vfloat16m1_t wei_vec_f16
                        = __riscv_vle16_v_f16m1(weight_col_ptr + k0, k_vl);
                vfloat32m2_t wei_vec
                        = __riscv_vfwcvt_f_f_v_f32m2(wei_vec_f16, k_vl);

                vfloat32m2_t prod
                        = __riscv_vfmul_vv_f32m2(src_vec, wei_vec, k_vl);
                vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m2_f32m1(
                        prod, __riscv_vfmv_v_f_f32m1(0.0f, k_vl), k_vl);
                float partial = __riscv_vfmv_f_s_f32m1_f32(reduced);

                out_vals[ni] += partial;
            }

            k0 += k_vl;
        }

        vfloat32m1_t acc = __riscv_vle32_v_f32m1(out_vals.data(), vl);

        if (bias) {
            if (bias_d.nelems() == 1) {
                acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
            } else {
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                std::vector<size_t> bias_strides(bias_ndims);
                bias_strides[bias_ndims - 1] = 1;
                for (int d = bias_ndims - 2; d >= 0; --d)
                    bias_strides[d]
                            = bias_strides[d + 1] * (size_t)bias_dims[d + 1];

                size_t base_bias_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx = (bias_dims[d] == 1)
                            ? 0
                            : dst_idx_prefix[dst_dim_idx];
                    base_bias_off += idx * bias_strides[d];
                }

                if (bias_dims[bias_ndims - 1] == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[base_bias_off], vl);
                } else {
                    const float *bias_ptr = bias + base_bias_off + n0;
                    vfloat32m1_t bias_vec = __riscv_vle32_v_f32m1(bias_ptr, vl);
                    acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                }
            }
        }

        acc = postops_handler.apply(acc, vl);
        __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
        n0 += vl;
    }
}

static inline void rvv_matmul_colmajor_compute_kernel(const void *src_base_ptr,
        const void *weights_base_ptr, void *dst_base_ptr, dim_t K, dim_t N,
        const void *bias, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    using namespace dnnl::impl::data_type;
    const auto sdt = src_d.data_type();
    const auto wdt = weights_d.data_type();
    const auto ddt = dst_d.data_type();

    if (sdt == f32 && wdt == f32 && ddt == f32) {
        rvv_matmul_colmajor_kernel_f32_f32_f32(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, postops_handler);
    } else if (sdt == f16 && wdt == f16 && ddt == f16) {
        rvv_matmul_colmajor_kernel_f16_f16_f16(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, postops_handler);
    } else if (sdt == f16 && wdt == f16 && ddt == f32) {
        rvv_matmul_colmajor_kernel_f16_f16_f32(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, postops_handler);
    }
}

static inline void rvv_matmul_rowmajor_kernel_f32_f32_f32(
        const void *src_base_void_ptr, const void *weights_base_void_ptr,
        void *dst_base_void_ptr, dim_t K, dim_t N, const void *bias_void_ptr,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    const float *src_base_ptr = static_cast<const float *>(src_base_void_ptr);
    const float *weights_base_ptr
            = static_cast<const float *>(weights_base_void_ptr);
    float *dst_base_ptr = static_cast<float *>(dst_base_void_ptr);
    const float *bias = static_cast<const float *>(bias_void_ptr);

    for (dim_t n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m1(N - n0);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (dim_t k = 0; k < K; ++k) {
            vfloat32m1_t a_vec = __riscv_vfmv_v_f_f32m1(src_base_ptr[k], vl);
            const float *b_ptr = weights_base_ptr + (size_t)k * N + n0;
            vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b_ptr, vl);
            acc = __riscv_vfmacc_vv_f32m1(acc, a_vec, b_vec, vl);
        }

        if (bias) {
            if (bias_d.nelems() == 1) {
                acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
            } else {
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                std::vector<size_t> bias_strides(bias_ndims);
                bias_strides[bias_ndims - 1] = 1;
                for (int d = bias_ndims - 2; d >= 0; --d)
                    bias_strides[d]
                            = bias_strides[d + 1] * (size_t)bias_dims[d + 1];

                size_t base_bias_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx = (bias_dims[d] == 1)
                            ? 0
                            : dst_idx_prefix[dst_dim_idx];
                    base_bias_off += idx * bias_strides[d];
                }

                if (bias_dims[bias_ndims - 1] == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[base_bias_off], vl);
                } else {
                    const float *bias_ptr = bias + base_bias_off + n0;
                    vfloat32m1_t bias_vec = __riscv_vle32_v_f32m1(bias_ptr, vl);
                    acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                }
            }
        }

        acc = postops_handler.apply(acc, vl);
        __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
        n0 += vl;
    }
}

static inline void rvv_matmul_rowmajor_kernel_f16_f16_f16(
        const void *src_base_void_ptr, const void *weights_base_void_ptr,
        void *dst_base_void_ptr, dim_t K, dim_t N, const void *bias_void_ptr,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    const _Float16 *src_base_ptr
            = static_cast<const _Float16 *>(src_base_void_ptr);
    const _Float16 *weights_base_ptr
            = static_cast<const _Float16 *>(weights_base_void_ptr);
    _Float16 *dst_base_ptr = static_cast<_Float16 *>(dst_base_void_ptr);
    const _Float16 *bias = static_cast<const _Float16 *>(bias_void_ptr);

    for (dim_t n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m1(N - n0);
        vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (dim_t k = 0; k < K; ++k) {
            float a_scalar = (float)src_base_ptr[k];
            vfloat32m2_t a_vec = __riscv_vfmv_v_f_f32m2(a_scalar, vl);
            const _Float16 *b_ptr = weights_base_ptr + (size_t)k * N + n0;
            vfloat16m1_t b_vec_f16 = __riscv_vle16_v_f16m1(b_ptr, vl);
            vfloat32m2_t b_vec = __riscv_vfwcvt_f_f_v_f32m2(b_vec_f16, vl);
            acc = __riscv_vfmacc_vv_f32m2(acc, a_vec, b_vec, vl);
        }

        if (bias) {
            if (bias_d.nelems() == 1) {
                float b = (float)bias[0];
                acc = __riscv_vfadd_vf_f32m2(acc, b, vl);
            } else {
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                std::vector<size_t> bias_strides(bias_ndims);
                bias_strides[bias_ndims - 1] = 1;
                for (int d = bias_ndims - 2; d >= 0; --d)
                    bias_strides[d]
                            = bias_strides[d + 1] * (size_t)bias_dims[d + 1];

                size_t base_bias_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx = (bias_dims[d] == 1)
                            ? 0
                            : dst_idx_prefix[dst_dim_idx];
                    base_bias_off += idx * bias_strides[d];
                }

                if (bias_dims[bias_ndims - 1] == 1) {
                    acc = __riscv_vfadd_vf_f32m2(acc, bias[base_bias_off], vl);
                } else {
                    const _Float16 *bias_ptr = bias + base_bias_off + n0;
                    vfloat16m1_t bias_vec_f16
                            = __riscv_vle16_v_f16m1(bias_ptr, vl);
                    vfloat32m2_t bias_vec
                            = __riscv_vfwcvt_f_f_v_f32m2(bias_vec_f16, vl);
                    acc = __riscv_vfadd_vv_f32m2(acc, bias_vec, vl);
                }
            }
        }

        acc = postops_handler.apply(acc, vl);
        std::vector<float> tmp(vl);
        __riscv_vse32_v_f32m2(tmp.data(), acc, vl);
        for (size_t i = 0; i < vl; ++i)
            dst_base_ptr[n0 + i] = (_Float16)tmp[i];
        n0 += vl;
    }
}

static inline void rvv_matmul_rowmajor_kernel_f16_f16_f32(
        const void *src_base_void_ptr, const void *weights_base_void_ptr,
        void *dst_base_void_ptr, dim_t K, dim_t N, const void *bias_void_ptr,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    const _Float16 *src_base_ptr
            = static_cast<const _Float16 *>(src_base_void_ptr);
    const _Float16 *weights_base_ptr
            = static_cast<const _Float16 *>(weights_base_void_ptr);
    float *dst_base_ptr = static_cast<float *>(dst_base_void_ptr);
    const float *bias = static_cast<const float *>(bias_void_ptr);

    for (dim_t n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m1(N - n0);
        vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (dim_t k = 0; k < K; ++k) {
            float a_scalar = (float)src_base_ptr[k];
            vfloat32m2_t a_vec = __riscv_vfmv_v_f_f32m2(a_scalar, vl);
            const _Float16 *b_ptr = weights_base_ptr + (size_t)k * N + n0;
            vfloat16m1_t b_vec_f16 = __riscv_vle16_v_f16m1(b_ptr, vl);
            vfloat32m2_t b_vec = __riscv_vfwcvt_f_f_v_f32m2(b_vec_f16, vl);
            acc = __riscv_vfmacc_vv_f32m2(acc, a_vec, b_vec, vl);
        }

        if (bias) {
            if (bias_d.nelems() == 1) {
                acc = __riscv_vfadd_vf_f32m2(acc, bias[0], vl);
            } else {
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                std::vector<size_t> bias_strides(bias_ndims);
                bias_strides[bias_ndims - 1] = 1;
                for (int d = bias_ndims - 2; d >= 0; --d)
                    bias_strides[d]
                            = bias_strides[d + 1] * (size_t)bias_dims[d + 1];

                size_t base_bias_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx = (bias_dims[d] == 1)
                            ? 0
                            : dst_idx_prefix[dst_dim_idx];
                    base_bias_off += idx * bias_strides[d];
                }

                if (bias_dims[bias_ndims - 1] == 1) {
                    acc = __riscv_vfadd_vf_f32m2(acc, bias[base_bias_off], vl);
                } else {
                    const float *bias_ptr = bias + base_bias_off + n0;
                    vfloat32m2_t bias_vec = __riscv_vle32_v_f32m2(bias_ptr, vl);
                    acc = __riscv_vfadd_vv_f32m2(acc, bias_vec, vl);
                }
            }
        }

        acc = postops_handler.apply(acc, vl);
        __riscv_vse32_v_f32m2(&dst_base_ptr[n0], acc, vl);
        n0 += vl;
    }
}

static inline void rvv_matmul_rowmajor_compute_kernel(const void *src_base_ptr,
        const void *weights_base_ptr, void *dst_base_ptr, dim_t K, dim_t N,
        const void *bias, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &bias_d,
        const std::vector<dim_t> &dst_idx_prefix,
        const rvv_postops_t &postops_handler) {
    using namespace dnnl::impl::data_type;
    const auto sdt = src_d.data_type();
    const auto wdt = weights_d.data_type();
    const auto ddt = dst_d.data_type();

    if (sdt == f32 && wdt == f32 && ddt == f32) {
        rvv_matmul_rowmajor_kernel_f32_f32_f32(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, postops_handler);
    } else if (sdt == f16 && wdt == f16 && ddt == f16) {
        rvv_matmul_rowmajor_kernel_f16_f16_f16(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, postops_handler);
    } else if (sdt == f16 && wdt == f16 && ddt == f32) {
        rvv_matmul_rowmajor_kernel_f16_f16_f32(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, postops_handler);
    }
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_MATMUL_KERNEL_HPP