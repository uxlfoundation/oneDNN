/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 SpacemiT Corporation
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

#ifndef CPU_RV64_JIT_RVV_SOFTMAX_KERNEL_HPP
#define CPU_RV64_JIT_RVV_SOFTMAX_KERNEL_HPP

#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_softmax_affine_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t len;
        float sub;
        float mul;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_affine_kernel_t)

    jit_rvv_softmax_affine_kernel_t();

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;
};

struct jit_rvv_softmax_f16_affine_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        dnnl::impl::float16_t *dst;
        dim_t len;
        float sub;
        float mul;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_f16_affine_kernel_t)

    explicit jit_rvv_softmax_f16_affine_kernel_t(bool src_f32);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool src_f32_;
};

struct jit_rvv_softmax_f16_strided_kernel_t : public jit_generator_t {
    struct call_params_t {
        const dnnl::impl::float16_t *src;
        dnnl::impl::float16_t *dst;
        dim_t len;
        dim_t stride_bytes;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_f16_strided_kernel_t)

    explicit jit_rvv_softmax_f16_strided_kernel_t(bool gather);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool gather_;
};

struct jit_rvv_softmax_f16_exp_sub_sum_kernel_t : public jit_generator_t {
    struct call_params_t {
        const uint16_t *src;
        float *tmp;
        dim_t len;
        float sub;
        float *sum;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_f16_exp_sub_sum_kernel_t)

    jit_rvv_softmax_f16_exp_sub_sum_kernel_t();

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;
};

void jit_rvv_softmax_f16_affine_from_f16(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, float sub, float mul);

void jit_rvv_softmax_f16_affine_from_f32(const float *src,
        dnnl::impl::float16_t *dst, dim_t len, float sub, float mul);

void jit_rvv_softmax_f16_gather(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, dim_t stride_bytes);

void jit_rvv_softmax_f16_scatter(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, dim_t stride_bytes);

void jit_rvv_softmax_f16_exp_sub_sum(const dnnl::impl::float16_t *src, float *tmp,
        dim_t len, float sub, float *sum);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_SOFTMAX_KERNEL_HPP
