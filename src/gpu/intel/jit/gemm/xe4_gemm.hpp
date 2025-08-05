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

#ifndef GPU_INTEL_JIT_GEMM_XE4_GEMM_HPP
#define GPU_INTEL_JIT_GEMM_XE4_GEMM_HPP

#include <assert.h>
#include <limits>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"
#include "gpu/intel/jit/gemm/jit_gemm_pd.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct xe4_gemm_t : public gpu_gemm_t {
    struct kernel_desc_t : public trivially_serializable_t<kernel_desc_t> {
        ngen::DataType a_type;
        ngen::DataType b_type;
        ngen::DataType c_type;
        uint8_t pad[1];

        transpose_t a_trans = transpose::notrans;
        transpose_t b_trans = transpose::notrans;

        uint32_t bm;
        uint32_t bn;
        uint32_t bk;
        uint32_t stages;

        uint32_t slm_size() const {
            uint32_t a_bytes = (bm * bk * ngen::getBytes(a_type)) * stages;
            uint32_t b_bytes = (bk * bn * ngen::getBytes(b_type)) * stages;
            return a_bytes + b_bytes
                    + (ngen::getBytes(acc_type()) + ngen::getBytes(c_type))
                    * (bm * bn);
        }

        ngen::DataType acc_type() const {
            if (a_type == ngen::DataType::s8 || a_type == ngen::DataType::u8)
                return ngen::DataType::s32;
            return ngen::DataType::f32;
        }

        status_t create_generator(const compute::compute_engine_t &engine,
                compute::kernel_t &kernel) const;
    };

    struct pd_t : public jit_gemm_pd_t {
        kernel_desc_t kernel_desc;
        bool swap_ab = false;

        using jit_gemm_pd_t::jit_gemm_pd_t;

        DECLARE_COMMON_PD_T("jit:gemm:xe4", xe4_gemm_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using namespace alg_kind;
            using arch_t = compute::gpu_arch_t;

            const uint32_t mma_bm_min = 32;
            const uint32_t mma_bn_min = 32;
            const uint32_t mma_bk_min = 32;
            const uint32_t mma_bm_max = 256;
            const uint32_t mma_bn_max = 512;
            const uint32_t mma_bk_max_d8 = 256;
            const uint32_t max_slm_size = 1280 * 1024;

            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto *dev_info = compute_engine->device_info();

            bool arch_ok = (dev_info->gpu_arch() == arch_t::xe4);
            const auto d = desc();

            VDISPATCH_GEMM(arch_ok, VERBOSE_UNSUPPORTED_ARCH, "gpu");
            VDISPATCH_GEMM(compute_engine->mayiuse_ngen_kernels(),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "ngen_kernels");
            VDISPATCH_GEMM(!has_blocks(), VERBOSE_BLOCKING_FAIL, "");
            VDISPATCH_GEMM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(!d->is_batched(), VERBOSE_BAD_DIM, "batch",
                    d->c_desc.ndims - 2);
            VDISPATCH_GEMM(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_GEMM(memory_desc_wrapper(d->a_desc).is_dense(),
                    VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "A");
            VDISPATCH_GEMM(memory_desc_wrapper(d->b_desc).is_dense(),
                    VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "B");

            // Check data types.
            if (utils::one_of(d->a_type(), s8, u8)) {
                VDISPATCH_GEMM(utils::one_of(d->b_type(), s8, u8),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(d->c_type() == s32, VERBOSE_UNSUPPORTED_DT);
            } else if (d->a_type() == bf16) {
                VDISPATCH_GEMM(d->b_type() == bf16, VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(utils::one_of(d->c_type(), bf16, f32),
                        VERBOSE_UNSUPPORTED_DT);
            } else if (d->a_type() == f16) {
                VDISPATCH_GEMM(d->b_type() == f16, VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(utils::one_of(d->c_type(), f16, f32),
                        VERBOSE_UNSUPPORTED_DT);
            } else {
                return status::unimplemented;
            }
            kernel_desc.a_type = convert_dnnl_type_to_ngen(d->b_type());
            kernel_desc.b_type = convert_dnnl_type_to_ngen(d->a_type());
            kernel_desc.c_type = convert_dnnl_type_to_ngen(d->c_type());
            uint32_t a_size = ngen::getBytes(kernel_desc.a_type);
            uint32_t b_size = ngen::getBytes(kernel_desc.b_type);
            uint32_t ab_size = std::max(a_size, b_size);
            uint32_t mma_bk_max = mma_bk_max_d8 / ab_size;
            kernel_desc.a_trans = d->transb();
            kernel_desc.b_trans = d->transa();
            kernel_desc.bm = select_block(
                    (uint32_t)desc()->n(), mma_bm_min, mma_bm_max);
            kernel_desc.bn = select_block(
                    (uint32_t)desc()->m(), mma_bn_min, mma_bn_max);
            kernel_desc.bk = select_block(
                    (uint32_t)desc()->k(), mma_bk_min, mma_bk_max);
            kernel_desc.stages = 3;

            auto trans_other = [](transpose_t t) {
                return (t == transpose::notrans) ? transpose::trans
                                                 : transpose::notrans;
            };

            if (d->transc() == transpose::trans) {
                swap_ab = true;
                kernel_desc.a_trans = trans_other(d->transa());
                kernel_desc.b_trans = trans_other(d->transb());
                std::swap(kernel_desc.a_type, kernel_desc.b_type);
                std::swap(kernel_desc.bm, kernel_desc.bn);
                // Fix up M/N blocks.
                kernel_desc.bm = std::min(mma_bm_max, kernel_desc.bm);
                kernel_desc.bn = std::min(mma_bn_max, kernel_desc.bn);
            }

            while (kernel_desc.slm_size() > max_slm_size) {
                if (kernel_desc.bn > mma_bn_min) {
                    kernel_desc.bn /= 2;
                    continue;
                }
                if (kernel_desc.bm > mma_bm_min) {
                    kernel_desc.bm /= 2;
                    continue;
                }
                if (kernel_desc.bk > mma_bk_min) {
                    kernel_desc.bk /= 2;
                    continue;
                }
                return status::unimplemented;
            }

            return status::success;
        }

        static uint32_t select_block(
                uint32_t size, uint32_t b_min, uint32_t b_max) {
            uint32_t b = b_min;
            while (b < b_max && size > b)
                b *= 2;
            return b;
        }
    };

    xe4_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    status_t init(impl::engine_t *engine) override {
        using namespace data_type;
        CHECK(create_kernel(engine, kernel_, "xe4_gemm", pd()->kernel_desc));
        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override {
        const auto &a = pd()->swap_ab ? GEMM_CTX_ARG_STORAGE(b)
                                      : GEMM_CTX_ARG_STORAGE(a);
        const auto &b = pd()->swap_ab ? GEMM_CTX_ARG_STORAGE(a)
                                      : GEMM_CTX_ARG_STORAGE(b);
        const auto &c = GEMM_CTX_ARG_STORAGE(c);
        uint32_t M = (uint32_t)pd()->desc()->n();
        uint32_t N = (uint32_t)pd()->desc()->m();
        uint32_t K = (uint32_t)pd()->desc()->k();
        if (pd()->swap_ab) std::swap(M, N);
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, a);
        arg_list.set(1, b);
        arg_list.set(2, c);
        arg_list.set(3, M);
        arg_list.set(4, N);
        arg_list.set(5, K);
        size_t m_wg = utils::div_up(M, pd()->kernel_desc.bm);
        size_t n_wg = utils::div_up(N, pd()->kernel_desc.bn);
        const compute::range_t gws
                = {(size_t)n_wg * 128, (size_t)m_wg, (size_t)1};
        const compute::range_t lws = {128, 1, 1};
        const auto nd_range = compute::nd_range_t(gws, lws);
        CHECK(parallel_for(ctx, nd_range, kernel_, arg_list));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
