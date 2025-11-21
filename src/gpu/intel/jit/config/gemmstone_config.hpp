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

#ifndef GPU_INTEL_JIT_CONFIG_GEMMSTONE_CONFIG_HPP
#define GPU_INTEL_JIT_CONFIG_GEMMSTONE_CONFIG_HPP

#include <bitset>

#include "common/primitive_attr.hpp"
#include "common/serialization.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/jit/post_op_injector.hpp"
#include "gpu/intel/microkernels/entrance_agent.hpp"
#include "gpu/intel/microkernels/package.hpp"
#include "gpu/intel/post_ops.hpp"
#include "ngen_register_allocator.hpp"

// TODO: Work with upstream to prefix defines with GEMMSTONE
#define DNNL
#define MICROKERNEL_INTERFACE

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#define ZEBIN_OUTPUT
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define OPENCL_OUTPUT
#endif

#ifdef DNNL_DEV_MODE
#define GEMMSTONE_ASSERTIONS 1
#endif

namespace gemmstone {

#define GENERATOR_SUPER(hw) ngen::ELFCodeGenerator<hw>
#define GENERATOR_BASE(hw) dnnl::impl::gpu::intel::jit::generator_t<hw>
#define FORWARD(hw) NGEN_FORWARD_ELF(hw)
#define GENERATOR_DEBUGINFO \
    dnnl::impl::gpu::intel::jit::debug_config_t { \
        GENERATOR_NAME, GENERATOR_LINE \
    }

inline int getEnv(const char *s, int def) {
    return dnnl::impl::gpu::intel::gpu_utils::dev_getenv(s, def);
}

enum class GEMMVerbose { DebugInfo = dnnl::impl::verbose_t::debuginfo };

inline int getVerbose(GEMMVerbose v) {
    return dnnl::impl::get_verbose(
            static_cast<dnnl::impl::verbose_t::flag_kind>(v));
}

template <typename... Args>
inline void verbosePrintf(const char *fmtStr, Args... args) {
    return dnnl::impl::verbose_printf(fmtStr, args...);
}

namespace micro = dnnl::impl::gpu::intel::micro;

using SerializationStream = dnnl::impl::serialization_stream_t;

enum class BinaryOp;
using PostOps = dnnl::impl::gpu::intel::gpu_post_ops_t;
// NOLINTBEGIN(readability-identifier-naming)
struct PostOpsProblem {
    PostOpsProblem() = default;
    PostOpsProblem(PostOps &&ops) : ops(std::move(ops)) {};

    static const int maxPostOps = dnnl::impl::post_ops_t::post_ops_limit;

    template <ngen::HW hw>
    using Injector = dnnl::impl::gpu::intel::jit::post_op_injector_t<
            typename GENERATOR_BASE(hw)::RootCodeGenerator>;
    static BinaryOp toBinaryOp(const PostOps::entry_t &e);

    bool empty() const { return ops.empty(); }
    size_t len() const { return ops.len(); }
    PostOps::entry_t &operator[](size_t idx) { return ops[idx]; }
    const PostOps::entry_t &operator[](size_t idx) const { return ops[idx]; }

    void transpose() {
        std::swap(binaryRow, binaryCol);
        binaryTrans.flip();
    }

    void serialize(SerializationStream &s) const {
        s.append(ops);
        s.append(binaryRow);
        s.append(binaryCol);
        s.append(binaryBatch);
        s.append(binaryTrans);
        s.append(fwd);
        s.append(cStochasticRound);
    }

    PostOps ops;
    std::bitset<maxPostOps> binaryRow; // Broadcasts row data if false
    std::bitset<maxPostOps> binaryCol; // Broadcasts column data if false
    std::bitset<maxPostOps> binaryBatch; // Broadcasts batch data if false
    std::bitset<maxPostOps> binaryTrans; // Used to compute GEMMProblem::binary

    bool fwd = true;
    bool cStochasticRound = false;

    template <ngen::HW hw>
    void injectNonBinaryPostOps(const PostOps::entry_t &entry,
            GENERATOR_BASE(hw) * g, ngen::RegisterAllocator ra,
            int C_grfs[ngen::GRF::maxRegs()], int C_ngrf) const {
        namespace jit = dnnl::impl::gpu::intel::jit;
        switch (entry.kind()) {
            case dnnl::impl::gpu::intel::post_op::kind_t::eltwise: {
                using Injector = jit::eltwise_injector_f32_t<
                        typename jit::generator_t<hw>::RootCodeGenerator>;
                auto &ee = entry.as_eltwise();
                Injector injector {g, ee.alg, ee.alpha, ee.beta, ee.scale,
                        ngen::GRFRange(), fwd};

                auto scratch
                        = ra.try_alloc_range(injector.preferred_scratch_regs());
                if (scratch.isInvalid())
                    scratch = ra.alloc_range(injector.min_scratch_regs());

                injector.set_scratch(scratch);
                injector.prepare();
                injector.compute(C_grfs, C_ngrf);
                break;
            }
            default: gpu_error_not_expected();
        }
    }

    template <ngen::HW hw>
    void injectStochasticRound(GENERATOR_BASE(hw) * g,
            ngen::RegisterAllocator ra, int C_grfs[ngen::GRF::maxRegs()],
            int C_ngrf, const ngen::Subregister &seed, ngen::DataType t) const {
        namespace jit = dnnl::impl::gpu::intel::jit;
        using Injector = jit::eltwise_injector_f32_t<
                typename jit::generator_t<hw>::RootCodeGenerator>;
        Injector injector {g, dnnl::impl::alg_kind::eltwise_stochastic_round,
                0.0, 0.0, 1.0, ngen::GRFRange(), fwd};
        auto scratch = ra.try_alloc_range(injector.preferred_scratch_regs());
        if (scratch.isInvalid())
            scratch = ra.alloc_range(injector.min_scratch_regs());
        if (scratch.isInvalid()) gpu_error_not_expected();

        injector.set_scratch(scratch);
        injector.prepare();
        injector.compute(C_grfs, C_ngrf, seed.getBase(), seed.getOffset(), t);
    }

    template <ngen::HW hw>
    void injectMXScale(GENERATOR_BASE(hw) * g, ngen::RegisterAllocator ra,
            int C_grfs[ngen::GRF::maxRegs()], int C_ngrf,
            const ngen::Subregister &scaleDst, ngen::DataType t,
            int unroll) const {
        namespace jit = dnnl::impl::gpu::intel::jit;
        using Injector = jit::eltwise_injector_f32_t<
                typename jit::generator_t<hw>::RootCodeGenerator>;
        Injector injector {g, dnnl::impl::alg_kind::eltwise_mx_scale, 0.0, 0.0,
                1.0, ngen::GRFRange(), fwd};
        auto scratch = ra.try_alloc_range(injector.preferred_scratch_regs());
        if (scratch.isInvalid())
            scratch = ra.alloc_range(injector.min_scratch_regs());
        if (scratch.isInvalid()) gpu_error_not_expected();

        injector.set_scratch(scratch);
        injector.prepare();
        injector.compute(C_grfs, C_ngrf, scaleDst.getBase(), unroll, t);
    }
};
// NOLINTEND(readability-identifier-naming)

} // namespace gemmstone
#endif
