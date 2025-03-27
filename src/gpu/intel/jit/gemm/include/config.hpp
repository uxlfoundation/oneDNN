/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_CONFIG_HPP
#define GEMMSTONE_GUARD_CONFIG_HPP

#include "common/serialization.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/gpu_post_ops.hpp"
#include "gpu/intel/microkernels/package.hpp"
#include "gpu/intel/microkernels/entrance_agent.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/jit/post_op_injector.hpp"
#include "gpu/intel/utils.hpp"
#include "ngen_register_allocator.hpp"

#include "pre_config.hpp"

#include "internal/namespace_start.hxx"

#define GENERATOR_BASE(hw) dnnl::impl::gpu::intel::jit::generator_t<hw>

enum class GEMMVerbose {
    DebugInfo = dnnl::impl::verbose_t::debuginfo
};

inline int getVerbose(GEMMVerbose v) {
    return dnnl::impl::get_verbose(static_cast<dnnl::impl::verbose_t::flag_kind>(v));
}

template <typename... Args>
inline void verbosePrintf(const char *fmtStr, Args... args) {
    return dnnl::impl::verbose_printf(fmtStr, args...);
}

namespace micro = dnnl::impl::gpu::intel::micro;

using SerializationStream = dnnl::impl::serialization_stream_t;

using PostOps = dnnl::impl::gpu::intel::gpu_post_ops_t;
template <ngen::HW hw>
using PostOpInjector = dnnl::impl::gpu::intel::jit::post_op_injector_t<GENERATOR_BASE(hw)>;
constexpr int maxPostOps = dnnl::impl::post_ops_t::post_ops_limit;

static inline BinaryOp toBinaryOp(const PostOps::entry_t &e)
{
    using namespace dnnl::impl;
    switch (e.as_binary().alg) {
        case alg_kind::binary_add:   return BinaryOp::Add;
        case alg_kind::binary_sub:   return BinaryOp::Sub;
        case alg_kind::binary_mul:   return BinaryOp::Mul;
        case alg_kind::binary_div:   return BinaryOp::Div;
        case alg_kind::binary_min:   return BinaryOp::Min;
        case alg_kind::binary_max:   return BinaryOp::Max;
        case alg_kind::binary_prelu: return BinaryOp::Prelu;
        default: gpu_error_not_expected();
    }
    return BinaryOp::Add;
}

template <ngen::HW hw>
void injectNonBinaryPostOps(const PostOps::entry_t & entry, GENERATOR_BASE(hw) *g,
                            ngen::RegisterAllocator ra, int C_grfs[ngen::GRF::maxRegs()],
                            int C_ngrf, bool postOpFwd) {
    namespace jit = dnnl::impl::gpu::intel::jit;
    switch (entry.kind()) {
        case dnnl::impl::gpu::intel::post_op::kind_t::eltwise: {
            using Injector = jit::eltwise_injector_f32_t<jit::generator_t<hw>>;
            int euCount = 0; /* only used for a DG2 W/A for conv */
            auto &ee = entry.as_eltwise();
            Injector injector{g, ee.alg, ee.alpha, ee.beta, ee.scale,
                              euCount, ngen::GRFRange(), postOpFwd};

            auto scratch = ra.try_alloc_range(injector.preferred_scratch_regs());
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
void injectStochasticRound(GENERATOR_BASE(hw) *g,
                            ngen::RegisterAllocator ra, int C_grfs[ngen::GRF::maxRegs()],
                           int C_ngrf, bool postOpFwd, const ngen::Subregister &seed, ngen::DataType t) {
    namespace jit = dnnl::impl::gpu::intel::jit;
    using Injector = jit::eltwise_injector_f32_t<jit::generator_t<hw>>;
    int euCount = 0; /* only used for a DG2 W/A for conv */
    Injector injector{g, dnnl::impl::alg_kind::eltwise_stochastic_round, 0.0, 0.0, 1.0,
                      euCount, ngen::GRFRange(), postOpFwd};
    auto scratch = ra.try_alloc_range(injector.preferred_scratch_regs());
    if (scratch.isInvalid())
        scratch = ra.alloc_range(injector.min_scratch_regs());
    if (scratch.isInvalid())
        gpu_error_not_expected();

    injector.set_scratch(scratch);
    injector.prepare();
    injector.compute(C_grfs, C_ngrf, seed.getBase(), seed.getOffset(), t);

}


#include "internal/namespace_end.hxx"

#endif /* header guard */
