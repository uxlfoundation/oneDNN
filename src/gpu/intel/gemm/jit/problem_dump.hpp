/*******************************************************************************
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

#ifndef GPU_INTEL_GEMM_JIT_PROBLEM_DUMP_HPP
#define GPU_INTEL_GEMM_JIT_PROBLEM_DUMP_HPP

// Diagnostic GEMMProblem dump. Gated by env var ONEDNN_DUMP_GEMM_PROBLEM=<tag>.
// When enabled, dumps every field of a GEMMProblem to stderr prefixed with the
// given tag, one field per line. Intended for diffing against a sibling build
// (e.g. ../base) to triage migration regressions.
//
// Header-only so it can be cherry-picked verbatim into pre-migration trees:
// no dependencies beyond gemmstone/problem.hpp and the C stdlib.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gemmstone/problem.hpp"
#include "gpu/intel/post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

inline const char *dump_gemm_problem_tag() {
    const char *t = std::getenv("ONEDNN_DUMP_GEMM_PROBLEM");
    return (t && *t) ? t : nullptr;
}

inline const char *dump_layout_str(::gemmstone::MatrixLayout l) {
    switch (l) {
        case ::gemmstone::MatrixLayout::N: return "N";
        case ::gemmstone::MatrixLayout::T: return "T";
        case ::gemmstone::MatrixLayout::Pc: return "Pc";
        case ::gemmstone::MatrixLayout::Pr: return "Pr";
    }
    return "?";
}

inline const char *dump_aboffset_str(::gemmstone::ABOffset v) {
    switch (v) {
        case ::gemmstone::ABOffset::None: return "None";
        case ::gemmstone::ABOffset::Calc: return "Calc";
        case ::gemmstone::ABOffset::Load: return "Load";
    }
    return "?";
}

inline const char *dump_coffset_str(::gemmstone::COffset v) {
    switch (v) {
        case ::gemmstone::COffset::None: return "None";
        case ::gemmstone::COffset::Post: return "Post";
        case ::gemmstone::COffset::Pre: return "Pre";
    }
    return "?";
}

inline const char *dump_batch_str(::gemmstone::BatchMode v) {
    switch (v) {
        case ::gemmstone::BatchMode::None: return "None";
        case ::gemmstone::BatchMode::Strided: return "Strided";
        case ::gemmstone::BatchMode::Nonstrided: return "Nonstrided";
        case ::gemmstone::BatchMode::Variable: return "Variable";
    }
    return "?";
}

inline int dump_type_id(::gemmstone::Type t) {
    return static_cast<int>(static_cast<::gemmstone::Type::_Type>(t));
}

inline int dump_scalar_value(const ::gemmstone::Scalar &s) {
    // Print fixed scalar value if fixed, else -1 sentinel.
    return s.fixed() ? int(s) : -1;
}

inline const char *dump_scalar_type(const ::gemmstone::Scalar &s) {
    switch (s.getType()) {
        case ::gemmstone::Scalar::Fixed: return "Fixed";
        case ::gemmstone::Scalar::Variable: return "Var";
        case ::gemmstone::Scalar::Pointer: return "Ptr";
        case ::gemmstone::Scalar::RealPointer: return "RPtr";
    }
    return "?";
}

inline void dump_matrix_addressing(
        const char *tag, const char *name, const ::gemmstone::MatrixAddressing &m) {
    std::fprintf(stderr,
            "[GPDUMP:%s] %s layout=%s packSize=%u tileR=%u tileC=%u "
            "panelLength=%u crosspack=%u alignment=%u needA64=%d\n",
            tag, name, dump_layout_str(m.layout), (unsigned)m.packSize,
            (unsigned)m.tileR, (unsigned)m.tileC, (unsigned)m.panelLength,
            (unsigned)m.crosspack, (unsigned)m.alignment, (int)m.needA64);
}

inline void dump_gemm_problem(
        const char *tag, const ::gemmstone::GEMMProblem &p) {
    using namespace ::gemmstone;
    std::fprintf(stderr, "[GPDUMP:%s] === BEGIN GEMMProblem ===\n", tag);
    std::fprintf(stderr,
            "[GPDUMP:%s] Ta=%d Tb=%d Tc=%d Ts=%d\n", tag, dump_type_id(p.Ta),
            dump_type_id(p.Tb), dump_type_id(p.Tc), dump_type_id(p.Ts));
    std::fprintf(stderr,
            "[GPDUMP:%s] Ta_ext=%d Tb_ext=%d Tc_ext=%d\n", tag,
            dump_type_id(p.Ta_ext), dump_type_id(p.Tb_ext),
            dump_type_id(p.Tc_ext));
    std::fprintf(stderr, "[GPDUMP:%s] Tao=%d Tbo=%d Tco=%d\n", tag,
            dump_type_id(p.Tao), dump_type_id(p.Tbo), dump_type_id(p.Tco));
    std::fprintf(stderr,
            "[GPDUMP:%s] Ta_scale=%d Tb_scale=%d Tc_scale=%d\n", tag,
            dump_type_id(p.Ta_scale), dump_type_id(p.Tb_scale),
            dump_type_id(p.Tc_scale));
    std::fprintf(stderr, "[GPDUMP:%s] Tag=%d Tbg=%d\n", tag,
            dump_type_id(p.Tag), dump_type_id(p.Tbg));
    std::fprintf(stderr,
            "[GPDUMP:%s] alpha:type=%s value=%d  beta:type=%s value=%d\n", tag,
            dump_scalar_type(p.alpha), dump_scalar_value(p.alpha),
            dump_scalar_type(p.beta), dump_scalar_value(p.beta));
    dump_matrix_addressing(tag, "A", p.A);
    dump_matrix_addressing(tag, "B", p.B);
    dump_matrix_addressing(tag, "C", p.C);
    dump_matrix_addressing(tag, "AO", p.AO);
    dump_matrix_addressing(tag, "BO", p.BO);
    dump_matrix_addressing(tag, "CO", p.CO);
    dump_matrix_addressing(tag, "A_scale", p.A_scale);
    dump_matrix_addressing(tag, "B_scale", p.B_scale);
    dump_matrix_addressing(tag, "C_scale", p.C_scale);
    dump_matrix_addressing(tag, "Ag", p.Ag);
    dump_matrix_addressing(tag, "Bg", p.Bg);
    dump_matrix_addressing(tag, "sroundSeed", p.sroundSeed);
    std::fprintf(stderr, "[GPDUMP:%s] checkBeta0=%d\n", tag,
            (int)p.checkBeta0);
    std::fprintf(stderr, "[GPDUMP:%s] aOffset=%s bOffset=%s cOffset=%s\n", tag,
            dump_aboffset_str(p.aOffset), dump_aboffset_str(p.bOffset),
            dump_coffset_str(p.cOffset));
    std::fprintf(stderr, "[GPDUMP:%s] aoPtrDims=%d boPtrDims=%d coPtrDims=%d\n",
            tag, p.aoPtrDims, p.boPtrDims, p.coPtrDims);
    std::fprintf(stderr,
            "[GPDUMP:%s] asPtrDims=%d bsPtrDims=%d csPtrDims=%d\n", tag,
            p.asPtrDims, p.bsPtrDims, p.csPtrDims);
    std::fprintf(stderr, "[GPDUMP:%s] aqGroupM=%d aqGroupK=%d\n", tag,
            p.aqGroupM, p.aqGroupK);
    std::fprintf(stderr, "[GPDUMP:%s] bqGroupN=%d bqGroupK=%d\n", tag,
            p.bqGroupN, p.bqGroupK);
    std::fprintf(stderr, "[GPDUMP:%s] cqGroupM=%d cqGroupN=%d\n", tag,
            p.cqGroupM, p.cqGroupN);
    std::fprintf(stderr, "[GPDUMP:%s] batch=%s batchDims=%d\n", tag,
            dump_batch_str(p.batch), p.batchDims);
    std::fprintf(stderr,
            "[GPDUMP:%s] sumA=%d sumB=%d forceGroupSumsA=%d forceGroupSumsB=%d\n",
            tag, (int)p.sumA, (int)p.sumB, (int)p.forceGroupSumsA,
            (int)p.forceGroupSumsB);
    std::fprintf(stderr,
            "[GPDUMP:%s] bdpasEnabled=%d cMXScale=%d\n", tag,
            (int)p.bdpasEnabled, (int)p.cMXScale);
    // PostOps summary.
    std::fprintf(stderr,
            "[GPDUMP:%s] postOps.len=%zu fwd=%d cStochasticRound=%d\n", tag,
            p.postOps.len(), (int)p.postOps.fwd,
            (int)p.postOps.cStochasticRound);
    const size_t po_limit = p.postOps.len();
    for (size_t i = 0; i < po_limit; ++i) {
        const auto &e = p.postOps.ops[i];
        const int kind = static_cast<int>(e.kind());
        std::fprintf(stderr,
                "[GPDUMP:%s] postOps[%zu] kind=%d binaryRow=%d binaryCol=%d "
                "binaryBatch=%d binaryTrans=%d",
                tag, i, kind, (int)p.postOps.binaryRow[i],
                (int)p.postOps.binaryCol[i], (int)p.postOps.binaryBatch[i],
                (int)p.postOps.binaryTrans[i]);
        if (e.is_sum()) {
            const auto &s = e.as_sum();
            std::fprintf(stderr,
                    " sum{dt=%d inline_scale=%d scale=%g inline_zp=%d zp=%d}",
                    (int)s.dt, (int)s.inline_scale, (double)s.scale,
                    (int)s.inline_zero_point, s.zero_point);
        } else if (e.is_eltwise()) {
            const auto &el = e.as_eltwise();
            std::fprintf(stderr,
                    " eltwise{alg=%d scale=%g alpha=%g beta=%g}",
                    (int)el.alg, (double)el.scale, (double)el.alpha,
                    (double)el.beta);
        } else if (e.is_binary()) {
            const auto &b = e.as_binary();
            std::fprintf(stderr,
                    " binary{alg=%d src1.dt=%d src1.bcast=0x%x src1.inner_dim=%s}",
                    (int)b.alg, (int)b.src1_desc.dt,
                    (unsigned)b.src1_desc.broadcast_mask,
                    b.src1_desc.inner_dim.str().c_str());
        } else if (e.is_depthwise_conv()) {
            const auto &c = e.as_depthwise_conv();
            std::fprintf(stderr,
                    " conv{k=%ld s=%ld p=%ld wei=%d bias=%d dst=%d}",
                    (long)c.kernel, (long)c.stride, (long)c.padding,
                    (int)c.wei_dt, (int)c.bias_dt, (int)c.dst_dt);
        }
        std::fprintf(stderr, "\n");
    }
    // Derived binary[] addressings.
    for (size_t i = 0; i < p.binary.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "binary[%zu]", i);
        dump_matrix_addressing(tag, name, p.binary[i]);
        std::fprintf(stderr, "[GPDUMP:%s] binary[%zu].Tbinary=%d\n", tag, i,
                i < p.Tbinary.size() ? dump_type_id(p.Tbinary[i]) : -1);
    }
    std::fprintf(stderr, "[GPDUMP:%s] === END GEMMProblem ===\n", tag);
    std::fflush(stderr);
}

// Convenience: dump only if env var is set, using its value as the tag.
inline void maybe_dump_gemm_problem(const ::gemmstone::GEMMProblem &p) {
    const char *tag = dump_gemm_problem_tag();
    if (!tag) return;
    dump_gemm_problem(tag, p);
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
