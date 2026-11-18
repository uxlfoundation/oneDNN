//
// SPDX-FileCopyrightText: Copyright 2017-2022, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stdio.h>
#include <limits>

#include "gemm_driver_common.hpp"

#ifdef CYCLE_PROFILING
#include "kai/ops/profiler.hpp"
#endif

namespace kai {
namespace ops {

// Implementation of the GemmCommon abstract class.
//
// This is implementation is for GEMV with pretransposition.
//
// batches are not supported as a batched GEMV makes no sense (can be converted to a GEMM).
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing>
class GemvPretransposed : public GemmDriverCommon<strategy, Tlo, Tro, Tr, OutputStage, typename strategy::operand_type> {
    using Base = GemmDriverCommon<strategy, Tlo, Tro, Tr, OutputStage, typename strategy::operand_type>;
    using Base::_args;
    using Base::_os;
    using Base::_B_transposed;
    using Base::get_col_sum_size;

    typedef typename strategy::operand_type Troi;
    typedef typename strategy::result_type Tri;

    const unsigned int _buffer_per_multi;

    unsigned int k_block=0;
    unsigned int n_block=0;

public:
    GemvPretransposed(GemvPretransposed &) = delete;
    GemvPretransposed & operator= (GemvPretransposed &) = delete;

    GemvPretransposed(const GemmArgs &args, const OutputStage &os = {})
                      : Base(args, os),
                        _buffer_per_multi(roundup(args._Ksize, strategy::k_unroll()) * roundup(args._Nsize, strategy::out_width())) {
        assert(!_args._fixed_format);
        assert(_args._Ksections == 1);

        /* For now don't do any blocking. TODO: figure out if we should. */
        if (strategy::supports_accumulate() && args._cfg && args._cfg->inner_block_size) {
            k_block = args._cfg->inner_block_size;
        } else {
            k_block = args._Ksize;
        }

        if (args._cfg && args._cfg->outer_block_size) {
            n_block = args._cfg->outer_block_size;
        } else {
            n_block = args._Nsize;
        }
    }

    // Window is number of out_width blocks, times number of multis.
    ndrange_t get_window_size() const override {
        return { iceildiv(_args._Nsize, strategy::out_width()) * _args._nmulti };
    }

    // Actually execute the GEMV.
    void execute(const ndcoord_t &work_range, const ndcoord_t &, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_args._ci);

        auto& g_arrays = this->_gemm_arrays;

        const auto start = work_range.get_position(0);
        const auto end   = work_range.get_position_end(0);

        /* Break the window values down into multis of interest... */
        const unsigned int window_per_multi = iceildiv(_args._Nsize, strategy::out_width());
        const unsigned int multi_0    = start / window_per_multi;
        const unsigned int multi_end  = end   / window_per_multi;

        /* ... and figure out where we start and end in the first and last multi. */
        const unsigned int n_0   = (start - (multi_0 * window_per_multi)) * strategy::out_width();
        const unsigned int n_max = (end - (multi_end * window_per_multi)) * strategy::out_width();

        static_assert(std::is_same<Tr, Tri>::value, "GemvPretransposed: Result types must be the same.");

        for (unsigned int multi=multi_0; multi<=multi_end; multi++) {
            const unsigned int n_start = (multi==multi_0) ? n_0 : 0;
            const unsigned int n_end = (multi==multi_end) ? n_max : _args._Nsize;

            if (n_end <= n_start)
                continue;

            for (unsigned int k0=0; k0<_args._Ksize; k0+=k_block) {
                unsigned int kmax = std::min(k0 + k_block, _args._Ksize);

                for (unsigned int n=n_start; n<n_end; n+=n_block) {
                    unsigned int nmax = std::min(n + n_block, n_end);
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (kmax-k0) * (nmax-n));
#endif
                    KernelArgs ka{};
                    ka.M = 1;
                    ka.N = nmax - n;
                    ka.K = kmax - k0;
                    ka.lhs = g_arrays._Aptr + multi * g_arrays._A_multi_stride + k0;
                    ka.rhs = _B_transposed + multi * _buffer_per_multi +
                             n * roundup(_args._Ksize, strategy::k_unroll()) + k0 * strategy::out_width();
                    ka.dst = g_arrays._Cptr + multi * g_arrays._C_multi_stride + n;
                    ka.bias = (g_arrays._bias && k0 == 0 && !_args._accumulate) ?
                        g_arrays._bias + multi * g_arrays._bias_multi_stride + n : nullptr;
                    ka.accumulate = (k0 != 0) || _args._accumulate;
                    ka.clamp_min = std::is_integral<Tri>::value ? std::numeric_limits<Tri>::lowest() :
                        static_cast<Tri>(-std::numeric_limits<float>::infinity());
                    ka.clamp_max = std::is_integral<Tri>::value ? std::numeric_limits<Tri>::max() :
                        static_cast<Tri>(std::numeric_limits<float>::infinity());
                    if (kmax == _args._Ksize) {
                        switch (_args._act.type) {
                            default:
                            case Activation::Type::None:
                                break;
                            case Activation::Type::BoundedReLU:
                                ka.clamp_max = static_cast<Tri>(_args._act.param1);
                                /* fall through */
                            case Activation::Type::ReLU:
                                ka.clamp_min = static_cast<Tri>(0);
                        }
                    }
                    this->set_kernel_output_args(ka, multi, n, _os);
                    strat.kernel(&ka);
                }
            }
        }
    }

    /* Pretransposed interface implementation */
    size_t get_B_pretransposed_array_size() const override {
        return _buffer_per_multi * _args._nmulti * sizeof(Troi) + get_col_sum_size();
    }

    GemmConfig get_config() override {
        return this->get_config_common(k_block, n_block);
    }
};

}  // namespace ops
}  // namespace kai
