//
// SPDX-FileCopyrightText: Copyright 2017-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

#include "gemm_driver_common.hpp"
#include "convolver.hpp"
#include "kernel_weight_format.hpp"
#include "kai/ops/gemm/ndrange.hpp"
#include "kai/ops/float16.hpp"
#include "performance_parameters.hpp"
#include "common_internal/quantized.hpp"
#include "common_internal/utils.hpp"

#ifdef CYCLE_PROFILING
#include "kai/ops/profiler.hpp"
#endif

namespace kai {
namespace ops {

// FixedFormat describes the kernel layout; args._fixed_format describes the
// supplied weights. Ordinary weights are packed into that layout on setup.
// Implementation of the GemmCommon abstract class.
template<typename strategy, typename To, typename Tw, typename Tr, typename OutputStage=Nothing, bool SeparateQuantize=false, bool FixedFormat=false>
class GemmHybridIndirect : public GemmDriverCommon<strategy, To, Tw, Tr, OutputStage, typename strategy::rhs_operand_type, FixedFormat> {
    using Base = GemmDriverCommon<strategy, To, Tw, Tr, OutputStage, typename strategy::rhs_operand_type, FixedFormat>;
    using Base::_args;
    using Base::_os;
    using Base::_B_transposed;
    using Base::col_bias;
    using Base::get_col_sum_size;
    using Base::_Ktotal;
    using Base::_rounded_Ksize;
    using Base::get_ktotal;

    typedef typename strategy::lhs_operand_type Tloi;
    typedef typename strategy::rhs_operand_type Troi;
    typedef typename strategy::result_type Tri;

    /* Blocking info */
    const int64_t _k_block;
    const int64_t _n_block;
    const int64_t _Mround;

    /* Indirect parameters.  _indirect_buf doubles as a flag to indicate that "indirect" transform should be used. */
    const To * const * const * _indirect_buf = nullptr;

    /* Convolver - only set up for convolution problems, so also doubles as a flag. */
    std::unique_ptr<convolver<To>>  _convolver = nullptr;

    // Array of pointers to output rows
//    Tr * const *        _output_ptrs;

    const NDRange<4> _window_range;

    // Ordinary output, including kernels which fuse requantization.
    void execute_kernel(
#ifdef CYCLE_PROFILING
                        profiler &,
#endif
                        const strategy &strat, KernelArgs ka,
                        int64_t, int64_t, std::false_type) const {
        strat.kernel(&ka);
    }

    // C++11 tag dispatch keeps the Requantize32-only path out of other
    // output-stage instantiations. Input setup is shared through KernelArgs.
    void execute_kernel(
#ifdef CYCLE_PROFILING
                        profiler &prof,
#endif
                        const strategy &strat, KernelArgs ka,
                        int64_t multi, int64_t n0, std::true_type) const {
        const Requantize32 &qp = _os;
        assert(ka.M <= strategy::out_height());
        const int64_t width = roundup<int64_t>(ka.N, strategy::out_width());
        auto *result = reinterpret_cast<int32_t *>(alloca(width * strategy::out_height() * sizeof(int32_t)));
        int32_t row_sums[strategy::out_height()];
        Tr *dst = static_cast<Tr *>(ka.dst);
        const int64_t dst_stride = ka.dst_stride / sizeof(Tr);
        ka.dst = result;
        ka.dst_stride = width * sizeof(int32_t);
        ka.bias = nullptr;
        ka.accumulate = false;
        ka.clamp_min = std::numeric_limits<int32_t>::lowest();
        ka.clamp_max = std::numeric_limits<int32_t>::max();
        strat.kernel(&ka);
        if (qp.b_offset) {
#ifdef CYCLE_PROFILING
            // Count the unpadded input elements consumed by the row sums.
            int64_t depth = ka.K;
            if (ka.K_lengths) {
                depth = 0;
                for (int64_t section = 0; section < ka.K; ++section) {
                    depth += ka.K_lengths[section];
                }
            }
            auto p = prof.ScopedProfiler(PROFILE_ROWSUMS, ka.M * depth);
#endif
            row_sums_indirect<To>(&ka, row_sums, qp.b_offset);
        } else {
            std::fill(row_sums, row_sums + strategy::out_height(), 0);
        }
        {
#ifdef CYCLE_PROFILING
            auto p = prof.ScopedProfiler(PROFILE_QUANTIZE, ka.M * ka.N);
#endif
            requantize_block_32(qp, ka.N, ka.M, result, width, dst, dst_stride, row_sums,
                               col_bias + multi * _args._Nsize + n0, n0);
        }
    }

    static int64_t compute_k_block(const GemmArgs &args) {
        // Some kernels don't support accumulate mode - these can't do K blocking at all.
        if (!strategy::supports_accumulate() || std::is_same<OutputStage, Requantize32>::value) {
            return get_ktotal(args);
        }

        if (args._cfg && args._cfg->inner_block_size) {
            return roundup<int64_t>(args._cfg->inner_block_size, strategy::k_unroll());
        }

        // Hybrid FP16 or BF16 kernels running in non-fast mode can't do K
        // blocking by default due to accuracy issues (but allow explicitly
        // configured blocking to work with the clause above).
        if (std::is_same<To, __fp16>::value && std::is_same<Tr, __fp16>::value && !args._fast_mode) {
            return get_ktotal(args);
        }

        if (std::is_same<To, bfloat16>::value && std::is_same<Tr, bfloat16>::value && !args._fast_mode) {
            return get_ktotal(args);
        }

        // Experimental data suggests an optimal block size of 512 for FP32 (scaling accordingly for other
        // datatypes); but don't divide into blocks until we hit 1.5X this size.
        int64_t target_block_size = 2048 / sizeof(To);
        const int64_t ktotal = get_ktotal(args);

        if (ktotal > ((target_block_size*3)/2)) {
            int64_t target_blocks = iceildiv<int64_t>(ktotal, target_block_size);

            int64_t block_size = iceildiv<int64_t>(ktotal, target_blocks);

            block_size = roundup<int64_t>(block_size, strategy::k_unroll());

            return block_size;
        }

        return ktotal;
    }

    // New N blocking strategy: if it's narrow, or much taller than it is wide, do the full width.  Otherwise do a
    // single block.
    static int64_t compute_n_block(const GemmArgs &args, const OutputStage os = {}) {
        if (args._cfg && args._cfg->outer_block_size) {
            return args._cfg->outer_block_size;
        }

        if (args._Nsize <= 64) {
            return args._Nsize;
        }

        if ((args._Msize / args._Nsize) > 155) {
            return args._Nsize;
        }

        // "Asymmetric" quantizing GEMMs require a different approach - the tall skinny blocks we would otherwise
        // use imply a great deal of repeated work performing the row sums.  If row sums are involved, work out how
        // much "column" parallelism is going to be required and set the block size accordingly.
        if (std::is_same<OutputStage, Requantize32>::value) {
            const Requantize32 *qp = reinterpret_cast<const Requantize32 *>(&os);

            // Row sums only needed if b_offset isn't 0
            if (qp->b_offset != 0) {
                // We can already parallelize across batches, multis and rows (in units of 'out_height')
                int64_t multi_row_parallelism = static_cast<int64_t>(args._nmulti) * args._nbatches * iceildiv<int64_t>(args._Msize, strategy::out_height());

                // If this isn't enough, we will need to split up the columns too.
                if (multi_row_parallelism < args._maxthreads) {
                    int64_t columns_needed = iceildiv<int64_t>(args._maxthreads, multi_row_parallelism);

                    int64_t n_block = iceildiv<int64_t>(args._Nsize, columns_needed);

                    return roundup<int64_t>(n_block, strategy::out_width());
                }

                // Multi/Batch/Row parallelism is enough - don't split up the columns.
                return args._Nsize;
            }
        }

        if (args._Ksize <= 128 && args._maxthreads <= 16) {
            return strategy::out_width() * 3;
        }

        return strategy::out_width();
    }

    // Return types follow the shared packing interface until it is widened.
    unsigned int get_rhs_packing_k_block() const override { return _k_block; }
    unsigned int get_rhs_packing_n_block() const override { return roundup<int64_t>(_n_block, strategy::out_width()); }

public:
    GemmHybridIndirect(GemmHybridIndirect &) = delete;
    GemmHybridIndirect & operator= (GemmHybridIndirect &) = delete;

    /* Constructor */
    GemmHybridIndirect(const GemmArgs &args, const OutputStage &os)
              : Base(args, os),
                _k_block(compute_k_block(args)), _n_block(compute_n_block(args, os)),
                _Mround(roundup<int64_t>(args._Msize, strategy::out_height())),
                _window_range(static_cast<unsigned int>(iceildiv<int64_t>(args._Msize, strategy::out_height())), args._nbatches,
                              static_cast<unsigned int>(iceildiv<int64_t>(args._Nsize, _n_block)), args._nmulti)
    {
        assert(!_args._fixed_format || FixedFormat);
    }

    /* Constructor without OutputStage */
    GemmHybridIndirect(const GemmArgs &args)
              : GemmHybridIndirect(args, OutputStage{}) {}

    // Interface implementation - Compulsory functions
    ndrange_t get_window_size() const override {
        return { _window_range.total_size() };
    }

    // This kernel can always be dynamically scheduled.
    bool supports_dynamic_scheduling() const override {
        return true;
    }

    // Common execution logic.
    void execute_common(const ndcoord_t &work_range, const ndcoord_t &, int, GemmArrays<To, Tw, Tr>& g_arrays) {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_args._ci);

        std::vector<const To *>         in_row_ptrs;
        std::vector<int64_t>            kernel_lengths;
        std::vector<const To * const *> kernel_strings;

        // Indirect and convolution modes share the section pointers and lengths.
        if (_args._indirect_input) {
            kernel_lengths.resize(_args._Ksections);
            kernel_strings.resize(_args._Ksections);
        }

        // In convolution mode, we need input pointers.
        if (_convolver) {
            in_row_ptrs = std::vector<const To *>(static_cast<int64_t>(strategy::out_height()) * _args._Ksections, nullptr);

            for (int64_t i=0; i<_args._Ksections; i++) {
                kernel_strings[i] = &(in_row_ptrs.data()[i * strategy::out_height()]);
            }
        }

        /* Make sure we've been set up correctly. */
        assert(_args._fixed_format || _B_transposed);
        static_assert(std::is_same<To, Tloi>::value, "gemm_native: Operand types must be the same.");

        const int64_t rounded_width = roundup<int64_t>(_args._Nsize, strategy::out_width());
        const Troi *rhs_base = _args._fixed_format ? reinterpret_cast<const Troi *>(g_arrays._Bptr) : _B_transposed;
        const int64_t rhs_multi_stride = _args._fixed_format ? g_arrays._B_multi_stride : rounded_width * _Ktotal;
        const int64_t rhs_row_stride = _args._fixed_format ? g_arrays._ldb : static_cast<int64_t>(_Ktotal) * driver_stripe_width<strategy, FixedFormat>::get();

        /* For now, each work item implies all the K for a given output
         * pixel (so we don't need to synchronize access to the output
         * array).  So separate the loop over K blocks here.  */
        for (int64_t k0=0; k0<_Ktotal; k0+=_k_block) {
            int64_t kmax   = std::min<int64_t>(k0 + _k_block, _Ktotal);
            int64_t kern_k = roundup<int64_t>(kmax-k0, strategy::k_unroll());

            const bool first_pass = (k0 == 0);
            const bool last_pass = (kmax == _Ktotal);

            int64_t first_section = (k0 / _rounded_Ksize);
            int64_t first_offset  = (k0 % _rounded_Ksize);
            int64_t kleft = kern_k;
            int64_t sections=0;
            int64_t offset = first_offset;

            if (_args._indirect_input) {
                while (kleft) {
                    // When chopping into sections: the amount that goes into 'kernel_lengths' is the amount to be
                    // processed (excluding padding).  But the amount we subtract from 'kleft' takes account of any
                    // padding applied.
                    kernel_lengths[sections] = std::min<int64_t>(kleft, _args._Ksize - offset);
                    kleft -= std::min<int64_t>(kleft, _rounded_Ksize - offset);
                    sections++;
                    offset=0;
                }
            }

            auto p = _window_range.iterator(work_range.get_position(0), work_range.get_position_end(0));

            if (p.done()) {
                return;
            }

            // Process rows either 'out_height' rows at a time, or do all valid rows at once with a single kernel call.
            // The separate quantizer path only handles one block of rows at a time (as it has to store sums and intermediate results).
            // THe convolution path only generates the pointers for one block of rows at a time.
            const bool process_all_rows = (!SeparateQuantize && !_convolver);

            do {
                const int64_t m_start = static_cast<int64_t>(p.dim(0)) * strategy::out_height();
                const int64_t m_end   = process_all_rows ? std::min<int64_t>(static_cast<int64_t>(p.dim0_max()) * strategy::out_height(), _args._Msize) : std::min<int64_t>(m_start + strategy::out_height(), _args._Msize);
                const int64_t batch   = p.dim(1);
                const int64_t n0      = p.dim(2) * _n_block;
                const int64_t nmax    = std::min<int64_t>(n0 + _n_block, _args._Nsize);
                const int64_t multi   = p.dim(3);

                const Troi *b_panel;
                if (FixedFormat) {
                    b_panel = rhs_base +
                               (multi * rhs_multi_stride) +
                               ((n0 / driver_stripe_width<strategy, FixedFormat>::get()) * rhs_row_stride) +
                               (k0 * driver_stripe_width<strategy, FixedFormat>::get());
                } else {
                    b_panel = _B_transposed +
                               (multi * roundup<int64_t>(_args._Nsize, strategy::out_width()) * _Ktotal) +
                               (k0 * roundup<int64_t>(_args._Nsize, strategy::out_width())) +
                               (n0 * kern_k);
                }

                KernelArgs ka{};
                ka.M = m_end - m_start;
                ka.N = nmax - n0;

#ifdef CYCLE_PROFILING
                auto p = prof.ScopedProfiler(PROFILE_KERNEL, (m_end - m_start) * kern_k * roundup<int64_t>(nmax-n0, strategy::out_width()));
#endif
                if (_indirect_buf) {
                    const auto strings = _indirect_buf + (multi * _args._nbatches + batch) * _args._Ksections + first_section;
                    for (int64_t section = 0; section < sections; ++section) {
                        kernel_strings[section] = strings[section] + m_start;
                    }
                    ka.lhs = kernel_strings.data();
                    // Indirect offsets are in elements, and apply only to the first section.
                    ka.lhs_stride = first_offset;
                    ka.K = sections;
                    ka.K_lengths = kernel_lengths.data();
                } else if (_convolver) {
                    auto conv_cols = _convolver->process_columns(g_arrays._Aptr + (multi * g_arrays._A_multi_stride) + (batch * g_arrays._A_batch_stride), g_arrays._lda, k0, kmax, _rounded_Ksize);

                    int64_t pos=0;
                    auto conv_rows = conv_cols.process_rows(m_start, m_end - m_start);

                    while (!conv_rows.finished()) {
                        int64_t width, conv_offset;

                        assert(pos < sections);

                        std::tie(width, conv_offset) = conv_rows.next_block(&(in_row_ptrs[pos * strategy::out_height()]));

                        if (pos==0) {
                            assert(conv_offset == first_offset);
                        }
                        assert(width == kernel_lengths[pos]);
                        pos++;
                    }
                    assert(pos == sections);

                    ka.lhs = kernel_strings.data();
                    ka.lhs_stride = first_offset;
                    ka.K = sections;
                    ka.K_lengths = kernel_lengths.data();
                } else {
                    ka.lhs = g_arrays._Aptr + multi * g_arrays._A_multi_stride +
                        batch * g_arrays._A_batch_stride + m_start * g_arrays._lda + k0;
                    ka.lhs_stride = g_arrays._lda * sizeof(To);
                    // Direct depth excludes any padding in the last K block.
                    ka.K = std::min<int64_t>(_args._Ksize, kmax) - k0;
                }
                ka.rhs = b_panel;
                ka.rhs_stride = (FixedFormat ? rhs_row_stride : kern_k * strategy::out_width()) * sizeof(Troi);
                ka.dst = g_arrays._Cptr + multi * g_arrays._C_multi_stride + batch * g_arrays._C_batch_stride +
                         m_start * g_arrays._ldc + n0;
                ka.dst_stride = g_arrays._ldc * sizeof(Tr);
                ka.bias = (g_arrays._bias && first_pass && !_args._accumulate) ?
                    g_arrays._bias + multi * g_arrays._bias_multi_stride + n0 : nullptr;
                ka.accumulate = !first_pass || _args._accumulate;
                ka.clamp_min = std::is_integral<Tri>::value ? std::numeric_limits<Tri>::lowest() :
                    static_cast<Tri>(-std::numeric_limits<float>::infinity());
                ka.clamp_max = std::is_integral<Tri>::value ? std::numeric_limits<Tri>::max() :
                    static_cast<Tri>(std::numeric_limits<float>::infinity());
                if (last_pass) {
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
                if (!SeparateQuantize) {
                    this->set_kernel_output_args(ka, multi, n0, _os);
                }
                execute_kernel(
#ifdef CYCLE_PROFILING
                    prof,
#endif
                    strat, ka, multi, n0, std::integral_constant<bool, SeparateQuantize>{});
            } while (process_all_rows ? p.next_dim1() : p.next_dim0());
        }
    }

    // Stateless execute
    void execute_stateless(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid, GemmArrays<To, Tw, Tr>& g_arrays) override {
        assert(_args._fixed_format);

        return execute_common(work_range, thread_locator, threadid, g_arrays);
    }

    // Execute
    void execute(const ndcoord_t &work_range, const ndcoord_t & thread_locator, int threadid) override {
        execute_common(work_range, thread_locator, threadid, this->_gemm_arrays);
    }

    // Interface implementation - pretransposed
    size_t get_B_pretransposed_array_size() const override {
        if (_args._fixed_format) {
            return 0;
        }

        // Start with actual pretransposed buffer...
        size_t size = roundup<int64_t>(_args._Nsize, strategy::out_width()) * _Ktotal * _args._nmulti * sizeof(Troi);

        // Space for result row pointers retained for existing buffer sizing.
        size += static_cast<int64_t>(_args._Msize) * _args._nbatches * _args._nmulti * sizeof(const Tr *);

        if (std::is_same<OutputStage, Requantize32>::value) {
            size += get_col_sum_size();
        }

        return size;
    }

    bool B_pretranspose_supports_transpose() const override {
        strategy strat(_args._ci);
        return strat.transforms.PrepareB_supports_transpose();
    }

    // Estimate cycles for given problem given provided parameters.
    // "perf_type" is a type to pass along to get_performance_parameters to get the right set of performance
    // parameters - it's arbitrary but usually either the input or output type.
    template <typename perf_type>
    static uint64_t estimate_cycles(const GemmArgs &args, const OutputStage &os = {}) {
        const PerformanceParameters params = strategy::template get_performance_parameters<perf_type>(args._ci);

        // Note: Current hybrid kernels don't actually round up height (they
        // have paths for each possible height).  Might need to make this
        // configurable in future.
        uint64_t total_macs = static_cast<uint64_t>(args._nbatches) * args._nmulti * args._Msize * roundup<int64_t>(args._Nsize, strategy::out_width()) * get_ktotal(args);

        float mac_cycles = static_cast<float>(total_macs) / params.kernel_macs_cycle;

        // TODO: A bit of a kludge here: current hybrid kernels incur extra
        // overhead where the width is not a multiple of kernel width.  It's
        // most noticable where the overall width is quite low, so add 15%
        // penalty for such widths.
        if ((args._Nsize < strategy::out_width()) || (args._Nsize > strategy::out_width() && args._Nsize < 2*strategy::out_width())) {
            mac_cycles *= 1.15f;
        }

        uint64_t total_cycles = mac_cycles;

        // Quantizing kernels with separate quantize need to add in the extra stages.
        if (std::is_same<OutputStage, Requantize32>::value && SeparateQuantize) {
            const Requantize32 *qp = reinterpret_cast<const Requantize32 *>(&os);

            // Row sums: need to consider each value in A (batch * multi * M * K)...
            uint64_t rowsum_bytes = static_cast<uint64_t>(args._nbatches) * args._nmulti * args._Msize * get_ktotal(args);

            // ... but row sums are skipped if B offset==0.
            if (qp->b_offset == 0) {
                rowsum_bytes = 0;
            }

            // Use "prepare bytes per cycle" to store "row sum values per cycle".
            float rowsum_cycles = static_cast<float>(rowsum_bytes) / params.prepare_bytes_cycle;

            // Requantize: need to consider each value in C (batch * multi * M * N)
            uint64_t requantize_bytes = static_cast<uint64_t>(args._nbatches) * args._nmulti * args._Msize * args._Nsize;

            // Use "merge bytes per cycle" to store "requantize values per cycle".
            float requantize_cycles = static_cast<float>(requantize_bytes) / params.merge_bytes_cycle;

            // Recalculate total_cycles with the extra components.
            total_cycles = mac_cycles + rowsum_cycles + requantize_cycles;
        }

        return total_cycles;
    }

    void set_indirect_parameters(size_t string_len, const To * const * const *ptr) override {
        assert(string_len == _args._Ksize);
        _indirect_buf = ptr;
    }

    void set_convolution_parameters(ConvolutionParameters parms) override {
        assert(parms.input_channels == _args._Ksize);
        _convolver = std::unique_ptr<convolver<To>>(new convolver<To>(parms));
    }

    GemmConfig get_config() override {
        return this->get_config_common(_k_block, _n_block);
    }
};

template<typename strategy, typename To, typename Tr, typename OutputStage=Nothing>
using GemmHybridIndirectFixedFormat = GemmHybridIndirect<strategy, To, To, Tr, OutputStage, false, true>;

}  // namespace ops
}  // namespace kai
