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

#include "kai/ops/gemm/kai_ops.hpp"
#include "convolver.hpp"
#include "kernel_weight_format.hpp"
#include "kai/ops/gemm/ndrange.hpp"
#include "performance_parameters.hpp"
#include "common_internal/utils.hpp"

#ifdef CYCLE_PROFILING
#include "kai/ops/profiler.hpp"
#endif

#ifndef UNUSED
#define __I_DEFINED_UNUSED
#define UNUSED(x)  ((void)(x))
#endif

namespace kai {
namespace ops {

namespace {

// We need to invoke the kernel differently for quantizing and non-quantizing cases, so here is a shim class to do
// that.

template<typename OutputStage, bool SeparateQuantize, bool FixedFormat>
class run_hybrid_kernel {
public:
    template<typename strategy, typename Tlo, typename Tro, typename Tr>
    static inline void run (
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        const strategy &strat, unsigned int num_strings, const unsigned int *string_ptr, IndirectInputArg<Tlo> A_arg, unsigned int M, unsigned int N,
        unsigned int kern_k, const Tro *b_ptr, size_t b_stride, OutputArg<Tr> output_arg, const Tr *bias_ptr, Activation act, bool accumulate,
        const OutputStage &os, const int32_t *col_bias, unsigned int n_0 );
};

template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr>
inline void run_hybrid_kernel<Nothing, false, false>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        const strategy &strat, unsigned int num_strings, const unsigned int *string_ptr, IndirectInputArg<Tlo> A_arg, unsigned int M, unsigned int N,
        unsigned int kern_k, const Tro *b_ptr, size_t, OutputArg<Tr> output_arg, const Tr *bias_ptr, Activation act, bool accumulate,
        const Nothing &, const int32_t *, unsigned int) {
#ifdef CYCLE_PROFILING
    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)M * kern_k * roundup(N, strategy::out_width()));
#endif
    UNUSED(kern_k);

    /* Indirect hybrid kernels read the full width of the bias.  So we need to detect the case where we are writing
     * a partial block and pad the bias for that block. */
    if (bias_ptr && !accumulate && (N % strategy::out_width() != 0)) {
        /* Break N into "N_bulk" (a multiple of output width) and "N_remainder" */
        unsigned int N_remainder = N % strategy::out_width();
        unsigned int N_bulk = N - N_remainder;

        /* Output argument to be used for the tail */
        OutputArg<Tr> offset_output = output_arg;

        /* If there is a "bulk" to be processed, handle that and update "offset_output" appropriately. */
        if (N_bulk > 0) {
            strat.kernel(num_strings, string_ptr, A_arg, M, N_bulk, b_ptr, output_arg, bias_ptr, act, accumulate);

            offset_output = OutputArg<Tr>(output_arg.base + N_bulk, output_arg.stride);
        }

        /* Pad the bias buffer for the remainder */
        Tr *bias_pad_buffer = reinterpret_cast<Tr *>(alloca(strategy::out_width() * sizeof(Tr)));
        memcpy(bias_pad_buffer, bias_ptr + N_bulk, N_remainder * sizeof(Tr));

        /* Process the remainder, offsetting the B pointer as needed. */
        strat.kernel(num_strings, string_ptr, A_arg, M, N_remainder, b_ptr + (N_bulk * kern_k), offset_output, bias_pad_buffer, act, accumulate);
    } else {
        strat.kernel(num_strings, string_ptr, A_arg, M, N, b_ptr, output_arg, bias_ptr, act, accumulate);
    }
}

template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr>
inline void run_hybrid_kernel<Nothing, false, true>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        const strategy &strat, unsigned int num_strings, const unsigned int *string_ptr, IndirectInputArg<Tlo> A_arg, unsigned int M, unsigned int N,
        unsigned int kern_k, const Tro *b_ptr, size_t b_stride, OutputArg<Tr> output_arg, const Tr *bias_ptr, Activation act, bool accumulate,
        const Nothing &, const int32_t *, unsigned int) {
#ifdef CYCLE_PROFILING
    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)M * kern_k * roundup(N, strategy::out_width()));
#endif
    UNUSED(kern_k);

    /* Indirect hybrid kernels read the full width of the bias.  So we need to detect the case where we are writing
     * a partial block and pad the bias for that block. */
    if (bias_ptr && !accumulate && (N % strategy::out_width() != 0)) {
        /* Break N into "N_bulk" (a multiple of output width) and "N_remainder" */
        unsigned int N_remainder = N % strategy::out_width();
        unsigned int N_bulk = N - N_remainder;

        /* Output argument to be used for the tail */
        OutputArg<Tr> offset_output = output_arg;

        /* If there is a "bulk" to be processed, handle that and update "offset_output" appropriately. */
        if (N_bulk > 0) {
            strat.kernel(num_strings, string_ptr, A_arg, M, N_bulk, b_ptr, b_stride, output_arg, bias_ptr, act, accumulate);

            offset_output = OutputArg<Tr>(output_arg.base + N_bulk, output_arg.stride);
        }

        /* Pad the bias buffer for the remainder */
        Tr *bias_pad_buffer = reinterpret_cast<Tr *>(alloca(strategy::out_width() * sizeof(Tr)));
        memcpy(bias_pad_buffer, bias_ptr + N_bulk, N_remainder * sizeof(Tr));

        /* Process the remainder, offsetting the B pointer as needed. */
        strat.kernel(num_strings, string_ptr, A_arg, M, N_remainder,
                     b_ptr + (N_bulk / strategy::stripe_width()) * b_stride, b_stride, offset_output,
                     bias_pad_buffer, act, accumulate);
    } else {
        strat.kernel(num_strings, string_ptr, A_arg, M, N, b_ptr, b_stride, output_arg, bias_ptr, act, accumulate);
    }
}

template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr>
inline void run_hybrid_kernel<Requantize32, false, false>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        const strategy &strat, unsigned int num_strings, const unsigned int *string_ptr, IndirectInputArg<Tlo> A_arg, unsigned int M, unsigned int N,
        unsigned int kern_k, const Tro *b_ptr, size_t, OutputArg<Tr> output_arg, const Tr *, Activation, bool,
        const Requantize32 &os, const int32_t *col_bias, unsigned int n_0 ) {
#ifdef CYCLE_PROFILING
    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)M * kern_k * roundup(N, strategy::out_width()));
#endif
    UNUSED(kern_k);

    strat.kernel(num_strings, string_ptr, A_arg, M, N, b_ptr, output_arg, &os, col_bias + n_0, n_0);
}

template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr>
inline void run_hybrid_kernel<Requantize32, true, false>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        const strategy &strat, unsigned int num_strings, const unsigned int *string_ptr, IndirectInputArg<Tlo> A_arg, unsigned int M, unsigned int N,
        unsigned int kern_k, const Tro *b_ptr, size_t, OutputArg<Tr> output_arg, const Tr *, Activation, bool,
        const Requantize32 &os, const int32_t *col_bias, unsigned int n_0 ) {
    UNUSED(kern_k);
    // On this route we will only process one kernel height at a time and will make sure this happens in the driver loop.
    assert(M <= strategy::out_height());
    // We need a row sum buffer and intermediate output buffer.
    // These go on the stack as they are not too large, using an automatic array and alloca() respectively.
    int32_t row_sums[strategy::out_height()];
    typename strategy::result_type *result_buffer;

    unsigned int output_width = roundup(N, strategy::out_width());

    result_buffer = reinterpret_cast<typename strategy::result_type *>(alloca(output_width * strategy::out_height() * sizeof(typename strategy::result_type)));

    {
#ifdef CYCLE_PROFILING
        auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)M * kern_k * roundup(N, strategy::out_width()));
#endif
        // Perform the GEMM, into the output buffer.
        strat.kernel(num_strings, string_ptr, A_arg, M, N, b_ptr, OutputArg<typename strategy::result_type>(result_buffer, output_width), nullptr, Activation(), false);
    }

    if (os.b_offset != 0) {
#ifdef CYCLE_PROFILING
        auto p = prof.ScopedProfiler(PROFILE_ROWSUMS, (unsigned long)M * kern_k);
#endif
        row_sums_indirect(num_strings, string_ptr, A_arg, M, row_sums, &os);
    } else {
        memset(row_sums, 0, sizeof(int32_t) * strategy::out_height());
    }

    {
#ifdef CYCLE_PROFILING
        auto p = prof.ScopedProfiler(PROFILE_QUANTIZE, (unsigned long)M * N);
#endif
        // Quantize
        requantize_block_32(os, N, M, result_buffer, output_width, output_arg.base, output_arg.stride, row_sums, col_bias + n_0, n_0);
    }
}

template<typename strategy, bool FixedFormat>
struct stripe_width {
    static unsigned int get() {
        return strategy::stripe_width();
    }
};

template<typename strategy>
struct stripe_width<strategy, false> {
    static unsigned int get() {
        return 0;
    }
};

template<typename strategy, bool FixedFormat>
struct kernel_weight_format {
    static KernelWeightFormat get() {
        return strategy::kernel_weight_format();
    }
};

template<typename strategy>
struct kernel_weight_format<strategy, false> {
    static KernelWeightFormat get() {
        return KernelWeightFormat::NON_FIXED;
    }
};

} // anonymous namespace

// Implementation of the GemmCommon abstract class.
template<typename strategy, typename To, typename Tw, typename Tr, typename OutputStage=Nothing, bool SeparateQuantize=false, bool FixedFormat=false>
class GemmHybridIndirect : public GemmCommon<To, Tw, Tr> {
    typedef typename strategy::lhs_operand_type Tloi;
    typedef typename strategy::rhs_operand_type Troi;
    typedef typename strategy::result_type Tri;

    GemmArgs           _args;
    OutputStage        _os = {};

    /* Quantized support (in addition to 'output stage' above) */
    int32_t *_col_bias = nullptr;

    const unsigned int _Ktotal;
    const unsigned int _rounded_Ksize;

    /* Blocking info */
    const unsigned int _k_block;
    const unsigned int _n_block;
    const unsigned int _Mround;

    /* Pretransposed buffer. */
    const Troi *_B_transposed=nullptr;

    /* Indirect parameters.  _indirect_buf doubles as a flag to indicate that "indirect" transform should be used. */
    const To * const * const * _indirect_buf = nullptr;

    /* Convolver - only set up for convolution problems, so also doubles as a flag. */
    std::unique_ptr<convolver<To>>  _convolver = nullptr;

    // Array of pointers to output rows
//    Tr * const *        _output_ptrs;

    const NDRange<4> _window_range;

    unsigned int get_col_sum_size() const {
        if (std::is_same<OutputStage, Requantize32>::value) {
            return _args._Nsize * _args._nmulti * sizeof(int32_t);
        } else {
            return 0;
        }
    }

    static unsigned int get_ktotal(const GemmArgs &args) {
        return args._Ksections * roundup(args._Ksize, strategy::k_unroll());
    }

    static unsigned int compute_k_block(const GemmArgs &args) {
        // Some kernels don't support accumulate mode - these can't do K blocking at all.
        if (!strategy::supports_accumulate() || std::is_same<OutputStage, Requantize32>::value) {
            return get_ktotal(args);
        }

        if (args._cfg && args._cfg->inner_block_size) {
            return roundup(args._cfg->inner_block_size, strategy::k_unroll());
        }

        // Hybrid FP16 kernels running in non-fast mode can't do K blocking
        // by default due to accuracy issues (but allow explicitly
        // configured blocking to work with the clause above).
        if (std::is_same<To, __fp16>::value && std::is_same<Tr, __fp16>::value && !args._fast_mode) {
            return get_ktotal(args);
        }

        // Experimental data suggests an optimal block size of 512 for FP32 (scaling accordingly for other
        // datatypes); but don't divide into blocks until we hit 1.5X this size.
        unsigned int target_block_size = 2048 / sizeof(To);
        auto ktotal = get_ktotal(args);

        if (ktotal > ((target_block_size*3)/2)) {
            unsigned int target_blocks = iceildiv(ktotal, target_block_size);

            unsigned int block_size = iceildiv(ktotal, target_blocks);

            block_size = roundup(block_size, strategy::k_unroll());

            return block_size;
        }

        return ktotal;
    }

    // New N blocking strategy: if it's narrow, or much taller than it is wide, do the full width.  Otherwise do a
    // single block.
    static unsigned int compute_n_block(const GemmArgs &args, const OutputStage os = {}) {
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
                int multi_row_parallelism = args._nmulti * args._nbatches * iceildiv(args._Msize, strategy::out_height());

                // If this isn't enough, we will need to split up the columns too.
                if (multi_row_parallelism < args._maxthreads) {
                    unsigned int columns_needed = iceildiv(args._maxthreads, multi_row_parallelism);

                    unsigned int n_block = iceildiv(args._Nsize, columns_needed);

                    return roundup(n_block, strategy::out_width());
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

public:
    GemmHybridIndirect(GemmHybridIndirect &) = delete;
    GemmHybridIndirect & operator= (GemmHybridIndirect &) = delete;

    /* Constructor */
    GemmHybridIndirect(const GemmArgs &args, const OutputStage &os)
              : _args(args), _os(os), _Ktotal(get_ktotal(args)),
                _rounded_Ksize(roundup(args._Ksize, strategy::k_unroll())),
                _k_block(compute_k_block(args)), _n_block(compute_n_block(args, os)),
                _Mround(roundup(args._Msize, strategy::out_height())),
                _window_range(iceildiv(args._Msize, strategy::out_height()), args._nbatches,
                              iceildiv(args._Nsize, _n_block), args._nmulti)
    {
        // We take a copy of the arguments (not a pointer or reference), but there is no lifetime requirement on the
        // GemmConfig.  Clear out the pointer to avoid accidents.
        _args._cfg = nullptr;
    }

    /* Constructor without OutputStage */
    GemmHybridIndirect(const GemmArgs &args)
              : _args(args), _Ktotal(get_ktotal(args)),
                _rounded_Ksize(roundup(args._Ksize, strategy::k_unroll())),
                _k_block(compute_k_block(args)), _n_block(compute_n_block(args)),
                _Mround(roundup(args._Msize, strategy::out_height())),
                _window_range(iceildiv(args._Msize, strategy::out_height()), args._nbatches,
                              iceildiv(args._Nsize, _n_block), args._nmulti)
    {
        // We take a copy of the arguments (not a pointer or reference), but there is no lifetime requirement on the
        // GemmConfig.  Clear out the pointer to avoid accidents.
        _args._cfg = nullptr;
    }

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
        std::vector<const To * const *> in_row_strings;
        std::vector<unsigned int>       string_lengths;

        // In convolution mode, we need input pointers.
        if (_convolver) {
            in_row_ptrs = std::vector<const To *>(strategy::out_height() * _args._Ksections, nullptr);
            in_row_strings = std::vector<const To * const *>(_args._Ksections, nullptr);

            for (unsigned int i=0; i<_args._Ksections; i++) {
                in_row_strings[i] = &(in_row_ptrs.data()[i * strategy::out_height()]);
            }
        }

        // In any indirect mode, we need the string lengths.
        if (_args._indirect_input) {
            string_lengths = std::vector<unsigned int>(_args._Ksections, 0);
        }

        /* Make sure we've been set up correctly. */
        assert(FixedFormat || _B_transposed);
        static_assert(std::is_same<To, Tloi>::value, "gemm_native: Operand types must be the same.");

        /* For now, each work item implies all the K for a given output
         * pixel (so we don't need to synchronize access to the output
         * array).  So separate the loop over K blocks here.  */
        for (unsigned int k0=0; k0<_Ktotal; k0+=_k_block) {
            unsigned int kmax   = std::min(k0 + _k_block, _Ktotal);
            unsigned int kern_k = roundup(kmax-k0, strategy::k_unroll());

            const bool first_pass = (k0 == 0);
            const bool last_pass = (kmax == _Ktotal);

            unsigned int first_section = (k0 / _rounded_Ksize);
            unsigned int first_offset  = (k0 % _rounded_Ksize);
            unsigned int kleft = kern_k;
            unsigned int sections=0;
            unsigned int offset = first_offset;

            if (_args._indirect_input) {
                while (kleft) {
                    // When chopping into sections: the amount that goes into 'string_lengths' is the amount to be
                    // processed (excluding padding).  But the amount we subtract from 'kleft' takes account of any
                    // padding applied.
                    string_lengths[sections] = std::min(kleft, _args._Ksize - offset);
                    kleft -= std::min(kleft, _rounded_Ksize - offset);
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
                const unsigned int m_start = p.dim(0) * strategy::out_height();
                const unsigned int m_end   = process_all_rows ? std::min(p.dim0_max() * strategy::out_height(), _args._Msize) : std::min(m_start + strategy::out_height(), _args._Msize);
//                const unsigned int m_end   = std::min(m_start + strategy::out_height(), _args._Msize);
                const unsigned int batch   = p.dim(1);
                const unsigned int n0      = p.dim(2) * _n_block;
                const unsigned int nmax    = std::min(n0 + _n_block, _args._Nsize);
                const unsigned int multi   = p.dim(3);


                // We need to select the appropriate part of the column bias
                // for quantized cases, but not manipulate a nullptr in
                // other cases.
                const int32_t * const offset_col_bias = (_col_bias==nullptr) ? nullptr : _col_bias+(multi * _args._Nsize);

                const Troi *b_panel;
                if (FixedFormat) {
                    b_panel = reinterpret_cast<const Troi *>(g_arrays._Bptr) +
                               (multi * g_arrays._B_multi_stride) +
                               ((n0 / stripe_width<strategy, FixedFormat>::get()) * g_arrays._ldb) +
                               (k0 * stripe_width<strategy, FixedFormat>::get());
                } else {
                    b_panel = _B_transposed +
                               (multi * roundup(_args._Nsize, strategy::out_width()) * _Ktotal) +
                               (k0 * roundup(_args._Nsize, strategy::out_width())) +
                               (n0 * kern_k);
                }

                OutputArg<Tr> out_arg(g_arrays._Cptr + (multi * g_arrays._C_multi_stride) + (batch * g_arrays._C_batch_stride) + (m_start * g_arrays._ldc) + n0, g_arrays._ldc);

#ifdef CYCLE_PROFILING
                auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(m_end - m_start) * kern_k * roundup(nmax-n0, strategy::out_width()));
#endif
                if (_indirect_buf) {
                    run_hybrid_kernel<OutputStage, SeparateQuantize, FixedFormat>::run(
#ifdef CYCLE_PROFILING
                                 prof,
#endif
                                 strat, sections, string_lengths.data(),
                                 IndirectInputArg<To>(_indirect_buf + (multi * _args._nbatches * _args._Ksections) + (batch * _args._Ksections) + first_section, m_start, first_offset),
                                 (m_end - m_start), (nmax - n0), kern_k, b_panel, g_arrays._ldb, out_arg,
                                 (g_arrays._bias && first_pass) ? g_arrays._bias + (multi * g_arrays._bias_multi_stride) + n0 : nullptr,
                                 last_pass ? _args._act : Activation(),
                                 !first_pass || _args._accumulate,
                                 // Quantization parameters
                                 _os, offset_col_bias, n0);
                } else if (_convolver) {
                    auto conv_cols = _convolver->process_columns(g_arrays._Aptr + (multi * g_arrays._A_multi_stride) + (batch * g_arrays._A_batch_stride), g_arrays._lda, k0, kmax, _rounded_Ksize);

                    unsigned int pos=0;
                    auto conv_rows = conv_cols.process_rows(m_start, m_end - m_start);

                    while (!conv_rows.finished()) {
                        unsigned int width, conv_offset;

                        assert(pos < sections);

                        std::tie(width, conv_offset) = conv_rows.next_block(&(in_row_ptrs[pos * strategy::out_height()]));

                        if (pos==0) {
                            assert(conv_offset == first_offset);
                        }
                        assert(width == string_lengths[pos]);
                        pos++;
                    }
                    assert(pos == sections);

                    run_hybrid_kernel<OutputStage, SeparateQuantize, FixedFormat>::run(
#ifdef CYCLE_PROFILING
                                 prof,
#endif
                                 strat, sections, string_lengths.data(),
                                 IndirectInputArg<To>(in_row_strings.data(), 0, first_offset),
                                 (m_end - m_start), (nmax - n0), kern_k, b_panel, g_arrays._ldb, out_arg,
                                 (g_arrays._bias && first_pass) ? g_arrays._bias + (multi * g_arrays._bias_multi_stride) + n0 : nullptr,
                                 last_pass ? _args._act : Activation(),
                                 !first_pass || _args._accumulate,
                                 // Quantization parameters
                                 _os, offset_col_bias, n0);
                } else {
                    // Length to process.  This needs to exclude padding, but 'kmax' potentially includes it.
                    const unsigned int len = (std::min(_args._Ksize, kmax) - k0);

                    run_hybrid_kernel<OutputStage, SeparateQuantize, FixedFormat>::run(
#ifdef CYCLE_PROFILING
                                 prof,
#endif
                                 strat, 1, &len,
                                 IndirectInputArg<To>(g_arrays._Aptr + (multi * g_arrays._A_multi_stride) + (batch * g_arrays._A_batch_stride) + m_start * g_arrays._lda + k0, g_arrays._lda),
                                 (m_end - m_start), (nmax - n0), kern_k, b_panel, g_arrays._ldb, out_arg,
                                 (g_arrays._bias && first_pass) ? g_arrays._bias + (multi * g_arrays._bias_multi_stride) + n0 : nullptr,
                                 last_pass ? _args._act : Activation(),
                                 !first_pass || _args._accumulate,
                                 // Quantization parameters
                                 _os, offset_col_bias, n0);
                }
            } while (process_all_rows ? p.next_dim1() : p.next_dim0());
        }
    }

    // Stateless execute
    void execute_stateless(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid, GemmArrays<To, Tw, Tr>& g_arrays) override {
        assert(FixedFormat);

        return execute_common(work_range, thread_locator, threadid, g_arrays);
    }

    // Execute
    void execute(const ndcoord_t &work_range, const ndcoord_t & thread_locator, int threadid) override {
        execute_common(work_range, thread_locator, threadid, this->_gemm_arrays);
    }

    // Interface implementation - pretransposed
    bool B_is_pretransposed() const override {
        return (FixedFormat == false);
    }

    bool B_pretranspose_required() const override {
        return (FixedFormat == false) && (_B_transposed==nullptr);
    }

    size_t get_B_pretransposed_array_size() const override {
        if (FixedFormat) {
            return 0;
        }

        // Start with actual pretransposed buffer...
        size_t size = roundup(_args._Nsize, strategy::out_width()) * _Ktotal * _args._nmulti * sizeof(Troi);

        // Space for result row pointers retained for existing buffer sizing.
        size += _args._Msize * _args._nbatches * _args._nmulti * sizeof(const Tr *);

        if (std::is_same<OutputStage, Requantize32>::value) {
            size += get_col_sum_size();
        }

        return size;
    }

    size_t get_B_pretranspose_window_size() const override {
        return _args._nmulti * iceildiv(_args._Nsize, strategy::out_width());
    }

    void requantize_bias(void *in_buffer, const Tw *B, const int ldb, const int B_multi_stride) override {
        if (std::is_same<OutputStage, Requantize32>::value) {
            _col_bias = reinterpret_cast<int32_t *>(in_buffer);

            Requantize32 *qp_ptr = reinterpret_cast<Requantize32 *>(&_os);

            for (unsigned int i=0; i<_args._nmulti; i++) {
                // The input is assumed not to have any padding between sections, so straightforward Ksize * Ksections computation gets the total size.
                compute_col_sums(*qp_ptr, _args._Nsize, _args._Ksize * _args._Ksections, B + (i * B_multi_stride), ldb, _col_bias + (i * _args._Nsize), _args._Ksize * _args._Ksections, i, 0);
            }
        }
    }

    bool B_pretranspose_supports_transpose() const override {
        strategy strat(_args._ci);
        return strat.transforms.PrepareB_supports_transpose();
    }

    void pretranspose_B_array(void *in_buffer, const Tw *B, const int ldb, const int B_multi_stride, bool transposed) override {
        pretranspose_B_array_part(in_buffer, B, ldb, B_multi_stride, transposed, 0, get_B_pretranspose_window_size());
    }

    void pretranspose_B_array_part(void *in_buffer, const Tw *B, const int ldb, const int B_multi_stride, bool transposed, size_t start, size_t end) override {
        if (end >= get_B_pretranspose_window_size()) {
            requantize_bias(in_buffer, B, ldb, B_multi_stride);
        }

        // Put the transposed data after the column sums - in non-transposing cases get_col_sum_size() == 0
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        Troi *buffer_base = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        _B_transposed = buffer_base;

        strategy strat(_args._ci);
        size_t work_per_multi = iceildiv(_args._Nsize, strategy::out_width());

        for (unsigned int multi=(start / work_per_multi); multi<_args._nmulti; multi++) {
            // Work out which part of the window space this multi occupies,
            // skip to the next multi or exit as needed.
            size_t wk_start = multi * work_per_multi;
            size_t wk_end = (multi + 1) * work_per_multi;

            assert(wk_end > start);

            if (wk_start >= end) {
                break;
            }

            for (unsigned int k0=0; k0<_Ktotal; k0+=_k_block) {
                const unsigned int kmax=std::min(k0 + _k_block, _Ktotal);

                /* Figure out the size of each block. */
                unsigned int k_size = kmax - k0;

                // Correct the N range and buffer base if we are not processing the whole block.
                size_t n_start = 0;
                size_t n_end = _args._Nsize;

                // If we are not doing the first columns, update the buffer write position and starting N value.
                if (start > wk_start) {
                    n_start = (start - wk_start) * strategy::out_width();
                }

                // If we are not doing the last items, update the final N value.
                if (end < wk_end) {
                    n_end = (end - wk_start) * strategy::out_width();
                }

                // Set the buffer pointer
                Troi *buffer = buffer_base +
                               (roundup(_args._Nsize, strategy::out_width()) * (multi * _Ktotal + k0)) +
                               (n_start * roundup(k_size, strategy::k_unroll()));

                if (_args._Ksections > 1) {
                    // We need to insert padding at the end of each K section.
                    // The computation needed is a little delicate - the k0/kmax coordinates are expressed in
                    // terms of the full, padded, _Ktotal.
                    // But we need to transform each section with reference to the original, unpadded, input, letting the
                    // transform pad each section as needed.

                    // This is needed for computations below.
                    const unsigned int rounded_section_size = roundup(_args._Ksize, strategy::k_unroll());

                    // The expected output format is also an entire <out_width> columns interleaved, then the next set of
                    // columns, and so on.  This means, as we are breaking it up vertically, we have to do it one column at
                    // a time.
                    for (unsigned int x0 = n_start; x0 < n_end; x0 += strategy::out_width()) {
                        unsigned int xmax = std::min(x0 + strategy::out_width(), _args._Nsize);

                        // Track where we are and how much work is left.
                        unsigned int kpos  = k0;
                        unsigned int kleft = k_size;

                        while (kleft) {
                            // Which section are we in?  Based on the rounded-up section size.
                            unsigned int k_section_base = kpos / rounded_section_size;
                            // How far into the section are we?
                            unsigned int k_offset = kpos - (k_section_base * rounded_section_size);

                            // We will either copy the rest of this section, or to the end of the requested length.
                            unsigned int k_length = std::min(_args._Ksize - k_offset, kleft);

                            strat.transforms.PrepareB(buffer, B + (multi * B_multi_stride), ldb,
                                                      x0, xmax,
                                                      (k_section_base * _args._Ksize) + k_offset,               // K starting point - compute row to read based on our section and the true section length.
                                                      (k_section_base * _args._Ksize) + k_offset + k_length,    // K end point - starting point plus length computed above.
                                                      transposed);

                            // We need to modify our position based on the ROUNDED version of what we just did.
                            unsigned int padded_length = roundup(k_length, strategy::k_unroll());

                            buffer += strategy::out_width() * padded_length;

                            kpos  += padded_length;
                            kleft -= padded_length;
                        }
                    }
                } else {
                    // In the single K section case, can process the whole lot in one go.
                    strat.transforms.PrepareB(buffer, B + (multi * B_multi_stride), ldb,
                                              n_start, n_end, k0, std::min(kmax, _args._Ksize), transposed);
                }
            }
        }
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        // Put the transposed data after the column sums - in non-transposing cases get_col_sum_size() == 0
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        _B_transposed = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        _col_bias = reinterpret_cast<int32_t *>(in_buffer);
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
        uint64_t total_macs = static_cast<uint64_t>(args._nbatches) * args._nmulti * args._Msize * roundup(args._Nsize, strategy::out_width()) * get_ktotal(args);

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

    void set_quantized_bias(const int32_t *bias, size_t bias_multi_stride) override {
        if (std::is_same<OutputStage, Requantize32>::value) {
            Requantize32 *qp = reinterpret_cast<Requantize32 *>(&_os);

            qp->bias = bias;
            qp->bias_multi_stride = bias_multi_stride;
        }
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
        GemmConfig c;

        c.inner_block_size = _k_block;
        c.outer_block_size = _n_block;
        c.filter = get_type_name<strategy>();
        c.weight_format = get_weight_format(kernel_weight_format<strategy, FixedFormat>::get(), sizeof(To));

        return c;
    }
};

template<typename strategy, typename To, typename Tr, typename OutputStage=Nothing>
using GemmHybridIndirectFixedFormat = GemmHybridIndirect<strategy, To, To, Tr, OutputStage, false, true>;

}  // namespace ops
}  // namespace kai

#ifdef __I_DEFINED_UNUSED
#undef UNUSED
#endif
