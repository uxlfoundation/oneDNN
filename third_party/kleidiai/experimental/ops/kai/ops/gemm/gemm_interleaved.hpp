//
// SPDX-FileCopyrightText: Copyright 2017-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cassert>

#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/bfloat.hpp"
#include "convolver.hpp"
#include "kernel_traits.hpp"
#include "kernel_weight_format.hpp"
#include "performance_parameters.hpp"
#include "common_internal/quantized.hpp"
#include "common_internal/utils.hpp"

#ifdef CYCLE_PROFILING
#include "kai/ops/profiler.hpp"
#endif

// Some macros used to decide how much working space to allocate.
// Round allocations up to the next cache line.
#define ALLOC_ROUND	64
#define ROUND_UP(x)	((((x) + ALLOC_ROUND-1) / ALLOC_ROUND) * ALLOC_ROUND)

// Implementation of the GemmCommon abstract class.
//
// This implementation interleaves the source matrices in blocks - good for
// larger matrices.

namespace kai {
namespace ops {

namespace {

// Some kernels output to a linear buffer and require a separate merge step.
// Others output directly to the matrix result.  This helper class calls the
// appropriate functions, using templating to avoid calling non-existent
// functions.
template<bool MergeStep, bool FixedFormat, typename OutputStage>
class kernel_and_merge {
public:
    template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
    static void run (
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t b_stride, Tri *c_panel,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0,
        unsigned int m_max, unsigned int n_0, unsigned int n_max, const Tr *biasptr,
        const Activation &act, bool accumulate, const OutputStage &os, const int32_t *col_bias,
        Tab *acc_buff);
};

// Run a kernel and call the separate merge step
template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<true, false, Nothing>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t, Tri *c_panel,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0,
        unsigned int m_max, unsigned int n_0, unsigned int n_max, const Tr *biasptr,
        const Activation &act, bool accumulate, const Nothing &, const int32_t *, Tab *)
{
    const int bblocks = iceildiv(n_max - n_0, strategy::out_width());

    {
#ifdef CYCLE_PROFILING
        auto p=prof.ScopedProfiler(PROFILE_KERNEL, (strategy::out_height() * bblocks * strategy::out_width() * kern_k));
#endif

        strat.kernel(a_ptr, b_panel, c_panel, 1, bblocks, kern_k);
    }

    {
#ifdef CYCLE_PROFILING
        auto p=prof.ScopedProfiler(PROFILE_MERGE, (strategy::out_height() * bblocks * strategy::out_width() * sizeof(Tr)));
#endif
        strat.transforms.Merge(c_ptr, c_panel, ldc, m_0, m_max, n_0, n_max, biasptr, act, accumulate);
    }
}

// Run a fixed-format kernel and call the separate merge step
template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<true, true, Nothing>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t b_stride, Tri *c_panel,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0,
        unsigned int m_max, unsigned int n_0, unsigned int n_max, const Tr *biasptr,
        const Activation &act, bool accumulate, const Nothing &, const int32_t *, Tab *)
{
    {
#ifdef CYCLE_PROFILING
        const int bblocks = iceildiv(n_max - n_0, strategy::out_width());
        auto p=prof.ScopedProfiler(PROFILE_KERNEL, (strategy::out_height() * bblocks * strategy::out_width() * kern_k));
#endif

        strat.kernel(a_ptr, b_panel, b_stride, c_panel, 1, (n_max - n_0), kern_k);
    }

    {
#ifdef CYCLE_PROFILING
        const int bblocks = iceildiv(n_max - n_0, strategy::out_width());
        auto p=prof.ScopedProfiler(PROFILE_MERGE, (strategy::out_height() * bblocks * strategy::out_width() * sizeof(Tr)));
#endif
        strat.transforms.Merge(c_ptr, c_panel, ldc, m_0, m_max, n_0, n_max, biasptr, act, accumulate);
    }
}

// Run a kernel with integrated merge
template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<false, false, Nothing>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t, Tri *,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0, unsigned int m_max,
        unsigned int n_0, unsigned int n_max, const Tr *biasptr,
        const Activation &act, bool accumulate, const Nothing &, const int32_t *,
        Tab *acc_buff)
{
#ifdef CYCLE_PROFILING
    auto p=prof.ScopedProfiler(PROFILE_KERNEL, (m_max - m_0) * (n_max - n_0) * kern_k);
#endif

    // We need to offset the C pointer, but as it might be NULL (requesting output to accumulation buffer) we need
    // to be careful not to offset a null pointer.
    Tri *offset_c_ptr;

    if (c_ptr == nullptr) {
        offset_c_ptr = nullptr;
    } else {
        offset_c_ptr = c_ptr + m_0 * ldc + n_0;
    }

    strat.kernel(// A and B pointers are just the packed panels.
                 a_ptr, b_panel,
                 // Provide relevant part of output array and row stride.
                 offset_c_ptr, ldc,
                 // M, N, K sizes
                 m_max-m_0, n_max - n_0, kern_k,
                 // Bias, activation, accumulation.  Need to offset the bias as needed.
                 biasptr ? biasptr + n_0 : nullptr, act, accumulate,
                 // Accumulation buffer.
                 acc_buff );
}

// Run a kernel with integrated merge, quantizing
template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<false, false, Requantize32>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t, Tri *,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0, unsigned int m_max,
        unsigned int n_0, unsigned int n_max, const Tr *,
        const Activation &, bool accumulate, const Requantize32 &qp, const int32_t *col_bias,
        Tab *acc_buff)
{
#ifdef CYCLE_PROFILING
    auto p=prof.ScopedProfiler(PROFILE_KERNEL, (m_max - m_0) * (n_max - n_0) * kern_k);
#endif

    // Offset C pointer in a similar way to non-quantized case above.
    Tri *offset_c_ptr;

    if (c_ptr == nullptr) {
        offset_c_ptr = nullptr;
    } else {
        offset_c_ptr = c_ptr + m_0 * ldc + n_0;
    }

    strat.kernel(// A and B pointers are just the packed panels.
                 a_ptr, b_panel,
                 // Provide relevant part of output array and row stride.
                 offset_c_ptr, ldc,
                 // M, N, K sizes
                 m_max-m_0, n_max - n_0, kern_k,
                 // Bias, activation, accumulation.  Need to offset the bias as needed.
                 col_bias + n_0, qp, n_0, accumulate, acc_buff);
}

// Run a kernel and call the separate quantize step
template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<true, false, Requantize32>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t, Tri *c_panel,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0,
        unsigned int m_max, unsigned int n_0, unsigned int n_max, const Tr *,
        const Activation &, bool, const Requantize32 &qp, const int32_t *col_bias,
        Tab *)
{
    const int bblocks = iceildiv(n_max - n_0, strategy::out_width());

    {
#ifdef CYCLE_PROFILING
        auto p=prof.ScopedProfiler(PROFILE_KERNEL, (strategy::out_height() * bblocks * strategy::out_width() * kern_k));
#endif

        strat.kernel(a_ptr, b_panel, c_panel, 1, bblocks, kern_k);
    }

    {
#ifdef CYCLE_PROFILING
        auto p=prof.ScopedProfiler(PROFILE_QUANTIZE, ((m_max-m_0) * bblocks * strategy::out_width() * sizeof(Tr)));
#endif
        // The interleaved kernel outputs in blocks - each block is a
        // row-major matrix of size out_width * out_height.  The merge
        // kernels are designed to deal with this but the requantizer is
        // not, so we need to requantize one block at a time.
        for (int i=0; i<bblocks; i++) {
            unsigned int n_start = n_0 + (strategy::out_width() * i);
            unsigned int n_end = std::min(n_start + strategy::out_width(), n_max);

            // The row bias is interleaved with the transposed A data, get a pointer to it here.
            const int32_t *row_bias = reinterpret_cast<const int32_t *>(a_ptr + strategy::out_height() * kern_k);

            requantize_block_32(qp, (n_end - n_start), (m_max-m_0),
                                c_panel + (i * strategy::out_width() * strategy::out_height()), strategy::out_width(),
                                c_ptr + m_0 * ldc + n_start, ldc,
                                row_bias, col_bias + n_start, n_start);
        }
    }
}

// Run a kernel with integrated merge, quantizing to FP32
template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<false, false, DequantizeFloat>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t, Tri *,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0, unsigned int m_max,
        unsigned int n_0, unsigned int n_max, const Tr *bias,
        const Activation &act, bool accumulate, const DequantizeFloat &dq, const int32_t *col_bias,
        Tab *acc_buff)
{
#ifdef CYCLE_PROFILING
    auto p=prof.ScopedProfiler(PROFILE_KERNEL, (m_max - m_0) * (n_max - n_0) * kern_k);
#endif

    const int32_t *offset_col_bias = nullptr;
    const Tr *offset_bias = nullptr;

    if (col_bias) {
        offset_col_bias = col_bias + n_0;
    }

    if (bias) {
        offset_bias = bias + n_0;
    }

    strat.kernel(// A and B pointers are just the packed panels.
                 a_ptr, b_panel,
                 // Provide relevant part of output array and row stride.
                 c_ptr ? (c_ptr + m_0 * ldc + n_0) : nullptr, ldc,
                 // M, N, K sizes
                 m_max-m_0, n_max - n_0, kern_k,
                 // Bias, activation, accumulation.  Need to offset the bias as needed.
                 offset_col_bias, dq, offset_bias, act, accumulate, acc_buff);
}

template<>
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename Tri, typename Tab>
void kernel_and_merge<true, false, DequantizeFloat>::run(
#ifdef CYCLE_PROFILING
        profiler &prof,
#endif
        strategy &strat, const Tlo *a_ptr, const Tro *b_panel, size_t, Tri *c_panel,
        Tr *c_ptr, int ldc, int kern_k, unsigned int m_0,
        unsigned int m_max, unsigned int n_0, unsigned int n_max, const Tr *bias,
        const Activation &act, bool not_first_pass, const DequantizeFloat &qp, const int32_t *,
        Tab *)
{
    const int bblocks = iceildiv(n_max - n_0, strategy::out_width());

    {
#ifdef CYCLE_PROFILING
        auto p=prof.ScopedProfiler(PROFILE_KERNEL, (strategy::out_height() * bblocks * strategy::out_width() * kern_k));
#endif

        strat.kernel(a_ptr, b_panel, c_panel, 1, bblocks, kern_k);
    }

    {
#ifdef CYCLE_PROFILING
        auto p=prof.ScopedProfiler(PROFILE_QUANTIZE, ((m_max-m_0) * bblocks * strategy::out_width() * sizeof(Tr)));
#endif
        for (int i=0; i<bblocks; i++) {
            unsigned int n_start = n_0 + (strategy::out_width() * i);
            unsigned int n_end = std::min(n_start + strategy::out_width(), n_max);

            dequantize_block_32(qp, (n_end - n_start), (m_max - m_0),
                            c_panel + (i * strategy::out_width() * strategy::out_height()), strategy::out_width(),
                            c_ptr + m_0 * ldc + n_start, ldc,
                            bias != nullptr ? bias + n_start : nullptr, not_first_pass, act);

        }
    }
}

// Integer GEMMs can be used in two contexts - "normal" where the full 32-bit output is required, or in
// "requantizing" context where the output will be requantized.
//
// These require different input transforms, as if we are requantizing we want to sum the rows of the A input, and
// if we are not we don't.
//
// This helper class allows the appropriate transforms to be found, without requiring kernels that don't support
// quantization to define useless "quantized" transforms.
template<typename strategy, bool quantized>
class transform_type {
public:
    typedef decltype(strategy::transforms) type;
};

template<typename strategy>
class transform_type<strategy, true> {
public:
    typedef decltype(strategy::transforms_quantized) type;
};

// We need a similar trick here to figure out what type the accumulator buffer should be.
template<typename strategy, typename OutputStage, bool ForceFloat>
class accumulate_buffer_type {
public:
    typedef typename strategy::result_type type;
};

template<typename strategy>
class accumulate_buffer_type<strategy, Requantize32, false> {
public:
    typedef int32_t type;
};

template<typename strategy>
class accumulate_buffer_type<strategy, DequantizeFloat, false> {
public:
    typedef int32_t type;
};

template<typename strategy, typename OutputStage>
class accumulate_buffer_type<strategy, OutputStage, true> {
public:
    typedef float type;
};

// Stripe width is a concept only needed for FixedFormat kernels.  Use an accessor to avoid issues in other scenarios.
template<typename strategy, bool FixedFormat>
struct get_stripe_width {
    static unsigned int get() {
        return strategy::out_width();
    }
};

template<typename strategy>
struct get_stripe_width<strategy, true> {
    static unsigned int get() {
        return strategy::stripe_width();
    }
};

// KernelWeightFormat is a similar story.
template<typename strategy, bool FixedFormat, typename Tro>
struct get_kernel_weight_format {
    static KernelWeightFormat get() {
        return KernelWeightFormat::NON_FIXED;
    }
};

template<typename strategy, typename Tro>
struct get_kernel_weight_format<strategy, true, Tro> {
    static KernelWeightFormat get() {
        KernelWeightFormat kwf = strategy::kernel_weight_format();

        // If we are using a BF16 kernel to do an FP32 problem (fast mode) then we need to set the BF16 flag on the
        // weight format.
        if (std::is_same<Tro, float>::value && std::is_same<typename strategy::rhs_operand_type, bfloat16>::value) {
            uint32_t kwf_i = static_cast<uint32_t>(kwf);
            kwf_i |= 0x10;
            kwf = static_cast<KernelWeightFormat>(kwf_i);
        }

        return kwf;
    }
};

// Calculate the offset needed if the address is not a multiple of cache-line length
inline size_t get_cache_align_offset(uintptr_t addr) {
    return (addr & 0x3F) ? 0x40 - (addr & 0x3F) : 0;
}

} // anonymous namespace

template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing, bool MergeStep=true, bool FixedFormat=false, bool ForceThreadColumns=false, bool ForceFloatAccumulate=false>
class GemmInterleaved : public GemmCommon<Tlo, Tro, Tr> {
    typedef typename strategy::lhs_operand_type Tloi;
    typedef typename strategy::rhs_operand_type Troi;
    typedef typename strategy::result_type Tri;
    typedef typename accumulate_buffer_type<strategy, OutputStage, ForceFloatAccumulate>::type Tab;

    /* const properties set by constructor */
    const CPUInfo * const _ci;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;
    const unsigned int _Ksections;
    const unsigned int _Ktotal;
    const unsigned int _rounded_Ksize;

    const unsigned int _nbatches;
    const unsigned int _nmulti;

    const bool _thread_columns;

    const Activation _act;
    const bool _accumulate;

    const int _maxthreads;
    int _nthreads;

    /* Blocking info */
    unsigned int _k_block=0;
    unsigned int _x_block=0;
    unsigned int _Mround=0;

    /* Working space, pretransposed buffer, buffer manager */
    const Troi *_B_transposed=nullptr;

    /* Output stage */
    OutputStage  _os;

    /* Quantized support (in addition to 'output stage' above */
    int32_t *col_bias = nullptr;

    /* Indirect parameters.  _indirect_buf doubles as a flag to indicate that "indirect" transform should be used. */
    const Tlo * const * const * _indirect_buf = nullptr;

    /* Convolver - only set up for convolution problems, so also doubles as a flag. */
    std::unique_ptr<convolver<Tlo>>  _convolver = nullptr;

    unsigned int get_col_sum_size() const {
        if (std::is_same<OutputStage, Requantize32>::value) {
            return _Nsize * _nmulti * sizeof(int32_t);
        }

        return 0;
    }

    /* We will need to walk through the blocks of B in a few contexts, so
     * factor that out.  */
    class blockwalker {
    private:
        /* Size loops, etc. based on our parent's configuration */
        const GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, MergeStep, FixedFormat, ForceThreadColumns, ForceFloatAccumulate> &_parent;

        /* K, X and multi parameters for current iteration. */
        unsigned int _k0=0, _x0=0, _multi=0;

        /* Range of X to iterate over - used in "ForceThreadColumns" cases */
        unsigned int _x_start=0;
        unsigned int _x_end=_parent._Nsize;

        unsigned int _index=0;
        bool _done=false;
        bool _newkblock=true;
        bool _newmulti=true;

    public:
        blockwalker(const GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, MergeStep, FixedFormat, ForceThreadColumns, ForceFloatAccumulate> &parent) : _parent(parent) { }

        blockwalker(const GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, MergeStep, FixedFormat, ForceThreadColumns, ForceFloatAccumulate> &parent,
                    unsigned int x_start, unsigned int x_end) : _parent(parent), _x0 (_x_start), _x_start(x_start), _x_end(x_end) { }

        unsigned int xmax() {
            return std::min(_x0 + _parent._x_block, _x_end);
        }

        unsigned int kmax() {
            return std::min(_k0 + _parent._k_block, _parent._Ktotal);
        }

        /* Advance to the next block, return false at the end. */
        bool advance(void) {
            if (_done) {
                return false;
            }

            _newkblock=false;
            _x0 += _parent._x_block;
            if (_x0 >= _x_end) {
                _x0=_x_start;
                _k0 += _parent._k_block;
                if (_k0 >= _parent._Ktotal) {
                    _k0=0;
                    _multi++;
                    if (_multi >= _parent._nmulti) {
                        _done=true;
                        return false;
                    }
                    _newmulti=true;
                }
                _newkblock=true;
            }
            _index++;

            return true;
        }

        unsigned int k0(void) { return _k0; }
        unsigned int x0(void) { return _x0; }
        unsigned int multi(void) { return _multi; }
        unsigned int index(void) { return _index; }
        bool done(void) { return _done; }
        bool newkblock(void) { return _newkblock; }
    };

    // "k block" has two distinct uses: figuring out which iterations of K
    // to actually process, but also various size/pointer computations.  The
    // latter needs to take account of the extra space needed for the row
    // sums, if appropriate.
    unsigned int get_total_k_depth() const {
        unsigned int k_depth = _k_block;

        if (std::is_same<OutputStage, Requantize32>::value) {
            k_depth += sizeof(int32_t) / sizeof(Tloi);
        }

        return k_depth;
    }

    // A working size.
    size_t get_a_working_size() const {
        if (_thread_columns) {
            // For 2D threading: allocate a buffer of one block of rows per thread
            return ROUND_UP(sizeof(Tloi) * get_total_k_depth() * strategy::out_height() * _maxthreads);
        } else {
            // For 1D threaded: one of these needed, regardless of thread count.  Divided according to window.
            return ROUND_UP(sizeof(Tloi) * get_total_k_depth() * _Mround * _nbatches);
        }
    }

    // C working size: One needed per thread.  Not needed if there is no merge step.
    size_t get_c_working_size() const {
        if (MergeStep) {
            return ROUND_UP(sizeof(Tri) * _x_block * strategy::out_height());
        }

        return 0;
    }

    // Accumulation buffer size
    size_t get_accumulation_buffer_size() const {
        // We only support an accumulation buffer for non-merge cases.
        if (MergeStep) {
            return 0;
        }

        // Check if we are actually blocking
        if (_k_block == _Ktotal) {
            return 0;
        }

        // We are no-merge, non-quantized with active blocking: accumulation buffer needed.
        size_t size_per_buffer = sizeof(Tab) * strategy::out_height() * strategy::out_width();
        size_t num_buffers = iceildiv(_Msize, strategy::out_height()) * iceildiv(_Nsize, strategy::out_width()) * _nbatches * _nmulti;

        return num_buffers * size_per_buffer;
    }

    // Set up accumulation buffer
    Tab *get_accumulation_buffer_offset(void *working_space) const {
        Tab *accumulation_buffer = nullptr;

        auto working_space_addr = reinterpret_cast<uintptr_t>(working_space);

        if (get_accumulation_buffer_size() > 0) {
            auto acc_buff_addr = working_space_addr + get_a_working_size() + (get_c_working_size() * _maxthreads);
            acc_buff_addr += get_cache_align_offset(acc_buff_addr);
            accumulation_buffer = reinterpret_cast<Tab *>(acc_buff_addr);
        }

        return accumulation_buffer;
    }

    // Get pointer into accumulation buffer
    Tab *get_accumulation_buffer(Tab *accumulation_buffer, unsigned int M, unsigned int N, unsigned int batch, unsigned int multi) const {
        // Don't do anything if there's no buffer.
        if (accumulation_buffer == nullptr) {
            return nullptr;
        }

        // Here we are indexing an appropriately sized pointer, so no sizeof() needed to convert to bytes.
        size_t size_per_buffer = strategy::out_height() * strategy::out_width();

        size_t buffer_rows = iceildiv(_Msize, strategy::out_height());
        size_t buffer_cols = iceildiv(_Nsize, strategy::out_width());
        size_t buffers_per_batch = (buffer_rows * buffer_cols);
        size_t buffers_per_multi = buffers_per_batch * _nbatches;

        // M/N must reference the top-left corner of a block.
        size_t row = M / strategy::out_height();
        assert(M % strategy::out_height() == 0);
        size_t col = N / strategy::out_width();
        assert(N % strategy::out_width() == 0);

        size_t buffer_index = multi * buffers_per_multi + batch * buffers_per_batch + row * buffer_cols + col;

        return accumulation_buffer + (buffer_index * size_per_buffer);
    }

    int32_t row_sum_multiplier() const {
        if (std::is_same<OutputStage, Requantize32>::value) {
            const Requantize32 *qp = reinterpret_cast<const Requantize32 *>(&_os);

            return -qp->b_offset;
        }

        return 0;
    }

    // Heuristics to decide whether to use the 'thread columns' regime
    static bool is_thread_columns(const GemmArgs &args) {
        // For now, there is a templace parameter to force it.
        if (ForceThreadColumns) {
            return true;
        }

        // Never do this for single threaded cases.
        if (args._maxthreads == 1) {
            return false;
        }

        // How many blocks of work are available for threading on M?
        int m_blocks = iceildiv(args._Msize, strategy::out_height()) * args._nbatches;

        // If we just can't share the work across threads with the row threading regime.
        if (args._maxthreads > m_blocks) {
            return true;
        }

        // If the row threading regime is too wasteful (20% threshold)
        if (((roundup(m_blocks, args._maxthreads) * 100) / m_blocks) > 120) {
            return true;
        }

        return false;
    }

    static unsigned int get_ktotal(const GemmArgs &args) {
        return args._Ksections * roundup(args._Ksize, strategy::k_unroll());
    }

    static unsigned int get_k_block_size(const GemmArgs &args) {
        if (args._cfg && args._cfg->inner_block_size) {
            return roundup(args._cfg->inner_block_size, strategy::k_unroll());
        }

        // K blocking not supported if we are requantizing with the merging
        // kernels.
        if (std::is_same<OutputStage, Requantize32>::value && MergeStep) {
            return get_ktotal(args);
        }

        // We can't K block non-fast FP16 cases without an accumulation buffer.
        if (std::is_same<Tlo, __fp16>::value && std::is_same<Tr, __fp16>::value && !args._fast_mode && MergeStep) {
            return get_ktotal(args);
        }

        // Same for BF16BF16 cases.
        if (std::is_same<Tlo, bfloat16>::value && std::is_same<Tr, bfloat16>::value && MergeStep) {
            return get_ktotal(args);
        }

        // And for int8-requantizing-fp16
        if (std::is_same<Tro, int8_t>::value && std::is_same<Tr, __fp16>::value && MergeStep) {
            return get_ktotal(args);
        }

        const unsigned int L1_size = args._ci->get_L1_cache_size();

        // Special blocking for SME
        if (is_sme<strategy>::value) {
            // Target 512 bytes for 64kB L1, or 1024 bytes for 128kB L1.
            unsigned int target_bytes_per_block = L1_size / 128;

            // Default cache size in gemm-linux is 32kB though - so make
            // sure minimum is 512
            if (target_bytes_per_block < 512) {
                target_bytes_per_block = 512;
            }

            // Don't bother to block below this size threshold (1.25X target size)
            unsigned int scaling_threshold = ((target_bytes_per_block * 5) / 4) / sizeof(Tloi);

            if (get_ktotal(args) <= scaling_threshold) {
                return get_ktotal(args);
            }

            // Once we are blocking, this (lower) threshold determines when we should use more blocks
            // NOTE: Could be that some factor-based solution would work better here.
            unsigned int max_block_size = target_bytes_per_block / sizeof(Tloi);

            unsigned int num_k_blocks = iceildiv(get_ktotal(args), max_block_size);

            unsigned int k_block = roundup(iceildiv(get_ktotal(args), num_k_blocks), strategy::k_unroll());

            return k_block;
        }

        unsigned int k_block;

        // k_block: Find out how much of the larger array can be loaded into half the cache.
        // This should account for associative caches.
        k_block = (L1_size / 2) / (sizeof(Tloi) * (std::max(strategy::out_width(), strategy::out_height())));

        // Needs to be (at least a single) multiple of the K unroll level.
        k_block /= strategy::k_unroll();
        k_block = std::max(k_block, 1U) * strategy::k_unroll();

        // Now tune to presented problem size; this is how many blocks we need.
        unsigned int num_k_blocks = iceildiv(get_ktotal(args), k_block);

        // So divide the space equally into that many blocks.
        k_block = iceildiv(get_ktotal(args), num_k_blocks);

        // And round UP to the K unroll level required.
        k_block = roundup(k_block, strategy::k_unroll());

        assert(k_block > 0);

        return k_block;
    }

    static unsigned int get_x_block_size(const GemmArgs &args) {
        if (is_thread_columns(args)) {
            // In 2D mode, override X block, because we will process width first.
            return roundup(args._Nsize, strategy::out_width());
        }

        if (args._cfg && args._cfg->outer_block_size) {
            return roundup(args._cfg->outer_block_size, strategy::out_width());
        }

        // Special blocking for SME
        if (is_sme<strategy>::value) {
            // If total width is less than 4x kernel width, return the entire width.
            if (args._Nsize < strategy::out_width()*4) {
                return roundup(args._Nsize, strategy::out_width());
            }

            // Otherwise block to single kernel width.
            return strategy::out_width();
        }

        unsigned int x_block;
        const unsigned int L2_size = args._ci->get_L2_cache_size();
        const unsigned int k_block = get_k_block_size(args);

        // x_block: Work out how many rows (of length k_block) will fit in the L2
        // Don't allocate more than 90% of the L2 to allow for overheads, and subtract off the L1 contents.
        const unsigned int scaled_l2_size = (L2_size * 9) / 10;
        const unsigned int k_block_area = k_block * sizeof(Tloi) * (strategy::out_width() + strategy::out_height());

        // .. if the L1 contents is bigger than the L2, just return a minimal size block.
        if (k_block_area > scaled_l2_size) {
            return strategy::out_width();
        }

        x_block = (scaled_l2_size - k_block_area) / (sizeof(Tloi) * k_block);

        // Needs to be (at least a single) multiple of the kernel output width.
        x_block /= strategy::out_width();
        x_block = std::max(x_block, 1u) * strategy::out_width();

        // And tune to the presented problem size.
        unsigned int num_x_blocks = iceildiv(args._Nsize, x_block);
        x_block = iceildiv(args._Nsize, num_x_blocks);

        x_block = roundup(x_block, strategy::out_width());

        assert(x_block > 0);

        return x_block;
    }

public:
    GemmInterleaved(GemmInterleaved &) = delete;
    GemmInterleaved & operator= (GemmInterleaved &) = delete;

    /* Constructor */
    GemmInterleaved(const GemmArgs &args, const OutputStage &os)
                    : _ci(args._ci), _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize),
                      _Ksections(args._Ksections), _Ktotal(get_ktotal(args)),
                      _rounded_Ksize(roundup(_Ksize, strategy::k_unroll())),
                      _nbatches(args._nbatches), _nmulti(args._nmulti), _thread_columns(is_thread_columns(args)),
                      _act(args._act), _accumulate(args._accumulate), _maxthreads(args._maxthreads), _nthreads(args._maxthreads),
                      _k_block(get_k_block_size(args)), _x_block(get_x_block_size(args)), _Mround(roundup(args._Msize, strategy::out_height())),
                      _os(os) { }

    /* Constructor without OutputStage */
    GemmInterleaved(const GemmArgs &args)
                    : _ci(args._ci), _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize),
                      _Ksections(args._Ksections), _Ktotal(get_ktotal(args)),
                      _rounded_Ksize(roundup(_Ksize, strategy::k_unroll())),
                      _nbatches(args._nbatches), _nmulti(args._nmulti), _thread_columns(is_thread_columns(args)),
                      _act(args._act), _accumulate(args._accumulate), _maxthreads(args._maxthreads), _nthreads(args._maxthreads),
                      _k_block(get_k_block_size(args)), _x_block(get_x_block_size(args)), _Mround(roundup(args._Msize, strategy::out_height())),
                      _os() { }

    // Interface implementation - Compulsory functions

    // Window size: Only the last thread should do a ragged block, so dole
    // out work in units of out_height.  Factor batches into the window, but
    // not multi for now (as this would cause problems with the buffer
    // manager).
    ndrange_t get_window_size() const override {
        unsigned int row_blocks = (_Mround / strategy::out_height()) * _nbatches;

        if (_thread_columns) {
            return { row_blocks, iceildiv(_Nsize, strategy::out_width()) };
        } else {
            // _Mround is a multiple of out_height by definition.
            return { row_blocks };
        }
    }

    // set_nthreads: pass on to buffer manager to avoid it waiting for non-existant threads.
    void set_nthreads(int nthreads) override {
        _nthreads = std::min(nthreads, _maxthreads);
    }

    // Common execution logic.
    void execute_common(const ndcoord_t &work_range, const ndcoord_t &, int threadid, GemmArrays<Tlo, Tro, Tr> &g_arrays) {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif

        void *working_space = g_arrays._workspace;

        // Make sure we've been set up correctly.
        assert(FixedFormat || _B_transposed);
        assert(working_space);

        // Align if needed.
        auto *working_space_bytes = reinterpret_cast<int8_t *>(working_space);
        auto working_space_addr = reinterpret_cast<uintptr_t>(working_space);
        working_space_bytes += get_cache_align_offset(working_space_addr);
        working_space = reinterpret_cast<void *>(working_space_bytes);

        auto *accumulation_buffer = get_accumulation_buffer_offset(working_space);

        strategy strat(_ci);

        const auto start = work_range.get_position(0);
        const auto end   = work_range.get_position_end(0);

        /* Translate 'start' and 'end' into a position within the batches and rows. */
        const unsigned int window_per_batch = _Mround / strategy::out_height();
        unsigned int batch_0   = start / window_per_batch;
        unsigned int batch_end = end   / window_per_batch;

        // In ThreadColumns mode, process work one horizontal strip at a time.
        // Transpose the block of needed rows at the start, then do all the work on that block.
        if (_thread_columns) {
            const auto start_x = work_range.get_position(1) * strategy::out_width();
            const auto end_x = std::min(work_range.get_position_end(1) * strategy::out_width(), _Nsize);

            Tri * const c_panel = reinterpret_cast<Tri *>(working_space_bytes + (threadid * get_c_working_size()));
            Tloi * const a_panel = reinterpret_cast<Tloi *>(working_space_bytes + (_maxthreads * get_c_working_size()) +
                                       (threadid * sizeof(Tloi) * get_total_k_depth() * strategy::out_height()));

            for (unsigned int multi=0; multi<_nmulti; multi++) {
                for (unsigned int k0=0; k0<_Ktotal; k0+=_k_block) {
                    unsigned int kmax=std::min(k0+_k_block, _Ktotal);

                    unsigned int rounded_width = roundup(_Nsize, strategy::out_width());

                    const bool first_pass = (k0==0);
                    const bool last_pass  = (kmax==_Ktotal);

                    // Bias is passed for the first pass only, except for dequantizefloat nomerge cases where it's the last pass.
                    const bool bias_pass = (std::is_same<OutputStage, DequantizeFloat>::value && !MergeStep) ? last_pass : first_pass;

                    // Figure out how many "K" the kernel will actually process.
                    unsigned int kern_k = roundup(kmax - k0, strategy::k_unroll());

                    const Troi *b_ptr = FixedFormat ?
                        reinterpret_cast<const Troi *>(g_arrays._Bptr) + (multi * g_arrays._B_multi_stride) +
                                                     ((start_x / get_stripe_width<strategy, FixedFormat>::get()) * g_arrays._ldb) +
                                                     (k0 * get_stripe_width<strategy, FixedFormat>::get()) :
                        _B_transposed + (rounded_width * _Ktotal * multi) + (k0 * rounded_width) + (start_x * kern_k);

                    unsigned int batch     = batch_0;
                    unsigned int start_row = (start - (batch_0 * window_per_batch)) * strategy::out_height();

                    for (unsigned int p=start; p<end; p++) {
                        unsigned int end_row = std::min(start_row + strategy::out_height(), _Msize);

                        // Set up transposed 'A' block
                        {
#ifdef CYCLE_PROFILING
                            auto p=prof.ScopedProfiler(PROFILE_PREPA, strategy::out_height() * (kmax-k0) * sizeof(Tloi));
#endif
                            // See comment above on transform_type<> class: this extracts either 'transforms' or
                            // 'transforms_quantized' as appropriate.
                            typename transform_type<strategy, MergeStep && std::is_same<OutputStage, Requantize32>::value>::type transforms;

                            if (_indirect_buf != nullptr) {
                                transforms.PrepareA_indirect(a_panel,
                                                             _indirect_buf + (multi * _nbatches * _Ksections) + (batch * _Ksections), _Ksize,
                                                             _rounded_Ksize, start_row, end_row, k0, kmax, row_sum_multiplier());
                            } else if (_convolver) {
                                transforms.PrepareA_convolution(a_panel,
                                                                g_arrays._Aptr + (batch * g_arrays._A_batch_stride) + (multi * g_arrays._A_multi_stride),
                                                                g_arrays._lda, *_convolver, _rounded_Ksize, start_row, end_row, k0, kmax, row_sum_multiplier());
                            } else {
                                transforms.PrepareA(a_panel,
                                                    g_arrays._Aptr + (batch * g_arrays._A_batch_stride) + (multi * g_arrays._A_multi_stride),
                                                    g_arrays._lda, start_row, end_row, k0, std::min(kmax, _Ksize), row_sum_multiplier());
                            }
                        }

                        Tr *result_ptr = g_arrays._Cptr + (batch * g_arrays._C_batch_stride) + (multi * g_arrays._C_multi_stride);

                        // If we are using an accumulation buffer and this isn't the last pass, don't pass a result pointer.
                        if (accumulation_buffer && !last_pass) {
                            result_ptr = nullptr;
                        }

                        // Perform the kernel and merge step, either separately or together as required.
                        kernel_and_merge<MergeStep, FixedFormat, OutputStage>::run(
                        #ifdef CYCLE_PROFILING
                            prof,
                        #endif
                            // Strategy and panel pointers
                            strat, a_panel, b_ptr, g_arrays._ldb, c_panel,
                            // Result buffer pointers
                            result_ptr, g_arrays._ldc,
                            // K size, and M/N ranges
                            kern_k, start_row, end_row, start_x, end_x,
                            // Only do bias on the first pass
                            ((bias_pass && g_arrays._bias) ? g_arrays._bias + (multi * g_arrays._bias_multi_stride) : nullptr),
                            // Only do activation on the last pass, and accumulation on any non-first pass.
                            (last_pass ? _act : Activation()), (!first_pass || _accumulate),
                            // Pass in quantization parameters for requantizing kernels (others will ignore)
                            _os, (col_bias ? col_bias + (multi * _Nsize) : nullptr),
                            // Accumulation buffer
                            get_accumulation_buffer(accumulation_buffer, start_row, start_x, batch, multi));

                        /* Increment to the next block */
                        start_row += strategy::out_height();
                        if (start_row >= _Msize) {
                            start_row = 0;
                            batch++;
                        }
                    }
                }
            }
        } else {
            blockwalker current(*this);

            /* Compute the M values to operate on */
            unsigned int m_0   = (start - (batch_0 * window_per_batch)) * strategy::out_height();
            unsigned int m_max = (end - (batch_end * window_per_batch)) * strategy::out_height();

            // Private buffers.  Treat working_space as an array of C buffers
            // (one per thread) first, followed by the (window-divided) A
            // buffer.
            // Set a_panel to the base of the A buffers - compute offsets into it based on M/batches later.
            Tloi * const a_panel = reinterpret_cast<Tloi *>(working_space_bytes + (_maxthreads * get_c_working_size()));
            Tri * const c_panel = reinterpret_cast<Tri *>(working_space_bytes + (threadid * get_c_working_size()));

            const Troi *b_panel;
            b_panel = _B_transposed;

            // newkblock() is always true on the first iteration, so these will be set properly on the first loop.

            // kern_k tracks the accumulation depth for the CURRENT K block a_panel_stride similarly tracks the total
            // stride of the A panel (i.e.  with 4 added for cases with embedded row sums)

            // These are distinct from k_block and get_total_k_depth() which are based on the target K block size, and
            // used for addressing inside a_panel.

            // In cases where K blocking is in use and the blocks are not all the same size, the (smaller) final block
            // won't use all the memory allocated.
            unsigned int kern_k = 0;
            unsigned int a_panel_stride = 0;

            for (;!current.done();current.advance()) {
                if (current.newkblock()) {
#ifdef CYCLE_PROFILING
                    auto p=prof.ScopedProfiler(PROFILE_PREPA, (end - start) * strategy::out_height() * (current.kmax()-current.k0()) * sizeof(Tloi));
#endif
                    // See comment above on transform_type<> class: this extracts either 'transforms' or
                    // 'transforms_quantized' as appropriate.
                    typename transform_type<strategy, MergeStep && std::is_same<OutputStage, Requantize32>::value>::type transforms;

                    for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                        unsigned int first_m = (batch == batch_0)   ? m_0   : 0;
                        unsigned int last_m  = (batch == batch_end) ? m_max : _Msize;

                        if (first_m >= last_m)
                            continue;

                        if (_indirect_buf != nullptr) {
                            transforms.PrepareA_indirect(a_panel + ((batch * _Mround + first_m) * get_total_k_depth()),
                                                      _indirect_buf + (current.multi() * _nbatches * _Ksections) + (batch * _Ksections), _Ksize,
                                                      _rounded_Ksize, first_m, last_m, current.k0(), current.kmax(), row_sum_multiplier());
                        } else if (_convolver) {
                            transforms.PrepareA_convolution(a_panel + ((batch * _Mround + first_m) * get_total_k_depth()),
                                                      g_arrays._Aptr + (batch * g_arrays._A_batch_stride) + (current.multi() * g_arrays._A_multi_stride),
                                                      g_arrays._lda, *_convolver, _rounded_Ksize, first_m, last_m, current.k0(), current.kmax(), row_sum_multiplier());
                        } else {
                            transforms.PrepareA(a_panel + ((batch * _Mround + first_m) * get_total_k_depth()),
                                                      g_arrays._Aptr + (batch * g_arrays._A_batch_stride) + (current.multi() * g_arrays._A_multi_stride),
                                                      g_arrays._lda, first_m, last_m, current.k0(), std::min(_Ksize, current.kmax()), row_sum_multiplier());
                        }
                    }

                    // Figure out how many "K" the kernel will actually process.
                    kern_k = roundup(current.kmax() - current.k0(), strategy::k_unroll());

                    // Requantizing GEMMs have the row sums built in to the
                    // transposed data, so the stride between rows is 4 bytes
                    // larger than the (rounded) K value.

                    if(std::is_same<OutputStage, Requantize32>::value) {
                        a_panel_stride = kern_k + (sizeof(int32_t) / sizeof(Tloi));
                    } else {
                        a_panel_stride = kern_k;
                    }
                }

                // For FixedFormat cases, figure out the B pointer.  The loop below moves through batches and vertically through the output so this will be the same throughout.
                if (FixedFormat) {
                    b_panel = reinterpret_cast<const Troi *>(g_arrays._Bptr) + (current.multi() * g_arrays._B_multi_stride) +
                                                                           ((current.x0() / get_stripe_width<strategy, FixedFormat>::get()) * g_arrays._ldb) +
                                                                           (current.k0() * get_stripe_width<strategy, FixedFormat>::get());
                }

                /* Do the actual work. */
                for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                    unsigned int first_m = (batch == batch_0)   ? m_0   : 0;
                    unsigned int last_m  = (batch == batch_end) ? m_max : _Msize;

                    const Tloi *a_ptr = a_panel + (batch * _Mround + first_m) * get_total_k_depth();

                    if (first_m >= last_m)
                        continue;

                    // For the merge case we need to do this out_height() rows
                    // at a time, as that is the size of our intermediate
                    // buffer.  If we are not doing that, we can do all the
                    // relevant rows in one go.
                    unsigned int m_step = MergeStep ? strategy::out_height() : (last_m - first_m);

                    // But in the case where we have an accumulation buffer, we can't do that after all, unless
                    // there is no N blocking.
                    if (accumulation_buffer && ((current.x0() != 0) || (current.xmax() < _Nsize))) {
                        m_step = strategy::out_height();
                    }

                    for (unsigned int y=first_m; y<last_m; y+=m_step) {
                        unsigned int ymax = std::min(_Msize, y + m_step);

                        const bool first_pass = (current.k0() == 0);
                        const bool last_pass  = (current.kmax() == _Ktotal);

                        // Bias is passed for the first pass only, except for dequantizefloat nomerge cases where it's the last pass.
                        const bool bias_pass = (std::is_same<OutputStage, DequantizeFloat>::value && !MergeStep) ? last_pass : first_pass;

                        // Pointer to appropriate part of result array.
                        Tr *result_ptr = g_arrays._Cptr + (batch * g_arrays._C_batch_stride) + (current.multi() * g_arrays._C_multi_stride);

                        // If we are using an accumulation buffer, we don't pass the result buffer to ask the kernel
                        // to write things into the accumulation buffer instead, except on the last pass.
                        if (accumulation_buffer && !last_pass) {
                            result_ptr = nullptr;
                        }

                        // Perform the kernel and merge step, either separately or together as required.
                        kernel_and_merge<MergeStep, FixedFormat, OutputStage>::run(
                        #ifdef CYCLE_PROFILING
                            prof,
                        #endif
                            // Strategy and panel pointers
                            strat, a_ptr, b_panel, g_arrays._ldb, c_panel,
                            // Result buffer pointers
                            result_ptr, g_arrays._ldc,
                            // K size, and M/N ranges
                            kern_k, y, ymax, current.x0(), current.xmax(),
                            // Only do bias on the first pass
                            ((bias_pass && g_arrays._bias) ? g_arrays._bias + (current.multi() * g_arrays._bias_multi_stride) : nullptr),
                            // Only do activation on the last pass, and accumulation on any non-first pass.
                            (last_pass ? _act : Activation()), (!first_pass || _accumulate),
                            // Pass in quantization parameters for requantizing kernels (others will ignore)
                            _os, col_bias ? col_bias + (current.multi() * _Nsize) : nullptr,
                            // Accumulation buffer
                            get_accumulation_buffer(accumulation_buffer, y, current.x0(), batch, current.multi()) );

                        a_ptr += (strategy::out_height() * a_panel_stride);
                    }
                }

                if (FixedFormat == false) {
                    b_panel += (roundup(current.xmax() - current.x0(), strategy::out_width()) * kern_k);
                }
            }
        }
    }

    // Stateless execute
    void execute_stateless(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid, GemmArrays<Tlo, Tro, Tr> &g_arrays) override {
        assert(FixedFormat);

        return execute_common(work_range, thread_locator, threadid, g_arrays);
    }

    // Execute
    void execute(const ndcoord_t &work_range, const ndcoord_t & thread_locator, int threadid) override {
        execute_common(work_range, thread_locator, threadid, this->_gemm_arrays);
    }

    // Interface implementation - working space
    size_t get_working_size() const override {
        // In all cases, we need one A buffer plus a C buffer per thread, plus an accumulation buffer.
        size_t size = get_a_working_size() + (get_c_working_size() * _maxthreads) + get_accumulation_buffer_size();

        size += 128; // Add on two cache lines extra for alignment.

        return size;
    }

    void set_working_space(void *working_space) override {
        // Make sure everything ends up cache line aligned
        auto *working_space_bytes = reinterpret_cast<int8_t *>(working_space);
        auto  working_space_addr = reinterpret_cast<uintptr_t>(working_space);
        working_space_bytes += get_cache_align_offset(working_space_addr);

        // Pretransposed case: just set internal pointer to parameter value.
        this->_gemm_arrays._workspace = reinterpret_cast<void *>(working_space_bytes);
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

        unsigned int x_size = roundup(_Nsize, strategy::out_width());

        return (x_size * _Ktotal * _nmulti * sizeof(Troi)) + get_col_sum_size();
    }

    size_t get_B_pretranspose_window_size() const override {
        size_t n_blocks = iceildiv(_Nsize, _x_block);
        size_t k_blocks = iceildiv(_Ktotal, _k_block);

        return n_blocks * k_blocks * _nmulti;
    }

    void requantize_bias(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride) override {
        if (std::is_same<OutputStage, Requantize32>::value) {
            col_bias = reinterpret_cast<int32_t *>(in_buffer);

            Requantize32 *qp_ptr = reinterpret_cast<Requantize32 *>(&_os);

            for (unsigned int i=0; i<_nmulti; i++) {
                // The input is assumed not to have any padding between sections, so straightforward Ksize * Ksections computation gets the total size.
                compute_col_sums(*qp_ptr, _Nsize, _Ksize * _Ksections, B + (i * B_multi_stride), ldb, col_bias + (i * _Nsize), _Ksize * _Ksections, i, 0);
            }
        }
    }

    // Support for transposed B is a property of the strategy::transpose type
    bool B_pretranspose_supports_transpose() const override {
        typename transform_type<strategy, MergeStep && std::is_same<OutputStage, Requantize32>::value>::type transforms;

        return transforms.PrepareB_supports_transpose();
    }

    void pretranspose_B_array(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride, const bool transposed) override {
        pretranspose_B_array_part(in_buffer, B, ldb, B_multi_stride, transposed, 0, get_B_pretranspose_window_size());
    }

    void pretranspose_B_array_part(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride, const bool transposed, size_t start, size_t end) override {
        // Perform column sums etc as part of the last block.
        if (end >= get_B_pretranspose_window_size()) {
            requantize_bias(in_buffer, B, ldb, B_multi_stride);
        }

        // Put the transposed data after the column sums - in non-quantized cases get_col_sum_size() == 0
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        Troi *buffer = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        _B_transposed = buffer;

        blockwalker current(*this);
        strategy strat(_ci);

        // Skip over blocks we aren't doing
        for(size_t i = 0; i < start; i++) {
            buffer += roundup(current.xmax() - current.x0(), strategy::out_width()) * roundup(current.kmax() - current.k0(), strategy::k_unroll());
            current.advance();
        }

        size_t blocks_left = (end - start);

        // Double check that we haven't run out of work
        if (current.done()) {
            blocks_left = 0;
        }

        for (/* blocks_left initialized above */; blocks_left > 0; blocks_left--) {
            /* Figure out the size of each block. */
            unsigned int k_size = (current.kmax() - current.k0());

            if (_Ksections > 1) {
                // We need to insert padding at the end of each K section.
                // The computation needed is a little delicate - the coordinates from the block walker are expressed in
                // terms of the full, padded, _Ktotal.
                // But we need to transform each section with reference to the original, unpadded, input, letting the
                // transform pad each section as needed.

                // This is needed for computations below.
                const unsigned int rounded_section_size = roundup(_Ksize, strategy::k_unroll());

                // The expected output format is also an entire <out_width> columns interleaved, then the next set of
                // columns, and so on.  This means, as we are breaking it up vertically, we have to do it one column at
                // a time.
                for (unsigned int x0=current.x0(); x0 < current.xmax(); x0 += strategy::out_width() ) {
                    unsigned int xmax = std::min(x0 + strategy::out_width(), current.xmax());

                    // Track where we are and how much work is left.
                    unsigned int kpos  = current.k0();
                    unsigned int kleft = k_size;

                    while (kleft) {
                        // Which section are we in?  Based on the rounded-up section size.
                        unsigned int k_section_base = kpos / rounded_section_size;
                        // How far into the section are we?
                        unsigned int k_offset = kpos - (k_section_base * rounded_section_size);

                        // We will either copy the rest of this section, or to the end of the requested length.
                        unsigned int k_length = std::min(_Ksize - k_offset, kleft);

                        strat.transforms.PrepareB(buffer, B + (current.multi() * B_multi_stride), ldb,
                                                  x0, xmax,
                                                  (k_section_base * _Ksize) + k_offset,               // K starting point - compute row to read based on our section and the true section length.
                                                  (k_section_base * _Ksize) + k_offset + k_length,    // K end point - starting point plus length computed above.
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
                // Caution: 'blockwalker::kmax()' rounds up, so clamp to valid _Ksize.
                strat.transforms.PrepareB(buffer, B + (current.multi() * B_multi_stride), ldb,
                                          current.x0(), current.xmax(), current.k0(), std::min(current.kmax(), _Ksize), transposed);
                buffer += roundup(current.xmax() - current.x0(), strategy::out_width()) * roundup(current.kmax() - current.k0(), strategy::k_unroll());
            }

            // Advance to the next block, break if we run off the end.
            if (!current.advance()) {
                break;
            }
        }
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        // Put the transposed data after the column sums - in non-quantized cases get_col_sum_size() == 0
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        _B_transposed = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        col_bias = reinterpret_cast<int32_t *>(in_buffer);
    }

    void set_quantized_bias(const int32_t *bias, size_t bias_multi_stride) override {
        if (std::is_same<OutputStage, Requantize32>::value) {
            Requantize32 *qp = reinterpret_cast<Requantize32 *>(&_os);

            qp->bias = bias;
            qp->bias_multi_stride = bias_multi_stride;
        }
    }

    void set_dequantize_scale(const float scale) override {
        if(std::is_same<OutputStage, DequantizeFloat>::value) {
            DequantizeFloat* df = reinterpret_cast<DequantizeFloat *>(&_os);
            df->scale = scale;
        }
    }

    void set_indirect_parameters(size_t string_len, const Tlo * const * const *ptr) override {
        assert(string_len == _Ksize);
        _indirect_buf = ptr;
    }

    void set_convolution_parameters(ConvolutionParameters parms) override {
        assert(parms.input_channels == _Ksize);
        _convolver = std::unique_ptr<convolver<Tlo>>(new convolver<Tlo>(parms));
    }

    // Estimate cycles for given problem given provided parameters
    template<typename perf_type>
    static uint64_t estimate_cycles(const GemmArgs &args) {
        unsigned int k_blocks = iceildiv(args._Ksize, get_k_block_size(args));

        const PerformanceParameters &params = strategy::template get_performance_parameters<perf_type>(args._ci);

        uint64_t total_macs    = static_cast<uint64_t>(args._nbatches) * args._nmulti * roundup(args._Msize, strategy::out_height()) * roundup(args._Nsize, strategy::out_width()) * get_ktotal(args);
        uint64_t prepare_bytes = static_cast<uint64_t>(args._nbatches) * args._nmulti * roundup(args._Msize, strategy::out_height()) * get_ktotal(args) * sizeof(Tloi);
        uint64_t merge_bytes   = static_cast<uint64_t>(args._nbatches) * args._nmulti * k_blocks * args._Msize * roundup(args._Nsize, strategy::out_width()) * sizeof(Tr);

        float mac_cycles     = static_cast<float>(total_macs) / params.kernel_macs_cycle;
        float prepare_cycles = static_cast<float>(prepare_bytes) / params.prepare_bytes_cycle;
        float merge_cycles   = static_cast<float>(merge_bytes) / params.merge_bytes_cycle;

        float total_cycles = mac_cycles + prepare_cycles + merge_cycles;

        // We can't thread over multis or width, which makes this a poor
        // choice in many threaded cases.  Penalize that here.
        float parallelism_available = static_cast<float>(iceildiv(args._Msize, strategy::out_height()) * args._nbatches) * 0.9f;

        if (parallelism_available < args._maxthreads) {
            total_cycles *= (static_cast<float>(args._maxthreads) / parallelism_available);
        }

        return static_cast<uint64_t>(total_cycles);
    }

    GemmConfig get_config() override {
        GemmConfig c;

        c.inner_block_size = _k_block;
        c.outer_block_size = _x_block;
        c.filter = get_type_name<strategy>();
        c.weight_format = get_weight_format(get_kernel_weight_format<strategy, FixedFormat, Tro>::get(), sizeof(Tro));

        return c;
    }
};

// Aliases for the variations
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing>
using GemmInterleavedNoMerge = GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, false>;

template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing>
using GemmInterleavedNoMergeFloatAcc = GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, false, false, false, true>;

template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing>
using GemmInterleavedFixedFormat = GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, true, true>;

template<typename strategy, typename Tlo, typename Tro, typename Tr>
using GemmInterleavedPretransposedNoMergeQuantizedInline = GemmInterleaved<strategy, Tlo, Tro, Tr, Requantize32, false>;

template<typename strategy, typename Tlo, typename Tro, typename Tr>
using GemmInterleavedQuantized = GemmInterleaved<strategy, Tlo, Tro, Tr, Requantize32>;

template<typename strategy, typename Tlo, typename Tro, typename Tr>
using GemmInterleavedNoMergeDequantized = GemmInterleaved<strategy, Tlo, Tro, Tr, DequantizeFloat, false>;

template<typename strategy, typename Tlo, typename Tro, typename Tr>
using GemmInterleavedDequantized = GemmInterleaved<strategy, Tlo, Tro, Tr, DequantizeFloat>;

}  // namespace ops
}  // namespace kai
