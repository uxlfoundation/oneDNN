//
// SPDX-FileCopyrightText: Copyright 2017-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cassert>
#include <limits>

#include "gemm_driver_common.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai/ops/float16.hpp"
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

// Select row-summing transforms only for a separate requantization step.
template<typename Strategy, bool Quantized>
struct interleaved_transform_type {
    using type = decltype(Strategy::transforms);
};

template<typename Strategy>
struct interleaved_transform_type<Strategy, true> {
    using type = decltype(Strategy::transforms_quantized);
};

// Direct-output kernels consume the driver's KernelArgs unchanged. Separate
// merge kernels write one row of tiles to scratch, using element counts too.
template<bool MergeStep, typename OutputStage>
struct interleaved_kernel_and_merge {
    template<typename Strategy, typename Tr>
    static void run(
#ifdef CYCLE_PROFILING
                    profiler &prof,
#endif
                    Strategy &strat, KernelArgs ka, typename Strategy::result_type *,
                    const OutputStage &, const Activation &, unsigned int) {
#ifdef CYCLE_PROFILING
        auto p = prof.ScopedProfiler(PROFILE_KERNEL, ka.M * ka.N * ka.K);
#endif
        strat.kernel(&ka);
    }
};

template<typename OutputStage>
struct interleaved_kernel_and_merge<true, OutputStage> {
    template<typename Strategy, typename Tr>
    static void merge(Strategy &strat, const KernelArgs &ka, typename Strategy::result_type *panel,
                      Tr *dst, unsigned int rows, const Nothing &, const Activation &, unsigned int) {
        KernelArgs merge_args = ka;
        merge_args.lhs = panel;
        merge_args.dst = dst;
        merge_args.M = rows;
        strat.transforms.template Merge<Tr>(&merge_args);
    }

    template<typename Strategy, typename Tr>
    static void merge(Strategy &, const KernelArgs &ka, typename Strategy::result_type *panel,
                      Tr *dst, unsigned int rows, const Requantize32 &qp, const Activation &, unsigned int column) {
        const auto *row_bias = reinterpret_cast<const int32_t *>(
            static_cast<const typename Strategy::lhs_operand_type *>(ka.lhs) + Strategy::out_height() * ka.K);
        for (unsigned int n = 0; n < ka.N; n += Strategy::out_width()) {
            requantize_block_32(qp, std::min<unsigned int>(Strategy::out_width(), ka.N - n), rows,
                               panel + n * Strategy::out_height(), Strategy::out_width(),
                               dst + n, ka.dst_stride / sizeof(Tr), row_bias,
                               static_cast<const int32_t *>(ka.bias) + n, column + n);
        }
    }

    template<typename Strategy, typename Tr>
    static void merge(Strategy &, const KernelArgs &ka, typename Strategy::result_type *panel,
                      Tr *dst, unsigned int rows, const DequantizeFloat &dq, const Activation &act, unsigned int) {
        const auto *bias = static_cast<const Tr *>(ka.late_bias);
        for (unsigned int n = 0; n < ka.N; n += Strategy::out_width()) {
            dequantize_block_32(dq, std::min<unsigned int>(Strategy::out_width(), ka.N - n), rows,
                               panel + n * Strategy::out_height(), Strategy::out_width(),
                               dst + n, ka.dst_stride / sizeof(Tr), bias ? bias + n : nullptr,
                               ka.accumulate, act);
        }
    }

    template<typename Strategy, typename Tr>
    static void run(
#ifdef CYCLE_PROFILING
                    profiler &prof,
#endif
                    Strategy &strat, KernelArgs ka, typename Strategy::result_type *panel,
                    const OutputStage &os, const Activation &act, unsigned int column) {
        const unsigned int height = ka.M;
        auto *dst = static_cast<Tr *>(ka.dst);
        for (unsigned int m = 0; m < height; m += Strategy::out_height()) {
            KernelArgs kernel_args = ka;
            kernel_args.M = std::min(Strategy::out_height(), height - m);
            kernel_args.dst = panel;
            {
#ifdef CYCLE_PROFILING
                auto p = prof.ScopedProfiler(PROFILE_KERNEL, Strategy::out_height() *
                    roundup<int64_t>(ka.N, Strategy::out_width()) * ka.K);
#endif
                strat.kernel(&kernel_args);
            }
            {
#ifdef CYCLE_PROFILING
                auto p = prof.ScopedProfiler(std::is_same<OutputStage, Nothing>::value ? PROFILE_MERGE : PROFILE_QUANTIZE,
                    kernel_args.M * ka.N * sizeof(Tr));
#endif
                merge(strat, ka, panel, dst, kernel_args.M, os, act, column);
            }
            ka.lhs = static_cast<const typename Strategy::lhs_operand_type *>(ka.lhs) +
                     Strategy::out_height() * (ka.K + (std::is_same<OutputStage, Requantize32>::value ?
                         sizeof(int32_t) / sizeof(typename Strategy::lhs_operand_type) : 0));
            dst += Strategy::out_height() * ka.dst_stride / sizeof(Tr);
        }
    }
};

// Calculate the offset needed if the address is not a multiple of cache-line length
inline size_t get_cache_align_offset(uintptr_t addr) {
    return (addr & 0x3F) ? 0x40 - (addr & 0x3F) : 0;
}

} // anonymous namespace

// FixedFormat describes the kernel's RHS layout. GemmArgs::_fixed_format
// describes the supplied weights: when false, the strategy's transforms pack
// them into either the fixed-format stripes or the ordinary blocked layout.
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing, bool MergeStep=true, bool FixedFormat=false, bool ForceThreadColumns=false>
class GemmInterleaved : public GemmDriverCommon<strategy, Tlo, Tro, Tr, OutputStage, typename strategy::rhs_operand_type, FixedFormat> {
    using Base = GemmDriverCommon<strategy, Tlo, Tro, Tr, OutputStage, typename strategy::rhs_operand_type, FixedFormat>;
    using Base::_args;
    using Base::_os;
    using Base::_B_transposed;
    using Base::col_bias;
    using Base::get_col_sum_size;
    using Base::_Ktotal;
    using Base::_rounded_Ksize;
    using Base::get_ktotal;
    using Base::set_kernel_output_args;

    typedef typename strategy::lhs_operand_type Tloi;
    typedef typename strategy::rhs_operand_type Troi;

    void set_activation_args(KernelArgs &ka, bool last_pass) const {
        ka.clamp_min = std::is_integral<Tr>::value ? std::numeric_limits<Tr>::lowest() :
            static_cast<Tr>(-std::numeric_limits<float>::infinity());
        ka.clamp_max = std::is_integral<Tr>::value ? std::numeric_limits<Tr>::max() :
            static_cast<Tr>(std::numeric_limits<float>::infinity());
        // Intermediate K passes must remain unclamped.
        if (last_pass) {
            switch (_args._act.type) {
                default:
                case Activation::Type::None:
                    break;
                case Activation::Type::BoundedReLU:
                    ka.clamp_max = static_cast<Tr>(_args._act.param1);
                    /* fall through */
                case Activation::Type::ReLU:
                    ka.clamp_min = static_cast<Tr>(0);
            }
        }
    }

    const bool _thread_columns;
    int _nthreads;

    /* Blocking info */
    unsigned int _k_block=0;
    unsigned int _x_block=0;
    unsigned int _Mround=0;

    /* Indirect parameters.  _indirect_buf doubles as a flag to indicate that "indirect" transform should be used. */
    const Tlo * const * const * _indirect_buf = nullptr;

    /* Convolver - only set up for convolution problems, so also doubles as a flag. */
    std::unique_ptr<convolver<Tlo>>  _convolver = nullptr;

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
            return ROUND_UP(sizeof(Tloi) * get_total_k_depth() * strategy::out_height() * _args._maxthreads);
        } else {
            // For 1D threaded: one of these needed, regardless of thread count.  Divided according to window.
            return ROUND_UP(sizeof(Tloi) * get_total_k_depth() * _Mround * _args._nbatches);
        }
    }

    // One row of output tiles per thread for a separate merge step.
    size_t get_c_working_size() const {
        return MergeStep ? ROUND_UP(sizeof(typename strategy::result_type) * _x_block * strategy::out_height()) : 0;
    }

    // Accumulation buffer size
    size_t get_accumulation_buffer_size() const {
        // Check if we are actually blocking
        if (_k_block == _Ktotal) {
            return 0;
        }

        // Check if this kernel uses an accumulation buffer.
        if (MergeStep || !strategy::accumulation_buffer_element_size()) {
            return 0;
        }

        // Active K blocking requires an accumulation buffer for this strategy.
        size_t size_per_buffer = strategy::accumulation_buffer_element_size() * strategy::out_height() * strategy::out_width();
        size_t num_buffers = iceildiv(_args._Msize, strategy::out_height()) * iceildiv(_args._Nsize, strategy::out_width()) * _args._nbatches * _args._nmulti;

        return num_buffers * size_per_buffer;
    }

    // Set up accumulation buffer
    uint8_t *get_accumulation_buffer_offset(void *working_space) const {
        uint8_t *accumulation_buffer = nullptr;

        auto working_space_addr = reinterpret_cast<uintptr_t>(working_space);

        if (get_accumulation_buffer_size() > 0) {
            auto acc_buff_addr = working_space_addr + get_a_working_size();
            acc_buff_addr += get_cache_align_offset(acc_buff_addr);
            accumulation_buffer = reinterpret_cast<uint8_t *>(acc_buff_addr);
        }

        return accumulation_buffer;
    }

    // Get pointer into accumulation buffer
    uint8_t *get_accumulation_buffer(uint8_t *accumulation_buffer, int64_t M, int64_t N, int64_t batch, int64_t multi) const {
        // Don't do anything if there's no buffer.
        if (accumulation_buffer == nullptr) {
            return nullptr;
        }

        // Buffer pointers and tile offsets are measured in bytes.
        size_t size_per_buffer = strategy::accumulation_buffer_element_size() * strategy::out_height() * strategy::out_width();

        size_t buffer_rows = iceildiv(_args._Msize, strategy::out_height());
        size_t buffer_cols = iceildiv(_args._Nsize, strategy::out_width());
        size_t buffers_per_batch = (buffer_rows * buffer_cols);
        size_t buffers_per_multi = buffers_per_batch * _args._nbatches;

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

    static unsigned int get_k_block_size(const GemmArgs &args) {
        // Separate requantization cannot accumulate partial K results. Avoid
        // repeated rounding for reduced-precision outputs as in the old driver.
        if (MergeStep && (std::is_same<OutputStage, Requantize32>::value ||
            (std::is_same<Tr, __fp16>::value && (!args._fast_mode || std::is_same<Tro, int8_t>::value)) ||
            std::is_same<Tr, bfloat16>::value)) {
            return get_ktotal(args);
        }

        if (args._cfg && args._cfg->inner_block_size) {
            return roundup(args._cfg->inner_block_size, strategy::k_unroll());
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


    unsigned int get_rhs_packing_k_block() const override { return _k_block; }
    unsigned int get_rhs_packing_n_block() const override { return _x_block; }

public:
    GemmInterleaved(GemmInterleaved &) = delete;
    GemmInterleaved & operator= (GemmInterleaved &) = delete;

    /* Constructor */
    GemmInterleaved(const GemmArgs &args, const OutputStage &os)
                    : Base(args, os), _thread_columns(is_thread_columns(args)), _nthreads(args._maxthreads),
                      _k_block(get_k_block_size(args)), _x_block(get_x_block_size(args)), _Mround(roundup(args._Msize, strategy::out_height())) {
        assert(!_args._fixed_format || FixedFormat);
    }

    /* Constructor without OutputStage */
    GemmInterleaved(const GemmArgs &args)
                    : GemmInterleaved(args, OutputStage{}) {}

    // Interface implementation - Compulsory functions

    // Window size: Only the last thread should do a ragged block, so dole
    // out work in units of out_height.  Factor batches into the window, but
    // not multi for now (as this would cause problems with the buffer
    // manager).
    ndrange_t get_window_size() const override {
        unsigned int row_blocks = (_Mround / strategy::out_height()) * _args._nbatches;

        if (_thread_columns) {
            return { row_blocks, iceildiv(_args._Nsize, strategy::out_width()) };
        } else {
            // _Mround is a multiple of out_height by definition.
            return { row_blocks };
        }
    }

    // set_nthreads: pass on to buffer manager to avoid it waiting for non-existant threads.
    void set_nthreads(int nthreads) override {
        _nthreads = std::min(nthreads, _args._maxthreads);
    }

    // Common execution logic.
    void execute_common(const ndcoord_t &work_range, const ndcoord_t &, int threadid, GemmArrays<Tlo, Tro, Tr> &g_arrays) {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        KernelArgs ka{};

        void *working_space = g_arrays._workspace;

        // Make sure we've been set up correctly.
        assert(_args._fixed_format || _B_transposed);
        assert(working_space);

        // Align if needed.
        auto *working_space_bytes = reinterpret_cast<int8_t *>(working_space);
        auto working_space_addr = reinterpret_cast<uintptr_t>(working_space);
        working_space_bytes += get_cache_align_offset(working_space_addr);
        working_space = reinterpret_cast<void *>(working_space_bytes);

        auto *c_panel = reinterpret_cast<typename strategy::result_type *>(
            working_space_bytes + get_a_working_size() + threadid * get_c_working_size());
        auto *accumulation_buffer = get_accumulation_buffer_offset(working_space);
        // Byte stride between rows of accumulator tiles, including columns
        // outside the current work range.
        ka.accumulation_buffer_stride = strategy::accumulation_buffer_element_size() * strategy::out_height() *
                                        roundup(_args._Nsize, strategy::out_width());

        strategy strat(_args._ci);

        const int64_t start = work_range.get_position(0);
        const int64_t end   = work_range.get_position_end(0);

        /* Translate 'start' and 'end' into a position within the batches and rows. */
        const unsigned int window_per_batch = _Mround / strategy::out_height();
        unsigned int batch_0   = start / window_per_batch;
        unsigned int batch_end = end   / window_per_batch;

        const int64_t rounded_width = roundup(_args._Nsize, strategy::out_width());
        const unsigned int stripe_width = driver_stripe_width<strategy, FixedFormat>::get();

        // RHS pointers and stride come from either g_arrays or
        // _B_transposed depending on the fixed_format setting.
        const Troi    *rhs_base         = _args._fixed_format ? reinterpret_cast<const Troi *>(g_arrays._Bptr)           : _B_transposed;
        const int64_t  rhs_multi_stride = _args._fixed_format ? g_arrays._B_multi_stride : (rounded_width * _Ktotal);
        const int64_t  rhs_row_stride   = _args._fixed_format ? g_arrays._ldb            : (_Ktotal * stripe_width);

        // In ThreadColumns mode, process work one horizontal strip at a time.
        // Transpose the block of needed rows at the start, then do all the work on that block.
        if (_thread_columns) {
            const int64_t start_x = work_range.get_position(1) * strategy::out_width();
            const int64_t end_x = std::min(work_range.get_position_end(1) * strategy::out_width(), _args._Nsize);

            Tloi * const a_panel = reinterpret_cast<Tloi *>(working_space_bytes +
                                       (threadid * sizeof(Tloi) * get_total_k_depth() * strategy::out_height()));

            for (unsigned int multi=0; multi<_args._nmulti; multi++) {
                for (unsigned int k0=0; k0<_Ktotal; k0+=_k_block) {
                    unsigned int kmax=std::min(k0+_k_block, _Ktotal);

                    const bool first_pass = (k0==0);
                    const bool last_pass  = (kmax==_Ktotal);

                    set_activation_args(ka, last_pass);

                    // Fused dequantization applies bias on the last pass; separate merges use the first.
                    const bool bias_pass = (std::is_same<OutputStage, DequantizeFloat>::value && !MergeStep) ? last_pass : first_pass;

                    // Figure out how many "K" the kernel will actually process.
                    int64_t kern_k = roundup(kmax - k0, strategy::k_unroll());

                    const Troi * const b_ptr = rhs_base + (multi * rhs_multi_stride) +
                                        (FixedFormat ?
                                            ((start_x / stripe_width) * rhs_row_stride + k0 * stripe_width) :
                                            (k0 * rounded_width + start_x * kern_k));

                    unsigned int batch     = batch_0;
                    unsigned int start_row = (start - (batch_0 * window_per_batch)) * strategy::out_height();

                    for (unsigned int p=start; p<end; p++) {
                        unsigned int end_row = std::min(start_row + strategy::out_height(), _args._Msize);

                        // Set up transposed 'A' block
                        {
#ifdef CYCLE_PROFILING
                            auto p=prof.ScopedProfiler(PROFILE_PREPA, strategy::out_height() * (kmax-k0) * sizeof(Tloi));
#endif
                            typename interleaved_transform_type<strategy, MergeStep && std::is_same<OutputStage, Requantize32>::value>::type transforms;

                            if (_indirect_buf != nullptr) {
                                transforms.PrepareA_indirect(a_panel,
                                                             _indirect_buf + (multi * _args._nbatches * _args._Ksections) + (batch * _args._Ksections), _args._Ksize,
                                                             _rounded_Ksize, start_row, end_row, k0, kmax, row_sum_multiplier());
                            } else if (_convolver) {
                                transforms.PrepareA_convolution(a_panel,
                                                                g_arrays._Aptr + (batch * g_arrays._A_batch_stride) + (multi * g_arrays._A_multi_stride),
                                                                g_arrays._lda, *_convolver, _rounded_Ksize, start_row, end_row, k0, kmax, row_sum_multiplier());
                            } else {
                                transforms.PrepareA(a_panel,
                                                    g_arrays._Aptr + (batch * g_arrays._A_batch_stride) + (multi * g_arrays._A_multi_stride),
                                                    g_arrays._lda, start_row, end_row, k0, std::min(kmax, _args._Ksize), row_sum_multiplier());
                            }
                        }

                        Tr *result_ptr = g_arrays._Cptr + (batch * g_arrays._C_batch_stride) + (multi * g_arrays._C_multi_stride) + (start_row * g_arrays._ldc) + start_x;

                        // If we are using an accumulation buffer and this isn't the last pass, don't pass a result pointer.
                        if (accumulation_buffer && !last_pass) {
                            result_ptr = nullptr;
                        }

                        // Set KernelArgs parameter for kernel call
                        ka.M = (end_row - start_row);
                        ka.N = (end_x - start_x);
                        ka.K = kern_k;
                        ka.lhs = a_panel;
                        ka.lhs_stride = 0;
                        ka.rhs = b_ptr;
                        ka.rhs_stride = rhs_row_stride * sizeof(Troi);
                        ka.bias = ((bias_pass && g_arrays._bias) ? g_arrays._bias + (multi * g_arrays._bias_multi_stride) + start_x : nullptr);
                        ka.dst = result_ptr;
                        ka.dst_stride = g_arrays._ldc * sizeof(Tr);
                        ka.accumulate = (!first_pass || _args._accumulate);
                        ka.accumulation_buffer = get_accumulation_buffer(accumulation_buffer, start_row, start_x, batch, multi);

                        // Set OutputStage specific parameters
                        set_kernel_output_args(ka, multi, start_x, _os);

                        // Run the kernel
                        interleaved_kernel_and_merge<MergeStep, OutputStage>::template run<strategy, Tr>(
#ifdef CYCLE_PROFILING
                            prof,
#endif
                            strat, ka, c_panel, _os, last_pass ? _args._act : Activation(), start_x);

                        /* Increment to the next block */
                        start_row += strategy::out_height();
                        if (start_row >= _args._Msize) {
                            start_row = 0;
                            batch++;
                        }
                    }
                }
            }
        } else {
            /* Compute the M values to operate on. */
            const unsigned int m_0 = (start - (batch_0 * window_per_batch)) * strategy::out_height();
            const unsigned int m_max = (end - (batch_end * window_per_batch)) * strategy::out_height();

            // The row-threaded path has one shared A buffer, divided according
            // to the window, rather than one single-row block per thread.
            Tloi * const a_panel = reinterpret_cast<Tloi *>(working_space_bytes);

            for (unsigned int multi = 0; multi < _args._nmulti; multi++) {
                for (unsigned int k0 = 0; k0 < _Ktotal; k0 += _k_block) {
                    const unsigned int kmax = std::min(k0 + _k_block, _Ktotal);
                    const int64_t kern_k = roundup(kmax - k0, strategy::k_unroll());

#ifdef CYCLE_PROFILING
                    auto p=prof.ScopedProfiler(PROFILE_PREPA, (end - start) * strategy::out_height() * (kmax-k0) * sizeof(Tloi));
#endif
                    typename interleaved_transform_type<strategy, MergeStep && std::is_same<OutputStage, Requantize32>::value>::type transforms;

                    for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                        const unsigned int first_m = (batch == batch_0) ? m_0 : 0;
                        const unsigned int last_m = (batch == batch_end) ? m_max : _args._Msize;

                        if (first_m >= last_m) {
                            continue;
                        }

                        Tloi * const panel = a_panel + ((batch * _Mround + first_m) * get_total_k_depth());

                        if (_indirect_buf != nullptr) {
                            transforms.PrepareA_indirect(panel,
                                                        _indirect_buf + (multi * _args._nbatches * _args._Ksections) + (batch * _args._Ksections),
                                                        _args._Ksize, _rounded_Ksize, first_m, last_m,
                                                        k0, kmax, row_sum_multiplier());
                        } else if (_convolver) {
                            transforms.PrepareA_convolution(panel,
                                                           g_arrays._Aptr + (batch * g_arrays._A_batch_stride) +
                                                               (multi * g_arrays._A_multi_stride),
                                                           g_arrays._lda, *_convolver, _rounded_Ksize, first_m, last_m,
                                                           k0, kmax, row_sum_multiplier());
                        } else {
                            transforms.PrepareA(panel,
                                                g_arrays._Aptr + (batch * g_arrays._A_batch_stride) +
                                                    (multi * g_arrays._A_multi_stride),
                                                g_arrays._lda, first_m, last_m, k0,
                                                std::min(_args._Ksize, kmax), row_sum_multiplier());
                        }
                    }

                    const bool first_pass = (k0 == 0);
                    const bool last_pass = (kmax == _Ktotal);
                    const bool bias_pass = (std::is_same<OutputStage, DequantizeFloat>::value && !MergeStep) ? last_pass : first_pass;

                    set_activation_args(ka, last_pass);

                    for (unsigned int x0 = 0; x0 < _args._Nsize; x0 += _x_block) {
                        const unsigned int xmax = std::min(x0 + _x_block, _args._Nsize);
                        const Troi * const b_ptr = rhs_base + (multi * rhs_multi_stride) +
                                                    (FixedFormat ?
                                                        ((x0 / stripe_width) * rhs_row_stride + k0 * stripe_width) :
                                                        (k0 * rounded_width + x0 * kern_k));

                        for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                            const unsigned int first_m = (batch == batch_0) ? m_0 : 0;
                            const unsigned int last_m = (batch == batch_end) ? m_max : _args._Msize;

                            if (first_m >= last_m) {
                                continue;
                            }

                            Tr *result_ptr = g_arrays._Cptr + (batch * g_arrays._C_batch_stride) +
                                             (multi * g_arrays._C_multi_stride) +
                                             (first_m * g_arrays._ldc) + x0;

                            if (accumulation_buffer && !last_pass) {
                                result_ptr = nullptr;
                            }

                            ka.M = last_m - first_m;
                            ka.N = xmax - x0;
                            ka.K = kern_k;
                            ka.lhs = a_panel + ((batch * _Mround + first_m) * get_total_k_depth());
                            ka.lhs_stride = 0;
                            ka.rhs = b_ptr;
                            ka.rhs_stride = (FixedFormat ? rhs_row_stride : kern_k * strategy::out_width()) * sizeof(Troi);
                            ka.bias = ((bias_pass && g_arrays._bias) ?
                                           g_arrays._bias + (multi * g_arrays._bias_multi_stride) + x0 :
                                           nullptr);
                            ka.dst = result_ptr;
                            ka.dst_stride = g_arrays._ldc * sizeof(Tr);
                            ka.accumulate = (!first_pass || _args._accumulate);
                            ka.accumulation_buffer = get_accumulation_buffer(accumulation_buffer, first_m, x0, batch, multi);

                            set_kernel_output_args(ka, multi, x0, _os);
                            interleaved_kernel_and_merge<MergeStep, OutputStage>::template run<strategy, Tr>(
#ifdef CYCLE_PROFILING
                            prof,
#endif
                                strat, ka, c_panel, _os, last_pass ? _args._act : Activation(), x0);
                        }
                    }
                }
            }
        }
    }

    // Stateless execute
    void execute_stateless(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid, GemmArrays<Tlo, Tro, Tr> &g_arrays) override {
        assert(_args._fixed_format);

        return execute_common(work_range, thread_locator, threadid, g_arrays);
    }

    // Execute
    void execute(const ndcoord_t &work_range, const ndcoord_t & thread_locator, int threadid) override {
        execute_common(work_range, thread_locator, threadid, this->_gemm_arrays);
    }

    // Interface implementation - working space
    size_t get_working_size() const override {
        // Workspace contains packed A, per-thread merge scratch and any accumulation buffer.
        size_t size = get_a_working_size() + _args._maxthreads * get_c_working_size() + get_accumulation_buffer_size();

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
    size_t get_B_pretransposed_array_size() const override {
        if (_args._fixed_format) {
            return 0;
        }

        unsigned int x_size = roundup(_args._Nsize, strategy::out_width());

        return (x_size * _Ktotal * _args._nmulti * sizeof(Troi)) + get_col_sum_size();
    }

    // Support for transposed B is a property of the strategy::transpose type
    bool B_pretranspose_supports_transpose() const override {
        typename interleaved_transform_type<strategy, MergeStep && std::is_same<OutputStage, Requantize32>::value>::type transforms;

        return transforms.PrepareB_supports_transpose();
    }

    void set_indirect_parameters(size_t string_len, const Tlo * const * const *ptr) override {
        assert(string_len == _args._Ksize);
        _indirect_buf = ptr;
    }

    void set_convolution_parameters(ConvolutionParameters parms) override {
        assert(parms.input_channels == _args._Ksize);
        _convolver = std::unique_ptr<convolver<Tlo>>(new convolver<Tlo>(parms));
    }

    // Estimate cycles for given problem given provided parameters
    template<typename perf_type>
    static uint64_t estimate_cycles(const GemmArgs &args) {
        const PerformanceParameters &params = strategy::template get_performance_parameters<perf_type>(args._ci);

        uint64_t total_macs    = static_cast<uint64_t>(args._nbatches) * args._nmulti * roundup(args._Msize, strategy::out_height()) * roundup(args._Nsize, strategy::out_width()) * get_ktotal(args);
        uint64_t prepare_bytes = static_cast<uint64_t>(args._nbatches) * args._nmulti * roundup(args._Msize, strategy::out_height()) * get_ktotal(args) * sizeof(Tloi);

        float mac_cycles     = static_cast<float>(total_macs) / params.kernel_macs_cycle;
        float prepare_cycles = static_cast<float>(prepare_bytes) / params.prepare_bytes_cycle;

        float total_cycles = mac_cycles + prepare_cycles;
        if (MergeStep) {
            uint64_t merge_bytes = static_cast<uint64_t>(args._nbatches) * args._nmulti *
                iceildiv(get_ktotal(args), get_k_block_size(args)) * args._Msize *
                roundup(args._Nsize, strategy::out_width()) * sizeof(Tr);
            total_cycles += static_cast<float>(merge_bytes) / params.merge_bytes_cycle;
        }

        // We can't thread over multis or width, which makes this a poor
        // choice in many threaded cases.  Penalize that here.
        float parallelism_available = static_cast<float>(iceildiv(args._Msize, strategy::out_height()) * args._nbatches) * 0.9f;

        if (parallelism_available < args._maxthreads) {
            total_cycles *= (static_cast<float>(args._maxthreads) / parallelism_available);
        }

        return static_cast<uint64_t>(total_cycles);
    }

    GemmConfig get_config() override {
        return this->get_config_common(_k_block, _x_block);
    }
};

// Aliases for the variations.
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage=Nothing>
using GemmInterleavedNoMerge = GemmInterleaved<strategy, Tlo, Tro, Tr, OutputStage, false>;

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
