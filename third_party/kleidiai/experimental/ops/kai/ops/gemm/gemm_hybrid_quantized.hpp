//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <assert.h>

#include <algorithm>

#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"
#include "common_internal/utils.hpp"

#ifdef CYCLE_PROFILING
#include "kai/ops/profiler.hpp"
#endif

namespace kai {
namespace ops {

// Implementation of the GemmCommon abstract class.
template<typename strategy, typename Tlo, typename Tro, typename Tr>
class GemmHybridQuantized : public GemmCommon<Tlo, Tro, Tr> {
    typedef typename strategy::operand_type Tloi;
    typedef typename strategy::operand_type Troi;
    typedef typename strategy::result_type Tri;

    /* const properties set by constructor */
    const CPUInfo * const _ci;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nbatches;
    const unsigned int _nmulti;

    /* Blocking info */
    const unsigned int _k_block;
    const unsigned int _n_block;
    const unsigned int _Mround;

    /* Pretransposed buffer. */
    const Troi *_B_transposed=nullptr;

    const NDRange<4> _window_range;

    Requantize32  _qp;
    int32_t *row_bias = nullptr;
    int32_t *col_bias = nullptr;

    unsigned int _nthreads;

    unsigned int get_col_sum_size() const {
        return _Nsize * _nmulti * sizeof(int32_t);
    }

    static unsigned int compute_k_block(const GemmArgs &args) {
        // We don't support K blocks as we only temporarily store 32 bit results.
        return args._Ksize;

        if (args._cfg && args._cfg->inner_block_size) {
            return args._cfg->inner_block_size;
        }

        const unsigned int L1_size = args._ci->get_L1_cache_size();

        // k_block: Find out how much of the larger array can be loaded into half the cache.
        // This should account for associative caches.
        unsigned int k_block = (L1_size / 2) / (sizeof(Tloi) * (std::max(strategy::out_width(), strategy::out_height())));

        // Needs to be (at least a single) multiple of the K unroll level.
        k_block /= strategy::k_unroll();
        k_block = std::max(k_block, 1U) * strategy::k_unroll();

        // Now tune to presented problem size; this is how many blocks we need.
        unsigned int numk_blocks = iceildiv(args._Ksize, k_block);

        // So divide the space equally into that many blocks.
        k_block = iceildiv(args._Ksize, numk_blocks);

        // And round UP to the K unroll level required.
        k_block = roundup(k_block, strategy::k_unroll());

        return k_block;
    }

    static unsigned int compute_n_block(const GemmArgs &args) {
        if (args._cfg && args._cfg->outer_block_size) {
            unsigned int n_block = args._cfg->outer_block_size;

            // Needs to be (at least a single) multiple of the kernel output width.
            n_block /= strategy::out_width();
            n_block = std::max(n_block, 1u) * strategy::out_width();

            return n_block;
        }

        const unsigned int k_block = compute_k_block(args);
        const unsigned int L2_size = args._ci->get_L2_cache_size();

        // n_block: Work out how many rows (of length k_block) will fit in the L2
        // Don't allocate more than 90% of the L2 to allow for overheads, and subtract off the L1 contents.
        const unsigned int scaled_l2_size = (L2_size * 9) / 10;
        const unsigned int k_block_area = k_block * ((sizeof(Tloi) * strategy::out_height()) + (sizeof(Troi) * strategy::out_width()));

        // .. if the L1 contents is bigger than the L2, just return a minimal size block.
        if (k_block_area > scaled_l2_size) {
            return strategy::out_width();
        }

        unsigned int n_block = (scaled_l2_size - k_block_area) / (sizeof(Troi) * k_block);

        // Needs to be (at least a single) multiple of the kernel output width.
        n_block /= strategy::out_width();
        n_block = std::max(n_block, 1u) * strategy::out_width();

        // And tune to the presented problem size.
        unsigned int numblocks = iceildiv(args._Nsize, n_block);
        n_block = iceildiv(args._Nsize, numblocks);
        n_block = roundup(n_block, strategy::out_width());

        assert(n_block > 0);

        return n_block;
    }

public:
    GemmHybridQuantized(GemmHybridQuantized &) = delete;
    GemmHybridQuantized & operator= (GemmHybridQuantized &) = delete;

    /* Constructor */
    GemmHybridQuantized(const GemmArgs &args, const Requantize32 &qp)
              : _ci(args._ci), _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize),
                _nbatches(args._nbatches), _nmulti(args._nmulti),
                _k_block(compute_k_block(args)), _n_block(compute_n_block(args)),
                _Mround(roundup(args._Msize, strategy::out_height())),
                _window_range(iceildiv(args._Msize, strategy::out_height()), _nbatches, iceildiv(_Nsize, _n_block), _nmulti),
                _qp (qp), _nthreads(args._maxthreads) { }

    // Interface implementation - Compulsory functions
    ndrange_t get_window_size() const override {
        return { _window_range.total_size() };
    }

    // This kernel can always be dynamically scheduled.
    bool supports_dynamic_scheduling() const override {
        return true;
    }

    // Execute
    void execute(const ndcoord_t &work_range, const ndcoord_t &, int threadid) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        auto &g_arrays = this->_gemm_arrays;

        void *working_space = g_arrays._workspace;
        auto working_int = reinterpret_cast<uintptr_t>(working_space);

        auto *result_buffer = reinterpret_cast<Tri *>(working_int + (threadid * strategy::out_height() * _Nsize * sizeof(Tri)));

        /* Make sure we've been set up correctly. */
        assert(_B_transposed);
        static_assert(std::is_same<Tlo, Tloi>::value, "gemm_native: LHS operand types must be the same.");

        /* For now, each work item implies all the K for a given output
         * pixel (so we don't need to synchronize access to the output
         * array).  So separate the loop over K blocks here.  */
        for (unsigned int k0=0; k0<_Ksize; k0+=_k_block) {
            unsigned int kmax   = std::min(k0 + _k_block, _Ksize);
            unsigned int kern_k = roundup(kmax-k0, strategy::k_unroll());

            auto p = _window_range.iterator(work_range.get_position(0), work_range.get_position_end(0));

            if (p.done()) {
                return;
            }

            do {
                const unsigned int m_start = p.dim(0) * strategy::out_height();
                const unsigned int m_end   = std::min((p.dim(0) + 1) * strategy::out_height(), _Msize);
                const unsigned int batch   = p.dim(1);
                const unsigned int n0      = p.dim(2) * _n_block;
                const unsigned int nmax    = std::min(n0 + _n_block, _Nsize);
                const unsigned int multi   = p.dim(3);

                int32_t local_row_sums[strategy::out_height()];

                const Troi *b_panel = _B_transposed +
                                     (multi * roundup(_Nsize, strategy::out_width()) * roundup(_Ksize, strategy::k_unroll())) +
                                     (k0 * roundup(_Nsize, strategy::out_width())) +
                                     (n0 * kern_k);

                {
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (m_end - m_start) * kern_k * roundup(nmax-n0, strategy::out_width()));
#endif
                    strat.kernel(g_arrays._Aptr + (multi * g_arrays._A_multi_stride) + (batch * g_arrays._A_batch_stride) + (m_start * g_arrays._lda) + k0, g_arrays._lda,
                                 b_panel,
                                 result_buffer, (nmax-n0),
                                 (m_end - m_start), (nmax - n0), (kmax-k0),
                                 nullptr, Activation(), false);
                }

                {
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_ROWSUMS, (m_end - m_start) * _Ksize);
#endif
                    compute_row_sums(_qp, _Ksize, (m_end - m_start),
                                     g_arrays._Aptr + (multi * g_arrays._A_multi_stride) + (batch * g_arrays._A_batch_stride) + (m_start * g_arrays._lda), g_arrays._lda,
                                     local_row_sums);
                }

                {
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_QUANTIZE, (m_end - m_start) * _Nsize);
#endif

                    requantize_block_32(_qp, (nmax - n0), (m_end - m_start), result_buffer, (nmax - n0),
                                        g_arrays._Cptr + (multi * g_arrays._C_multi_stride) + (batch * g_arrays._C_batch_stride) + (m_start * g_arrays._ldc) + n0, g_arrays._ldc,
                                        local_row_sums, col_bias + (multi * _Nsize) + n0, n0);
                }
            } while (p.next_dim0());
        }
    }

    // Working space needed for intermediate result buffers.
    size_t get_working_size() const override {
        return (_nthreads * strategy::out_height() * _Nsize * sizeof(Tri));
    }

    void set_working_space(void *buffer) override {
        this->_gemm_arrays._workspace = buffer;
    }

    // Interface implementation - pretransposed
    bool B_is_pretransposed() const override {
        return true;
    }

    bool B_pretranspose_required() const override {
        return (_B_transposed==nullptr);
    }

    size_t get_B_pretransposed_array_size() const override {
        return get_col_sum_size() + (roundup(_Nsize, strategy::out_width()) * roundup(_Ksize, strategy::k_unroll()) * _nmulti * sizeof(Troi));
    }

    void requantize_bias(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride) override {
        col_bias = reinterpret_cast<int32_t *>(in_buffer);

        for (unsigned int i=0; i<_nmulti; i++) {
            compute_col_sums(_qp, _Nsize, _Ksize, B + (i * B_multi_stride), ldb, col_bias + (i * _Nsize),  _Ksize, i, 0);
        }
    }

    void pretranspose_B_array(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride, bool transposed) override {
        assert(!transposed);

        requantize_bias(in_buffer, B, ldb, B_multi_stride);

        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        Troi *buffer = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        _B_transposed = buffer;
        strategy strat(_ci);

        for (unsigned int multi=0; multi<_nmulti; multi++) {
            for (unsigned int k0=0; k0<_Ksize; k0+=_k_block) {
                const unsigned int kmax = std::min(k0 + _k_block, _Ksize);
                const unsigned int k_size = roundup(kmax-k0, strategy::k_unroll());

                for (unsigned int x0=0; x0<_Nsize; x0+=_n_block) {
                    const unsigned int xmax = std::min(x0+_n_block, _Nsize);

                    const unsigned int size = roundup(xmax-x0, strategy::out_width()) * k_size;

                    strat.transforms.PrepareB( buffer, B + (multi * B_multi_stride), ldb,
                                               x0, xmax, k0, kmax, false);

                    buffer += size;
                }
            }
        }
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        _B_transposed = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        col_bias = reinterpret_cast<int32_t *>(in_buffer);
    }

    void set_quantized_bias(const int32_t *bias, size_t bias_multi_stride) override {
        _qp.bias = bias;
        _qp.bias_multi_stride = bias_multi_stride;
    }

    GemmConfig get_config() override {
        GemmConfig c;

        c.inner_block_size = _k_block;
        c.outer_block_size = _n_block;
        c.filter = get_type_name<strategy>();

        return c;
    }
};

}  // namespace ops
}  // namespace kai
