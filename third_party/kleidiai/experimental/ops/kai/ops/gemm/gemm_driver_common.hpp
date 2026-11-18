//
// SPDX-FileCopyrightText: Copyright 2017-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/bfloat.hpp"
#include "kernel_weight_format.hpp"
#include "common_internal/kernel_args.hpp"
#include "common_internal/quantized.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

// Stripe width is only exposed by fixed-format strategies.
template<typename strategy, bool FixedFormat>
struct driver_stripe_width {
    static unsigned int get() { return strategy::out_width(); }
};

template<typename strategy>
struct driver_stripe_width<strategy, true> {
    static unsigned int get() { return strategy::stripe_width(); }
};

// Non-fixed strategies need not expose a kernel weight format.
template<typename strategy, bool FixedFormat, typename Tro>
struct driver_kernel_weight_format {
    static KernelWeightFormat get() {
        return KernelWeightFormat::NON_FIXED;
    }
};

template<typename strategy, typename Tro>
struct driver_kernel_weight_format<strategy, true, Tro> {
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

// Shared state and setup for the packed-RHS GEMM/GEMV drivers. Blocking,
// workspace and execution remain the responsibility of each driver. Troi is
// explicit because GEMV strategies use operand_type whereas GEMM strategies
// distinguish lhs_operand_type and rhs_operand_type.
template<typename strategy, typename Tlo, typename Tro, typename Tr, typename OutputStage,
         typename Troi, bool FixedFormat=false>
class GemmDriverCommon : public GemmCommon<Tlo, Tro, Tr> {
protected:
    GemmArgs _args;
    OutputStage _os;
    const unsigned int _Ktotal;
    const unsigned int _rounded_Ksize;
    const Troi *_B_transposed = nullptr;
    int32_t *col_bias = nullptr;

    GemmDriverCommon(const GemmArgs &args, const OutputStage &os)
        : _args(args), _os(os), _Ktotal(get_ktotal(args)),
          _rounded_Ksize(roundup(args._Ksize, strategy::k_unroll())) {
        // Drivers consume configuration from the constructor argument. The
        // caller does not have to keep the GemmConfig alive afterwards.
        _args._cfg = nullptr;
    }

    GemmConfig get_config_common(unsigned int k_block, unsigned int n_block) const {
        GemmConfig config;
        config.inner_block_size = k_block;
        config.outer_block_size = n_block;
        config.filter = get_type_name<strategy>();
        config.weight_format = get_weight_format(_args._fixed_format ?
            driver_kernel_weight_format<strategy, FixedFormat, Tro>::get() : KernelWeightFormat::NON_FIXED,
            sizeof(Tro));
        return config;
    }

    unsigned int get_col_sum_size() const {
        if (std::is_same<OutputStage, Requantize32>::value) {
            return _args._Nsize * _args._nmulti * sizeof(int32_t);
        }

        return 0;
    }

    static unsigned int get_ktotal(const GemmArgs &args) {
        return args._Ksections * roundup(args._Ksize, strategy::k_unroll());
    }

    // Functions to set quantization-related kernel arguments.  This is
    // factored out into this overloaded function to enable the correct
    // version to be picked up without doing a bunch of is_same<> and casts
    // to handle it inline.
    void set_kernel_output_args(KernelArgs &, unsigned int, unsigned int, const Nothing &) const {}

    void set_kernel_output_args(KernelArgs &ka, unsigned int multi, unsigned int x,
                                const Requantize32 &qp) const {
        ka.bias = col_bias ? col_bias + multi * _args._Nsize + x : nullptr;
        ka.scale_rhs_offset = qp.b_offset;
        ka.scale_result_offset = qp.c_offset;
        ka.scale_tensor_mul = qp.per_layer_mul;
        // Drivers must route left-shift cases to a fallback before using this API.
        ka.scale_tensor_shift = qp.per_layer_right_shift;
        ka.scale_mul_rhs_ptr = qp.per_channel_requant ? qp.per_channel_muls + x : nullptr;
        ka.scale_shift_rhs_ptr = qp.per_channel_requant ? qp.per_channel_right_shifts + x : nullptr;
        ka.clamp_min = qp.minval;
        ka.clamp_max = qp.maxval;
    }

    void set_kernel_output_args(KernelArgs &ka, unsigned int, unsigned int, const DequantizeFloat &dq) const {
        ka.scale_tensor_mul = dq.scale;
        // The FP bias is supplied on the final K pass. Integer-domain offset
        // corrections are not currently supported by this output stage.
        ka.late_bias = ka.bias;
        ka.bias = nullptr;
    }

    // Selects column-sum computation by output stage so non-quantized drivers
    // do not instantiate unsupported row- or column-sum calls.
    template<class Stage>
    void requantize_bias_impl(void *, const Tro *, int, int, bool, const Stage &) {}

    void requantize_bias_impl(void *in_buffer, const Tro *B, int ldb, int B_multi_stride,
                              bool transposed, const Requantize32 &qp) {
        col_bias = reinterpret_cast<int32_t *>(in_buffer);

        // If a_offset is zero we don't need to do anything except
        // populate with the supplied bias, if present.
        if (qp.a_offset == 0) {
            for (unsigned int i = 0; i < _args._nmulti; i++) {
                int32_t *out = col_bias + i * _args._Nsize;
                if (qp.bias) {
                    std::copy_n(qp.bias + i * qp.bias_multi_stride, _args._Nsize, out);
                } else {
                    std::fill_n(out, _args._Nsize, 0);
                }
            }
            return;
        }

        // Compute total depth - the input is assumed to be packing with
        // no padding between sections.
        const int64_t depth = _args._Ksize * _args._Ksections;
        const int32_t offset_correction = qp.a_offset * qp.b_offset * depth;

        // Arguments to pass to the sum functions; row_sums reads LHS
        // parameters and col_sums reads RHS parameters, so populate
        // both for simplicity.
        KernelArgs ka{};
        ka.M = ka.N = _args._Nsize;
        ka.K = depth;
        ka.lhs_stride = ka.rhs_stride = ldb * sizeof(Tro);

        for (unsigned int i = 0; i < _args._nmulti; i++) {
            // Set base addresses for this multi
            ka.lhs = ka.rhs = B + i * B_multi_stride;
            int32_t *out = col_bias + i * _args._Nsize;

            // Sum rows or columns as appropriate
            if (transposed) {
                row_sums_indirect<Tro>(&ka, out, qp.a_offset);
            } else {
                compute_col_sums<Tro>(&ka, out, qp.a_offset);
            }

            // Add the supplied bias and/or offset correction if needed.
            if (qp.bias) {
                const int32_t *bias = qp.bias + i * qp.bias_multi_stride;
                for (unsigned int n = 0; n < _args._Nsize; n++) {
                    out[n] += offset_correction + bias[n];
                }
            } else if (qp.b_offset != 0) {
                for (unsigned int n = 0; n < _args._Nsize; n++) {
                    out[n] += offset_correction;
                }
            }
        }
    }

    // Packing depth is independent of execution blocking for GEMV: its
    // column panels contain the full K range. GEMM drivers override these
    // sizes to match their execution layout. N blocks must be whole panels.
    virtual unsigned int get_rhs_packing_k_block() const { return _Ktotal; }
    virtual unsigned int get_rhs_packing_n_block() const { return strategy::out_width(); }


public:
    bool B_is_pretransposed() const override {
        return !_args._fixed_format;
    }

    bool B_pretranspose_required() const override {
        return !_args._fixed_format && _B_transposed == nullptr;
    }

    // Work items follow multi, K block, N block. Fixed-format layouts use
    // full-depth stripes instead of K blocks.
    size_t get_B_pretranspose_window_size() const override {
        const unsigned int packing_k_block = get_rhs_packing_k_block();
        const unsigned int packing_n_block = get_rhs_packing_n_block();
        if (FixedFormat) {
            return iceildiv(_args._Nsize, driver_stripe_width<strategy, FixedFormat>::get()) * _args._nmulti;
        }
        return iceildiv(_args._Nsize, packing_n_block) * iceildiv(_Ktotal, packing_k_block) * _args._nmulti;
    }

    void pretranspose_B_array(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride, const bool transposed) override {
        pretranspose_B_array_part(in_buffer, B, ldb, B_multi_stride, transposed, 0, get_B_pretranspose_window_size());
    }

    void pretranspose_B_array_part(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride, const bool transposed, size_t start, size_t end) override {
        const unsigned int packing_k_block = get_rhs_packing_k_block();
        const unsigned int packing_n_block = get_rhs_packing_n_block();
        assert(!transposed || this->B_pretranspose_supports_transpose());

        strategy strat(_args._ci);

        // Perform column sums etc as part of the last block.
        if (end >= get_B_pretranspose_window_size()) {
            this->requantize_bias(in_buffer, B, ldb, B_multi_stride, transposed);
        }

        // Put the transposed data after the column sums - in non-quantized cases get_col_sum_size() == 0
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(in_buffer);
        Troi *buffer = reinterpret_cast<Troi *>(buffer_int + get_col_sum_size());
        _B_transposed = buffer;

        const size_t rounded_width = roundup(_args._Nsize, strategy::out_width());
        const size_t n_blocks = FixedFormat ? iceildiv(_args._Nsize, driver_stripe_width<strategy, FixedFormat>::get()) :
                                                 iceildiv(_args._Nsize, packing_n_block);
        const size_t k_blocks = FixedFormat ? 1 : iceildiv(_Ktotal, packing_k_block);
        end = std::min(end, get_B_pretranspose_window_size());

        for (size_t pos = start; pos < end;) {
            const size_t multi = pos / (n_blocks * k_blocks);
            const unsigned int k0 = FixedFormat ? 0 : ((pos / n_blocks) % k_blocks) * packing_k_block;
            const unsigned int kmax = FixedFormat ? _Ktotal : std::min(k0 + packing_k_block, _Ktotal);
            const unsigned int width = FixedFormat ? driver_stripe_width<strategy, FixedFormat>::get() : packing_n_block;
            const unsigned int x0 = (pos % n_blocks) * width;
            // Consecutive fixed-format stripes can be transformed together.
            const size_t blocks = FixedFormat ? std::min(n_blocks - pos % n_blocks, end - pos) : 1;
            const unsigned int xmax = std::min<size_t>(x0 + blocks * width, _args._Nsize);
            pos += blocks;
            const unsigned int depth = kmax - k0;
            Troi *out = buffer + multi * rounded_width * _Ktotal +
                        (FixedFormat ? x0 * _Ktotal : k0 * rounded_width + x0 * depth);

            if (_args._Ksections == 1) {
                strat.transforms.PrepareB(out, B + multi * B_multi_stride, ldb,
                                          x0, xmax, k0, std::min(kmax, _args._Ksize), transposed);
                continue;
            }

            const unsigned int stripe = driver_stripe_width<strategy, FixedFormat>::get();

            // Each packed stripe contains all its K sections, with padding at
            // section boundaries. K blocks may start or end within a section.
            for (unsigned int x = x0; x < xmax; x += stripe) {
                for (unsigned int k = k0; k < kmax;) {
                    const unsigned int section = k / _rounded_Ksize;
                    const unsigned int offset = k % _rounded_Ksize;
                    const unsigned int length = std::min(_args._Ksize - offset, kmax - k);
                    strat.transforms.PrepareB(out, B + multi * B_multi_stride, ldb,
                                              x, std::min(x + stripe, xmax),
                                              section * _args._Ksize + offset,
                                              section * _args._Ksize + offset + length, transposed);
                    const unsigned int padded_length = roundup(length, strategy::k_unroll());
                    out += stripe * padded_length;
                    k += padded_length;
                }
            }
        }
    }

    void requantize_bias(void *in_buffer, const Tro *B, const int ldb, const int B_multi_stride, const bool transposed) override {
        requantize_bias_impl(in_buffer, B, ldb, B_multi_stride, transposed, _os);
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
};

}  // namespace ops
}  // namespace kai
