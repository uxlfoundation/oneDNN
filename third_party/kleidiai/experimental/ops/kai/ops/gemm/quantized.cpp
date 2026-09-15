//
// SPDX-FileCopyrightText: Copyright 2019, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include "kai/ops/gemm/kai_ops.hpp"
#include "common_internal/utils.hpp"
#include "common_internal/quantized.hpp"

#include <arm_neon.h>

namespace kai {
namespace ops {

namespace {

/* Requantize a block of data, using the requantize parameters in 'qp'.
 *
 * row_bias and col_bias are assumed to be precomputed values which include
 * any externally supplied bias, plus the row/column contibution sums, plus
 * the overall constant offset (A_offset * B_offset * depth).
 *
 * Note that this function works equally well for uint8_t output: just set
 * minval/maxval appropriately and cast the output pointer.  It is caller's
 * responsibility to ensure that minval/maxval are representable in the
 * target type - the downcast to (u)int8_t is done by simply extracting the
 * LSB.
 *
 * The 'per_channel' template parameter selects between per channel and per
 * layer requantization - in the former case we need to load vectors of
 * shifts and multipliers for each column.  A separate vector for each
 * column is set up in any case (and it is hoped that the compiler can elide
 * the needless movs in the per-layer case).
 */
template<bool per_channel, bool do_left_shift>
void requantize_block_32_int(const Requantize32 &qp, unsigned int width, unsigned int height,
                             const int32_t *input, unsigned int in_stride, int8_t *output, unsigned int out_stride,
                             const int32_t *row_bias, const int32_t *col_bias, const unsigned int start_col) {
    const int32x4_t v_mul          = vdupq_n_s32(qp.per_layer_mul);
    const int32x4_t v_right_shift  = vdupq_n_s32(qp.per_layer_right_shift);
    const int32x4_t v_left_shift   = vdupq_n_s32(qp.per_layer_left_shift);
    const int32x4_t v_minval       = vdupq_n_s32(qp.minval);
    const int32x4_t v_maxval       = vdupq_n_s32(qp.maxval);
    const int32x4_t v_c_offset     = vdupq_n_s32(qp.c_offset);

    /* To make sure we have plenty of accumulators, compute two rows at a
     * time.  If the number of rows is odd, compute the bottom row twice to
     * avoid needing a duplicate codepath. */
    for (unsigned int row=0; row<height; row+=2) {
        /* Prefer to do 4 vectors (16 values) at once as this collapses
         * neatly to a single vector of output, failing that a vector at a
         * time and then the odd ones out at the end.  */
        unsigned int blocks=(width / 16);
        unsigned int regs=(width % 16) / 4;
        unsigned int odds=(width % 4);

        /* Pointers to various per-column data.  Take care to avoid manipulating a probably-null
         * pointer in cases these are not used */
        const int32_t *colptr = col_bias;
        const int32_t *perch_mul_ptr    = per_channel ? (qp.per_channel_muls + start_col) : nullptr;
        const int32_t *perch_shift_ptr  = per_channel ? (qp.per_channel_right_shifts + start_col) : nullptr;
        const int32_t *perch_shiftl_ptr = (per_channel && do_left_shift) ? (qp.per_channel_left_shifts + start_col) : nullptr;

        const int32_t *in_ptr = input + (row * in_stride);
        int8_t *out_ptr = output + (row * out_stride);
        int32_t row_sum = row_bias[row];

        const int32_t *in_ptr1;
        int8_t *out_ptr1;
        int32_t row_sum1;

        if (row == height-1) {
            in_ptr1  = in_ptr;
            out_ptr1 = out_ptr;
            row_sum1 = row_sum;
        } else {
            in_ptr1  = in_ptr + in_stride;
            out_ptr1 = out_ptr + out_stride;
            row_sum1 = row_bias[row+1];
        }

        const int32x4_t v_row_sum  = vdupq_n_s32(row_sum);
        const int32x4_t v_row_sum1 = vdupq_n_s32(row_sum1);

        while (blocks--) {
            int32x4_t v_mul0;
            int32x4_t v_mul1;
            int32x4_t v_mul2;
            int32x4_t v_mul3;

            int32x4_t v_shf0;
            int32x4_t v_shf1;
            int32x4_t v_shf2;
            int32x4_t v_shf3;

            int32x4_t v_shf0l;
            int32x4_t v_shf1l;
            int32x4_t v_shf2l;
            int32x4_t v_shf3l;

            if (per_channel) {
                v_mul0 = vld1q_s32(perch_mul_ptr);
                v_mul1 = vld1q_s32(perch_mul_ptr + 4);
                v_mul2 = vld1q_s32(perch_mul_ptr + 8);
                v_mul3 = vld1q_s32(perch_mul_ptr + 12);
                perch_mul_ptr += 16;

                v_shf0 = vld1q_s32(perch_shift_ptr);
                v_shf1 = vld1q_s32(perch_shift_ptr + 4);
                v_shf2 = vld1q_s32(perch_shift_ptr + 8);
                v_shf3 = vld1q_s32(perch_shift_ptr + 12);
                perch_shift_ptr += 16;

                if (do_left_shift) {
                    v_shf0l = vld1q_s32(perch_shiftl_ptr);
                    v_shf1l = vld1q_s32(perch_shiftl_ptr + 4);
                    v_shf2l = vld1q_s32(perch_shiftl_ptr + 8);
                    v_shf3l = vld1q_s32(perch_shiftl_ptr + 12);
                    perch_shiftl_ptr += 16;
                }
            } else {
                v_mul0=v_mul1=v_mul2=v_mul3=v_mul;
                v_shf0=v_shf1=v_shf2=v_shf3=v_right_shift;
                v_shf0l=v_shf1l=v_shf2l=v_shf3l=v_left_shift;
            }

            // Load column pointers
            int32x4_t v_col0 = vld1q_s32(colptr);
            int32x4_t v_col1 = vld1q_s32(colptr + 4);
            int32x4_t v_col2 = vld1q_s32(colptr + 8);
            int32x4_t v_col3 = vld1q_s32(colptr + 12);
            colptr += 16;

            // Load input data (row 0);
            int32x4_t v_in00 = vld1q_s32(in_ptr);
            int32x4_t v_in01 = vld1q_s32(in_ptr + 4);
            int32x4_t v_in02 = vld1q_s32(in_ptr + 8);
            int32x4_t v_in03 = vld1q_s32(in_ptr + 12);
            in_ptr += 16;

            // Load input data (row 1);
            int32x4_t v_in10 = vld1q_s32(in_ptr1);
            int32x4_t v_in11 = vld1q_s32(in_ptr1 + 4);
            int32x4_t v_in12 = vld1q_s32(in_ptr1 + 8);
            int32x4_t v_in13 = vld1q_s32(in_ptr1 + 12);
            in_ptr1 += 16;

            // Add on row bias and column bias
            v_in00 = vaddq_s32(v_in00, v_row_sum);
            v_in01 = vaddq_s32(v_in01, v_row_sum);
            v_in02 = vaddq_s32(v_in02, v_row_sum);
            v_in03 = vaddq_s32(v_in03, v_row_sum);

            v_in10 = vaddq_s32(v_in10, v_row_sum1);
            v_in11 = vaddq_s32(v_in11, v_row_sum1);
            v_in12 = vaddq_s32(v_in12, v_row_sum1);
            v_in13 = vaddq_s32(v_in13, v_row_sum1);

            v_in00 = vaddq_s32(v_in00, v_col0);
            v_in01 = vaddq_s32(v_in01, v_col1);
            v_in02 = vaddq_s32(v_in02, v_col2);
            v_in03 = vaddq_s32(v_in03, v_col3);

            v_in10 = vaddq_s32(v_in10, v_col0);
            v_in11 = vaddq_s32(v_in11, v_col1);
            v_in12 = vaddq_s32(v_in12, v_col2);
            v_in13 = vaddq_s32(v_in13, v_col3);

            // Quantize

            // If a left shift is needed it needs to happen first.
            if (do_left_shift) {
                v_in00 = vrshlq_s32(v_in00, v_shf0l);
                v_in01 = vrshlq_s32(v_in01, v_shf1l);
                v_in02 = vrshlq_s32(v_in02, v_shf2l);
                v_in03 = vrshlq_s32(v_in03, v_shf3l);

                v_in10 = vrshlq_s32(v_in10, v_shf0l);
                v_in11 = vrshlq_s32(v_in11, v_shf1l);
                v_in12 = vrshlq_s32(v_in12, v_shf2l);
                v_in13 = vrshlq_s32(v_in13, v_shf3l);
            }

            // Multiply
            v_in00 = vqdmulhq_s32(v_in00, v_mul0);
            v_in01 = vqdmulhq_s32(v_in01, v_mul1);
            v_in02 = vqdmulhq_s32(v_in02, v_mul2);
            v_in03 = vqdmulhq_s32(v_in03, v_mul3);

            v_in10 = vqdmulhq_s32(v_in10, v_mul0);
            v_in11 = vqdmulhq_s32(v_in11, v_mul1);
            v_in12 = vqdmulhq_s32(v_in12, v_mul2);
            v_in13 = vqdmulhq_s32(v_in13, v_mul3);

            v_in00 = vrshlq_s32(v_in00, v_shf0);
            v_in01 = vrshlq_s32(v_in01, v_shf1);
            v_in02 = vrshlq_s32(v_in02, v_shf2);
            v_in03 = vrshlq_s32(v_in03, v_shf3);

            v_in10 = vrshlq_s32(v_in10, v_shf0);
            v_in11 = vrshlq_s32(v_in11, v_shf1);
            v_in12 = vrshlq_s32(v_in12, v_shf2);
            v_in13 = vrshlq_s32(v_in13, v_shf3);

            v_in00 = vaddq_s32(v_in00, v_c_offset);
            v_in01 = vaddq_s32(v_in01, v_c_offset);
            v_in02 = vaddq_s32(v_in02, v_c_offset);
            v_in03 = vaddq_s32(v_in03, v_c_offset);

            v_in10 = vaddq_s32(v_in10, v_c_offset);
            v_in11 = vaddq_s32(v_in11, v_c_offset);
            v_in12 = vaddq_s32(v_in12, v_c_offset);
            v_in13 = vaddq_s32(v_in13, v_c_offset);

            v_in00 = vmaxq_s32(v_in00, v_minval);
            v_in01 = vmaxq_s32(v_in01, v_minval);
            v_in02 = vmaxq_s32(v_in02, v_minval);
            v_in03 = vmaxq_s32(v_in03, v_minval);

            v_in10 = vmaxq_s32(v_in10, v_minval);
            v_in11 = vmaxq_s32(v_in11, v_minval);
            v_in12 = vmaxq_s32(v_in12, v_minval);
            v_in13 = vmaxq_s32(v_in13, v_minval);

            v_in00 = vminq_s32(v_in00, v_maxval);
            v_in01 = vminq_s32(v_in01, v_maxval);
            v_in02 = vminq_s32(v_in02, v_maxval);
            v_in03 = vminq_s32(v_in03, v_maxval);

            v_in10 = vminq_s32(v_in10, v_maxval);
            v_in11 = vminq_s32(v_in11, v_maxval);
            v_in12 = vminq_s32(v_in12, v_maxval);
            v_in13 = vminq_s32(v_in13, v_maxval);

            int16x8_t v_uz00 = vuzp1q_s16(vreinterpretq_s16_s32(v_in00), vreinterpretq_s16_s32(v_in01));
            int16x8_t v_uz01 = vuzp1q_s16(vreinterpretq_s16_s32(v_in02), vreinterpretq_s16_s32(v_in03));

            int16x8_t v_uz10 = vuzp1q_s16(vreinterpretq_s16_s32(v_in10), vreinterpretq_s16_s32(v_in11));
            int16x8_t v_uz11 = vuzp1q_s16(vreinterpretq_s16_s32(v_in12), vreinterpretq_s16_s32(v_in13));

            int8x16_t v_uz0 = vuzp1q_s8(vreinterpretq_s8_s16(v_uz00), vreinterpretq_s8_s16(v_uz01));
            int8x16_t v_uz1 = vuzp1q_s8(vreinterpretq_s8_s16(v_uz10), vreinterpretq_s8_s16(v_uz11));

            vst1q_s8(out_ptr, v_uz0);
            out_ptr += 16;
            vst1q_s8(out_ptr1, v_uz1);
            out_ptr1 += 16;
        }

        // We are often quantizing one block of interleaved kernel output at a time - these are three registers
        // wide.  Special case that here.
        if (regs==3) {
            regs -= 3;

            int32x4_t v_mul0;
            int32x4_t v_mul1;
            int32x4_t v_mul2;

            int32x4_t v_shf0;
            int32x4_t v_shf1;
            int32x4_t v_shf2;

            int32x4_t v_shf0l;
            int32x4_t v_shf1l;
            int32x4_t v_shf2l;

            if (per_channel) {
                v_mul0 = vld1q_s32(perch_mul_ptr);
                v_mul1 = vld1q_s32(perch_mul_ptr + 4);
                v_mul2 = vld1q_s32(perch_mul_ptr + 8);
                perch_mul_ptr += 12;

                v_shf0 = vld1q_s32(perch_shift_ptr);
                v_shf1 = vld1q_s32(perch_shift_ptr + 4);
                v_shf2 = vld1q_s32(perch_shift_ptr + 8);
                perch_shift_ptr += 12;

                if (do_left_shift) {
                    v_shf0l = vld1q_s32(perch_shiftl_ptr);
                    v_shf1l = vld1q_s32(perch_shiftl_ptr + 4);
                    v_shf2l = vld1q_s32(perch_shiftl_ptr + 8);
                    perch_shiftl_ptr += 12;
                }
            } else {
                v_mul0=v_mul1=v_mul2=v_mul;
                v_shf0=v_shf1=v_shf2=v_right_shift;
                v_shf0l=v_shf1l=v_shf2l=v_left_shift;
            }

            // Load column pointers
            int32x4_t v_col0 = vld1q_s32(colptr);
            int32x4_t v_col1 = vld1q_s32(colptr + 4);
            int32x4_t v_col2 = vld1q_s32(colptr + 8);
            colptr += 12;

            // Load input data (row 0);
            int32x4_t v_in00 = vld1q_s32(in_ptr);
            int32x4_t v_in01 = vld1q_s32(in_ptr + 4);
            int32x4_t v_in02 = vld1q_s32(in_ptr + 8);
            in_ptr += 12;

            // Load input data (row 1);
            int32x4_t v_in10 = vld1q_s32(in_ptr1);
            int32x4_t v_in11 = vld1q_s32(in_ptr1 + 4);
            int32x4_t v_in12 = vld1q_s32(in_ptr1 + 8);
            in_ptr1 += 12;

            // Add on row bias and column bias
            v_in00 = vaddq_s32(v_in00, v_row_sum);
            v_in01 = vaddq_s32(v_in01, v_row_sum);
            v_in02 = vaddq_s32(v_in02, v_row_sum);

            v_in10 = vaddq_s32(v_in10, v_row_sum1);
            v_in11 = vaddq_s32(v_in11, v_row_sum1);
            v_in12 = vaddq_s32(v_in12, v_row_sum1);

            v_in00 = vaddq_s32(v_in00, v_col0);
            v_in01 = vaddq_s32(v_in01, v_col1);
            v_in02 = vaddq_s32(v_in02, v_col2);

            v_in10 = vaddq_s32(v_in10, v_col0);
            v_in11 = vaddq_s32(v_in11, v_col1);
            v_in12 = vaddq_s32(v_in12, v_col2);

            // Quantize

            // If a left shift is needed it needs to happen first.
            if (do_left_shift) {
                v_in00 = vrshlq_s32(v_in00, v_shf0l);
                v_in01 = vrshlq_s32(v_in01, v_shf1l);
                v_in02 = vrshlq_s32(v_in02, v_shf2l);

                v_in10 = vrshlq_s32(v_in10, v_shf0l);
                v_in11 = vrshlq_s32(v_in11, v_shf1l);
                v_in12 = vrshlq_s32(v_in12, v_shf2l);
            }

            // Multiply
            v_in00 = vqdmulhq_s32(v_in00, v_mul0);
            v_in01 = vqdmulhq_s32(v_in01, v_mul1);
            v_in02 = vqdmulhq_s32(v_in02, v_mul2);

            v_in10 = vqdmulhq_s32(v_in10, v_mul0);
            v_in11 = vqdmulhq_s32(v_in11, v_mul1);
            v_in12 = vqdmulhq_s32(v_in12, v_mul2);

            v_in00 = vrshlq_s32(v_in00, v_shf0);
            v_in01 = vrshlq_s32(v_in01, v_shf1);
            v_in02 = vrshlq_s32(v_in02, v_shf2);

            v_in10 = vrshlq_s32(v_in10, v_shf0);
            v_in11 = vrshlq_s32(v_in11, v_shf1);
            v_in12 = vrshlq_s32(v_in12, v_shf2);

            v_in00 = vaddq_s32(v_in00, v_c_offset);
            v_in01 = vaddq_s32(v_in01, v_c_offset);
            v_in02 = vaddq_s32(v_in02, v_c_offset);

            v_in10 = vaddq_s32(v_in10, v_c_offset);
            v_in11 = vaddq_s32(v_in11, v_c_offset);
            v_in12 = vaddq_s32(v_in12, v_c_offset);

            v_in00 = vmaxq_s32(v_in00, v_minval);
            v_in01 = vmaxq_s32(v_in01, v_minval);
            v_in02 = vmaxq_s32(v_in02, v_minval);

            v_in10 = vmaxq_s32(v_in10, v_minval);
            v_in11 = vmaxq_s32(v_in11, v_minval);
            v_in12 = vmaxq_s32(v_in12, v_minval);

            v_in00 = vminq_s32(v_in00, v_maxval);
            v_in01 = vminq_s32(v_in01, v_maxval);
            v_in02 = vminq_s32(v_in02, v_maxval);

            v_in10 = vminq_s32(v_in10, v_maxval);
            v_in11 = vminq_s32(v_in11, v_maxval);
            v_in12 = vminq_s32(v_in12, v_maxval);

            int16x8_t v_uz00 = vuzp1q_s16(vreinterpretq_s16_s32(v_in00), vreinterpretq_s16_s32(v_in01));
            int16x8_t v_uz01 = vuzp1q_s16(vreinterpretq_s16_s32(v_in02), vreinterpretq_s16_s32(v_in02));

            int16x8_t v_uz10 = vuzp1q_s16(vreinterpretq_s16_s32(v_in10), vreinterpretq_s16_s32(v_in11));
            int16x8_t v_uz11 = vuzp1q_s16(vreinterpretq_s16_s32(v_in12), vreinterpretq_s16_s32(v_in12));

            int8x16_t v_uz0 = vuzp1q_s8(vreinterpretq_s8_s16(v_uz00), vreinterpretq_s8_s16(v_uz01));
            int8x16_t v_uz1 = vuzp1q_s8(vreinterpretq_s8_s16(v_uz10), vreinterpretq_s8_s16(v_uz11));

            vst1q_lane_s64(reinterpret_cast<int64_t *>(out_ptr), vreinterpretq_s64_s8(v_uz0), 0);
            vst1q_lane_s32(reinterpret_cast<int32_t *>(out_ptr + 8), vreinterpretq_s32_s8(v_uz0), 2);
            out_ptr += 12;
            vst1q_lane_s64(reinterpret_cast<int64_t *>(out_ptr1), vreinterpretq_s64_s8(v_uz1), 0);
            vst1q_lane_s32(reinterpret_cast<int32_t *>(out_ptr1 + 8), vreinterpretq_s32_s8(v_uz1), 2);
            out_ptr1 += 12;
        }

        while (regs--) {
            int32x4_t v_mul0;
            int32x4_t v_shf0;
            int32x4_t v_shf0l;

            if (per_channel) {
                v_mul0 = vld1q_s32(perch_mul_ptr);
                perch_mul_ptr += 4;

                v_shf0 = vld1q_s32(perch_shift_ptr);
                perch_shift_ptr += 4;

                if (do_left_shift) {
                    v_shf0l = vld1q_s32(perch_shiftl_ptr);
                    perch_shiftl_ptr += 4;
                }
            } else {
                v_mul0=v_mul;
                v_shf0=v_right_shift;
                v_shf0l=v_left_shift;
            }
            // Load column pointers
            int32x4_t v_col0 = vld1q_s32(colptr);
            colptr += 4;

            // Load input data (row 0);
            int32x4_t v_in00 = vld1q_s32(in_ptr);
            in_ptr += 4;

            // Load input data (row 1);
            int32x4_t v_in10 = vld1q_s32(in_ptr1);
            in_ptr1 += 4;

            // Add on row sum and bias constant
            v_in00 = vaddq_s32(v_in00, v_row_sum);

            v_in10 = vaddq_s32(v_in10, v_row_sum1);

            // Subtract col sum * a_offset
            v_in00 = vaddq_s32(v_in00, v_col0);

            v_in10 = vaddq_s32(v_in10, v_col0);

            // Quantize - start with (optional) left shift
            if (do_left_shift) {
                v_in00 = vrshlq_s32(v_in00, v_shf0l);

                v_in10 = vrshlq_s32(v_in10, v_shf0l);
            }

            // Then multiply
            v_in00 = vqdmulhq_s32(v_in00, v_mul0);

            v_in10 = vqdmulhq_s32(v_in10, v_mul0);

            v_in00 = vrshlq_s32(v_in00, v_shf0);

            v_in10 = vrshlq_s32(v_in10, v_shf0);

            v_in00 = vaddq_s32(v_in00, v_c_offset);

            v_in10 = vaddq_s32(v_in10, v_c_offset);

            v_in00 = vmaxq_s32(v_in00, v_minval);

            v_in10 = vmaxq_s32(v_in10, v_minval);

            v_in00 = vminq_s32(v_in00, v_maxval);

            v_in10 = vminq_s32(v_in10, v_maxval);

            int16x8_t v_uz00 = vuzp1q_s16(vreinterpretq_s16_s32(v_in00), vreinterpretq_s16_s32(v_in10));

            int8x16_t v_uz0 = vuzp1q_s8(vreinterpretq_s8_s16(v_uz00), vreinterpretq_s8_s16(v_uz00));

            vst1q_lane_s32(reinterpret_cast<int32_t *>(out_ptr), vreinterpretq_s32_s8(v_uz0), 0);
            out_ptr += 4;
            vst1q_lane_s32(reinterpret_cast<int32_t *>(out_ptr1), vreinterpretq_s32_s8(v_uz0), 1);
            out_ptr1 += 4;
        }

        if (odds) {
            int32x4_t v_col0 = vdupq_n_s32(0);
            int32x4_t v_in00 = vdupq_n_s32(0);
            int32x4_t v_in10 = vdupq_n_s32(0);
            int32x4_t v_mul0 = vdupq_n_s32(0);
            int32x4_t v_shf0 = vdupq_n_s32(0);
            int32x4_t v_shf0l = vdupq_n_s32(0);

            if (!per_channel) {
                v_mul0 = v_mul;
                v_shf0 = v_right_shift;
                v_shf0l = v_left_shift;
            }

            do {
                v_col0 = vld1q_lane_s32(colptr, v_col0, 0);
                v_in00 = vld1q_lane_s32(in_ptr, v_in00, 0);
                v_in10 = vld1q_lane_s32(in_ptr1, v_in10, 0);
                if (per_channel) {
                    v_mul0 = vld1q_lane_s32(perch_mul_ptr, v_mul0, 0);
                    v_shf0 = vld1q_lane_s32(perch_shift_ptr, v_shf0, 0);
                    if (do_left_shift) {
                        v_shf0l = vld1q_lane_s32(perch_shiftl_ptr, v_shf0l, 0);
                    }
                }
                if (odds == 1) { break; }

                v_col0 = vld1q_lane_s32(colptr + 1, v_col0, 1);
                v_in00 = vld1q_lane_s32(in_ptr + 1, v_in00, 1);
                v_in10 = vld1q_lane_s32(in_ptr1 + 1, v_in10, 1);
                if (per_channel) {
                    v_mul0 = vld1q_lane_s32(perch_mul_ptr + 1, v_mul0, 1);
                    v_shf0 = vld1q_lane_s32(perch_shift_ptr + 1, v_shf0, 1);
                    if (do_left_shift) {
                        v_shf0l = vld1q_lane_s32(perch_shiftl_ptr + 1, v_shf0l, 1);
                    }
                }
                if (odds == 2) { break; }

                v_col0 = vld1q_lane_s32(colptr + 2, v_col0, 2);
                v_in00 = vld1q_lane_s32(in_ptr + 2, v_in00, 2);
                v_in10 = vld1q_lane_s32(in_ptr1 + 2, v_in10, 2);
                if (per_channel) {
                    v_mul0 = vld1q_lane_s32(perch_mul_ptr + 2, v_mul0, 2);
                    v_shf0 = vld1q_lane_s32(perch_shift_ptr + 2, v_shf0, 2);
                    if (do_left_shift) {
                        v_shf0l = vld1q_lane_s32(perch_shiftl_ptr + 2, v_shf0l, 2);
                    }
                }
            } while (0);

            // Add on row sum and bias constant
            v_in00 = vaddq_s32(v_in00, v_row_sum);

            v_in10 = vaddq_s32(v_in10, v_row_sum1);

            // Subtract col sum * a_offset
            v_in00 = vaddq_s32(v_in00, v_col0);

            v_in10 = vaddq_s32(v_in10, v_col0);

            // Quantize - start with (optional) left shift
            if (do_left_shift) {
                v_in00 = vrshlq_s32(v_in00, v_shf0l);

                v_in10 = vrshlq_s32(v_in10, v_shf0l);
            }

            // Then multiply
            v_in00 = vqdmulhq_s32(v_in00, v_mul0);

            v_in10 = vqdmulhq_s32(v_in10, v_mul0);

            v_in00 = vrshlq_s32(v_in00, v_shf0);

            v_in10 = vrshlq_s32(v_in10, v_shf0);

            v_in00 = vaddq_s32(v_in00, v_c_offset);

            v_in10 = vaddq_s32(v_in10, v_c_offset);

            v_in00 = vmaxq_s32(v_in00, v_minval);

            v_in10 = vmaxq_s32(v_in10, v_minval);

            v_in00 = vminq_s32(v_in00, v_maxval);

            v_in10 = vminq_s32(v_in10, v_maxval);

            do {
                vst1q_lane_s8(out_ptr, vreinterpretq_s8_s32(v_in00), 0);
                vst1q_lane_s8(out_ptr1, vreinterpretq_s8_s32(v_in10), 0);

                if (odds==1) { break; }

                vst1q_lane_s8(out_ptr + 1, vreinterpretq_s8_s32(v_in00), 4);
                vst1q_lane_s8(out_ptr1 + 1, vreinterpretq_s8_s32(v_in10), 4);

                if (odds==2) { break; }

                vst1q_lane_s8(out_ptr + 2, vreinterpretq_s8_s32(v_in00), 8);
                vst1q_lane_s8(out_ptr1 + 2, vreinterpretq_s8_s32(v_in10), 8);
            } while(0);
        }
    }
}

} // anonymous namespace

template<typename Tin, typename Tout>
void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const Tin *input, unsigned int in_stride, Tout *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col) {
    if (qp.per_channel_requant) {
        if (qp.per_channel_left_shifts) {
            requantize_block_32_int<true, true>(qp, width, height, reinterpret_cast<const int32_t *>(input), in_stride,
                             reinterpret_cast<int8_t *>(output), out_stride, row_bias, col_bias, start_col);
        } else {
            requantize_block_32_int<true, false>(qp, width, height, reinterpret_cast<const int32_t *>(input), in_stride,
                             reinterpret_cast<int8_t *>(output), out_stride, row_bias, col_bias, start_col);
        }
    } else {
        if (qp.per_layer_left_shift > 0) {
            requantize_block_32_int<false, true>(qp, width, height, reinterpret_cast<const int32_t *>(input), in_stride,
                             reinterpret_cast<int8_t *>(output), out_stride, row_bias, col_bias, start_col);
        } else {
            requantize_block_32_int<false, false>(qp, width, height, reinterpret_cast<const int32_t *>(input), in_stride,
                             reinterpret_cast<int8_t *>(output), out_stride, row_bias, col_bias, start_col);
        }
    }
}

template void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const int32_t *input, unsigned int in_stride, int8_t *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

template void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const uint32_t *input, unsigned int in_stride, uint8_t *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

template void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const int32_t *input, unsigned int in_stride, uint8_t *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

/*
 * Routine (and helpers) to compute row sums needed for offset correction.
 *
 * This routine is templated on the type to be accumulated, because the
 * innermost instruction used needs to be of the correct signedness.
 * However, beyond this point we always use signed values in both cases.
 * The instructions that need to be different are therefore wrapped in
 * helper functions below.
 *
 * The general strategy used is to load vectors of 16 bytes and accumulate
 * (using uadalp/sadalp or AArch32 equivalents) into 8x16-bit accumulators.
 * These are then reduced (using uadalp/sadalp again) into 4x32-bit
 * accumulators.  The 4 accumulators for up to 4 rows being processed are
 * then added together into a single output vector using pairwise adds.
 *
 * For odd lengths (not multiples of 16), the odd bytes are copied into a
 * temporary 16-byte buffer before calling the standard routine to avoid
 * overreading the input.
 *
 * This reduction from the 8x16-bit into the 4x32-bit accumulators needs to
 * occur before the 16-bit accumulators can overflow - which is every 32
 * iterations (512 total bytes processed).  This is explained more below.
 */
namespace {
    struct row_sum_helpers {
        /* Load a full 16 byte vector, pairwise accumulate into 'sum' with uadalp or sadalp */
        template<typename T>
        inline int16x8_t accumulate_16(const T *ptr, int16x8_t sum);

        /* Handle "odd" bytes by copying to a temporary buffer */
        template<typename T>
        inline int16x8_t accumulate_odds_16(const T *ptr, int16x8_t sum, size_t odds);

        /* This function does the actual work for up to 4 rows at a time.
         * It's pulled out so we can template on the row count to generate
         * the 4 different cases.  4 rows are computed at a time as this
         * reduces to a single vector write.  */
        template<unsigned int rows, typename T>
        void compute_some_rows(unsigned int blocks, const T *input, unsigned int in_stride, int32_t *row_bias, size_t odds, int32x4_t offset_mul) {
            int16x8_t sums[rows];
            int32x4_t finalsums[rows];

            for (unsigned int i=0; i<rows; i++) {
                sums[i]      = vdupq_n_s16(0);
                finalsums[i] = vdupq_n_s32(0);
            }

            for (unsigned int i=0; i<blocks; i++) {
                for (unsigned int r=0; r<rows; r++) {
                    /* If we add too many blocks together, we run the risk
                     * of overflowing the intermediate 16-bit accumulators,
                     * especially in the unsigned case where we later treat
                     * the accumulator as signed.
                     *
                     * In that case, the maximum (signed) value is 16383,
                     * which is safe for 64 (unsigned) accumulations (255*64
                     * = 16,320).
                     *
                     * Each invocation of pairwise add adds 2 values to the
                     * accumulator - so in the unsigned case we can do 32
                     * adds before we need to reset the 16-bit accumulator
                     * by adding into the 32-bit 'finalsums'.
                     *
                     * We could do 64 adds in the signed case, but that
                     * optimization is not worth the complexity.
                     */
                    if (i > 0 && ((i & 31) == 0)) {
                        finalsums[r] = vpadalq_s16(finalsums[r], sums[r]);
                        sums[r] = vdupq_n_s16(0);
                    }
                    sums[r] = accumulate_16(input + (r * in_stride) + (i * 16), sums[r]);
                }
            }

            /* Handle the final odd read if needed. */
            if (odds > 0) {
                for (unsigned int r=0; r<rows; r++) {
                    sums[r] = accumulate_odds_16(input + (r * in_stride) + (blocks * 16), sums[r], odds);
                }
            }

            for (unsigned int i=0; i<rows; i++) {
                finalsums[i] = vpadalq_s16(finalsums[i], sums[i]);
            }

            int32x4_t t0, t1;
            int32x2_t t2;

            /* Result writeback - need to write back one value per row
             * processed.  Multiply all the final totals by -b_offset so
             * that the terms can simply be added in the requantize code.
             * */
            switch (rows) {
                case 1:
                    /* If we only have one output, just use ADDV.  Multiply
                     * the offset into all four components separately so it
                     * can stay in the SIMD register file.  */
                    t0 = vmulq_s32(finalsums[0], offset_mul);
                    *row_bias = vaddvq_s32(t0);
                    break;

                case 2:
                    /* For two outputs, two rounds of pairwise adds will
                     * generate the result in a 2-vector we can store in one
                     * go.  */
                    t0 = vpaddq_s32(finalsums[0], finalsums[1]);
                    t0 = vpaddq_s32(t0, t0);
                    t2 = vmul_s32(vget_low_s32(t0), vget_low_s32(offset_mul));
                    vst1_s32(row_bias, t2);
                    break;

                case 3:
                    /* Three rows - need to store the low two words plus the odd value from lane 2 */
                    t0 = vpaddq_s32(finalsums[0], finalsums[1]);
                    t1 = vpaddq_s32(finalsums[2], finalsums[2]);

                    t0 = vpaddq_s32(t0, t1);
                    t0 = vmulq_s32(t0, offset_mul);

                    vst1_s32(row_bias, vget_low_s32(t0));
                    row_bias[2] = vgetq_lane_s32(t0, 2);
                    break;

                case 4:
                    /* Four rows (most common case) - reduce to a single
                     * vector with pairwise adds.  */
                    t0 = vpaddq_s32(finalsums[0], finalsums[1]);
                    t1 = vpaddq_s32(finalsums[2], finalsums[3]);

                    t0 = vpaddq_s32(t0, t1);
                    t0 = vmulq_s32(t0, offset_mul);

                    vst1q_s32(row_bias, t0);
                    break;

                default:
                    UNREACHABLE("Impossible.");
            }
        }
    };

    template<>
    int16x8_t row_sum_helpers::accumulate_16(const uint8_t *ptr, int16x8_t sum) {
        return vreinterpretq_s16_u16(vpadalq_u8(vreinterpretq_u16_s16(sum), vld1q_u8(ptr)));
    }

    template<>
    int16x8_t row_sum_helpers::accumulate_16(const int8_t *ptr, int16x8_t sum) {
        return vpadalq_s8(sum, vld1q_s8(ptr));
    }

    template<typename T>
    int16x8_t row_sum_helpers::accumulate_odds_16(const T *ptr, int16x8_t sum, size_t odds) {
        T buffer[16] = {};
        for(size_t i=0; i<odds; i++) {
            buffer[i] = ptr[i];
        }
        return accumulate_16(buffer, sum);
    }
} // anonymous namespace

template<typename T>
void compute_row_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                      const T *input, unsigned int in_stride, int32_t *row_bias) {
    /* If the 'b' offset is zero, just skip this entirely. */
    if (qp.b_offset == 0) {
        memset(row_bias, 0, height * sizeof(int32_t));
        return;
    }

    row_sum_helpers thehelpers;

    const int32x4_t offset_mul = vdupq_n_s32(-qp.b_offset);

    /* Work out how many full vectors of 16 bytes we will read, and how many
     * odd bytes at the end */
    unsigned int blocks = (width / 16);
    const unsigned int odds = width % 16;

    for (unsigned int row=0; row<height; row+=4) {
        switch(height-row) {
            default:
            case 4:
                thehelpers.compute_some_rows<4>(blocks, input + (row * in_stride), in_stride, row_bias + row, odds, offset_mul);
                break;
            case 3:
                thehelpers.compute_some_rows<3>(blocks, input + (row * in_stride), in_stride, row_bias + row, odds, offset_mul);
                break;
            case 2:
                thehelpers.compute_some_rows<2>(blocks, input + (row * in_stride), in_stride, row_bias + row, odds, offset_mul);
                break;
            case 1:
                thehelpers.compute_some_rows<1>(blocks, input + (row * in_stride), in_stride, row_bias + row, odds, offset_mul);
                break;
        }
    }
}

/* Instantiate the two versions for uint8_t and int8_t. */
template void compute_row_sums(const Requantize32 &, unsigned int, unsigned int, const int8_t *, unsigned int, int32_t *);
template void compute_row_sums(const Requantize32 &, unsigned int, unsigned int, const uint8_t *, unsigned int, int32_t *);

template<unsigned int active_rows, typename T>
inline void add_block(const T *input, unsigned int in_stride, int32_t *output);

template<unsigned int active_rows>
inline void add_block(const uint8_t *input, unsigned int in_stride, int32_t *output) {
    uint8x16_t inputs[4];

    for (unsigned int i=0; i<4; i++) {
        if (i < active_rows) {
            inputs[i] = vld1q_u8(input + i * in_stride);
        } else {
            inputs[i] = vdupq_n_u8(0);
        }
    }

    int16x8_t sums_16b[4];

    // Two adds for the low pairs
    sums_16b[0]=vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(inputs[0]), vget_low_u8(inputs[1])));
    sums_16b[1]=vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(inputs[2]), vget_low_u8(inputs[3])));
    // Two adds for the high pairs
    sums_16b[2]=vreinterpretq_s16_u16(vaddl_high_u8(inputs[0], inputs[1]));
    sums_16b[3]=vreinterpretq_s16_u16(vaddl_high_u8(inputs[2], inputs[3]));

    int32x4_t sums_32b[4];

    sums_32b[0]=vaddl_s16(vget_low_s16(sums_16b[0]), vget_low_s16(sums_16b[1]));
    sums_32b[1]=vaddl_high_s16(sums_16b[0], sums_16b[1]);
    sums_32b[2]=vaddl_s16(vget_low_s16(sums_16b[2]), vget_low_s16(sums_16b[3]));
    sums_32b[3]=vaddl_high_s16(sums_16b[2], sums_16b[3]);

    for (unsigned int i=0; i<4; i++) {
        vst1q_s32(output + 4*i, vaddq_s32(sums_32b[i], vld1q_s32(output + 4*i)));
    }
}

template<unsigned int active_rows>
inline void add_block(const int8_t *input, unsigned int in_stride, int32_t *output) {
    int8x16_t inputs[4];

    for (unsigned int i=0; i<4; i++) {
        if (i < active_rows) {
            inputs[i] = vld1q_s8(input + i * in_stride);
        } else {
            inputs[i] = vdupq_n_s8(0);
        }
    }

    int16x8_t sums_16b[4];

    // Two adds for the low pairs
    sums_16b[0]=vaddl_s8(vget_low_s8(inputs[0]), vget_low_s8(inputs[1]));
    sums_16b[1]=vaddl_s8(vget_low_s8(inputs[2]), vget_low_s8(inputs[3]));
    // Two adds for the high pairs
    sums_16b[2]=vaddl_high_s8(inputs[0], inputs[1]);
    sums_16b[3]=vaddl_high_s8(inputs[2], inputs[3]);

    int32x4_t sums_32b[4];

    sums_32b[0]=vaddl_s16(vget_low_s16(sums_16b[0]), vget_low_s16(sums_16b[1]));
    sums_32b[1]=vaddl_high_s16(sums_16b[0], sums_16b[1]);
    sums_32b[2]=vaddl_s16(vget_low_s16(sums_16b[2]), vget_low_s16(sums_16b[3]));
    sums_32b[3]=vaddl_high_s16(sums_16b[2], sums_16b[3]);

    for (unsigned int i=0; i<4; i++) {
        vst1q_s32(output + 4*i, vaddq_s32(sums_32b[i], vld1q_s32(output + 4*i)));
    }
}

/* "first_col" parameter is used to offset the read into the qp.bias array,
 * in cases where we are not computing the first columns of the output (i.e.
 * in multithreaded cases where we divide columns across threads) */
template<typename T>
void compute_col_sums(const Requantize32 &qp, unsigned int width, unsigned int height, const T *input, unsigned int in_stride, int32_t *col_bias, unsigned int depth, unsigned int multi, unsigned int first_col) {
    /* Only actually add up the columns if a_offset is non-zero. */
    if (qp.a_offset != 0) {
        memset(reinterpret_cast<void *>(col_bias), 0, width * sizeof(int32_t));

        for (unsigned int row=0; row<height; row+=4) {
            unsigned int numrows=std::min(height-row, 4u);

            for (unsigned int col=0; col<width; col+=16) {
                unsigned int numcols=std::min(width-col, 16u);

                if (numcols==16) {
                    switch(numrows) {
                        case 1:
                            add_block<1>(input + row * in_stride + col, in_stride, col_bias + col);
                            break;

                        case 2:
                            add_block<2>(input + row * in_stride + col, in_stride, col_bias + col);
                            break;

                        case 3:
                            add_block<3>(input + row * in_stride + col, in_stride, col_bias + col);
                            break;

                        case 4:
                            add_block<4>(input + row * in_stride + col, in_stride, col_bias + col);
                            break;

                        default:
                            UNREACHABLE("Impossible.");
                    }
                } else {
                    for (; col<width; col++) {
                        int32_t sum=0;
                        for (unsigned int r=0; r<numrows; r++) {
                            sum += input[(row + r)*in_stride + col];
                        }
                        col_bias[col] += sum;
                    }
                }
            }
        }
    }

    for (unsigned int col=0; col<width; col++) {
        int32_t result = col_bias[col];

        result = (qp.a_offset * qp.b_offset * depth) - (result * qp.a_offset);

        if (qp.bias != nullptr) {
            result += qp.bias[multi * qp.bias_multi_stride + col + first_col];
        }

        col_bias[col] = result;
    }
}

template void compute_col_sums(const Requantize32 &qp, unsigned int width, unsigned int height, const int8_t *input, unsigned int in_stride, int32_t *col_bias, unsigned int depth, unsigned int multi, unsigned int first_col);
template void compute_col_sums(const Requantize32 &qp, unsigned int width, unsigned int height, const uint8_t *input, unsigned int in_stride, int32_t *col_bias, unsigned int depth, unsigned int multi, unsigned int first_col);

template<>
void dequantize_block_32<float>(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                         const int32_t* in_ptr, unsigned int in_stride, float *out_ptr, unsigned int out_stride,
                         const float* bias_ptr, bool accumulate, const Activation &act)
{
    const float32x4_t vscale = vdupq_n_f32(qp.scale);
    float maxval = std::numeric_limits<float>::infinity();
    float minval = -std::numeric_limits<float>::infinity();

    switch(act.type) {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            maxval = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            minval = 0;
            break;
    }

    const float32x4_t vmin = vdupq_n_f32(minval);
    const float32x4_t vmax = vdupq_n_f32(maxval);

    for(unsigned int row=0; row<height; row++) {
        auto row_in_ptr = in_ptr + (row * in_stride);
        auto row_out_ptr = out_ptr + (row * out_stride);
        unsigned int col=0;
        if (width >= 4) {
            for(; col <= (width - 4); col+= 4) {
                const int32x4_t vin = vld1q_s32(row_in_ptr + col);
                float32x4_t vdeq = vmulq_f32(vcvtq_f32_s32(vin), vscale);
                if(bias_ptr) {
                    const float32x4_t bin = vld1q_f32(bias_ptr + col);
                    vdeq = vaddq_f32(vdeq, bin);
                }
                if(accumulate) {
                    vdeq = vaddq_f32(vdeq, vld1q_f32(row_out_ptr + col));
                }
                vdeq = vminq_f32(vmaxq_f32(vdeq, vmin), vmax);
                vst1q_f32(reinterpret_cast<float *>(row_out_ptr + col), vdeq);
            }
        }
        // left-over elements
        for(; col < width; ++col) {
            const int32_t val = *(row_in_ptr + col);
            float res = static_cast<float>(val * qp.scale);
            if(bias_ptr) {
                res += static_cast<float>(*(bias_ptr + col));
            }
            if(accumulate) {
                res += *(row_out_ptr + col);
            }
            res = std::min(std::max(res, minval), maxval);
            *(row_out_ptr + col) = res;
        }
    }
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
