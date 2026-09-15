//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "common_internal/utils.hpp"

#include "kai/ops/bfloat.hpp"

#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

namespace kai {
namespace ops {

/*
 * Generic transform.
 *
 * Assuming the untransposed case, this works by first reading <BlockBy>
 * consecutive values from the first input row.  This same number of values
 * are then read from the next <IntBy-1> rows.  Now return to the first
 * input row and repeat.
 *
 * Need to cope with the work requested in either dimension not actually
 * being a multiple of the block sizes.
 */
template <unsigned int tIntBy, unsigned int BlockBy, bool Transposed, size_t TOutSize, size_t TInSize, VLType vlt>
struct TransformImpl {
    template <typename TOut, typename TIn>
    static void Transform(TOut* out, const TIn* const in, const int stride,
                          const int y0, const int ymax, const int x0, const int xmax) {
        // NOTE: This code is disabled to avoid the call to get_vector_length(), so templated transforms will not be
        // correct for SVE.  This is not an issue as we have specializations for all SVE cases.
        // For SVE cases we multiply the interleave factor by the vector length.
        // const unsigned int IntBy = tIntBy * (vlt == VLType::SVE ? get_vector_length<TOut>() / BlockBy : 1);
        const unsigned int IntBy = tIntBy;

        const int n_whole_y_blocks = (ymax - y0) / IntBy;
        const int y_remainders = (ymax - y0) % IntBy;
        const int n_y_blocks = n_whole_y_blocks + (y_remainders ? 1 : 0);

        const int n_whole_x_blocks = (xmax - x0) / BlockBy;
        const int x_remainders = (xmax - x0) % BlockBy;
        const int n_x_blocks = n_whole_x_blocks + (x_remainders ? 1 : 0);

        // "Y" loop: advance down the rows of the source IntBy rows at a time.
        // Set up fill_rows to show the number rows to copy from, and blank_rows
        // for the number of blank rows to add.
        for (int y_block=0 ; y_block < n_y_blocks; y_block++) {
            int fill_rows = (y_block < n_whole_y_blocks) ? IntBy : y_remainders;
            int blank_rows = IntBy - fill_rows;

            int y_base = y0 + (y_block * IntBy);

            // So now advance along this block of rows, BlockBy columns at a time.
            for (int x_block=0 ; x_block < n_x_blocks; x_block++) {
                int fill_cols = (x_block < n_whole_x_blocks) ? BlockBy : x_remainders;
                int blank_cols = BlockBy - fill_cols;

                int x_base = x0 + (x_block * BlockBy);

                for (int row = 0; row < fill_rows; row++) {
                    for (int col = 0; col < fill_cols; col++) {
                        // In-range copy.  If it's transposed, we reverse the sense of rows and columns here.
                        if (Transposed) {
                            *out++ = static_cast<TOut>(in[(x_base + col) * stride + y_base + row]);
                        } else {
                            *out++ = static_cast<TOut>(in[(y_base + row) * stride + x_base + col]);
                        }
                    }
                    // "col" tail - row is in range but column is out of range.
                    for (int col=0; col < blank_cols; col++) {
                        *out++ = static_cast<TOut>(0);
                    }
                }
                // "row" tail - row is out of range so fill with zeros always.
                TOut zeroval = static_cast<TOut>(0);
                int pads = blank_rows * (fill_cols + blank_cols);

                for (int i=0; i<pads; i++) {
                    out[i] = zeroval;
                }

                out += pads;
            }
        }
    }

    template <typename T>
    static void Transform(T* out, const T* const in, const int stride,
                                 const int k0, const int kmax, const int x0, const int xmax) {
        Transform<T, T>(out, in, stride, k0, kmax, x0, xmax);
    }
};

/*****************************************************************************/
template <unsigned int IntBy, unsigned int BlockBy, bool Transposed, VLType vlt=VLType::None, typename TOut, typename TIn>
void Transform(
  TOut* out, const TIn* const in, const int stride,
  const int k0, const int kmax, const int x0, const int xmax
) {
  // Redirect to a specialised implementation predicated on argument size.
  TransformImpl<IntBy, BlockBy, Transposed, sizeof(TOut), sizeof(TIn), vlt>::Transform(
    out, in, stride, k0, kmax, x0, xmax
  );
}
/*****************************************************************************/

#include "transforms/list.hpp"

// We don't have assembler transforms for AArch32, generate templated ones here.
#ifdef __arm__
template void Transform<8, 1, true, VLType::None>(float *, const float *, int, int, int, int, int);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template void Transform<8, 1, true, VLType::None>(float *, const __fp16 *, int, int, int, int, int);
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template void Transform<8, 1, true, VLType::None>(float *, const bfloat16 *, int, int, int, int, int);
#endif // AArch32

template void Transform<12, 1, false, VLType::None>(float *, const __fp16 *, int, int, int, int, int);
template void Transform<12, 1, false, VLType::None>(float *, const bfloat16 *, int, int, int, int, int);

}  // namespace ops
}  // namespace kai
