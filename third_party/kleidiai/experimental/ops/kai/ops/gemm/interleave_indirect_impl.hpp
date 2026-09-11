//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Implementations of interleave functions
// These must be included with a "namespace kai::ops" block.

/*
 * Core function that does heavy lifting - interleave 'int_by' rows of width 'width' together.
 *
 * 'height' indicates the actual number of rows to interleave, so if it's less than int_by then the remaining
 * entries are padded (note that this is "GEMM" padding rather than convolution padding, so there is no need to pad
 * with a particular value.
 *
 * Note that it is not expected for this templated version to ever be used - all cases that matter should be
 * explicitly specialized with an optimized implementation.
 */
template<unsigned int height_vectors, unsigned int block, VLType vlt, bool integrate_sums, typename TIn, typename TOut>
void interleave_block( TOut * &out, const TIn * const *in, size_t width, size_t height, size_t row_offset, bool first,
                       int32_t sum_multiplier ) {
#ifdef __aarch64__
    const unsigned int int_by = height_vectors * (vlt == VLType::SVE ? get_vector_length<TOut>() / block :
                                                  (vlt == VLType::SME ? sme::get_vector_length<TOut>() / block : 1 ));
#else
    const unsigned int int_by = height_vectors;
#endif

    std::vector<int32_t> the_sums;

    if (integrate_sums) {
        the_sums = std::vector<int32_t>(int_by, 0);

        if (!first) {
            // In 'integrate sums' mode, we dump the sums at the end on each pass.

            // On the last pass this is correct, but on other passes it is not -
            // so on the subsequent pass we need to take the output written by
            // the previous pass as starting point for the sums, and then
            // overwrite them with new interleaved data.
            int32_t *out_int32 = reinterpret_cast<int32_t *>(out);

            // Rewind pointer to where we wrote out the sums last time.
            out_int32 -= int_by;

            // Restore the running sums.
            memcpy(the_sums.data(), out_int32, int_by * sizeof(int32_t));

            // Update the "real" pointer so that the next output will clobber the old sums.
            out = reinterpret_cast<TOut *>(out_int32);
        }
    }

    for (unsigned int pos=0; pos<width; pos+=block) {
        for (unsigned int row=0; row<int_by; row++) {
            // Row out of range - pad 'block' entries.
            if (row >= height) {
                for (unsigned int col=0; col<block; col++) {
                    *out++ = 0;
                }
                continue;
            }

            for (unsigned int col=0; col<block; col++) {
                // Column out of range - pad a single entry
                if (pos + col >= width) {
                    *out++ = 0;
                    continue;
                }

                if (integrate_sums) {
                    the_sums[row] += in[row][row_offset + pos + col];
                }

                *out++ = in[row][row_offset + pos + col];
            }
        }
    }

    if (integrate_sums) {
        for (auto &i : the_sums) {
            i *= sum_multiplier;
        }

        int32_t *out_int32 = reinterpret_cast<int32_t *>(out);

        memcpy(out_int32, the_sums.data(), int_by * sizeof(int32_t));

        out = reinterpret_cast<TOut *>(out_int32 + int_by);
    }
}

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TOut>
inline void FixupRowSums(TOut * &out, const int32_t row_sum_multiplier) {
#ifdef __aarch64__
    const unsigned int height = height_vectors * (vlt == VLType::SVE ? get_vector_length<TOut>() / block :
                                                  (vlt == VLType::SME ? sme::get_vector_length<TOut>() / block : 1 ));
#else
    const unsigned int height = height_vectors;
#endif

    // If we are integrating row sums, we need to do some fix up, depending on whether the multiplier is non-zero or not.
    if (row_sum_multiplier) {
        // Non-zero: interleave_block<>() will have done the sums and the multiplier, so no need to do anything now.
    } else {
        // Zero: interleave_block<>() will *not* have done the sums, so 'out' will point to the start of the
        // sum block.  We need to insert the (zero) sums, and advance 'out'.
        int32_t *out_int32 = reinterpret_cast<int32_t *>(out);

        for (unsigned int i=0; i<height; i++) {
            out_int32[i] = 0;
        }

        out_int32 += height;

        out = reinterpret_cast<TOut *>(out_int32);
    }
}

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void IndirectInterleave(TOut *out, const TIn * const * const *ptr, unsigned int stringlen,
                        unsigned int rounded_stringlen, const unsigned int y0, const unsigned int ymax,
                        const unsigned int k0, const unsigned int kmax, bool integrate_sums,
                        const int32_t row_sum_multiplier) {
#ifdef __aarch64__
    const unsigned int height = height_vectors * (vlt == VLType::SVE ? get_vector_length<TOut>() / block :
                                                  (vlt == VLType::SME ? sme::get_vector_length<TOut>() / block : 1 ));
#else
    const unsigned int height = height_vectors;
#endif

    // 'interleave_block' implementations are entitled to read a pointer for each row they handle from the input
    // pointer array, even for out of range rows (although they must not subsequently dereference those pointers for
    // out of range rows).  This allows interleave_block to use techniques like row predication, or loading all
    // pointers and conditionally overriding the out of range ones.

    // This is problematic in the "pure" indirect case when we get to the last rows, where it can lead to out of
    // range reads.  Avoid this with a local buffer to use in last-rows cases.  Use alloca as a std::vector can be
    // expensive in highly threaded scenarios.
    const TIn **row_ptrs = reinterpret_cast<const TIn **>(alloca(height * sizeof(const TIn *)));

    // Figure out the starting position based on k0 (with rounded length)
    unsigned int start_string      = k0 / rounded_stringlen;
    unsigned int start_stringpos   = k0 % rounded_stringlen;

    // Process blocks of 'height' height...
    for (unsigned int ybase = y0; ybase < ymax; ybase+=height) {
        // Height to process
        unsigned int active_height = std::min(ymax - ybase, height);

        // Track our progress through the various strings
        unsigned int k_left    = (kmax - k0);
        unsigned int string    = start_string;
        unsigned int stringpos = start_stringpos;

        bool first = true;

        // Prepare to call 'interleave_block' above for each string encompassed by K range
        while (k_left > 0) {
            // Width to process - and the width we will generate (with padding)
            unsigned int in_width   = std::min(k_left, stringlen - stringpos);
            unsigned int out_width  = std::min(k_left, rounded_stringlen - stringpos);

            const TIn * const *row_base = ptr[string] + ybase;

            // If not all rows are valid, copy the ones that are into local array (see above comment).
            if (active_height < height) {
                for (unsigned int i=0; i<active_height; i++) {
                    row_ptrs[i] = ptr[string][ybase + i];
                }

                row_base = row_ptrs;
            }

            // 'integrate_sums' is a function parameter rather than a template parameter to prevent duplicating too
            // much code.  However, integrated sums make no sense for non-integral types and won't ever be
            // requested.  So put a type trait check here to avoid generating pointless code.
            if (std::is_integral<TOut>::value && integrate_sums && row_sum_multiplier) {
                // Ask interleave_block to perform the scaling multiply iff this is the last block
                int32_t multiplier = (k_left > out_width) ? 1 : row_sum_multiplier;

                interleave_block<height_vectors, block, vlt, true>(out, row_base, in_width, active_height, stringpos, first, multiplier);
            } else {
                interleave_block<height_vectors, block, vlt, false>(out, row_base, in_width, active_height, stringpos, first, 0);
            }

            k_left -= out_width;
            string++;
            stringpos=0;
            first=false;
        }

        if (std::is_integral<TOut>::value && integrate_sums) {
            FixupRowSums<height_vectors, block, vlt>(out, row_sum_multiplier);
        }
    }
}

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void ConvolutionInterleave(TOut *out, const TIn *in, size_t in_stride, const convolver<TIn> &conv, const unsigned int rounded_stringlen,
        const unsigned int y0, const unsigned int ymax, const unsigned int k0, const unsigned int kmax, bool integrate_sums, const int32_t row_sum_multiplier) {
#ifdef __aarch64__
    const unsigned int height = height_vectors * (vlt == VLType::SVE ? get_vector_length<TOut>() / block :
                                                  (vlt == VLType::SME ? sme::get_vector_length<TOut>() / block : 1 ));
#else
    const unsigned int height = height_vectors;
#endif

    auto conv_cols = conv.process_columns(in, in_stride, k0, kmax, rounded_stringlen);

    // Use alloca here as a std::vector can be expensive in highly threaded scenarios.
    const TIn **row_ptrs = reinterpret_cast<const TIn **>(alloca(height * sizeof(const TIn *)));

    for (unsigned int ybase = y0; ybase < ymax; ybase += height) {
        // How many of the rows are active - the rest will get padded in interleave_block.
        unsigned int active_height   = std::min(ymax - ybase, height);
        bool first = true;

        auto conv_rows = conv_cols.process_rows(ybase, active_height);

        while (!conv_rows.finished()) {
            unsigned int width, offset;

            // Get next set of parameters
            std::tie(width, offset) = conv_rows.next_block(row_ptrs);

            // Perform the interleave
            if (std::is_integral<TOut>::value && integrate_sums && row_sum_multiplier) {
                // Ask interleave_block to perform the scaling multiply iff this is the last block
                int32_t multiplier = conv_rows.finished() ? row_sum_multiplier : 1;

                interleave_block<height_vectors, block, vlt, true>(out, row_ptrs, width, active_height, offset, first, multiplier);
            } else {
                interleave_block<height_vectors, block, vlt, false>(out, row_ptrs, width, active_height, offset, first, 0);
            }

            first=false;
        }

        if (std::is_integral<TOut>::value && integrate_sums) {
            FixupRowSums<height_vectors, block, vlt>(out, row_sum_multiplier);
        }
    }
}

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void Interleave(TOut *out, const TIn *in, size_t in_stride, const unsigned int y0, const unsigned int ymax, const unsigned int k0, const unsigned int kmax, bool integrate_sums, const int32_t row_sum_multiplier) {
#ifdef __aarch64__
    const unsigned int height = height_vectors * (vlt == VLType::SVE ? get_vector_length<TOut>() / block :
                                                  (vlt == VLType::SME ? sme::get_vector_length<TOut>() / block : 1 ));
#else
    const unsigned int height = height_vectors;
#endif

    // Use alloca here as a std::vector can be expensive in highly threaded scenarios.
    const TIn **row_ptrs = reinterpret_cast<const TIn **>(alloca(height * sizeof(const TIn *)));

    const unsigned int width=kmax-k0;

    for (unsigned int y=y0; y<ymax; y+=height) {
        for (unsigned int r=0; r<height; r++) {
            row_ptrs[r] = in + ((y + r) * in_stride);
        }

        if (std::is_integral<TOut>::value && integrate_sums && row_sum_multiplier) {
            interleave_block<height_vectors, block, vlt, true>(out, row_ptrs, width, std::min(height, ymax-y), k0, true, row_sum_multiplier);
        } else {
            interleave_block<height_vectors, block, vlt, false>(out, row_ptrs, width, std::min(height, ymax-y), k0, true, 0);
        }

        if (std::is_integral<TOut>::value && integrate_sums) {
            FixupRowSums<height_vectors, block, vlt>(out, row_sum_multiplier);
        }
    }
}
