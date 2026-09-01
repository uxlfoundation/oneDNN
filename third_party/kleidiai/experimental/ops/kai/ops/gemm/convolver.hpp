//
// SPDX-FileCopyrightText: Copyright 2020, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kai/ops/gemm/convolution_parameters.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

namespace kai {
namespace ops {

// Class to assist with convolution calculations.
//
// This is framed as a hierarchy of objects:
//
//  - Top level object which depends only on convolution parameters.  This sets up std::vectors for the padding and
//    kernel offset arrays.  From this you can request:
//
//  - Mid level object (e.g. instantiated at start of 'ConvolutionInterleave').  This holds specifics about the
//    input tensor, and the desired column range.  Calculations specific to this can be done once when this is set
//    up.  From this you can request:
//
//  - Low level object (instantiated for each range of rows).  This contains methods to actually populate a row
//    pointer array.


template<typename T>
class convolver {
private:
    const ConvolutionParameters  m_params;

    // Vector of padding data
    const std::vector<T>         m_pad_row;

    // X/Y offsets for each kernel position
    std::vector<int>             m_kernel_y;
    std::vector<int>             m_kernel_x;

    class column_handler {
    private:
        const convolver<T>          &m_parent;

        // Base/stride of input image
        const T * const              m_input_base;
        const size_t                 m_input_stride;

        // Starting kernel point and channel offset within that point
        const unsigned int           m_start_pos;
        const unsigned int           m_start_offset;

        // Total length to process, rounded length of each input channel block.
        const unsigned int           m_length;
        const unsigned int           m_rounded_stringlen;

        class row_handler {
        private:
            const convolver<T>          &m_convolver;
            const column_handler        &m_parent;

            // These variables track progress through the current block of rows
            unsigned int                 m_start_output_y=0;
            unsigned int                 m_start_output_x=0;

            unsigned int                 m_length_remaining=0;
            unsigned int                 m_current_pos=0;

            unsigned int                 m_active_height=0;

        public:
            row_handler(const column_handler &parent, unsigned int start_row, unsigned int active_height) :
                m_convolver(parent.m_parent),
                m_parent(parent),
                m_start_output_y(start_row / m_convolver.m_params.output_width),
                m_start_output_x(start_row % m_convolver.m_params.output_width),
                m_length_remaining(m_parent.m_length),
                m_current_pos(m_parent.m_start_pos),
                m_active_height(active_height) { }

            bool finished() const {
                return (m_length_remaining == 0);
            }

            // Compute a block of output pointers, accounting for padding.
            // This is performance critical.
            std::tuple<unsigned int, unsigned int> next_block(const T ** const row_ptr) {
                if (finished()) {
                    return std::make_tuple(0, 0);
                }

                const T *pad_ptr = m_convolver.m_pad_row.data();

                // "in_width" in the amount of data that will be read in (copied)
                // "out_width" is the total amount of data that will be produced (including padding)
                unsigned int offset = (m_current_pos == m_parent.m_start_pos) ? m_parent.m_start_offset : 0;
                unsigned int in_width = std::min(m_length_remaining, static_cast<unsigned int>(m_convolver.m_params.input_channels) - offset);
                unsigned int out_width = std::min(m_length_remaining, m_parent.m_rounded_stringlen - offset);

                unsigned int output_y = m_start_output_y;
                unsigned int output_x = m_start_output_x;

                // Loop over "row" (output points), but really there is one
                // trip through this outer loop per row of output to
                // minimize redundant padding calculations.
                unsigned int row=0;
                while (row < m_active_height) {
                    int input_y = (output_y * m_convolver.m_params.output_stride_h) + m_convolver.m_kernel_y[m_current_pos];
                    int input_x = (output_x * m_convolver.m_params.output_stride_w) + m_convolver.m_kernel_x[m_current_pos];

                    // Factor out base pointer computation.
                    const T *base_ptr = m_parent.m_input_base +
                                        (input_y * m_convolver.m_params.input_width * m_parent.m_input_stride);

                    // To start with, check the input row is in-bounds.  If
                    // not, (at least) this entire output row must be
                    // padding so handle accordingly.

                    // If input_y is off the bottom of the input, we are
                    // going to get padding for every remanining output
                    // point.
                    if (input_y >= m_convolver.m_params.input_height) {
                        while (row < m_active_height) {
                            row_ptr[row++] = pad_ptr;
                        }
                        break;
                    }

                    // If input_y is less than zero, we are going to get
                    // padding for the rest of this output row.
                    if (input_y < 0) {
                        while (output_x < m_convolver.m_params.output_width && row<m_active_height) {
                            row_ptr[row++] = pad_ptr;
                            output_x++;
                        }
                        goto next_row;
                    }

                    // The input row is in bounds - so handle left
                    // padding, then non-padding output, then right
                    // padding.

                    // Left padding
                    while (row < m_active_height && input_x < 0) {
                        row_ptr[row++] = pad_ptr;

                        output_x++;
                        input_x+=m_convolver.m_params.output_stride_w;

                        // Need to detect the end of the row, in case it's
                        // all padding.
                        if (output_x == m_convolver.m_params.output_width) {
                            goto next_row;
                        }
                    }

                    // Non-padding output.  Factor out base pointer calculation.
                    while (row < m_active_height && input_x < m_convolver.m_params.input_width) {
                        row_ptr[row++] = base_ptr + (input_x * m_parent.m_input_stride);

                        output_x++;
                        input_x+=m_convolver.m_params.output_stride_w;

                        if (output_x == m_convolver.m_params.output_width) {
                            goto next_row;
                        }
                    }

                    // Right padding.
                    while (row < m_active_height && output_x < m_convolver.m_params.output_width) {
                        row_ptr[row++] = pad_ptr;
                        output_x++;
                    }

                    // Update output indices for next row.  Used as a "goto"
                    // target due to end-of-row checks in nested loops.
next_row:
                    output_x=0;
                    output_y++;
                }

                m_current_pos++;
                m_length_remaining-=out_width;

                return std::make_tuple(in_width, offset);
            }
        }; // end of "row handler" class

    public:
        column_handler(const convolver<T> &parent, const T *input_base, size_t input_stride,
                       unsigned int k_start, unsigned int k_end, unsigned int rounded_stringlen)
                     : m_parent(parent), m_input_base(input_base), m_input_stride(input_stride),
                       m_start_pos(k_start / rounded_stringlen),
                       m_start_offset(k_start % rounded_stringlen),
                       m_length(k_end - k_start),
                       m_rounded_stringlen(rounded_stringlen) { }

        row_handler process_rows(unsigned int start_row, unsigned int active_height) const {
            return row_handler(*this, start_row, active_height);
        }
    }; // end of "column handler" class

public:
    convolver(ConvolutionParameters params) :
        m_params (params), m_pad_row(params.input_channels, static_cast<T>(params.padding_value)),
        m_kernel_y(params.kernel_width * params.kernel_height, 0),
        m_kernel_x(params.kernel_width * params.kernel_height, 0) {

        // Kernel points are addressed across, then down (assumed weight layout is WHIO)
        for (unsigned int ky=0; ky<params.kernel_height; ky++) {
            for (unsigned int kx=0; kx<params.kernel_width; kx++) {
                unsigned int n = (ky * params.kernel_width) + kx;
                m_kernel_y[n] = (ky * params.dilation_h) - params.padding_top;
                m_kernel_x[n] = (kx * params.dilation_w) - params.padding_left;
            }
        }
    }

    column_handler process_columns(const T *input_base, size_t input_stride,
                                   unsigned int k_start, unsigned int k_end, unsigned int rounded_stringlen) const {
        return column_handler(*this, input_base, input_stride, k_start, k_end, rounded_stringlen);
    }
};

}  // namespace ops
}  // namespace kai
