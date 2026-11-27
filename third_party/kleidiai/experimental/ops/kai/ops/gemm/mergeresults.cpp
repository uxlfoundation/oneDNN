//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

/* As some of the merges need these headers, but are all included in the
 * kai::ops namespace, put these headers here.  */
#include <algorithm>
#include <cstring>

#include "mergeresults.hpp"

#include <arm_neon.h>

#include "kai/ops/gemm/kai_ops.hpp"
#include "asmlib.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai/ops/float16.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

template<unsigned int twidth, unsigned int height, bool sve, typename Tin, typename Tout>
void MergeResults(const KernelArgs *args) {
    auto *out = static_cast<Tout *>(args->dst);
    const auto *in = static_cast<const Tin *>(args->lhs);
    const auto *bias = static_cast<const Tout *>(args->bias);
    const int ldc = args->dst_stride / sizeof(Tout);
    const int y0 = 0, ymax = args->M, x0 = 0, xmax = args->N;
    const bool append = args->accumulate;
    Tout output_min, output_max;
    std::memcpy(&output_min, &args->clamp_min.data, sizeof(Tout));
    std::memcpy(&output_max, &args->clamp_max.data, sizeof(Tout));
    const Tin minval = static_cast<Tin>(output_min);
    const Tin maxval = static_cast<Tin>(output_max);
    // NOTE: The following code is disabled to avoid calling get_vector_length(), so templated MergeResults will not
    // be correct for SVE cases.  This is OK as we have specialisations for all needed SVE cases anyway.
    //
    // For SVE cases, multiply the width up by the vector length.
    // Use the *input* type to determine this, since this will be what the kernel operated on.
    // const int width = twidth * (sve ? get_vector_length<Tin>() : 1);
    const int width = twidth;

    const int full_y_blocks = (ymax - y0) / height;
    const int y_remainder = (ymax - y0) % height;
    const int y_blocks = full_y_blocks + (y_remainder ? 1 : 0);

    const int full_x_blocks = (xmax - x0) / width;
    const int x_remainder = (xmax - x0) % width;
    const int x_blocks = full_x_blocks + (x_remainder ? 1 : 0);

    for (int y_block = 0; y_block < y_blocks; y_block++) {
        int ybase = y0 + (y_block * height);

        int fill_rows = (y_block < full_y_blocks) ? height : y_remainder;

        for (int x_block = 0; x_block < x_blocks; x_block++) {
            int xbase = x0 + (x_block * width);

            int fill_cols = (x_block < full_x_blocks) ? width : x_remainder;

            for (int row=0; row < fill_rows; row++) {
                for (int col=0; col < fill_cols; col++) {
                    Tout &r = out[(ybase + row) * ldc + xbase + col];
                    Tin v = in[row * width + col];

                    if (append) {
                        v += r;
                    }

                    if (bias) {
                        v += bias[xbase + col];
                    }

                    v = std::max(minval, std::min(maxval, v));

                    r = v;
                }
            }

            in += (width * height);
        }
    }
}

#include "merges/list.hpp"

}  // namespace ops
}  // namespace kai
