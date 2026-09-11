//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef __aarch64__

#include <arm_neon.h>

#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

#include <cstring>

#include "transform.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

namespace {

// Helper function to interleave a single 4x4 block of 32-bin values
// together.

// _full version doesn't need to worry about any padding.
static inline void transpose_block_32_full(const uint8_t * __restrict in_ptr0, const uint8_t * __restrict in_ptr1, const uint8_t * __restrict in_ptr2, const uint8_t * __restrict in_ptr3, uint8_t * __restrict out_ptr, long output_stride) {
    uint32x4_t inputs[4];
    uint32x4_t inters[4];
    uint32x4_t outputs[4];

    inputs[0] = vld1q_u32(reinterpret_cast<const uint32_t *>(in_ptr0));
    inputs[1] = vld1q_u32(reinterpret_cast<const uint32_t *>(in_ptr1));
    inputs[2] = vld1q_u32(reinterpret_cast<const uint32_t *>(in_ptr2));
    inputs[3] = vld1q_u32(reinterpret_cast<const uint32_t *>(in_ptr3));

    inters[0] = vzip1q_u32(inputs[0], inputs[2]);
    inters[1] = vzip2q_u32(inputs[0], inputs[2]);
    inters[2] = vzip1q_u32(inputs[1], inputs[3]);
    inters[3] = vzip2q_u32(inputs[1], inputs[3]);

    outputs[0] = vzip1q_u32(inters[0], inters[2]);
    outputs[1] = vzip2q_u32(inters[0], inters[2]);
    outputs[2] = vzip1q_u32(inters[1], inters[3]);
    outputs[3] = vzip2q_u32(inters[1], inters[3]);

    vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr), outputs[0]);
    vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr + output_stride), outputs[1]);
    vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr + output_stride*2), outputs[2]);
    vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr + output_stride*3), outputs[3]);
}

// _part version: Only read "bytes_in" bytes, not a full vector.  Only write
// out 4-byte blocks that have some live content (if bytes_in is not a
// multiple of 4 there will some padding in each 4-block)
static inline void transpose_block_32_part(const uint8_t *in_ptr0, const uint8_t *in_ptr1, const uint8_t *in_ptr2, const uint8_t *in_ptr3, uint8_t *out_ptr, long bytes_in, long output_stride) {
    uint32x4_t inputs[4];
    uint32x4_t inters[4];
    uint32x4_t outputs[4];
    uint8_t scratch[16] = {0};

    long num_outs = iceildiv<long>(bytes_in, 4);

    memcpy(scratch, in_ptr0, bytes_in);
    inputs[0] = vld1q_u32(reinterpret_cast<const uint32_t *>(scratch));
    memcpy(scratch, in_ptr1, bytes_in);
    inputs[1] = vld1q_u32(reinterpret_cast<const uint32_t *>(scratch));
    memcpy(scratch, in_ptr2, bytes_in);
    inputs[2] = vld1q_u32(reinterpret_cast<const uint32_t *>(scratch));
    memcpy(scratch, in_ptr3, bytes_in);
    inputs[3] = vld1q_u32(reinterpret_cast<const uint32_t *>(scratch));

    inters[0] = vzip1q_u32(inputs[0], inputs[2]);
    inters[1] = vzip2q_u32(inputs[0], inputs[2]);
    inters[2] = vzip1q_u32(inputs[1], inputs[3]);
    inters[3] = vzip2q_u32(inputs[1], inputs[3]);

    outputs[0] = vzip1q_u32(inters[0], inters[2]);
    outputs[1] = vzip2q_u32(inters[0], inters[2]);
    outputs[2] = vzip1q_u32(inters[1], inters[3]);
    outputs[3] = vzip2q_u32(inters[1], inters[3]);

    do {
        vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr), outputs[0]);
        if (num_outs < 2)
            break;
        vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr + output_stride), outputs[1]);
        if (num_outs < 3)
            break;
        vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr + output_stride*2), outputs[2]);
        if (num_outs < 4)
            break;
        vst1q_u32(reinterpret_cast<uint32_t *>(out_ptr + output_stride*3), outputs[3]);
    } while (0);
}

template<unsigned N>
struct Unroll {
    template<typename F>
    static void run(F f) {
        Unroll<N-1>::run(f);
        f(N-1);
    }
};

template<>
struct Unroll<0> {
    template<typename F>
    static void run(F) {
    }
};

// Interleave some multiple of 4 rows together.
//
// The template parameter BLOCKS controls the size of the inner loop - each BLOCK is 4 rows.
// The function parameter interleave_multiple controls the number of times the inner loop is run.

// The total interleave depth for a given run is therefore BLOCKS * interleave_multiple * 4.
template<unsigned BLOCKS>
void a64_interleave_1x4(uint8_t *out, const uint8_t *in, long width, long in_stride, long height, long interleave_multiple) {
    const long total_interleave_depth = BLOCKS * 4 * interleave_multiple;
    constexpr long loop_interleave_depth = BLOCKS * 4;

    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width));

    if (height % total_interleave_depth) {
        memset(pad_row, 0, width);
    }

    // Outer loop: process blocks of total_interleave_depth rows at a time.
    for (long y0_base=0; y0_base<height; y0_base+=total_interleave_depth) {
        // Middle loop: process each "interlave_multiple" block of rows.
        for (long block=0; block<interleave_multiple; block++) {
            const long y0 = y0_base + (block * loop_interleave_depth);
            uint8_t *out_ptr = out + (block * loop_interleave_depth * 4); // 4 is the blocking depth (we interleave 4 bytes at a time from each input)

            // Create and set up input row pointers.  The idea is that these
            // should entirely fit in the register file, so we don't have to
            // repeatedly load them (or perform the padding check)
#ifndef __COVERITY__
            const uint8_t *in_ptrs[loop_interleave_depth];
            Unroll<loop_interleave_depth>::run( [&](unsigned y) {
                in_ptrs[y] = (y+y0 < height) ? in + ((y+y0) * in_stride) : pad_row;
            });
#endif

            long bytes_left = width;
            // Process full vectors using transpose_block_32_full()
            while (bytes_left >= 16) { // 16 is the vector length in bytes
                Unroll<BLOCKS>::run( [&](unsigned u) {
                    transpose_block_32_full(in_ptrs[u*4 + 0],  in_ptrs[u*4 + 1],  in_ptrs[u*4 + 2],  in_ptrs[u*4 + 3],
                                            out_ptr + 16*u, total_interleave_depth * 4); // 4 is the blocking depth
                });

                Unroll<loop_interleave_depth>::run( [&](unsigned y) {
                    in_ptrs[y] += 16; // 16 is the vector length in bytes
                });

                out_ptr += total_interleave_depth * 16; // 16 is the vector length in bytes
                bytes_left -= 16; // 16 is the vector length in bytes
            }

            // Process any remaining bytes using transpose_block_32_part()
            if (bytes_left) {
                Unroll<BLOCKS>::run( [&](unsigned u) {
                    transpose_block_32_part(in_ptrs[u*4 + 0],  in_ptrs[u*4 + 1],  in_ptrs[u*4 + 2],  in_ptrs[u*4 + 3],
                                            out_ptr + 16*u, bytes_left, total_interleave_depth * 4);
                });
            }
        }

        // Update "out" pointer for next set of total_interleave_depth rows
        out += total_interleave_depth * roundup<long>(width, 4);
    }
}

} // anonymous namespace

template<>
void Transform<16, 4, false, VLType::None>(
    uint8_t *out, const uint8_t *in, int stride, int y0, int ymax, int x0, int xmax)
{
    a64_interleave_1x4<4>(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + y0 * stride + x0),
        (xmax - x0),
        stride,
        (ymax - y0),
        1
    );
}

template<>
void Transform<16, 4, false, VLType::None>(
    int8_t *out, const int8_t *in, int stride, int y0, int ymax, int x0, int xmax)
{
    a64_interleave_1x4<4>(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + y0 * stride + x0),
        (xmax - x0),
        stride,
        (ymax - y0),
        1
    );
}

template<>
void Transform<12, 1, false, VLType::None>(
    float *out, const float *in, int stride, int y0, int ymax, int x0, int xmax)
{
    a64_interleave_1x4<3>(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + y0 * stride + x0),
        (xmax - x0) * sizeof(float),
        stride * sizeof(float),
        (ymax - y0),
        1
    );
}

template<>
void Transform<16, 1, false, VLType::None>(
    float *out, const float *in, int stride, int y0, int ymax, int x0, int xmax)
{
    a64_interleave_1x4<4>(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + y0 * stride + x0),
        (xmax - x0) * sizeof(float),
        stride * sizeof(float),
        (ymax - y0),
        1
    );
}

template<>
void Transform<24, 1, false, VLType::None>(
    float *out, const float *in, int stride, int y0, int ymax, int x0, int xmax)
{
    a64_interleave_1x4<3>(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + y0 * stride + x0),
        (xmax - x0) * sizeof(float),
        stride * sizeof(float),
        (ymax - y0),
        2
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
