/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#if USE_LARGE_INDEX == 1
typedef long idx_t;
#else
typedef int idx_t;
#endif

#include "gpu/intel/ocl/concat_common.h"
#define unroll_for __attribute__((opencl_unroll_hint)) for

#define SRC_PARAM(n) \
    CS_PARAM(__global const DATA_T *src##n, const idx_t src_ext_offset##n, \
            const idx_t offset##n, const idx_t padded_offset##n, \
            const idx_t src_concat_axis##n)
#define SRC_PARAMS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PARAM)

idx_t get_concat_idx(idx_t inner_idx, idx_t inner_offset) {
    idx_t block = 1;
    idx_t idx = 0;
#define HANDLE(n) \
    idx += block * ((inner_idx / (BLOCK_S##n)) % (BLOCK_B##n)); \
    block *= BLOCK_B##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + block * (inner_idx / inner_offset);
}

idx_t get_concat_offset(idx_t concat_idx, idx_t inner_offset) {
    idx_t block = 1;
    idx_t idx = 0;
#define HANDLE(n) \
    idx += (BLOCK_S##n) * ((concat_idx / block) % (BLOCK_B##n)); \
    block *= BLOCK_B##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + inner_offset * (concat_idx / block);
}

struct write_info_t {
    idx_t idx;
    bool write;
};

struct write_info_t get_write_info(const idx_t src_idx,
        const idx_t src_ext_offset, const idx_t concat_offset,
        const idx_t dst_ext_offset, const idx_t thread_offset,
        const idx_t concat_axis, const idx_t last_concat_axis,
        const idx_t zero_pad_offset, const idx_t zero_pad_concat_axis,
        const idx_t inner_offset, bool must_compute_ext_idx) {

    idx_t inner_idx, ext_idx;
    if (must_compute_ext_idx) {
        // short circuit logic avoids significant assembly bloat
        // in the special case
        ext_idx = (src_ext_offset == 1) ? src_idx : src_idx / src_ext_offset;
        inner_idx = (src_ext_offset == 1) ? 0 : src_idx % src_ext_offset;
    } else {
        ext_idx = 0;
        inner_idx = src_idx;
    }

    struct write_info_t info;

#if BLOCK_DEPTH > 0
    idx_t concat_idx = get_concat_idx(inner_idx, inner_offset);
    bool write_value = concat_offset + concat_idx < concat_axis;

    idx_t write_offset;
    if (last_concat_axis < zero_pad_concat_axis) {
        idx_t zero_pad_start = zero_pad_offset - concat_axis + thread_offset;
        bool write_zeropad = zero_pad_start + concat_idx < zero_pad_concat_axis;

        write_offset = write_value ? concat_offset : zero_pad_start;
        info.write = write_value || write_zeropad;
    } else {
        write_offset = concat_offset;
        info.write = write_value;
    }
    info.idx = ext_idx * dst_ext_offset + inner_idx
            + get_concat_offset(concat_idx + write_offset, inner_offset)
            - get_concat_offset(concat_idx, inner_offset);
#else

    info.write = true;
    info.idx = ext_idx * dst_ext_offset + inner_offset * concat_offset
            + inner_idx;

#endif
    return info;
}

#if SIMD != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
reusable_simple_concat(__global DATA_T *dst, const ulong dst_offset0,
        const ulong dst_ext_offset, SRC_PARAMS, const idx_t zero_pad_offset,
        const idx_t zero_pad_concat_axis, const idx_t read_overlap,
        const idx_t gws0_block, const idx_t inner_offset,
        const uchar must_compute_ext_idx) {
    __global const DATA_T *src;
    idx_t src_ext_offset, input_offset;
    idx_t input_padded_offset, concat_axis_size;

#define CHECK_AND_GET(n) \
    if (get_global_id(2) >= padded_offset##n) { \
        src = src##n; \
        input_offset = offset##n; \
        input_padded_offset = padded_offset##n; \
        concat_axis_size = src_concat_axis##n; \
        src_ext_offset = src_ext_offset##n; \
    }
    REDUCE(N_INPUTS, JOIN_ELSE, CHECK_AND_GET);
#undef CHECK_AND_GET

    const idx_t block_offset
            = gws0_block * (get_global_id(2) - input_padded_offset)
            + (get_global_id(0) / SIMD) * READ_BLOCK;
    const idx_t thr_elems = READ_BLOCK / SIMD;
    src += read_overlap * get_global_id(1) * src_ext_offset + block_offset;

#define CONCAT_AXIS(n) src_concat_axis##n
#define RIGHT(x, y) y
#define WRITE_INFO(idx) \
    get_write_info(block_offset + (idx), src_ext_offset, input_offset, \
            dst_ext_offset, input_padded_offset, concat_axis_size, \
            REDUCE(N_INPUTS, RIGHT, CONCAT_AXIS), zero_pad_offset, \
            zero_pad_concat_axis, inner_offset, must_compute_ext_idx)

#if SIMD == 1
    dst += dst_offset0 + read_overlap * get_global_id(1) * dst_ext_offset;
    for (int i = 0; i < thr_elems; ++i) {
        struct write_info_t info = WRITE_INFO(i);
        if (info.write) dst[info.idx] = src[i];
    }
#else
    const uint lane = get_sub_group_local_id() % SIMD;
    buffer_t buf;

    if (READ_BLOCK * DATA_TYPE_SIZE % 4 != 0) {
        for (int i = 0; i < thr_elems; ++i)
            buf.v1[i] = src[i * SIMD + lane];
    } else {
#define MAYBE_BLOCK_READ(n, read, elems) \
    do { \
        if ((elems) & (n)) { \
            const uint rel = (elems) & ~(((n) << 1) - 1); \
            buf.v##n[rel / (n)] = read(&src[rel * SIMD]); \
        } \
    } while (0)

        for (int i = 0; i < thr_elems / 16; ++i)
            buf.v16[i] = BLOCK_READ16(&src[16 * i * SIMD]);
        MAYBE_BLOCK_READ(8, BLOCK_READ8, thr_elems);
        MAYBE_BLOCK_READ(4, BLOCK_READ4, thr_elems);
        MAYBE_BLOCK_READ(2, BLOCK_READ2, thr_elems);
        MAYBE_BLOCK_READ(1, BLOCK_READ, thr_elems);
#undef MAYBE_BLOCK_READ
    }
    if (lane < READ_BLOCK % SIMD)
        buf.v1[thr_elems] = src[thr_elems * SIMD + lane];

    dst += dst_offset0 + read_overlap * get_global_id(1) * dst_ext_offset;
    if (WRITE_BLOCK * DATA_TYPE_SIZE % 16 != 0) {
        for (int i = 0; i < thr_elems; ++i) {
            struct write_info_t info = WRITE_INFO(i * SIMD + lane);
            if (info.write) dst[info.idx] = buf.v1[i];
        }
        if (lane < READ_BLOCK % SIMD) {
            struct write_info_t info = WRITE_INFO(thr_elems * SIMD + lane);
            if (info.write) dst[info.idx] = buf.v1[thr_elems];
        }
    } else {
        // Break up the data that was read into several blocks that may be written
        // sequentially. If the block size is not divisible by the subgroup size,
        // borrow values from the next block(s), if available, to fill a scattered
        // write.
        const uint elems_per_iteration = MAX(SIMD, WRITE_BLOCK);
        const uint iterations = DIV_UP(READ_BLOCK, elems_per_iteration);

        unroll_for(int j = 0; j < iterations; ++j) {
            const uint buf_off = DIV_UP(j * elems_per_iteration, SIMD);
            const uint block_off = buf_off * SIMD;
            // Accounting for any values borrowed from the last iteration, this
            // block only has `iter_elems` values to write:
            const uint iter_elems = WRITE_BLOCK - (block_off % WRITE_BLOCK);
            const uint thr_iter_elems = iter_elems / SIMD;
            struct write_info_t info = WRITE_INFO(block_off);
            __global DATA_T *iter_dst = dst + info.idx;
#define MAYBE_BLOCK_WRITE(n, write, elems) \
    do { \
        if ((elems) & (n)) { \
            const uint rel = (elems) & ~(((n) << 1) - 1); \
            write(&iter_dst[rel * SIMD], load_vec##n(&buf, buf_off + rel)); \
        } \
    } while (0)

            if (info.write) {
                for (int i = 0; i < thr_iter_elems / 16; ++i)
                    BLOCK_WRITE16(&iter_dst[16 * i * SIMD],
                            load_vec16(&buf, buf_off + i * 16));
                MAYBE_BLOCK_WRITE(8, BLOCK_WRITE8, thr_iter_elems);
                MAYBE_BLOCK_WRITE(4, BLOCK_WRITE4, thr_iter_elems);
                MAYBE_BLOCK_WRITE(2, BLOCK_WRITE2, thr_iter_elems);
                MAYBE_BLOCK_WRITE(1, BLOCK_WRITE, thr_iter_elems);
            }
#undef MAYBE_BLOCK_WRITE

            // Write tail elements + the leading elements of the next block
            if (iter_elems % SIMD) {
                const uint written = block_off + thr_iter_elems * SIMD;
                struct write_info_t info = WRITE_INFO(written + lane);
                if (info.write && lane < MIN(SIMD, READ_BLOCK - written))
                    dst[info.idx] = buf.v1[buf_off + thr_iter_elems];
            }
        }
    }
#endif // SIMD == 1
}

#define DATA1_T DATA_T
#if DATA_TYPE_SIZE == 8
#define NPERSG 1
#define BBLOCK_READ BLOCK_READ
#define BBLOCK_WRITE BLOCK_WRITE
#define DDATA_T DATA1_T
#define AS_VEC as_ulong
#elif DATA_TYPE_SIZE == 4
#define NPERSG 2
#define BBLOCK_READ BLOCK_READ2
#define BBLOCK_WRITE BLOCK_WRITE2
#define DDATA_T DATA2_T
#define AS_VEC as_uint2
#elif DATA_TYPE_SIZE == 2
#define NPERSG 4
#define BBLOCK_READ BLOCK_READ4
#define BBLOCK_WRITE BLOCK_WRITE4
#define DDATA_T DATA4_T
#define AS_VEC as_ushort4
#elif DATA_TYPE_SIZE == 1
#define NPERSG 8
#define BBLOCK_READ BLOCK_READ8
#define BBLOCK_WRITE BLOCK_WRITE8
#define DDATA_T DATA8_T
#define AS_VEC as_uchar8
#endif

/*
 * This kernel handles internal padding cases by abstracting the problem into 3 cases:
 *     - Reads and writes fall completely into the first source and can be copied as is
 *
 *     - Reads and writes fall across the boundary between two sources, neighboring "read blocks"
 *       across the concat-axis boundary will need to be combined to form a single "write block" that will
 *       consist halfway of each of the source blocks
 *
 *     - Reads and writes that fall completely within the second source will continue to misalign along
 *       the halfway cutoff and will likewise require two reads per write, just with both reads within
 *       the same second source
 */
#if SIMD != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
internal_padding_block_concat2(__global DATA_T *dst,
        const idx_t dst_concat_axis, const idx_t dst_padded_concat_axis,
        __global const DATA_T *src0, const idx_t offset0,
        const idx_t padded_offset0, const idx_t src_concat_axis0,
        const idx_t src_padded_concat_axis0, __global const DATA_T *src1,
        const idx_t offset1, const idx_t padded_offset1,
        const idx_t src_concat_axis1, const idx_t src_padded_concat_axis1,
        const idx_t inner_dim) {

    const size_t dtsize = DATA_TYPE_SIZE;
    const int loads_per_sg = NPERSG;
    const int elems_per_sg = SIMD * loads_per_sg;

#if BLOCK_DEPTH > 0
    const unsigned B0 = BLOCK_B0;
#else
    const unsigned B0 = 1;
#endif

    const size_t first_boundary_block
            = ((DIV_UP(src_concat_axis0, B0) - 1) * inner_dim);
    const size_t last_boundary_block = first_boundary_block + inner_dim - 1;
    const size_t tot_boundary_block
            = (DIV_UP(dst_padded_concat_axis, B0) * inner_dim);

    // TODO: host side check, if blocks_per_sg > #blocks in src0, use legacy method instead
    unsigned blocks_per_sg;
    if (B0 > (elems_per_sg)) {
        blocks_per_sg = B0 / elems_per_sg;
    } else {
        blocks_per_sg = elems_per_sg
                / B0; // 32 / 2 = 16 layout blocks covered by simd
    }

    __global const DATA_T *src;

    // within block idx, ex: 0-7, 0-15, 0-23, 0-31
    size_t blid = get_local_id(0) % B0;

    size_t id_start = get_group_id(0) * elems_per_sg + get_local_id(0);
    long bii = id_start / B0;

    size_t sg_first_bii = get_group_id(0) * elems_per_sg / B0;

    // index along concat dimension
    long ic = (bii / inner_dim) * B0 + blid;
    long ic_end = ((bii + blocks_per_sg) / inner_dim) * B0 + blid;

    size_t sg_last_bii = sg_first_bii + blocks_per_sg
            - 1; //TODO: verify w/B0 > sg_tot_elems

    size_t ccsz
            = src_concat_axis0; // (31) padded concat dimension for calculating batched src
    size_t padded_ccsz
            = padded_offset1; // (32) padded concat dimension for calculating batched src

    size_t cutoff
            = 0; // determines boundary offset for write sg that spans multiple layout blocks
    bool boundary = false; // current sg spans boundary between two inputs?

    size_t batch_offset = inner_dim * padded_ccsz * get_global_id(1);
    size_t batch_offset1
            = inner_dim * src_padded_concat_axis1 * get_global_id(1);
    size_t ext_batch_offset
            = inner_dim * dst_padded_concat_axis * get_global_id(1);

    // completely aligned r/w blocks, no boundary src case nor misaligned reads
    if (sg_last_bii < first_boundary_block) {
        //TODO: no concretes src0, bii
        DDATA_T val = BBLOCK_READ(src0 + batch_offset + sg_first_bii * B0);
        BBLOCK_WRITE(dst + ext_batch_offset + sg_first_bii * B0, val);
    } else if (sg_first_bii > last_boundary_block) {
        // if sg_bii0 fully within second source, update src_bii+other idx vars to match src1
        long src_blid = (ic - src_concat_axis0) % B0;
        long src_bii = sg_first_bii
                - ((padded_offset1 / B0)
                        * inner_dim); // update inner logical block idx to 0-nblocks read range for src1
        if (src_bii < 0) { src_bii += inner_dim; }

        long sg_last_src_bii = src_bii + blocks_per_sg - 1;
        ccsz = src_concat_axis1 - src_concat_axis0;
        // TODO change dst_padded_concat_axis to padded_offset2? //TODO!!!! THIS IS WRONG
        padded_ccsz = dst_padded_concat_axis
                - padded_offset1; // update read padded size for batching
        batch_offset = inner_dim * padded_ccsz * get_global_id(1);

        cutoff = (B0 + (offset1 - padded_offset1) % B0)
                % B0; // positive modulo( offset/padded_offset , block size) |-|-------|
                // this cutoff will continue in following misaligned(multi-cacheline/ block src)

        DDATA_T aVal = BBLOCK_READ(src1 + batch_offset1 + src_bii * B0);
        long next_block = src_bii
                + inner_dim; //offset to read next required block for second half of cutoff
        long next_last_block = next_block + blocks_per_sg - 1;
        const size_t tot_src1_block
                = (DIV_UP(src_padded_concat_axis1, B0) * inner_dim);
        DDATA_T bVal;
        if (next_last_block > tot_src1_block) {
            int n_blocks_tail = tot_src1_block - next_block;
            int rem_elems = n_blocks_tail * B0;

            for (int b = get_local_id(0), vid = 0; b < rem_elems;
                    b += SIMD, vid++) {
#if NPERSG > 1
                bVal[vid] = src1[batch_offset1 + next_block * B0 + b];
#else
                bVal = src1[batch_offset1 + next_block * B0 + b];
#endif
            }
        } else {
            bVal = BBLOCK_READ(src1 + batch_offset1
                    + next_block
                            * B0); //TODO: potential read past end? check w/asan?. YES, causes segfault. develop a MAYBE read instead
        }

        int shift = (B0 - cutoff);
        int sg_shuffle_dt = shift; // TODO?:vec size per thread

        DDATA_T offset_vec_data;
        offset_vec_data = AS_VEC(intel_sub_group_shuffle_down(
                as_ulong(aVal), as_ulong(aVal), sg_shuffle_dt));

        unsigned bshift = (next_block * B0 - src_concat_axis0)
                % B0; //mismatch between and next_block cutoff? src1
        bVal = AS_VEC(intel_sub_group_shuffle_up(
                as_ulong(bVal), as_ulong(bVal), cutoff));
        if ((ic % B0)
                < cutoff) { // TODO: should change together with sg_shuffle_dt sizeof(B0)
            aVal = offset_vec_data;
        } else {
            aVal = bVal;
        }

        if (ic >= dst_concat_axis) {
            aVal = 0;
        } //depends todo: depends where each read falls, whether in range or not TODO: match for loop in BB case

        unsigned blocks_per_simd1 = SIMD / B0; //WRONG! TODO: what if SIMD > B0?
        if (ic_end >= dst_concat_axis) {
#if NPERSG > 1 // TODO: reformulate as &= 0x00FFFF...
            for (int i = 0; i < NPERSG; ++i) {
                if ((((bii + i * blocks_per_simd1) / inner_dim) * B0 + blid)
                        >= dst_concat_axis) {
                    aVal[i] = 0; // NOT OK! slow af
                }
            }
#else
            aVal = 0;
#endif
        }

        if (sg_last_bii > tot_boundary_block) {
            int n_blocks_tail = tot_boundary_block - sg_first_bii;
            int rem_elems = n_blocks_tail * B0;

            for (int b = get_local_id(0), vid = 0; b < rem_elems;
                    b += SIMD, vid++) {
#if NPERSG > 1
                dst[ext_batch_offset + sg_first_bii * B0 + b] = aVal
                        [vid]; // TODO: replace with union? non-standard behavior
#else
                dst[ext_batch_offset + sg_first_bii * B0 + b] = aVal;
#endif
            }
        } else {
            BBLOCK_WRITE(dst + ext_batch_offset + sg_first_bii * B0, aVal);
        }
    } else { // sg span falls within boundary blocks, handle overlaps
        // TODO: no concretes src0, bii
        // load blocks corresponding to first source
        DDATA_T aVal = BBLOCK_READ(src0 + batch_offset + sg_first_bii * B0);
        // since these are "boundary" blocks load corresponding blocks from next source, TODO: what if sg span > boundary? due to long sg reads, maybe set minimum problem sized to avoid edge case logic?

        long next_blid = (ic - src_concat_axis0) % B0;
        long next_bii = sg_first_bii
                - ((padded_offset1 / B0)
                        * inner_dim); // update inner logical block idx to 0-nblocks read range for src1
        if (next_bii < 0) {
            next_bii += inner_dim; // bump negative index to next src
            next_bii = (next_bii < 0)
                    ? 0
                    : next_bii; // if next src begins still "negative" clamp to beginning of next src, can happen if sg spans [n-2, n-1] vertical blocks
        }

        // with certain inner_dim sizes, sg may span before or after border blocks. additional shifting will be required in these cases
        int leading_boundary_shift = (sg_first_bii < first_boundary_block)
                ? first_boundary_block - sg_first_bii
                : 0; // sg span begins before boundary span
        int trailing_boundary_shift = (sg_last_bii > last_boundary_block)
                ? sg_last_bii - last_boundary_block
                : 0; //sg span ends after boundary span

        long sg_last_next_bii
                = next_bii + blocks_per_sg - 1; //sg span in next src

        ccsz = src_concat_axis1 - src_concat_axis0;
        // TODO: change dst_padded_concat_axis to padded_offset2?
        padded_ccsz = dst_padded_concat_axis
                - padded_offset1; // update read padded size for batching
        batch_offset = inner_dim * padded_ccsz * get_global_id(1);

        cutoff = (B0 + (offset1 - padded_offset1) % B0)
                % B0; // positive modulo( offset/padded_offset , block size) |-|-------|
                // this cutoff will continue in following misaligned(multi-cacheline/ block src)

        unsigned blocks_per_simd1 = SIMD / B0; //WRONG! TODO: what if SIMD > B0?
        DDATA_T bVal = BBLOCK_READ(src1 + batch_offset1 + next_bii * B0);
        if (leading_boundary_shift) {
            int block_scaled_leading_shift
                    = leading_boundary_shift / blocks_per_simd1;
            bVal = AS_VEC((as_ulong(bVal)
                    << block_scaled_leading_shift * DATA_TYPE_SIZE * 8));
        }

        int sg_shuffle_dt = cutoff;
        DDATA_T offset_vec_data;

        ulong zero = 0;
        offset_vec_data = AS_VEC(intel_sub_group_shuffle_up(
                as_ulong(zero), as_ulong(bVal), sg_shuffle_dt));

        DDATA_T cVal;
        if (trailing_boundary_shift) {
            // long trailing_bii = sg_last_next_bii - blocks_per_sg - trailing_boundary_shift;
            long trailing_bii = 0;
            cVal = BBLOCK_READ(src1 + batch_offset1 + trailing_bii * B0);

            int block_dt = (blocks_per_sg - trailing_boundary_shift);

            DDATA_T csval;
            csval = AS_VEC(intel_sub_group_shuffle_down(
                    as_ulong(cVal), zero, (B0 - cutoff)));

            int src_bank
                    = ((get_local_id(0) / B0) + block_dt % blocks_per_simd1)
                    % blocks_per_simd1;
            int ntrail = DIV_UP((blocks_per_sg - block_dt),
                    blocks_per_simd1); // at most how many trailing blocks
            ntrail = (src_bank >= (block_dt % blocks_per_simd1)
                             && blocks_per_simd1 > 1
                             && (block_dt % blocks_per_simd1 > 0))
                    ? ntrail - 1
                    : ntrail; // with multiple banks, can have varying trailing shift

            ulong trailmask;
            trailmask = (0xFFFFFFFFFFFFFFFF << (ntrail * DATA_TYPE_SIZE * 8))
                    >> (ntrail * DATA_TYPE_SIZE
                            * 8); // explicitly ignore any reads past boundary blocks

            if (cutoff > 0
                    && (ic % B0)
                            >= cutoff) { // TODO: should change together with sg_shuffle_dt sizeof(B0)
                aVal = AS_VEC(as_ulong(aVal) & trailmask);
                aVal |= offset_vec_data;
            }

            if (cutoff > 0
                    && (ic % B0)
                            < cutoff) { // TODO: should change together with sg_shuffle_dt sizeof(B0)
                aVal = AS_VEC(as_ulong(aVal) & trailmask);
            }

            // reorder according to source banks
            csval = AS_VEC(intel_sub_group_shuffle(
                    as_ulong(csval), src_bank * B0 + (get_local_id(0) % B0)));
            cVal = AS_VEC(
                    as_ulong(csval) << (NPERSG - ntrail) * DATA_TYPE_SIZE * 8);

            if (cutoff > 0
                    && (get_local_id(0) % B0)
                            < cutoff) { // TODO: should change together with sg_shuffle_dt sizeof(B0)
                aVal |= cVal;
            }

            if (ic_end >= dst_concat_axis) {
#if NPERSG > 1 // TODO: reformulate as &= 0x00FFFF...
                for (int i = 0; i < NPERSG; ++i) {
                    if ((((bii + i) / inner_dim) * B0 + blid)
                            >= dst_concat_axis) {
                        aVal[i] = 0;
                    }
                }
#else
                aVal = 0;
#endif
            }
        } else {
            // boundary case w/no trailing overlap
            if (cutoff > 0
                    && (ic % B0)
                            >= cutoff) { // TODO: should change together with sg_shuffle_dt sizeof(B0)
                aVal |= offset_vec_data;
            }
        }
        BBLOCK_WRITE(dst + ext_batch_offset + sg_first_bii * B0, aVal);
    }
}
