

#include "gpu/intel/include/conversion.h"
#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/sdpa/utils.h"

#include "gemm_grouped.h"

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
grouped_micro_gemm(const global half *src, int ldsrc, const global half *wei, int ldwei,
                   global float *dst, int lddst,
                   const global int *src_offsets, const global int *dst_offsets, int n, int k) {
    unsigned long batch = get_group_id(2);
    int2 src_offset = *(global int2*)(src_offsets+batch);
    src += src_offset.x * ldsrc; //A_offsets[batch];

    int m = (int)(src_offset.y - src_offset.x);
    //int n = n_array[batch];

    wei += batch * k * ldwei;
    dst += src_offset.x * lddst; //C_offsets[batch];

    int sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    int sg_j = sub_group_broadcast(get_local_id(1), 0);

    unsigned long wg_i0 = get_group_id(0) * ugemm_grouped_wg_tile_m;
    unsigned long wg_j0 = get_group_id(1) * ugemm_grouped_wg_tile_n;
    unsigned long sg_i0 = sg_i * ugemm_grouped_sg_tile_m;
    unsigned long sg_j0 = sg_j * ugemm_grouped_sg_tile_n;

    if (wg_i0 >= n || wg_j0 >= m) return; /* early exit if outside batch */

    ugemm_grouped_c_type c_tile
      = ugemm_grouped(wei, ldwei, src, ldsrc, n, m, k, wg_i0, wg_j0, 0, sg_i, sg_j);

    //if(get_group_id(2) == 1)
    //printf("(%2lu,%2lu,%2lu)(%2lu,%2lu): grouped_micro_gemm batch %2lu m %2d n %2d k %2d wg_i0 %2lu wg_j0 %2lu sg_i0 %2lu sg_j0 %2lu src_offset %3v2d tile[0]: %5.2v8hlf tile[1]: %5.2v8hlf ldc: %d\n",
           //get_group_id(0), get_group_id(1), get_group_id(2), get_local_id(0), get_local_id(1), batch, m, n, k, wg_i0, wg_j0, sg_i0, sg_j0, src_offset, c_tile.x[0], c_tile.x[1], lddst);
    //printf("(%2lu,%2lu,%2lu)(%2lu,%2lu): grouped_micro_gemm batch %2lu m %2d n %2d k %2d wg_i0 %2lu wg_j0 %2lu sg_i0 %2lu sg_j0 %2lu src_offset %3v2d tile[0]: %5.2v8hlf tile[1]: %5.2v8hlf tile[2]: %5.2v8hlf tile[3]: %5.2v8hlf ldc: %d\n",
    //       get_group_id(0), get_group_id(1), get_group_id(2), get_local_id(0), get_local_id(1), batch, m, n, k, wg_i0, wg_j0, sg_i0, sg_j0, src_offset, c_tile.x[0], c_tile.x[1], c_tile.x[2], c_tile.x[3], lddst);
    tile_store(c_tile, dst, n, m, lddst, wg_i0 + sg_i0, wg_j0 + sg_j0); // note oneDNN version needs a leading dimension parameter
}
