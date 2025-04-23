/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#ifndef CPU_PPC64_PPC64_GEMM_S8X8S32_HPP
#define CPU_PPC64_PPC64_GEMM_S8X8S32_HPP

#include <altivec.h>
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {

typedef __vector signed long long vec_i64 __attribute__((aligned(8)));
typedef __vector short vec_i16 __attribute__((aligned(2)));
typedef __vector unsigned char vec_t;
typedef __vector signed char vec_st;
typedef __vector_pair vecp_t;

inline int pack_N16_16bit(dim_t k, dim_t m, short *a, dim_t lda, short *ap) {
    int32_t i, j;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, k8, m4, m16;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;
    k8 = (k >> 3) << 3;
    m4 = (m >> 2) << 2;
    m16 = (m >> 4) << 4;

    // MAIN BLOCK
    for (j = 0; j < m16; j += 4) {
        for (i = 0; i < k8; i += 8) {
            kcell = i >> 1; // 0, 1, 2, 3
            mcell = j >> 2;
            short *dest = &ap[32 * ((mcell >> 2) * krows + kcell)
                    + 8 * (mcell & 3)];

            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29,
                    30, 31};
            vec_t swizL
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizR = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};

            V0 = *(vec_t *)&a[lda * (j + 0) + i];
            V1 = *(vec_t *)&a[lda * (j + 1) + i];
            V2 = *(vec_t *)&a[lda * (j + 2) + i];
            V3 = *(vec_t *)&a[lda * (j + 3) + i];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[32] = D1;
            *(vec_t *)&dest[64] = D2;
            *(vec_t *)&dest[96] = D3;
        }
    }

    for (j = m16; j < m4; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (j = m4; j < m_cap; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (j < m)
                ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < m4; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (i < k)
                ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (j = m4; j < m_cap; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (j < m && i < k)
                ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }
    return 0;
}

inline int pack_T16_16bit(dim_t k, dim_t m, short *a, dim_t lda, short *ap) {
    int32_t i, j;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, k4, m8, m16;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;
    k4 = (k >> 2) << 2;
    m16 = (m >> 4) << 4;
    m8 = (m >> 3) << 3;

    // MAIN BLOCK
    for (i = 0; i < k4; i += 4) {
        for (j = 0; j < m16; j += 16) {
            short *src = &a[lda * i + j];
            short *dst = &ap[2 * j * krows + 16 * i];
            vec_t V0, V1, V2, V3, V4, V5, V6, V7;
            vec_t D0, D1, D2, D3, D4, D5, D6, D7;
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            V0 = *(vec_t *)&src[0];
            V1 = *(vec_t *)&src[lda];
            V2 = *(vec_t *)&src[8];
            V3 = *(vec_t *)&src[lda + 8];
            V4 = *(vec_t *)&src[2 * lda];
            V5 = *(vec_t *)&src[3 * lda];
            V6 = *(vec_t *)&src[2 * lda + 8];
            V7 = *(vec_t *)&src[3 * lda + 8];
            D0 = vec_perm(V0, V1, swizL);
            D1 = vec_perm(V0, V1, swizR);
            D2 = vec_perm(V2, V3, swizL);
            D3 = vec_perm(V2, V3, swizR);
            D4 = vec_perm(V4, V5, swizL);
            D5 = vec_perm(V4, V5, swizR);
            D6 = vec_perm(V6, V7, swizL);
            D7 = vec_perm(V6, V7, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[8] = D1;
            *(vec_t *)&dst[16] = D2;
            *(vec_t *)&dst[24] = D3;
            *(vec_t *)&dst[32] = D4;
            *(vec_t *)&dst[40] = D5;
            *(vec_t *)&dst[48] = D6;
            *(vec_t *)&dst[56] = D7;
        }
    }

    for (i = 0; i < k4; ++i) {
        for (j = m16; j < m8; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (i = 0; i < k4; ++i) {
        for (j = m8; j < m_cap; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (j < m)
                ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k4; i < k_cap; ++i) {
        for (j = 0; j < m8; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (i < k)
                ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (i = k4; i < k_cap; ++i) {
        for (j = m8; j < m_cap; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (i < k && j < m)
                ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }
    return 0;
}

inline int pack_T8_16bit(dim_t k, dim_t n, short *b, dim_t ldb, short *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k4, n8, n16;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    k4 = (k >> 2) << 2;
    n8 = (n >> 3) << 3;
    n16 = (n >> 4) << 4;

    // MAIN BLOCK
    for (i = 0; i < k4; i += 4) {
        for (j = 0; j < n16; j += 16) {
            short *src = &b[ldb * i + j];
            short *dst0145 = &bp[2 * j * krows + 8 * i];
            short *dst2367 = &bp[2 * (j + 8) * krows + 8 * i];
            vec_t V0, V1, V2, V3, V4, V5, V6, V7;
            vec_t D0, D1, D2, D3, D4, D5, D6, D7;
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            V0 = *(vec_t *)&src[0];
            V1 = *(vec_t *)&src[ldb];
            V2 = *(vec_t *)&src[8];
            V3 = *(vec_t *)&src[ldb + 8];
            V4 = *(vec_t *)&src[2 * ldb];
            V5 = *(vec_t *)&src[3 * ldb];
            V6 = *(vec_t *)&src[2 * ldb + 8];
            V7 = *(vec_t *)&src[3 * ldb + 8];
            D0 = vec_perm(V0, V1, swizL);
            D1 = vec_perm(V0, V1, swizR);
            D2 = vec_perm(V2, V3, swizL);
            D3 = vec_perm(V2, V3, swizR);
            D4 = vec_perm(V4, V5, swizL);
            D5 = vec_perm(V4, V5, swizR);
            D6 = vec_perm(V6, V7, swizL);
            D7 = vec_perm(V6, V7, swizR);

            *(vec_t *)&dst0145[0] = D0;
            *(vec_t *)&dst0145[8] = D1;
            *(vec_t *)&dst2367[0] = D2;
            *(vec_t *)&dst2367[8] = D3;
            *(vec_t *)&dst0145[16] = D4;
            *(vec_t *)&dst0145[24] = D5;
            *(vec_t *)&dst2367[16] = D6;
            *(vec_t *)&dst2367[24] = D7;
        }
        for (j = n16; j < n8; j += 8) {
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            short *dst = &bp[8 * (columns_done * krows + (i & (~1)))];
            vec_t V0, V1, V2, V3;
            vec_t D0, D1, D2, D3;
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            V0 = *(vec_t *)&b[ldb * (i + 0) + j];
            V1 = *(vec_t *)&b[ldb * (i + 1) + j];
            V2 = *(vec_t *)&b[ldb * (i + 2) + j];
            V3 = *(vec_t *)&b[ldb * (i + 3) + j];
            D0 = vec_perm(V0, V1, swizL);
            D1 = vec_perm(V0, V1, swizR);
            D2 = vec_perm(V2, V3, swizL);
            D3 = vec_perm(V2, V3, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[8] = D1;
            *(vec_t *)&dst[16] = D2;
            *(vec_t *)&dst[24] = D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (i = 0; i < k4; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (j < n)
                bp[8 * cell + 2 * noff + koff] = b[ldb * i + j];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k4; i < k_cap; ++i) {
        for (j = 0; j < n8; ++j) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (i < k)
                bp[8 * cell + 2 * noff + koff] = b[ldb * i + j];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (i = k4; i < k_cap; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (i < k && j < n)
                bp[8 * cell + 2 * noff + koff] = b[ldb * i + j];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }
    return 0;
}

inline int pack_N8_16bit(dim_t k, dim_t n, short *b, dim_t ldb, short *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, k16, n4, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    k8 = (k >> 3) << 3;
    k16 = (k >> 4) << 4;
    n4 = (n >> 2) << 2;
    n8 = (n >> 3) << 3;

    // MAIN BLOCK
    for (j = 0; j < n8; j += 4) {
        for (i = 0; i < k16; i += 16) {
            kcell = i >> 1; // 0, 1, 2, 3
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t j_hiflag = (j & 4) >> 2;
            koff = i & 1;
            noff = j & 3;
            short *dst = &bp[8 * (columns_done * krows + kcell * 2 + j_hiflag)];

            vec_t V0, V1, V2, V3, V4, V5, V6, V7;
            vec_t D01A, D01B, D23A, D23B, D45A, D45B, D67A, D67B;
            vec_t D0, D1, D2, D3, D4, D5, D6, D7;
            vec_t swizA
                    = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29,
                    30, 31};
            vec_t swizL
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizR = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};

            V0 = *(vec_t *)&b[ldb * (j + 0) + i];
            V1 = *(vec_t *)&b[ldb * (j + 1) + i];
            V2 = *(vec_t *)&b[ldb * (j + 2) + i];
            V3 = *(vec_t *)&b[ldb * (j + 3) + i];
            V4 = *(vec_t *)&b[ldb * (j + 0) + i + 8];
            V5 = *(vec_t *)&b[ldb * (j + 1) + i + 8];
            V6 = *(vec_t *)&b[ldb * (j + 2) + i + 8];
            V7 = *(vec_t *)&b[ldb * (j + 3) + i + 8];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D45A = vec_perm(V4, V5, swizA);
            D45B = vec_perm(V4, V5, swizB);
            D67A = vec_perm(V6, V7, swizA);
            D67B = vec_perm(V6, V7, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            D4 = vec_perm(D45A, D67A, swizL);
            D5 = vec_perm(D45A, D67A, swizR);
            D6 = vec_perm(D45B, D67B, swizL);
            D7 = vec_perm(D45B, D67B, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[16] = D1;
            *(vec_t *)&dst[32] = D2;
            *(vec_t *)&dst[48] = D3;
            *(vec_t *)&dst[64] = D4;
            *(vec_t *)&dst[80] = D5;
            *(vec_t *)&dst[96] = D6;
            *(vec_t *)&dst[112] = D7;
        }
        for (i = k16; i < k8; i += 8) {
            kcell = i >> 1; // 0, 1, 2, 3
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t j_hiflag = (j & 4) >> 2;
            koff = i & 1;
            noff = j & 3;
            short *dst = &bp[8 * (columns_done * krows + kcell * 2 + j_hiflag)];

            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29,
                    30, 31};
            vec_t swizL
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizR = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};

            V0 = *(vec_t *)&b[ldb * (j + 0) + i];
            V1 = *(vec_t *)&b[ldb * (j + 1) + i];
            V2 = *(vec_t *)&b[ldb * (j + 2) + i];
            V3 = *(vec_t *)&b[ldb * (j + 3) + i];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[16] = D1;
            *(vec_t *)&dst[32] = D2;
            *(vec_t *)&dst[48] = D3;
        }
    }

    for (j = n8; j < n4; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (j = n4; j < n_cap; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (j < n)
                bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < n4; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (i < k)
                bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (j = n4; j < n_cap; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (j < n && i < k)
                bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }
    return 0;
}

template <typename VecType>
inline int pack_T16_8bit_V2(dim_t K_dim, dim_t M_dim, const int8_t *A,
        dim_t lda, int8_t *packA, int *row_sum) {

    vec_t mask = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

    while (M_dim >= 16) {

        const int8_t *a = A;

        VecType V0, V1, V2, V3;
        VecType D01A, D01B, D23A, D23B;
        VecType D0, D1, D2, D3;

        vec_t swizA = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
        vec_t swizB = {
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
        vec_t swizL = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
        vec_t swizR = {
                8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31};

        __vector signed int vsum = {0};
        __vector signed int vsum1 = {0};
        __vector signed int vsum2 = {0};
        __vector signed int vsum3 = {0};

        size_t y = K_dim;
        while (y >= 4) {
            V0 = *(VecType *)&a[lda * 0];
            V1 = *(VecType *)&a[lda * 1];
            V2 = *(VecType *)&a[lda * 2];
            V3 = *(VecType *)&a[lda * 3];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(VecType *)&packA[0] = D0;
            *(VecType *)&packA[16] = D1;
            *(VecType *)&packA[32] = D2;
            *(VecType *)&packA[48] = D3;

            vsum = vec_sum4s(D0, vsum);
            vsum1 = vec_sum4s(D1, vsum1);
            vsum2 = vec_sum4s(D2, vsum2);
            vsum3 = vec_sum4s(D3, vsum3);

            packA += 64;
            y -= 4;
            a += lda * 4;
        }
        if (y >= 1 && y <= 3) {
            V0 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            V1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            V2 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            V3 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            V0 = *(VecType *)&a[lda * 0];
            if (y == 2) { V1 = *(VecType *)&a[lda * 1]; }
            if (y == 3) {
                V1 = *(VecType *)&a[lda * 1];
                V2 = *(VecType *)&a[lda * 2];
            }

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(VecType *)&packA[0] = D0;
            *(VecType *)&packA[16] = D1;
            *(VecType *)&packA[32] = D2;
            *(VecType *)&packA[48] = D3;

            vsum = vec_sum4s(D0, vsum);
            vsum1 = vec_sum4s(D1, vsum1);
            vsum2 = vec_sum4s(D2, vsum2);
            vsum3 = vec_sum4s(D3, vsum3);
            packA += 64;
            y -= 4;
            a += lda * 4;
        }

        row_sum[0] = vsum[0];
        row_sum[1] = vsum[1];
        row_sum[2] = vsum[2];
        row_sum[3] = vsum[3];
        row_sum[4] = vsum1[0];
        row_sum[5] = vsum1[1];
        row_sum[6] = vsum1[2];
        row_sum[7] = vsum1[3];

        row_sum[8] = vsum2[0];
        row_sum[9] = vsum2[1];
        row_sum[10] = vsum2[2];
        row_sum[11] = vsum2[3];
        row_sum[12] = vsum3[0];
        row_sum[13] = vsum3[1];
        row_sum[14] = vsum3[2];
        row_sum[15] = vsum3[3];

        row_sum += 16;
        A += 16;
        M_dim -= 16;
    }
    if (M_dim > 12 && M_dim < 16) {
        const int8_t *a = A;
        size_t y = K_dim;
        size_t tail_M = M_dim - 12;

        __vector signed int vsum = {0};
        __vector signed int vsum1 = {0};
        __vector signed int vsum2 = {0};
        __vector signed int vsum3 = {0};

        while (y >= 4) {

            int b1 = *reinterpret_cast<const int *>(&a[0]);
            int b2 = *reinterpret_cast<const int *>(&a[lda * 1]);
            int b3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int b4 = *reinterpret_cast<const int *>(&a[lda * 3]);
            __vector int vb = {b1, b2, b3, b4};
            VecType vx = vec_perm(reinterpret_cast<VecType>(vb),
                    reinterpret_cast<VecType>(vb), mask);
            vsum = vec_sum4s(vx, vsum);
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx);
            packA += 16;

            b1 = *reinterpret_cast<const int *>(&a[4]);
            b2 = *reinterpret_cast<const int *>(&a[lda * 1 + 4]);
            b3 = *reinterpret_cast<const int *>(&a[lda * 2 + 4]);
            b4 = *reinterpret_cast<const int *>(&a[lda * 3 + 4]);
            __vector int vb1 = {b1, b2, b3, b4};
            vx = vec_perm(reinterpret_cast<VecType>(vb1),
                    reinterpret_cast<VecType>(vb1), mask);
            vsum1 = vec_sum4s(vx, vsum1);
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx);
            packA += 16;

            b1 = *reinterpret_cast<const int *>(&a[8]);
            b2 = *reinterpret_cast<const int *>(&a[lda * 1 + 8]);
            b3 = *reinterpret_cast<const int *>(&a[lda * 2 + 8]);
            b4 = *reinterpret_cast<const int *>(&a[lda * 3 + 8]);
            __vector int vb2 = {b1, b2, b3, b4};
            vx = vec_perm(reinterpret_cast<VecType>(vb2),
                    reinterpret_cast<VecType>(vb2), mask);
            vsum2 = vec_sum4s(vx, vsum2);
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx);
            packA += 16;

            if (tail_M >= 1) {
                VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
                va1[0] = a[12];
                va1[1] = a[lda * 1 + 12];
                va1[2] = a[lda * 2 + 12];
                va1[3] = a[lda * 3 + 12];

                if (tail_M == 2) {
                    va1[4] = a[13];
                    va1[5] = a[lda * 1 + 13];
                    va1[6] = a[lda * 2 + 13];
                    va1[7] = a[lda * 3 + 13];
                }
                if (tail_M == 3) {
                    va1[4] = a[13];
                    va1[5] = a[lda * 1 + 13];
                    va1[6] = a[lda * 2 + 13];
                    va1[7] = a[lda * 3 + 13];

                    va1[8] = a[14];
                    va1[9] = a[lda * 1 + 14];
                    va1[10] = a[lda * 2 + 14];
                    va1[11] = a[lda * 3 + 14];
                }
                *reinterpret_cast<VecType *>(&packA[0])
                        = reinterpret_cast<VecType>(va1);
                vsum3 = vec_sum4s(va1, vsum3);
                packA += 16;
            }
            y -= 4;
            a += lda * 4;
        }
        if (y >= 1 && y <= 3) {
            VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType va2 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType va3 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            va1[0] = a[0];
            va1[4] = a[1];
            va1[8] = a[2];
            va1[12] = a[3];

            va2[0] = a[4];
            va2[4] = a[5];
            va2[8] = a[6];
            va2[12] = a[7];

            va3[0] = a[8];
            va3[4] = a[9];
            va3[8] = a[10];
            va3[12] = a[11];

            if (y == 2) {
                va1[1] = a[lda];
                va1[5] = a[lda + 1];
                va1[9] = a[lda + 2];
                va1[13] = a[lda + 3];

                va2[1] = a[lda + 4];
                va2[5] = a[lda + 5];
                va2[9] = a[lda + 6];
                va2[13] = a[lda + 7];

                va3[1] = a[lda + 8];
                va3[5] = a[lda + 9];
                va3[9] = a[lda + 10];
                va3[13] = a[lda + 11];
            }
            if (y == 3) {
                va1[1] = a[lda];
                va1[5] = a[lda + 1];
                va1[9] = a[lda + 2];
                va1[13] = a[lda + 3];

                va2[1] = a[lda + 4];
                va2[5] = a[lda + 5];
                va2[9] = a[lda + 6];
                va2[13] = a[lda + 7];

                va3[1] = a[lda + 8];
                va3[5] = a[lda + 9];
                va3[9] = a[lda + 10];
                va3[13] = a[lda + 11];

                va1[2] = a[lda * 2];
                va1[6] = a[lda * 2 + 1];
                va1[10] = a[lda * 2 + 2];
                va1[14] = a[lda * 2 + 3];

                va2[2] = a[lda * 2 + 4];
                va2[6] = a[lda * 2 + 5];
                va2[10] = a[lda * 2 + 6];
                va2[14] = a[lda * 2 + 7];

                va3[2] = a[lda * 2 + 8];
                va3[6] = a[lda * 2 + 9];
                va3[10] = a[lda * 2 + 10];
                va3[14] = a[lda * 2 + 11];
            }
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(va1);
            vsum = vec_sum4s(va1, vsum);
            vsum1 = vec_sum4s(va2, vsum1);
            vsum2 = vec_sum4s(va3, vsum2);
            *reinterpret_cast<VecType *>(&packA[16])
                    = reinterpret_cast<VecType>(va2);
            *reinterpret_cast<VecType *>(&packA[32])
                    = reinterpret_cast<VecType>(va3);
            packA += 48;
            a += 12;

            if (tail_M > 0) {
                VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

                if (tail_M == 1) {
                    va1[0] = a[0];

                    if (y == 2) { va1[1] = a[lda]; }
                    if (y == 3) {
                        va1[1] = a[lda];
                        va1[2] = a[lda * 2];
                    }
                }
                if (tail_M == 2) {
                    va1[0] = a[0];
                    va1[4] = a[1];
                    if (y == 2) {
                        va1[1] = a[lda];
                        va1[5] = a[lda + 1];
                    }
                    if (y == 3) {
                        va1[1] = a[lda];
                        va1[5] = a[lda + 1];

                        va1[2] = a[lda * 2];
                        va1[6] = a[lda * 2 + 1];
                    }
                }
                if (tail_M == 3) {
                    va1[0] = a[0];
                    va1[4] = a[1];
                    va1[8] = a[2];
                    if (y == 2) {
                        va1[1] = a[lda];
                        va1[5] = a[lda + 1];
                        va1[9] = a[lda + 2];
                    }
                    if (y == 3) {
                        va1[1] = a[lda];
                        va1[5] = a[lda + 1];
                        va1[9] = a[lda + 2];

                        va1[2] = a[lda * 2];
                        va1[6] = a[lda * 2 + 1];
                        va1[10] = a[lda * 2 + 2];
                    }
                }
                *reinterpret_cast<VecType *>(&packA[0])
                        = reinterpret_cast<VecType>(va1);
                vsum3 = vec_sum4s(va1, vsum3);
                packA += 16;
            }
        }
        row_sum[0] = vsum[0];
        row_sum[1] = vsum[1];
        row_sum[2] = vsum[2];
        row_sum[3] = vsum[3];
        row_sum[4] = vsum1[0];
        row_sum[5] = vsum1[1];
        row_sum[6] = vsum1[2];
        row_sum[7] = vsum1[3];

        row_sum[8] = vsum2[0];
        row_sum[9] = vsum2[1];
        row_sum[10] = vsum2[2];
        row_sum[11] = vsum2[3];

        if (tail_M == 1) { row_sum[12] = vsum3[0]; }
        if (tail_M == 2) {
            row_sum[12] = vsum3[0];
            row_sum[13] = vsum3[1];
        }
        if (tail_M == 3) {
            row_sum[12] = vsum3[0];
            row_sum[13] = vsum3[1];
            row_sum[14] = vsum3[2];
        }
        M_dim = 0;
    }

    while (M_dim >= 8) {
        const int8_t *a = A;
        __vector signed int vsum = {0};
        __vector signed int vsum1 = {0};
        size_t y = K_dim;
        while (y >= 8) {
            VecType V0, V1, V2, V3;
            VecType D01A, D01B, D23A, D23B;
            VecType D0, D1, D2, D3;
            vec_t swizA
                    = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
            vec_t swizB = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                    15, 31};
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            *(signed long long *)&V0[0] = *(signed long long *)&a[0];
            *(signed long long *)&V1[0] = *(signed long long *)&a[lda * 1];
            *(signed long long *)&V2[0] = *(signed long long *)&a[lda * 2];
            *(signed long long *)&V3[0] = *(signed long long *)&a[lda * 3];
            *(signed long long *)&V0[8] = *(signed long long *)&a[lda * 4];
            *(signed long long *)&V1[8] = *(signed long long *)&a[lda * 5];
            *(signed long long *)&V2[8] = *(signed long long *)&a[lda * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&a[lda * 7];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(VecType *)&packA[0] = D0;
            *(VecType *)&packA[16] = D1;
            *(VecType *)&packA[32] = D2;
            *(VecType *)&packA[48] = D3;

            vsum = vec_sum4s(D0, vsum);
            vsum = vec_sum4s(D2, vsum);
            vsum1 = vec_sum4s(D1, vsum1);
            vsum1 = vec_sum4s(D3, vsum1);
            packA += 64;
            y -= 8;
            a += lda * 8;
        }
        if (y >= 4) {
            int b1 = *reinterpret_cast<const int *>(&a[0]);
            int b2 = *reinterpret_cast<const int *>(&a[lda]);
            int b3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int b4 = *reinterpret_cast<const int *>(&a[lda * 3]);
            __vector int vb = {b1, b2, b3, b4};
            VecType vx = vec_perm(reinterpret_cast<VecType>(vb),
                    reinterpret_cast<VecType>(vb), mask);
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx);
            packA += 16;

            vsum = vec_sum4s(vx, vsum);

            b1 = *reinterpret_cast<const int *>(&a[4]);
            b2 = *reinterpret_cast<const int *>(&a[lda + 4]);
            b3 = *reinterpret_cast<const int *>(&a[lda * 2 + 4]);
            b4 = *reinterpret_cast<const int *>(&a[lda * 3 + 4]);

            __vector int vb1 = {b1, b2, b3, b4};
            VecType vx1 = vec_perm(reinterpret_cast<VecType>(vb1),
                    reinterpret_cast<VecType>(vb1), mask);
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx1);
            packA += 16;
            vsum1 = vec_sum4s(vx1, vsum1);
            y -= 4;
            a += lda * 4;
        }

        if (y >= 1 && y <= 3) {
            VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType va2 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            va1[0] = a[0];
            va1[4] = a[1];
            va1[8] = a[2];
            va1[12] = a[3];

            va2[0] = a[4];
            va2[4] = a[5];
            va2[8] = a[6];
            va2[12] = a[7];
            if (y == 2) {
                va1[1] = a[lda];
                va1[5] = a[lda + 1];
                va1[9] = a[lda + 2];
                va1[13] = a[lda + 3];

                va2[1] = a[lda + 4];
                va2[5] = a[lda + 5];
                va2[9] = a[lda + 6];
                va2[13] = a[lda + 7];
            }
            if (y == 3) {
                va1[1] = a[lda];
                va1[5] = a[lda + 1];
                va1[9] = a[lda + 2];
                va1[13] = a[lda + 3];

                va2[1] = a[lda + 4];
                va2[5] = a[lda + 5];
                va2[9] = a[lda + 6];
                va2[13] = a[lda + 7];

                va1[2] = a[lda * 2];
                va1[6] = a[lda * 2 + 1];
                va1[10] = a[lda * 2 + 2];
                va1[14] = a[lda * 2 + 3];

                va2[2] = a[lda * 2 + 4];
                va2[6] = a[lda * 2 + 5];
                va2[10] = a[lda * 2 + 6];
                va2[14] = a[lda * 2 + 7];
            }
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(va1);
            *reinterpret_cast<VecType *>(&packA[16])
                    = reinterpret_cast<VecType>(va2);
            packA += 32;
            vsum = vec_sum4s(va1, vsum);
            vsum1 = vec_sum4s(va2, vsum1);
        }
        row_sum[0] = vsum[0];
        row_sum[1] = vsum[1];
        row_sum[2] = vsum[2];
        row_sum[3] = vsum[3];

        row_sum[4] = vsum1[0];
        row_sum[5] = vsum1[1];
        row_sum[6] = vsum1[2];
        row_sum[7] = vsum1[3];
        row_sum += 8;
        A += 8;
        M_dim -= 8;
    }

    if (M_dim < 8 && M_dim >= 4) {
        const int8_t *a = A;
        __vector signed int vsum = {0};
        __vector signed int vsum1 = {0};
        size_t y = K_dim;
        size_t tail_M = M_dim - 4;
        while (y >= 4) {
            int b1 = *reinterpret_cast<const int *>(&a[0]);
            int b2 = *reinterpret_cast<const int *>(&a[lda]);
            int b3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int b4 = *reinterpret_cast<const int *>(&a[lda * 3]);
            __vector int vb = {b1, b2, b3, b4};
            VecType vx = vec_perm(reinterpret_cast<VecType>(vb),
                    reinterpret_cast<VecType>(vb), mask);
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx);
            packA += 16;
            vsum = vec_sum4s(vx, vsum);

            if (tail_M >= 1) {
                VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
                va1[0] = a[4];
                va1[1] = a[lda + 4];
                va1[2] = a[lda * 2 + 4];
                va1[3] = a[lda * 3 + 4];
                if (tail_M == 2) {
                    va1[4] = a[5];
                    va1[5] = a[lda + 5];
                    va1[6] = a[lda * 2 + 5];
                    va1[7] = a[lda * 3 + 5];
                }
                if (tail_M == 3) {
                    va1[4] = a[5];
                    va1[5] = a[lda + 5];
                    va1[6] = a[lda * 2 + 5];
                    va1[7] = a[lda * 3 + 5];
                    va1[8] = a[6];
                    va1[9] = a[lda + 6];
                    va1[10] = a[lda * 2 + 6];
                    va1[11] = a[lda * 3 + 6];
                }
                *reinterpret_cast<VecType *>(&packA[0])
                        = reinterpret_cast<VecType>(va1);
                packA += 16;
                vsum1 = vec_sum4s(va1, vsum1);
            }
            a += lda * 4;
            y -= 4;
        }

        if (y >= 1 && y <= 3) {

            int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
            a1 = *reinterpret_cast<const int *>(&a[0]);
            if (y == 2) { a2 = *reinterpret_cast<const int *>(&a[lda]); }
            if (y == 3) {
                a2 = *reinterpret_cast<const int *>(&a[lda]);
                a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            }
            __vector int vb = {a1, a2, a3, a4};
            VecType vx = vec_perm(reinterpret_cast<VecType>(vb),
                    reinterpret_cast<VecType>(vb), mask);

            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(vx);
            packA += 16;
            vsum = vec_sum4s(vx, vsum);

            if (tail_M >= 1) {
                VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
                if (y == 1) {
                    va1[0] = a[4];
                    if (tail_M == 2) { va1[4] = a[5]; }
                    if (tail_M == 3) {
                        va1[4] = a[5];
                        va1[8] = a[6];
                    }
                }
                if (y == 2) {
                    va1[0] = a[4];
                    va1[1] = a[lda + 4];
                    if (tail_M == 2) {
                        va1[4] = a[5];
                        va1[5] = a[lda + 5];
                    }
                    if (tail_M == 3) {
                        va1[4] = a[5];
                        va1[5] = a[lda + 5];
                        va1[8] = a[6];
                        va1[9] = a[lda + 6];
                    }
                }
                if (y == 3) {
                    va1[0] = a[4];
                    va1[1] = a[lda + 4];
                    va1[2] = a[lda * 2 + 4];
                    if (tail_M == 2) {
                        va1[4] = a[5];
                        va1[5] = a[lda + 5];
                        va1[6] = a[lda * 2 + 5];
                    }
                    if (tail_M == 3) {
                        va1[4] = a[5];
                        va1[5] = a[lda + 5];
                        va1[6] = a[lda * 2 + 5];
                        va1[8] = a[6];
                        va1[9] = a[lda + 6];
                        va1[10] = a[lda * 2 + 6];
                    }
                }
                *reinterpret_cast<VecType *>(&packA[0])
                        = reinterpret_cast<VecType>(va1);
                packA += 4;
                vsum1 = vec_sum4s(va1, vsum1);
            }
        }
        row_sum[0] = vsum[0];
        row_sum[1] = vsum[1];
        row_sum[2] = vsum[2];
        row_sum[3] = vsum[3];
        row_sum += 4;
        if (tail_M > 0) {
            row_sum[0] = vsum1[0];
            row_sum[1] = vsum1[1];
            row_sum[2] = vsum1[2];
            row_sum[3] = vsum1[3];
            row_sum += 4;
        }
    }
    if (M_dim >= 1 && M_dim <= 3) {
        const int8_t *a = A;
        __vector signed int vsum = {0};
        size_t y = K_dim;
        while (y >= 4) {
            VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            va1[0] = a[0];
            va1[1] = a[lda];
            va1[2] = a[lda * 2];
            va1[3] = a[lda * 3];
            if (M_dim == 2) {
                va1[4] = a[1];
                va1[5] = a[lda + 1];
                va1[6] = a[lda * 2 + 1];
                va1[7] = a[lda * 3 + 1];
            }
            if (M_dim == 3) {
                va1[4] = a[1];
                va1[5] = a[lda + 1];
                va1[6] = a[lda * 2 + 1];
                va1[7] = a[lda * 3 + 1];
                va1[8] = a[2];
                va1[9] = a[lda + 2];
                va1[10] = a[lda * 2 + 2];
                va1[11] = a[lda * 3 + 2];
            }
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(va1);
            vsum = vec_sum4s(va1, vsum);
            packA += 16;
            a += lda * 4;
            y -= 4;
        }
        if (y >= 1 && y <= 3) {
            VecType va1 = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            if (y == 1) {
                va1[0] = a[0];
                if (M_dim == 2) { va1[4] = a[1]; }
                if (M_dim == 3) {
                    va1[4] = a[1];
                    va1[8] = a[2];
                }
            }
            if (y == 2) {
                va1[0] = a[0];
                va1[1] = a[lda];
                if (M_dim == 2) {
                    va1[4] = a[1];
                    va1[5] = a[lda + 1];
                }
                if (M_dim == 3) {
                    va1[4] = a[1];
                    va1[5] = a[lda + 1];
                    va1[8] = a[2];
                    va1[9] = a[lda + 2];
                }
            }
            if (y == 3) {
                va1[0] = a[0];
                va1[1] = a[lda];
                va1[2] = a[lda * 2];
                if (M_dim == 2) {
                    va1[4] = a[1];
                    va1[5] = a[lda + 1];
                    va1[6] = a[lda * 2 + 1];
                }
                if (M_dim == 3) {
                    va1[4] = a[1];
                    va1[5] = a[lda + 1];
                    va1[6] = a[lda * 2 + 1];
                    va1[8] = a[2];
                    va1[9] = a[lda + 2];
                    va1[10] = a[lda * 2 + 2];
                }
            }
            *reinterpret_cast<VecType *>(&packA[0])
                    = reinterpret_cast<VecType>(va1);
            vsum = vec_sum4s(va1, vsum);
            packA += 16;
        }
        row_sum[0] = vsum[0];
        row_sum[1] = vsum[1];
        row_sum[2] = vsum[2];
        row_sum[3] = vsum[3];
        row_sum += 4;
    }
    return 0;
}

inline int pack_T16_8bit(
        dim_t k, dim_t m, const int8_t *a, dim_t lda, int8_t *ap) {
    int32_t i, j;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, m16, k4;
    m16 = (m >> 4) << 4;
    k4 = (k >> 2) << 2;
    krows = (k + 3) >> 2;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;

    // MAIN BLOCK
    for (i = 0; i < k4; i += 4) {
        for (j = 0; j < m16; j += 16) {
            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
            vec_t swizB = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                    15, 31};
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};
            int8_t *dest;

            V0 = *(vec_t *)&a[lda * (i + 0) + j];
            V1 = *(vec_t *)&a[lda * (i + 1) + j];
            V2 = *(vec_t *)&a[lda * (i + 2) + j];
            V3 = *(vec_t *)&a[lda * (i + 3) + j];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            dest = &ap[16 * (((j >> 4) * block4) + i)];

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (i = 0; i < k4; ++i) {
        for (j = m16; j < m_cap; ++j) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (j < m)
                ap[16 * cell + 4 * moff + koff] = a[lda * i + j];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k4; i < k_cap; ++i) {
        for (j = 0; j < m16; ++j) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (i < k)
                ap[16 * cell + 4 * moff + koff] = a[lda * i + j];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (i = k4; i < k_cap; ++i) {
        for (j = m16; j < m_cap; ++j) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (i < k && j < m)
                ap[16 * cell + 4 * moff + koff] = a[lda * i + j];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    return 0;
}

template <typename VecType>
int pack_N8_8bit_V2_lxvp(
        dim_t K_dim, dim_t N_dim, const uint8_t *B, dim_t ldb, uint8_t *Bp) {

    while (N_dim >= 8) {
        uint8_t *b = const_cast<uint8_t *>(B);
        size_t y = K_dim;
        while (y >= 32) {
            __vector_pair row1, row2, row3, row4, row5, row6, row7, row8;
            VecType r1[2], r2[2], r3[2], r4[2], r5[2], r6[2], r7[2], r8[2];
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            VecType swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            VecType swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

            row1 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[0]));
            row2 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb]));
            row3 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 2]));
            row4 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 3]));
            row5 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 4]));
            row6 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
            row7 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 6]));
            row8 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 7]));

            __builtin_vsx_disassemble_pair(r1, &row1);
            __builtin_vsx_disassemble_pair(r2, &row2);
            __builtin_vsx_disassemble_pair(r3, &row3);
            __builtin_vsx_disassemble_pair(r4, &row4);
            __builtin_vsx_disassemble_pair(r5, &row5);
            __builtin_vsx_disassemble_pair(r6, &row6);
            __builtin_vsx_disassemble_pair(r7, &row7);
            __builtin_vsx_disassemble_pair(r8, &row8);

            // First 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[0] = D0;
            *(VecType *)&Bp[32] = D1;
            *(VecType *)&Bp[64] = D2;
            *(VecType *)&Bp[96] = D3;

            // Next (ldb * 4) 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r5[0]),
                            reinterpret_cast<__vector int>(r6[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r7[0]),
                            reinterpret_cast<__vector int>(r8[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r5[0]),
                            reinterpret_cast<__vector int>(r6[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r7[0]),
                            reinterpret_cast<__vector int>(r8[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[16] = D0;
            *(VecType *)&Bp[48] = D1;
            *(VecType *)&Bp[80] = D2;
            *(VecType *)&Bp[112] = D3;

            // First 4 Rows and Second 16 columns

            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[128] = D0;
            *(VecType *)&Bp[160] = D1;
            *(VecType *)&Bp[192] = D2;
            *(VecType *)&Bp[224] = D3;

            // Next (ldb * 4) 4 Rows and Second 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r5[1]),
                            reinterpret_cast<__vector int>(r6[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r7[1]),
                            reinterpret_cast<__vector int>(r8[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r5[1]),
                            reinterpret_cast<__vector int>(r6[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r7[1]),
                            reinterpret_cast<__vector int>(r8[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[144] = D0;
            *(VecType *)&Bp[176] = D1;
            *(VecType *)&Bp[208] = D2;
            *(VecType *)&Bp[240] = D3;

            y -= 32;
            b += 32;
            Bp += 8 * 32;
        }
        while (y >= 16) {
            // First 4th row and 16 Columns
            VecType b1 = *reinterpret_cast<const VecType *>(&b[0]);
            VecType b2 = *reinterpret_cast<const VecType *>(&b[ldb]);
            VecType b3 = *reinterpret_cast<const VecType *>(&b[ldb * 2]);
            VecType b4 = *reinterpret_cast<const VecType *>(&b[ldb * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[0]) = vx_row1;
            *reinterpret_cast<VecType *>(&Bp[32]) = vx_row3;
            *reinterpret_cast<VecType *>(&Bp[64]) = vx_row5;
            *reinterpret_cast<VecType *>(&Bp[96]) = vx_row7;

            // Second 4th Row and 16 Columns
            b1 = *reinterpret_cast<const VecType *>(&b[ldb * 4]);
            b2 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
            b3 = *reinterpret_cast<const VecType *>(&b[ldb * 6]);
            b4 = *reinterpret_cast<const VecType *>(&b[ldb * 7]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row2 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row4 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row6 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row8 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[16]) = vx_row2;
            *reinterpret_cast<VecType *>(&Bp[48]) = vx_row4;
            *reinterpret_cast<VecType *>(&Bp[80]) = vx_row6;
            *reinterpret_cast<VecType *>(&Bp[112]) = vx_row8;

            b += 16;
            Bp += 128;
            y -= 16;
        }
        while (y >= 8) {
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            VecType swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            VecType swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 3];
            *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 4];
            *(signed long long *)&V2[8] = *(signed long long *)&b[ldb * 5];
            *(signed long long *)&V3[0] = *(signed long long *)&b[ldb * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&b[ldb * 7];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(VecType *)&Bp[0] = D0;
            *(VecType *)&Bp[16] = D1;
            *(VecType *)&Bp[32] = D2;
            *(VecType *)&Bp[48] = D3;

            Bp += 64;
            b += 8;
            y -= 8;
        }
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            int a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int a4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_row1;

            a1 = *reinterpret_cast<const int *>(&b[ldb * 4]);
            a2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
            a3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
            a4 = *reinterpret_cast<const int *>(&b[ldb * 7]);
            __vector int vec_a1 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a1);
            *reinterpret_cast<VecType *>(&Bp[16]) = vec_row1;
            Bp += 32;
            y -= 4;
            b += 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];

            vec_tail2[0] = b[ldb * 4];
            vec_tail2[4] = b[ldb * 5];
            vec_tail2[8] = b[ldb * 6];
            vec_tail2[12] = b[ldb * 7];

            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];

                vec_tail2[2] = b[ldb * 4 + 2];
                vec_tail2[6] = b[ldb * 5 + 2];
                vec_tail2[10] = b[ldb * 6 + 2];
                vec_tail2[14] = b[ldb * 7 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];
            }
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
            *reinterpret_cast<VecType *>(&Bp[16]) = vec_tail2;
            Bp += 32;
        }
        N_dim -= 8;
        B += 8 * ldb;
    }
    if (N_dim >= 4 && N_dim < 8) {
        uint8_t *b = const_cast<uint8_t *>(B);
        size_t y = K_dim;
        int tail_N = N_dim - 4;
        while (y >= 32) {

            __vector_pair row1, row2, row3, row4, row5, row6, row7, row8;
            VecType r1[2] = {0}, r2[2] = {0}, r3[2] = {0}, r4[2] = {0},
                    r5[2] = {0}, r6[2] = {0}, r7[2] = {0}, r8[2] = {0};
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            VecType swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            VecType swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
            row1 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[0]));
            row2 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb]));
            row3 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 2]));
            row4 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 3]));

            __builtin_vsx_disassemble_pair(r1, &row1);
            __builtin_vsx_disassemble_pair(r2, &row2);
            __builtin_vsx_disassemble_pair(r3, &row3);
            __builtin_vsx_disassemble_pair(r4, &row4);

            // First 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[0] = D0;
            if (tail_N == 0) {
                *(VecType *)&Bp[16] = D1;
                *(VecType *)&Bp[32] = D2;
                *(VecType *)&Bp[48] = D3;
            }
            if (tail_N >= 1) {
                *(VecType *)&Bp[32] = D1;
                *(VecType *)&Bp[64] = D2;
                *(VecType *)&Bp[96] = D3;
            }

            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            if (tail_N == 0) {
                *(VecType *)&Bp[64] = D0;
                *(VecType *)&Bp[80] = D1;
                *(VecType *)&Bp[96] = D2;
                *(VecType *)&Bp[112] = D3;
            }
            if (tail_N >= 1) {
                *(VecType *)&Bp[128] = D0;
                *(VecType *)&Bp[160] = D1;
                *(VecType *)&Bp[192] = D2;
                *(VecType *)&Bp[224] = D3;
            }
            if (tail_N >= 1) {
                row5 = __builtin_vsx_lxvp(
                        0, reinterpret_cast<__vector_pair *>(&b[ldb * 4]));
                __builtin_vsx_disassemble_pair(r5, &row5);
                if (tail_N == 3) {
                    row6 = __builtin_vsx_lxvp(
                            0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
                    row7 = __builtin_vsx_lxvp(
                            0, reinterpret_cast<__vector_pair *>(&b[ldb * 6]));
                    __builtin_vsx_disassemble_pair(r6, &row6);
                    __builtin_vsx_disassemble_pair(r7, &row7);
                }
                if (tail_N == 2) {
                    row6 = __builtin_vsx_lxvp(
                            0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
                    __builtin_vsx_disassemble_pair(r6, &row6);
                }

                V0 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r5[0]),
                                reinterpret_cast<__vector int>(r6[0])));
                V1 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r7[0]),
                                reinterpret_cast<__vector int>(r8[0])));
                V2 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r5[0]),
                                reinterpret_cast<__vector int>(r6[0])));
                V3 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r7[0]),
                                reinterpret_cast<__vector int>(r8[0])));

                D0 = vec_xxpermdi(V0, V1, 0);
                D1 = vec_xxpermdi(V2, V3, 0);
                D2 = vec_xxpermdi(V0, V1, 3);
                D3 = vec_xxpermdi(V2, V3, 3);

                *(VecType *)&Bp[16] = D0;
                *(VecType *)&Bp[48] = D1;
                *(VecType *)&Bp[80] = D2;
                *(VecType *)&Bp[112] = D3;

                //Next 16 Columns
                V0 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r5[1]),
                                reinterpret_cast<__vector int>(r6[1])));
                V1 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r7[1]),
                                reinterpret_cast<__vector int>(r8[1])));
                V2 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r5[1]),
                                reinterpret_cast<__vector int>(r6[1])));
                V3 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r7[1]),
                                reinterpret_cast<__vector int>(r8[1])));

                D0 = vec_xxpermdi(V0, V1, 0);
                D1 = vec_xxpermdi(V2, V3, 0);
                D2 = vec_xxpermdi(V0, V1, 3);
                D3 = vec_xxpermdi(V2, V3, 3);

                *(VecType *)&Bp[144] = D0;
                *(VecType *)&Bp[176] = D1;
                *(VecType *)&Bp[208] = D2;
                *(VecType *)&Bp[240] = D3;
            }
            if (tail_N == 0) {
                Bp += 128;
            } else {
                Bp += 256;
            }
            b += 32;
            y -= 32;
        }

        while (y >= 16) {
            VecType b1 = *reinterpret_cast<const VecType *>(&b[0]);
            VecType b2 = *reinterpret_cast<const VecType *>(&b[ldb]);
            VecType b3 = *reinterpret_cast<const VecType *>(&b[ldb * 2]);
            VecType b4 = *reinterpret_cast<const VecType *>(&b[ldb * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[0]) = vx_row1;
            if (tail_N == 0) {
                *reinterpret_cast<VecType *>(&Bp[16]) = vx_row3;
                *reinterpret_cast<VecType *>(&Bp[32]) = vx_row5;
                *reinterpret_cast<VecType *>(&Bp[48]) = vx_row7;
            }
            if (tail_N >= 1) {
                *reinterpret_cast<VecType *>(&Bp[32]) = vx_row3;
                *reinterpret_cast<VecType *>(&Bp[64]) = vx_row5;
                *reinterpret_cast<VecType *>(&Bp[96]) = vx_row7;
            }

            if (tail_N >= 1) {
                VecType b5 = {0}, b6 = {0}, b7 = {0}, b8 = {0};
                b5 = *reinterpret_cast<const VecType *>(&b[ldb * 4]);

                if (tail_N == 3) {
                    b6 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
                    b7 = *reinterpret_cast<const VecType *>(&b[ldb * 6]);
                }
                if (tail_N == 2) {
                    b6 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
                }

                VecType vec_even12 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(b5),
                                reinterpret_cast<__vector int>(b6)));
                VecType vec_even34 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(b7),
                                reinterpret_cast<__vector int>(b8)));
                VecType vec_odd12 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(b5),
                                reinterpret_cast<__vector int>(b6)));
                VecType vec_odd34 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(b7),
                                reinterpret_cast<__vector int>(b8)));

                VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
                VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
                VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
                VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

                *reinterpret_cast<VecType *>(&Bp[16]) = vx_row1;
                *reinterpret_cast<VecType *>(&Bp[48]) = vx_row3;
                *reinterpret_cast<VecType *>(&Bp[80]) = vx_row5;
                *reinterpret_cast<VecType *>(&Bp[112]) = vx_row7;
            }
            b += 16;
            if (tail_N >= 1) {
                Bp += 16 * 8;
            } else {
                Bp += 16 * 4;
            }
            y -= 16;
        }

        while (y >= 8) {
            VecType V0 = {0}, V1 = {0}, V2 = {0}, V3 = {0};
            VecType D0, D1, D2, D3;

            VecType swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            VecType swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 3];

            D0 = vec_perm(V0, V1, swizA);
            D2 = vec_perm(V0, V1, swizB);

            *reinterpret_cast<VecType *>(&Bp[0]) = D0;
            if (tail_N == 0) {
                *reinterpret_cast<VecType *>(&Bp[16]) = D2;
            } else {
                *reinterpret_cast<VecType *>(&Bp[32]) = D2;
            }

            if (tail_N >= 1) {
                *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 4];
                if (tail_N == 3) {
                    *(signed long long *)&V2[8]
                            = *(signed long long *)&b[ldb * 5];
                    *(signed long long *)&V3[0]
                            = *(signed long long *)&b[ldb * 6];
                }
                if (tail_N == 2) {
                    *(signed long long *)&V2[8]
                            = *(signed long long *)&b[ldb * 5];
                }
                D1 = vec_perm(V2, V3, swizA);
                D3 = vec_perm(V2, V3, swizB);

                *(VecType *)&Bp[16] = D1;
                *(VecType *)&Bp[48] = D3;
            }
            b += 8;

            if (tail_N >= 1) {
                Bp += 8 * 8;
            } else {
                Bp += 4 * 8;
            }
            y -= 8;
        }

        while (y >= 4) {
            int b1 = *reinterpret_cast<int *>(&b[0]);
            int b2 = *reinterpret_cast<int *>(&b[ldb * 1]);
            int b3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int b4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {b1, b2, b3, b4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(VecType *)&Bp[0] = vec_row1;
            Bp += 16;

            if (tail_N >= 1) {
                b2 = 0;
                b3 = 0;
                b4 = 0;
                b1 = *reinterpret_cast<const int *>(&b[ldb * 4]);

                if (tail_N == 3) {
                    b2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
                    b3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
                }
                if (tail_N == 2) {
                    b2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
                }
                __vector int vec_a1 = {b1, b2, b3, b4};
                VecType vec_row2 = reinterpret_cast<VecType>(vec_a1);
                *(VecType *)&Bp[0] = vec_row2;
                Bp += 16;
            }
            y -= 4;
            b += 4;
        }

        if (y >= 1 && y <= 3) {
            VecType vec_tail1 = {0};
            VecType vec_tail2 = {0};
            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];
            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];
            }
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
            Bp += 16;

            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail2[0] = b[ldb * 4];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                    }
                    if (y == 2) { vec_tail2[1] = b[ldb * 4 + 1]; }
                }
                if (tail_N == 2) {
                    vec_tail2[0] = b[ldb * 4];
                    vec_tail2[4] = b[ldb * 5];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[6] = b[ldb * 5 + 2];
                    }
                    if (y == 2) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[5] = b[ldb * 5 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail2[0] = b[ldb * 4];
                    vec_tail2[4] = b[ldb * 5];
                    vec_tail2[8] = b[ldb * 6];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[6] = b[ldb * 5 + 2];
                        vec_tail2[9] = b[ldb * 6 + 1];
                        vec_tail2[10] = b[ldb * 6 + 2];
                    }
                    if (y == 2) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[9] = b[ldb * 6 + 1];
                    }
                }
                *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail2;
                Bp += 16;
            }
        }
    }
    if (N_dim <= 3 && N_dim >= 1) {

        const uint8_t *b = B;
        size_t y = K_dim;
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = 0, a3 = 0, a4 = 0;
            if (N_dim == 3) {
                a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
                a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            }
            if (N_dim == 2) {
                a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            }

            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(VecType *)&Bp[0] = vec_row1;

            Bp += 16;
            b += 4;
            y -= 4;
        }

        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            int tail_N = N_dim;

            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail1[0] = b[0];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];
                    }
                    if (y == 2) { vec_tail1[1] = b[1]; }
                }
                if (tail_N == 2) {
                    vec_tail1[0] = b[0];
                    vec_tail1[4] = b[ldb];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];

                        vec_tail1[5] = b[ldb * 1 + 1];
                        vec_tail1[6] = b[ldb * 1 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = b[1];
                        vec_tail1[5] = b[ldb * 1 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail1[0] = b[0];
                    vec_tail1[4] = b[ldb];
                    vec_tail1[8] = b[ldb * 2];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];

                        vec_tail1[5] = b[ldb + 1];
                        vec_tail1[6] = b[ldb + 2];

                        vec_tail1[9] = b[ldb * 2 + 1];
                        vec_tail1[10] = b[ldb * 2 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = b[1];

                        vec_tail1[5] = b[ldb + 1];
                        vec_tail1[9] = b[ldb * 2 + 1];
                    }
                }

                *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
                Bp += 16;
            }
        }
    }
    return 0;
}

template <typename VecType, typename BufType>
int pack_N8_8bit_V2_lxvp_signed(dim_t K_dim, dim_t N_dim, const BufType *B,
        dim_t ldb, uint8_t *Bp, bool is_signed) {
    const uint8_t BitFlipValue = (is_signed ? 0x80 : 0);
    VecType vmask = reinterpret_cast<VecType>(vec_splats(BitFlipValue));
    const int8_t Flip = (is_signed ? -128 : 0);

    typedef __vector unsigned char vec_t;

    while (N_dim >= 8) {
        BufType *b = const_cast<BufType *>(B);
        size_t y = K_dim;
        while (y >= 32) {
            __vector_pair row1, row2, row3, row4, row5, row6, row7, row8;
            VecType r1[2], r2[2], r3[2], r4[2], r5[2], r6[2], r7[2], r8[2];
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;

            row1 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[0]));
            row2 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb]));
            row3 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 2]));
            row4 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 3]));
            row5 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 4]));
            row6 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
            row7 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 6]));
            row8 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 7]));

            __builtin_vsx_disassemble_pair(r1, &row1);
            __builtin_vsx_disassemble_pair(r2, &row2);
            __builtin_vsx_disassemble_pair(r3, &row3);
            __builtin_vsx_disassemble_pair(r4, &row4);
            __builtin_vsx_disassemble_pair(r5, &row5);
            __builtin_vsx_disassemble_pair(r6, &row6);
            __builtin_vsx_disassemble_pair(r7, &row7);
            __builtin_vsx_disassemble_pair(r8, &row8);

            // First 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(vec_t *)&Bp[0] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            *(vec_t *)&Bp[32] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
            *(vec_t *)&Bp[64] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            *(vec_t *)&Bp[96] = reinterpret_cast<vec_t>(vec_add(D3, vmask));

            // Next (ldb * 4) 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r5[0]),
                            reinterpret_cast<__vector int>(r6[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r7[0]),
                            reinterpret_cast<__vector int>(r8[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r5[0]),
                            reinterpret_cast<__vector int>(r6[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r7[0]),
                            reinterpret_cast<__vector int>(r8[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(vec_t *)&Bp[16] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            *(vec_t *)&Bp[48] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
            *(vec_t *)&Bp[80] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            *(vec_t *)&Bp[112] = reinterpret_cast<vec_t>(vec_add(D3, vmask));

            // First 4 Rows and Second 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(vec_t *)&Bp[128] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            *(vec_t *)&Bp[160] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
            *(vec_t *)&Bp[192] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            *(vec_t *)&Bp[224] = reinterpret_cast<vec_t>(vec_add(D3, vmask));

            // Next (ldb * 4) 4 Rows and Second 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r5[1]),
                            reinterpret_cast<__vector int>(r6[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r7[1]),
                            reinterpret_cast<__vector int>(r8[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r5[1]),
                            reinterpret_cast<__vector int>(r6[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r7[1]),
                            reinterpret_cast<__vector int>(r8[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(vec_t *)&Bp[144] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            *(vec_t *)&Bp[176] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
            *(vec_t *)&Bp[208] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            *(vec_t *)&Bp[240] = reinterpret_cast<vec_t>(vec_add(D3, vmask));

            y -= 32;
            b += 32;
            Bp += 8 * 32;
        }
        while (y >= 16) {
            // First 4th row and 16 Columns
            VecType b1 = *reinterpret_cast<const VecType *>(&b[0]);
            VecType b2 = *reinterpret_cast<const VecType *>(&b[ldb]);
            VecType b3 = *reinterpret_cast<const VecType *>(&b[ldb * 2]);
            VecType b4 = *reinterpret_cast<const VecType *>(&b[ldb * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<vec_t *>(&Bp[0])
                    = reinterpret_cast<vec_t>(vec_add(vx_row1, vmask));
            *reinterpret_cast<vec_t *>(&Bp[32])
                    = reinterpret_cast<vec_t>(vec_add(vx_row3, vmask));
            *reinterpret_cast<vec_t *>(&Bp[64])
                    = reinterpret_cast<vec_t>(vec_add(vx_row5, vmask));
            *reinterpret_cast<vec_t *>(&Bp[96])
                    = reinterpret_cast<vec_t>(vec_add(vx_row7, vmask));

            // Second 4th Row and 16 Columns
            b1 = *reinterpret_cast<const VecType *>(&b[ldb * 4]);
            b2 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
            b3 = *reinterpret_cast<const VecType *>(&b[ldb * 6]);
            b4 = *reinterpret_cast<const VecType *>(&b[ldb * 7]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row2 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row4 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row6 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row8 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<vec_t *>(&Bp[16])
                    = reinterpret_cast<vec_t>(vec_add(vx_row2, vmask));
            *reinterpret_cast<vec_t *>(&Bp[48])
                    = reinterpret_cast<vec_t>(vec_add(vx_row4, vmask));
            *reinterpret_cast<vec_t *>(&Bp[80])
                    = reinterpret_cast<vec_t>(vec_add(vx_row6, vmask));
            *reinterpret_cast<vec_t *>(&Bp[112])
                    = reinterpret_cast<vec_t>(vec_add(vx_row8, vmask));

            b += 16;
            Bp += 128;
            y -= 16;
        }
        while (y >= 8) {
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            vec_t swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            vec_t swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 3];
            *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 4];
            *(signed long long *)&V2[8] = *(signed long long *)&b[ldb * 5];
            *(signed long long *)&V3[0] = *(signed long long *)&b[ldb * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&b[ldb * 7];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(vec_t *)&Bp[0] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            *(vec_t *)&Bp[16] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
            *(vec_t *)&Bp[32] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            *(vec_t *)&Bp[48] = reinterpret_cast<vec_t>(vec_add(D3, vmask));

            Bp += 64;
            b += 8;
            y -= 8;
        }
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            int a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int a4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<vec_t *>(&Bp[0])
                    = reinterpret_cast<vec_t>(vec_add(vec_row1, vmask));

            a1 = *reinterpret_cast<const int *>(&b[ldb * 4]);
            a2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
            a3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
            a4 = *reinterpret_cast<const int *>(&b[ldb * 7]);
            __vector int vec_a1 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a1);
            *reinterpret_cast<vec_t *>(&Bp[16])
                    = reinterpret_cast<vec_t>(vec_add(vec_row1, vmask));
            Bp += 32;
            y -= 4;
            b += 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1 = reinterpret_cast<VecType>(vec_splats(Flip));
            VecType vec_tail2 = reinterpret_cast<VecType>(vec_splats(Flip));
            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];

            vec_tail2[0] = b[ldb * 4];
            vec_tail2[4] = b[ldb * 5];
            vec_tail2[8] = b[ldb * 6];
            vec_tail2[12] = b[ldb * 7];

            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];

                vec_tail2[2] = b[ldb * 4 + 2];
                vec_tail2[6] = b[ldb * 5 + 2];
                vec_tail2[10] = b[ldb * 6 + 2];
                vec_tail2[14] = b[ldb * 7 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];
            }
            *reinterpret_cast<vec_t *>(&Bp[0])
                    = reinterpret_cast<vec_t>(vec_add(vec_tail1, vmask));
            *reinterpret_cast<vec_t *>(&Bp[16])
                    = reinterpret_cast<vec_t>(vec_add(vec_tail2, vmask));
            Bp += 32;
        }
        N_dim -= 8;
        B += 8 * ldb;
    }
    if (N_dim >= 4 && N_dim < 8) {
        BufType *b = const_cast<BufType *>(B);
        size_t y = K_dim;
        int tail_N = N_dim - 4;
        while (y >= 32) {

            __vector_pair row1, row2, row3, row4, row5, row6, row7;
            VecType r1[2] = {0}, r2[2] = {0}, r3[2] = {0}, r4[2] = {0},
                    r5[2] = {0}, r6[2] = {0}, r7[2] = {0}, r8[2] = {0};
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            row1 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[0]));
            row2 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb]));
            row3 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 2]));
            row4 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 3]));

            __builtin_vsx_disassemble_pair(r1, &row1);
            __builtin_vsx_disassemble_pair(r2, &row2);
            __builtin_vsx_disassemble_pair(r3, &row3);
            __builtin_vsx_disassemble_pair(r4, &row4);

            // First 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(vec_t *)&Bp[0] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            if (tail_N == 0) {
                *(vec_t *)&Bp[16] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[32] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
                *(vec_t *)&Bp[48] = reinterpret_cast<vec_t>(vec_add(D3, vmask));
            }
            if (tail_N >= 1) {
                *(vec_t *)&Bp[32] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[64] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
                *(vec_t *)&Bp[96] = reinterpret_cast<vec_t>(vec_add(D3, vmask));
            }

            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            if (tail_N == 0) {
                *(vec_t *)&Bp[64] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
                *(vec_t *)&Bp[80] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[96] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
                *(vec_t *)&Bp[112]
                        = reinterpret_cast<vec_t>(vec_add(D3, vmask));
            }
            if (tail_N >= 1) {
                *(vec_t *)&Bp[128]
                        = reinterpret_cast<vec_t>(vec_add(D0, vmask));
                *(vec_t *)&Bp[160]
                        = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[192]
                        = reinterpret_cast<vec_t>(vec_add(D2, vmask));
                *(vec_t *)&Bp[224]
                        = reinterpret_cast<vec_t>(vec_add(D3, vmask));
            }
            if (tail_N >= 1) {
                row5 = __builtin_vsx_lxvp(
                        0, reinterpret_cast<__vector_pair *>(&b[ldb * 4]));
                __builtin_vsx_disassemble_pair(r5, &row5);
                if (tail_N == 3) {
                    row6 = __builtin_vsx_lxvp(
                            0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
                    row7 = __builtin_vsx_lxvp(
                            0, reinterpret_cast<__vector_pair *>(&b[ldb * 6]));
                    __builtin_vsx_disassemble_pair(r6, &row6);
                    __builtin_vsx_disassemble_pair(r7, &row7);
                }
                if (tail_N == 2) {
                    row6 = __builtin_vsx_lxvp(
                            0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
                    __builtin_vsx_disassemble_pair(r6, &row6);
                }

                V0 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r5[0]),
                                reinterpret_cast<__vector int>(r6[0])));
                V1 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r7[0]),
                                reinterpret_cast<__vector int>(r8[0])));
                V2 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r5[0]),
                                reinterpret_cast<__vector int>(r6[0])));
                V3 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r7[0]),
                                reinterpret_cast<__vector int>(r8[0])));

                D0 = vec_xxpermdi(V0, V1, 0);
                D1 = vec_xxpermdi(V2, V3, 0);
                D2 = vec_xxpermdi(V0, V1, 3);
                D3 = vec_xxpermdi(V2, V3, 3);

                *(vec_t *)&Bp[16] = reinterpret_cast<vec_t>(vec_add(D0, vmask));
                *(vec_t *)&Bp[48] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[80] = reinterpret_cast<vec_t>(vec_add(D2, vmask));
                *(vec_t *)&Bp[112]
                        = reinterpret_cast<vec_t>(vec_add(D3, vmask));

                //Next 16 Columns
                V0 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r5[1]),
                                reinterpret_cast<__vector int>(r6[1])));
                V1 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(r7[1]),
                                reinterpret_cast<__vector int>(r8[1])));
                V2 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r5[1]),
                                reinterpret_cast<__vector int>(r6[1])));
                V3 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(r7[1]),
                                reinterpret_cast<__vector int>(r8[1])));

                D0 = vec_xxpermdi(V0, V1, 0);
                D1 = vec_xxpermdi(V2, V3, 0);
                D2 = vec_xxpermdi(V0, V1, 3);
                D3 = vec_xxpermdi(V2, V3, 3);

                *(vec_t *)&Bp[144]
                        = reinterpret_cast<vec_t>(vec_add(D0, vmask));
                *(vec_t *)&Bp[176]
                        = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[208]
                        = reinterpret_cast<vec_t>(vec_add(D2, vmask));
                *(vec_t *)&Bp[240]
                        = reinterpret_cast<vec_t>(vec_add(D3, vmask));
            }
            if (tail_N == 0) {
                Bp += 128;
            } else {
                Bp += 256;
            }
            b += 32;
            y -= 32;
        }

        while (y >= 16) {
            VecType b1 = *reinterpret_cast<const VecType *>(&b[0]);
            VecType b2 = *reinterpret_cast<const VecType *>(&b[ldb]);
            VecType b3 = *reinterpret_cast<const VecType *>(&b[ldb * 2]);
            VecType b4 = *reinterpret_cast<const VecType *>(&b[ldb * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<vec_t *>(&Bp[0])
                    = reinterpret_cast<vec_t>(vec_add(vx_row1, vmask));
            if (tail_N == 0) {
                *reinterpret_cast<vec_t *>(&Bp[16])
                        = reinterpret_cast<vec_t>(vec_add(vx_row3, vmask));
                *reinterpret_cast<vec_t *>(&Bp[32])
                        = reinterpret_cast<vec_t>(vec_add(vx_row5, vmask));
                *reinterpret_cast<vec_t *>(&Bp[48])
                        = reinterpret_cast<vec_t>(vec_add(vx_row7, vmask));
            }
            if (tail_N >= 1) {
                *reinterpret_cast<vec_t *>(&Bp[32])
                        = reinterpret_cast<vec_t>(vec_add(vx_row3, vmask));
                *reinterpret_cast<vec_t *>(&Bp[64])
                        = reinterpret_cast<vec_t>(vec_add(vx_row5, vmask));
                *reinterpret_cast<vec_t *>(&Bp[96])
                        = reinterpret_cast<vec_t>(vec_add(vx_row7, vmask));
            }

            if (tail_N >= 1) {
                VecType b5 = reinterpret_cast<VecType>(vec_splats(Flip));
                VecType b6 = reinterpret_cast<VecType>(vec_splats(Flip));
                VecType b7 = reinterpret_cast<VecType>(vec_splats(Flip));
                VecType b8 = reinterpret_cast<VecType>(vec_splats(Flip));
                b5 = *reinterpret_cast<const VecType *>(&b[ldb * 4]);

                if (tail_N == 3) {
                    b6 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
                    b7 = *reinterpret_cast<const VecType *>(&b[ldb * 6]);
                }
                if (tail_N == 2) {
                    b6 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
                }

                VecType vec_even12 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(b5),
                                reinterpret_cast<__vector int>(b6)));
                VecType vec_even34 = reinterpret_cast<VecType>(
                        vec_mergee(reinterpret_cast<__vector int>(b7),
                                reinterpret_cast<__vector int>(b8)));
                VecType vec_odd12 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(b5),
                                reinterpret_cast<__vector int>(b6)));
                VecType vec_odd34 = reinterpret_cast<VecType>(
                        vec_mergeo(reinterpret_cast<__vector int>(b7),
                                reinterpret_cast<__vector int>(b8)));

                VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
                VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
                VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
                VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

                *reinterpret_cast<vec_t *>(&Bp[16])
                        = reinterpret_cast<vec_t>(vec_add(vx_row1, vmask));
                *reinterpret_cast<vec_t *>(&Bp[48])
                        = reinterpret_cast<vec_t>(vec_add(vx_row3, vmask));
                *reinterpret_cast<vec_t *>(&Bp[80])
                        = reinterpret_cast<vec_t>(vec_add(vx_row5, vmask));
                *reinterpret_cast<vec_t *>(&Bp[112])
                        = reinterpret_cast<vec_t>(vec_add(vx_row7, vmask));
            }
            b += 16;
            if (tail_N >= 1) {
                Bp += 16 * 8;
            } else {
                Bp += 16 * 4;
            }
            y -= 16;
        }

        while (y >= 8) {
            VecType V0 = reinterpret_cast<VecType>(vec_splats(Flip));
            VecType V1 = reinterpret_cast<VecType>(vec_splats(Flip));
            VecType V2 = reinterpret_cast<VecType>(vec_splats(Flip));
            VecType V3 = reinterpret_cast<VecType>(vec_splats(Flip));
            VecType D0, D1, D2, D3;

            vec_t swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            vec_t swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 3];

            D0 = vec_perm(V0, V1, swizA);
            D2 = vec_perm(V0, V1, swizB);

            *reinterpret_cast<vec_t *>(&Bp[0])
                    = reinterpret_cast<vec_t>(vec_add(D0, vmask));
            if (tail_N == 0) {
                *reinterpret_cast<vec_t *>(&Bp[16])
                        = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            } else {
                *reinterpret_cast<vec_t *>(&Bp[32])
                        = reinterpret_cast<vec_t>(vec_add(D2, vmask));
            }

            if (tail_N >= 1) {
                *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 4];
                if (tail_N == 3) {
                    *(signed long long *)&V2[8]
                            = *(signed long long *)&b[ldb * 5];
                    *(signed long long *)&V3[0]
                            = *(signed long long *)&b[ldb * 6];
                }
                if (tail_N == 2) {
                    *(signed long long *)&V2[8]
                            = *(signed long long *)&b[ldb * 5];
                }
                D1 = vec_perm(V2, V3, swizA);
                D3 = vec_perm(V2, V3, swizB);

                *(vec_t *)&Bp[16] = reinterpret_cast<vec_t>(vec_add(D1, vmask));
                *(vec_t *)&Bp[48] = reinterpret_cast<vec_t>(vec_add(D3, vmask));
            }
            b += 8;

            if (tail_N >= 1) {
                Bp += 8 * 8;
            } else {
                Bp += 4 * 8;
            }
            y -= 8;
        }

        while (y >= 4) {
            int b1 = *reinterpret_cast<int *>(&b[0]);
            int b2 = *reinterpret_cast<int *>(&b[ldb * 1]);
            int b3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int b4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {b1, b2, b3, b4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(vec_t *)&Bp[0]
                    = reinterpret_cast<vec_t>(vec_add(vec_row1, vmask));
            Bp += 16;

            if (tail_N >= 1) {
                int value = (Flip & 0xFF) | ((Flip & 0xFF) << 8)
                        | ((Flip & 0xFF) << 16) | ((Flip & 0xFF) << 24);
                b2 = value;
                b3 = value;
                b4 = value;
                b1 = *reinterpret_cast<const int *>(&b[ldb * 4]);

                if (tail_N == 3) {
                    b2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
                    b3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
                }
                if (tail_N == 2) {
                    b2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
                }
                __vector int vec_a1 = {b1, b2, b3, b4};
                VecType vec_row2 = reinterpret_cast<VecType>(vec_a1);
                *(vec_t *)&Bp[0]
                        = reinterpret_cast<vec_t>(vec_add(vec_row2, vmask));
                Bp += 16;
            }
            y -= 4;
            b += 4;
        }

        if (y >= 1 && y <= 3) {
            VecType vec_tail1 = reinterpret_cast<VecType>(vec_splats(Flip));
            VecType vec_tail2 = reinterpret_cast<VecType>(vec_splats(Flip));
            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];
            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];
            }
            *reinterpret_cast<vec_t *>(&Bp[0])
                    = reinterpret_cast<vec_t>(vec_add(vec_tail1, vmask));
            Bp += 16;

            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail2[0] = b[ldb * 4];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                    }
                    if (y == 2) { vec_tail2[1] = b[ldb * 4 + 1]; }
                }
                if (tail_N == 2) {
                    vec_tail2[0] = b[ldb * 4];
                    vec_tail2[4] = b[ldb * 5];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[6] = b[ldb * 5 + 2];
                    }
                    if (y == 2) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[5] = b[ldb * 5 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail2[0] = b[ldb * 4];
                    vec_tail2[4] = b[ldb * 5];
                    vec_tail2[8] = b[ldb * 6];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[6] = b[ldb * 5 + 2];
                        vec_tail2[9] = b[ldb * 6 + 1];
                        vec_tail2[10] = b[ldb * 6 + 2];
                    }
                    if (y == 2) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[9] = b[ldb * 6 + 1];
                    }
                }
                *reinterpret_cast<vec_t *>(&Bp[0])
                        = reinterpret_cast<vec_t>(vec_add(vec_tail2, vmask));
                Bp += 16;
            }
        }
    }
    if (N_dim <= 3 && N_dim >= 1) {

        const BufType *b = B;
        size_t y = K_dim;
        while (y >= 4) {
            int value = (Flip & 0xFF) | ((Flip & 0xFF) << 8)
                    | ((Flip & 0xFF) << 16) | ((Flip & 0xFF) << 24);
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = value, a3 = value, a4 = value;
            if (N_dim == 3) {
                a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
                a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            }
            if (N_dim == 2) {
                a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            }

            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(vec_t *)&Bp[0]
                    = reinterpret_cast<vec_t>(vec_add(vec_row1, vmask));

            Bp += 16;
            b += 4;
            y -= 4;
        }

        if (y >= 1 && y <= 3) {
            VecType vec_tail1 = reinterpret_cast<VecType>(vec_splats(Flip));

            int tail_N = N_dim;

            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail1[0] = b[0];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];
                    }
                    if (y == 2) { vec_tail1[1] = b[1]; }
                }
                if (tail_N == 2) {
                    vec_tail1[0] = b[0];
                    vec_tail1[4] = b[ldb];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];

                        vec_tail1[5] = b[ldb * 1 + 1];
                        vec_tail1[6] = b[ldb * 1 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = b[1];
                        vec_tail1[5] = b[ldb * 1 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail1[0] = b[0];
                    vec_tail1[4] = b[ldb];
                    vec_tail1[8] = b[ldb * 2];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];

                        vec_tail1[5] = b[ldb + 1];
                        vec_tail1[6] = b[ldb + 2];

                        vec_tail1[9] = b[ldb * 2 + 1];
                        vec_tail1[10] = b[ldb * 2 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = b[1];

                        vec_tail1[5] = b[ldb + 1];
                        vec_tail1[9] = b[ldb * 2 + 1];
                    }
                }

                *reinterpret_cast<vec_t *>(&Bp[0])
                        = reinterpret_cast<vec_t>(vec_add(vec_tail1, vmask));
                Bp += 16;
            }
        }
    }
    return 0;
}

template <typename b_type>
inline int packB_N8bit(dim_t K_dim, dim_t N_dim, const b_type *B, dim_t ldb,
        uint8_t *Bp, bool is_signed) {
    if (is_signed) {
        pack_N8_8bit_V2_lxvp_signed<__vector signed char, b_type>(
                K_dim, N_dim, B, ldb, Bp, true);
    } else {
        pack_N8_8bit_V2_lxvp_signed<__vector unsigned char, b_type>(
                K_dim, N_dim, B, ldb, Bp, false);
    }
    return 0;
}

template <typename VecType>
int pack_N8_8bit_V2(
        dim_t K_dim, dim_t N_dim, const uint8_t *B, dim_t ldb, uint8_t *Bp) {
    int K_block = (K_dim + 3) & (~3);
    int N_block = (N_dim + 3) & (~3);

    while (N_dim >= 8) {
        const uint8_t *b = B;
        size_t y = K_dim;

        while (y >= 8) {
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            VecType swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            VecType swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 3];
            *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 4];
            *(signed long long *)&V2[8] = *(signed long long *)&b[ldb * 5];
            *(signed long long *)&V3[0] = *(signed long long *)&b[ldb * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&b[ldb * 7];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(VecType *)&Bp[0] = D0;
            *(VecType *)&Bp[16] = D1;
            *(VecType *)&Bp[32] = D2;
            *(VecType *)&Bp[48] = D3;

            Bp += 64;
            b += 8;
            y -= 8;
        }

        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            int a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int a4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_row1;

            a1 = *reinterpret_cast<const int *>(&b[ldb * 4]);
            a2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
            a3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
            a4 = *reinterpret_cast<const int *>(&b[ldb * 7]);
            __vector int vec_a1 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a1);
            *reinterpret_cast<VecType *>(&Bp[16]) = vec_row1;
            Bp += 32;
            y -= 4;
            b += 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];

            vec_tail2[0] = b[ldb * 4];
            vec_tail2[4] = b[ldb * 5];
            vec_tail2[8] = b[ldb * 6];
            vec_tail2[12] = b[ldb * 7];

            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];

                vec_tail2[2] = b[ldb * 4 + 2];
                vec_tail2[6] = b[ldb * 5 + 2];
                vec_tail2[10] = b[ldb * 6 + 2];
                vec_tail2[14] = b[ldb * 7 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];
            }

            *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
            *reinterpret_cast<VecType *>(&Bp[16]) = vec_tail2;
            Bp += 32;
        }

        N_dim -= 8;
        B += 8 * ldb;
    }

    if (N_dim >= 4 && N_dim < 8) {

        const uint8_t *b = B;
        size_t y = K_dim;
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            int a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int a4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(VecType *)&Bp[0] = vec_row1;
            Bp += 16;

            int tail_N = N_dim - 4;
            if (tail_N >= 1) {
                a2 = 0;
                a3 = 0;
                a4 = 0;
                a1 = *reinterpret_cast<const int *>(&b[ldb * 4]);
                if (tail_N == 3) {
                    a2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
                    a3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
                }
                if (tail_N == 2) {
                    a2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
                }
                __vector int vec_a1 = {a1, a2, a3, a4};
                VecType vec_row2 = reinterpret_cast<VecType>(vec_a1);
                *(VecType *)&Bp[0] = vec_row2;
                Bp += 16;
                //y -= 4;
            }
            y -= 4;
            b += 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];

            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];
            }
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
            Bp += 16;

            int tail_N = N_dim - 4;
            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail2[0] = b[ldb * 4];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];
                    }
                    if (y == 2) { vec_tail2[1] = b[ldb * 4 + 1]; }
                }
                if (tail_N == 2) {
                    vec_tail2[0] = b[ldb * 4];
                    vec_tail2[4] = b[ldb * 5];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];

                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[6] = b[ldb * 5 + 2];
                    }
                    if (y == 2) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[5] = b[ldb * 5 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail2[0] = b[ldb * 4];
                    vec_tail2[4] = b[ldb * 5];
                    vec_tail2[8] = b[ldb * 6];
                    if (y == 3) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[2] = b[ldb * 4 + 2];

                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[6] = b[ldb * 5 + 2];

                        vec_tail2[9] = b[ldb * 6 + 1];
                        vec_tail2[10] = b[ldb * 6 + 2];
                    }
                    if (y == 2) {
                        vec_tail2[1] = b[ldb * 4 + 1];
                        vec_tail2[5] = b[ldb * 5 + 1];
                        vec_tail2[9] = b[ldb * 6 + 1];
                    }
                }
                *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail2;
                Bp += 16;
            }
        }

        B += N_dim * ldb;
        N_dim = 0;
    }

    if (N_dim >= 1 && N_dim <= 3) {
        const uint8_t *b = B;
        size_t y = K_dim;
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = 0, a3 = 0, a4 = 0;
            if (N_dim == 3) {
                a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
                a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            }
            if (N_dim == 2) {
                a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            }

            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(VecType *)&Bp[0] = vec_row1;

            Bp += 16;
            b += 4;
            y -= 4;
        }

        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            int tail_N = N_dim;

            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail1[0] = b[0];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];
                    }
                    if (y == 2) { vec_tail1[1] = b[1]; }
                }
                if (tail_N == 2) {
                    vec_tail1[0] = b[0];
                    vec_tail1[4] = b[ldb];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];

                        vec_tail1[5] = b[ldb * 1 + 1];
                        vec_tail1[6] = b[ldb * 1 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = b[1];
                        vec_tail1[5] = b[ldb * 1 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail1[0] = b[0];
                    vec_tail1[4] = b[ldb];
                    vec_tail1[8] = b[ldb * 2];
                    if (y == 3) {
                        vec_tail1[1] = b[1];
                        vec_tail1[2] = b[2];

                        vec_tail1[5] = b[ldb + 1];
                        vec_tail1[6] = b[ldb + 2];

                        vec_tail1[9] = b[ldb * 2 + 1];
                        vec_tail1[10] = b[ldb * 2 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = b[1];

                        vec_tail1[5] = b[ldb + 1];
                        vec_tail1[9] = b[ldb * 2 + 1];
                    }
                }

                *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
                Bp += 16;
            }
        }
    }
    return 0;
}

template <typename T>
inline int pack_N8_8bit(dim_t k, dim_t n, const T *b, dim_t ldb, T *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    krows = (k + 3) >> 2;
    k8 = k >> 3;
    n8 = n >> 3;

    // MAIN BLOCK
    for (j = 0; j < (n8 << 3); j += 8) {
        for (i = 0; i < (k8 << 3); i += 8) {
            vec_t V0, V1, V2, V3;
            vec_t D0, D1, D2, D3;
            vec_t swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            vec_t swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
            const T *src = &b[ldb * j + i];
            T *dest = &bp[16 * (krows * (j >> 2) + (i >> 1))];

            *(signed long long *)&V0[0] = *(signed long long *)&src[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&src[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&src[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&src[ldb * 3];
            *(signed long long *)&V2[0] = *(signed long long *)&src[ldb * 4];
            *(signed long long *)&V2[8] = *(signed long long *)&src[ldb * 5];
            *(signed long long *)&V3[0] = *(signed long long *)&src[ldb * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&src[ldb * 7];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (j = (n8 << 3); j < n_cap; ++j) {
        for (i = 0; i < (k8 << 3); ++i) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * j + i];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < (n8 << 3); ++j) {
        for (i = (k8 << 3); i < k_cap; ++i) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * j + i];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (j = (n8 << 3); j < n_cap; ++j) {
        for (i = (k8 << 3); i < k_cap; ++i) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n && i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * j + i];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    return 0;
}
template <typename VecType>
void tailBlock16_12xK(int K_Dim, int N_Dim, const int8_t *A, int lda,
        int8_t *Apacked, int32_t *row_sum_eff) {

    __vector signed int vsum1 = {0};
    __vector signed int vsum2 = {0};
    __vector signed int vsum3 = {0};
    __vector signed int vsum4 = {0};

    if (N_Dim >= 13 && N_Dim < 16) {
        const int8_t *a = A;
        size_t y = K_Dim;

        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda * 1]);

            int a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda * 3]);

            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_row1;
            vsum1 = vec_sum4s(vec_row1, vsum1);

            a1 = *reinterpret_cast<const int *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 7]);

            __vector int vec_a2 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a2);
            *reinterpret_cast<VecType *>(&Apacked[16]) = vec_row1;
            vsum2 = vec_sum4s(vec_row1, vsum2);

            a1 = *reinterpret_cast<const int *>(&a[lda * 8]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 9]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 10]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 11]);

            __vector int vec_a3 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a3);
            *reinterpret_cast<VecType *>(&Apacked[32]) = vec_row1;
            vsum3 = vec_sum4s(vec_row1, vsum3);

            Apacked += 48;

            int tail_N = N_Dim - 12;
            if (tail_N >= 1) {
                a1 = 0, a2 = 0, a3 = 0, a4 = 0;
                if (tail_N == 1) {
                    a1 = *reinterpret_cast<const int *>(&a[lda * 12]);
                }
                if (tail_N == 2) {
                    a1 = *reinterpret_cast<const int *>(&a[lda * 12]);
                    a2 = *reinterpret_cast<const int *>(&a[lda * 13]);
                }
                if (tail_N == 3) {
                    a1 = *reinterpret_cast<const int *>(&a[lda * 12]);
                    a2 = *reinterpret_cast<const int *>(&a[lda * 13]);
                    a3 = *reinterpret_cast<const int *>(&a[lda * 14]);
                }
                __vector int vec_a4 = {a1, a2, a3, a4};
                vec_row1 = reinterpret_cast<VecType>(vec_a4);
                *reinterpret_cast<VecType *>(&Apacked[0]) = vec_row1;
                vsum4 = vec_sum4s(vec_row1, vsum4);
                Apacked += 16;
            }
            y -= 4;
            a += 4;
        }

        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail3
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail4
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            // 1st 4 rows
            vec_tail1[0] = a[0];
            vec_tail1[4] = a[lda];
            vec_tail1[8] = a[lda * 2];
            vec_tail1[12] = a[lda * 3];

            if (y == 3) {
                vec_tail1[1] = a[1];
                vec_tail1[5] = a[lda + 1];
                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[13] = a[lda * 3 + 1];

                vec_tail1[2] = a[2];
                vec_tail1[6] = a[lda + 2];
                vec_tail1[10] = a[lda * 2 + 2];
                vec_tail1[14] = a[lda * 3 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = a[1];
                vec_tail1[5] = a[lda + 1];
                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[13] = a[lda * 3 + 1];
            }

            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_tail1;
            vsum1 = vec_sum4s(vec_tail1, vsum1);

            vec_tail2[0] = a[lda * 4];
            vec_tail2[4] = a[lda * 5];
            vec_tail2[8] = a[lda * 6];
            vec_tail2[12] = a[lda * 7];
            if (y == 3) {
                vec_tail2[1] = a[lda * 4 + 1];
                vec_tail2[5] = a[lda * 5 + 1];
                vec_tail2[9] = a[lda * 6 + 1];
                vec_tail2[13] = a[lda * 7 + 1];

                vec_tail2[2] = a[lda * 4 + 2];
                vec_tail2[6] = a[lda * 5 + 2];
                vec_tail2[10] = a[lda * 6 + 2];
                vec_tail2[14] = a[lda * 7 + 2];
            }
            if (y == 2) {
                vec_tail2[1] = a[lda * 4 + 1];
                vec_tail2[5] = a[lda * 5 + 1];
                vec_tail2[9] = a[lda * 6 + 1];
                vec_tail2[13] = a[lda * 7 + 1];
            }

            *reinterpret_cast<VecType *>(&Apacked[16]) = vec_tail2;
            vsum2 = vec_sum4s(vec_tail2, vsum2);

            vec_tail3[0] = a[lda * 8];
            vec_tail3[4] = a[lda * 9];
            vec_tail3[8] = a[lda * 10];
            vec_tail3[12] = a[lda * 11];

            if (y == 3) {
                vec_tail3[1] = a[lda * 8 + 1];
                vec_tail3[5] = a[lda * 9 + 1];
                vec_tail3[9] = a[lda * 10 + 1];
                vec_tail3[13] = a[lda * 11 + 1];

                vec_tail3[2] = a[lda * 8 + 2];
                vec_tail3[6] = a[lda * 9 + 2];
                vec_tail3[10] = a[lda * 10 + 2];
                vec_tail3[14] = a[lda * 11 + 2];
            }
            if (y == 2) {
                vec_tail3[1] = a[lda * 8 + 1];
                vec_tail3[5] = a[lda * 9 + 1];
                vec_tail3[9] = a[lda * 10 + 1];
                vec_tail3[13] = a[lda * 11 + 1];
            }
            *reinterpret_cast<VecType *>(&Apacked[32]) = vec_tail3;
            vsum3 = vec_sum4s(vec_tail3, vsum3);

            int tail_N = N_Dim - 12;
            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail4[0] = a[lda * 12];
                    if (y == 3) {
                        vec_tail4[1] = a[lda * 12 + 1];
                        vec_tail4[2] = a[lda * 12 + 2];
                    }
                    if (y == 2) { vec_tail4[1] = a[lda * 12 + 1]; }
                }
                if (tail_N == 2) {
                    vec_tail4[0] = a[lda * 12];
                    vec_tail4[4] = a[lda * 13];
                    if (y == 3) {
                        vec_tail4[1] = a[lda * 12 + 1];
                        vec_tail4[2] = a[lda * 12 + 2];

                        vec_tail4[5] = a[lda * 13 + 1];
                        vec_tail4[6] = a[lda * 13 + 2];
                    }
                    if (y == 2) {
                        vec_tail4[1] = a[lda * 12 + 1];
                        vec_tail4[5] = a[lda * 13 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail4[0] = a[lda * 12];
                    vec_tail4[4] = a[lda * 13];
                    vec_tail4[8] = a[lda * 14];
                    if (y == 3) {
                        vec_tail4[1] = a[lda * 12 + 1];
                        vec_tail4[2] = a[lda * 12 + 2];

                        vec_tail4[5] = a[lda * 13 + 1];
                        vec_tail4[6] = a[lda * 13 + 2];

                        vec_tail4[9] = a[lda * 14 + 1];
                        vec_tail4[10] = a[lda * 14 + 2];
                    }
                    if (y == 2) {
                        vec_tail4[1] = a[lda * 12 + 1];
                        vec_tail4[5] = a[lda * 13 + 1];
                        vec_tail4[9] = a[lda * 14 + 1];
                    }
                }
                *reinterpret_cast<VecType *>(&Apacked[48]) = vec_tail4;
                vsum4 = vec_sum4s(vec_tail4, vsum4);
            }
            Apacked += 64;
        }

        row_sum_eff[0] = vsum1[0];
        row_sum_eff[1] = vsum1[1];
        row_sum_eff[2] = vsum1[2];
        row_sum_eff[3] = vsum1[3];

        row_sum_eff[4] = vsum2[0];
        row_sum_eff[5] = vsum2[1];
        row_sum_eff[6] = vsum2[2];
        row_sum_eff[7] = vsum2[3];

        row_sum_eff[8] = vsum3[0];
        row_sum_eff[9] = vsum3[1];
        row_sum_eff[10] = vsum3[2];
        row_sum_eff[11] = vsum3[3];

        row_sum_eff += 12;

        int tail_N = N_Dim - 12;

        if (tail_N == 1) { row_sum_eff[0] = vsum4[0]; }
        if (tail_N == 2) {
            row_sum_eff[0] = vsum4[0];
            row_sum_eff[1] = vsum4[1];
        }
        if (tail_N == 3) {
            row_sum_eff[0] = vsum4[0];
            row_sum_eff[1] = vsum4[1];
            row_sum_eff[2] = vsum4[2];
        }
        row_sum_eff += tail_N;
    }
}

template <typename VecType>
void tailBlock16_8xK(int K_Dim, int N_Dim, const int8_t *A, int lda,
        int8_t *Apacked, int32_t *row_sum_eff) {

    while (N_Dim >= 8) {

        const int8_t *a = A;
        size_t y = K_Dim;
        __vector signed int vsum1 = {0};
        __vector signed int vsum2 = {0};

        while (y >= 16) {
            VecType a1 = *reinterpret_cast<const VecType *>(&a[0]);
            VecType a2 = *reinterpret_cast<const VecType *>(&a[lda]);
            VecType a3 = *reinterpret_cast<const VecType *>(&a[lda * 2]);
            VecType a4 = *reinterpret_cast<const VecType *>(&a[lda * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row3 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Apacked[0]) = vx_row1;
            *reinterpret_cast<VecType *>(&Apacked[32]) = vx_row3;
            *reinterpret_cast<VecType *>(&Apacked[64]) = vx_row5;
            *reinterpret_cast<VecType *>(&Apacked[96]) = vx_row7;

            vsum1 = vec_sum4s(vx_row1, vsum1);
            vsum1 = vec_sum4s(vx_row3, vsum1);
            vsum1 = vec_sum4s(vx_row5, vsum1);
            vsum1 = vec_sum4s(vx_row7, vsum1);

            // 2nd 4 Columns
            a1 = *reinterpret_cast<const VecType *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const VecType *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const VecType *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const VecType *>(&a[lda * 7]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));

            VecType vx_row2 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row4 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row6 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row8 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Apacked[16]) = vx_row2;
            *reinterpret_cast<VecType *>(&Apacked[48]) = vx_row4;
            *reinterpret_cast<VecType *>(&Apacked[80]) = vx_row6;
            *reinterpret_cast<VecType *>(&Apacked[112]) = vx_row8;

            vsum2 = vec_sum4s(vx_row2, vsum2);
            vsum2 = vec_sum4s(vx_row4, vsum2);
            vsum2 = vec_sum4s(vx_row6, vsum2);
            vsum2 = vec_sum4s(vx_row8, vsum2);

            a += 16;
            Apacked += 128;
            y -= 16;
        }

        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda * 1]);
            int a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_row1;
            vsum1 = vec_sum4s(vec_row1, vsum1);

            // Next 4 Column
            a1 = *reinterpret_cast<const int *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 7]);
            __vector int vec_a1 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a1);
            *reinterpret_cast<VecType *>(&Apacked[16]) = vec_row1;
            vsum2 = vec_sum4s(vec_row1, vsum2);
            Apacked += 32;
            a += 4;
            y -= 4;
        }

        if (y <= 3 && y >= 1) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = a[0];
            vec_tail1[4] = a[lda];
            vec_tail1[8] = a[lda * 2];
            vec_tail1[12] = a[lda * 3];

            vec_tail2[0] = a[lda * 4];
            vec_tail2[4] = a[lda * 5];
            vec_tail2[8] = a[lda * 6];
            vec_tail2[12] = a[lda * 7];

            if (y == 3) {
                vec_tail1[1] = a[1];
                vec_tail1[2] = a[2];
                vec_tail1[5] = a[lda + 1];
                vec_tail1[6] = a[lda + 2];
                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[10] = a[lda * 2 + 2];
                vec_tail1[13] = a[lda * 3 + 1];
                vec_tail1[14] = a[lda * 3 + 2];

                vec_tail2[1] = a[lda * 4 + 1];
                vec_tail2[2] = a[lda * 4 + 2];
                vec_tail2[5] = a[lda * 5 + 1];
                vec_tail2[6] = a[lda * 5 + 2];
                vec_tail2[9] = a[lda * 6 + 1];
                vec_tail2[10] = a[lda * 6 + 2];
                vec_tail2[13] = a[lda * 7 + 1];
                vec_tail2[14] = a[lda * 7 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = a[1];
                vec_tail1[5] = a[lda + 1];
                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[13] = a[lda * 3 + 1];

                vec_tail2[1] = a[lda * 4 + 1];
                vec_tail2[5] = a[lda * 5 + 1];
                vec_tail2[9] = a[lda * 6 + 1];
                vec_tail2[13] = a[lda * 7 + 1];
            }
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_tail1;
            *reinterpret_cast<VecType *>(&Apacked[16]) = vec_tail2;
            vsum1 = vec_sum4s(vec_tail1, vsum1);
            vsum2 = vec_sum4s(vec_tail2, vsum2);

            Apacked += 32;
        }
        row_sum_eff[0] = vsum1[0];
        row_sum_eff[1] = vsum1[1];
        row_sum_eff[2] = vsum1[2];
        row_sum_eff[3] = vsum1[3];

        row_sum_eff[4] = vsum2[0];
        row_sum_eff[5] = vsum2[1];
        row_sum_eff[6] = vsum2[2];
        row_sum_eff[7] = vsum2[3];

        row_sum_eff += 8;
        A += 8 * lda;
        N_Dim -= 8;
    }
    while (N_Dim >= 4) {
        const int8_t *a = A;
        size_t y = K_Dim;
        __vector signed int vsum1 = {0};
        __vector signed int vsum2 = {0};
        int tail_N = N_Dim - 4;

        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda * 1]);
            int a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda * 3]);

            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_row;
            vsum1 = vec_sum4s(vec_row, vsum1);
            Apacked += 16;

            if (tail_N >= 1) {
                int a1 = *reinterpret_cast<const int *>(&a[lda * 4]);
                int a2 = 0, a3 = 0, a4 = 0;

                if (tail_N == 2) {
                    a2 = *reinterpret_cast<const int *>(&a[lda * 5]);
                }
                if (tail_N == 3) {
                    a2 = *reinterpret_cast<const int *>(&a[lda * 5]);
                    a3 = *reinterpret_cast<const int *>(&a[lda * 6]);
                }
                __vector int vec_a = {a1, a2, a3, a4};
                VecType vec_row = reinterpret_cast<VecType>(vec_a);
                *reinterpret_cast<VecType *>(&Apacked[0]) = vec_row;
                vsum2 = vec_sum4s(vec_row, vsum2);
                Apacked += 16;
            }

            a += 4;
            y -= 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = a[0];
            vec_tail1[4] = a[lda * 1];
            vec_tail1[8] = a[lda * 2];
            vec_tail1[12] = a[lda * 3];
            if (y == 3) {
                vec_tail1[1] = a[1];
                vec_tail1[2] = a[2];
                vec_tail1[5] = a[lda * 1 + 1];
                vec_tail1[6] = a[lda * 1 + 2];

                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[10] = a[lda * 2 + 2];
                vec_tail1[13] = a[lda * 3 + 1];
                vec_tail1[14] = a[lda * 3 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = a[1];
                vec_tail1[5] = a[lda * 1 + 1];
                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[13] = a[lda * 3 + 1];
            }
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_tail1;
            vsum1 = vec_sum4s(vec_tail1, vsum1);

            Apacked += 16;
            if (tail_N >= 1) {
                VecType vec_tail1
                        = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
                if (tail_N == 1) {
                    vec_tail1[0] = a[lda * 4];
                    if (y == 3) {
                        vec_tail1[1] = a[lda * 4 + 1];
                        vec_tail1[2] = a[lda * 4 + 2];
                    }
                    if (y == 2) { vec_tail1[1] = a[lda * 4 + 1]; }
                }
                if (tail_N == 2) {
                    vec_tail1[0] = a[lda * 4];
                    vec_tail1[4] = a[lda * 5];
                    if (y == 3) {
                        vec_tail1[1] = a[lda * 4 + 1];
                        vec_tail1[2] = a[lda * 4 + 2];

                        vec_tail1[5] = a[lda * 5 + 1];
                        vec_tail1[6] = a[lda * 5 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = a[lda * 4 + 1];
                        vec_tail1[5] = a[lda * 5 + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail1[0] = a[lda * 4];
                    vec_tail1[4] = a[lda * 5];
                    vec_tail1[8] = a[lda * 6];
                    if (y == 3) {
                        vec_tail1[1] = a[lda * 4 + 1];
                        vec_tail1[2] = a[lda * 4 + 2];
                        vec_tail1[5] = a[lda * 5 + 1];
                        vec_tail1[6] = a[lda * 5 + 2];
                        vec_tail1[9] = a[lda * 6 + 1];
                        vec_tail1[10] = a[lda * 6 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = a[lda * 4 + 1];
                        vec_tail1[5] = a[lda * 5 + 1];
                        vec_tail1[9] = a[lda * 6 + 1];
                    }
                }
                *reinterpret_cast<VecType *>(&Apacked[0]) = vec_tail1;
                vsum2 = vec_sum4s(vec_tail1, vsum2);
                Apacked += 16;
            }
        }
        row_sum_eff[0] = vsum1[0];
        row_sum_eff[1] = vsum1[1];
        row_sum_eff[2] = vsum1[2];
        row_sum_eff[3] = vsum1[3];
        row_sum_eff += 4;

        if (tail_N == 1) { row_sum_eff[0] = vsum2[0]; }
        if (tail_N == 2) {
            row_sum_eff[0] = vsum2[0];
            row_sum_eff[1] = vsum2[1];
        }
        if (tail_N == 3) {
            row_sum_eff[0] = vsum2[0];
            row_sum_eff[1] = vsum2[1];
            row_sum_eff[2] = vsum2[2];
        }
        row_sum_eff += tail_N;

        A += N_Dim * lda;
        N_Dim -= N_Dim;
    }

    if (N_Dim >= 1 && N_Dim <= 3) {

        const int8_t *a = A;
        size_t y = K_Dim;
        __vector signed int vsum1 = {0};
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = 0, a3 = 0, a4 = 0;
            if (N_Dim == 3) {
                a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
                a2 = *reinterpret_cast<const int *>(&a[lda * 1]);
            }
            if (N_Dim == 2) {
                a2 = *reinterpret_cast<const int *>(&a[lda * 1]);
            }
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *(VecType *)&Apacked[0] = vec_row1;

            vsum1 = vec_sum4s(vec_row1, vsum1);

            a += 4;
            Apacked += 16;
            y -= 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            int tail_N = N_Dim;

            if (tail_N >= 1) {
                if (tail_N == 1) {
                    vec_tail1[0] = a[0];
                    if (y == 3) {
                        vec_tail1[1] = a[1];
                        vec_tail1[2] = a[2];
                    }
                    if (y == 2) { vec_tail1[1] = a[1]; }
                }
                if (tail_N == 2) {
                    vec_tail1[0] = a[0];
                    vec_tail1[4] = a[lda];
                    if (y == 3) {
                        vec_tail1[1] = a[1];
                        vec_tail1[2] = a[2];

                        vec_tail1[5] = a[lda + 1];
                        vec_tail1[6] = a[lda + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = a[1];

                        vec_tail1[5] = a[lda + 1];
                    }
                }
                if (tail_N == 3) {
                    vec_tail1[0] = a[0];
                    vec_tail1[4] = a[lda];
                    vec_tail1[8] = a[lda * 2];

                    if (y == 3) {
                        vec_tail1[1] = a[1];
                        vec_tail1[2] = a[2];

                        vec_tail1[5] = a[lda + 1];
                        vec_tail1[6] = a[lda + 2];

                        vec_tail1[9] = a[lda * 2 + 1];
                        vec_tail1[10] = a[lda * 2 + 2];
                    }
                    if (y == 2) {
                        vec_tail1[1] = a[1];

                        vec_tail1[5] = a[lda + 1];

                        vec_tail1[9] = a[lda * 2 + 1];
                    }
                }
                *reinterpret_cast<VecType *>(&Apacked[0]) = vec_tail1;
                vsum1 = vec_sum4s(vec_tail1, vsum1);
                Apacked += 16;
            }
        }
        if (N_Dim == 3) {
            row_sum_eff[0] = vsum1[0];
            row_sum_eff[1] = vsum1[1];
            row_sum_eff[2] = vsum1[2];
        }
        if (N_Dim == 2) {
            row_sum_eff[0] = vsum1[0];
            row_sum_eff[1] = vsum1[1];
        }
        if (N_Dim == 1) { row_sum_eff[0] = vsum1[0]; }
        row_sum_eff += N_Dim;
    }
}

template <typename VecType>
int pack_N16_8bit_V2_lxvp(int K_dim, int N_dim, const int8_t *B, int ldb,
        int8_t *Bp, int32_t *row_sum_eff) {

    while (N_dim >= 16) {
        int8_t *b = const_cast<int8_t *>(B);
        size_t y = K_dim;
        __vector signed int vsum1 = {0};
        __vector signed int vsum2 = {0};
        __vector signed int vsum3 = {0};
        __vector signed int vsum4 = {0};
        while (y >= 32) {

            __vector_pair row1, row2, row3, row4, row5, row6, row7, row8;
            __vector_pair row9, row10, row11, row12, row13, row14, row15, row16;
            VecType r1[2] = {0}, r2[2] = {0}, r3[2] = {0}, r4[2] = {0},
                    r5[2] = {0}, r6[2] = {0}, r7[2] = {0}, r8[2] = {0};
            VecType r9[2] = {0}, r10[2] = {0}, r11[2] = {0}, r12[2] = {0},
                    r13[2] = {0}, r14[2] = {0}, r15[2] = {0}, r16[2] = {0};

            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;

            row1 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[0]));
            row2 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb]));
            row3 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 2]));
            row4 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 3]));
            row5 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 4]));
            row6 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 5]));
            row7 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 6]));
            row8 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 7]));
            row9 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 8]));
            row10 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 9]));
            row11 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 10]));
            row12 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 11]));
            row13 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 12]));
            row14 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 13]));
            row15 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 14]));
            row16 = __builtin_vsx_lxvp(
                    0, reinterpret_cast<__vector_pair *>(&b[ldb * 15]));

            __builtin_vsx_disassemble_pair(r1, &row1);
            __builtin_vsx_disassemble_pair(r2, &row2);
            __builtin_vsx_disassemble_pair(r3, &row3);
            __builtin_vsx_disassemble_pair(r4, &row4);
            __builtin_vsx_disassemble_pair(r5, &row5);
            __builtin_vsx_disassemble_pair(r6, &row6);
            __builtin_vsx_disassemble_pair(r7, &row7);
            __builtin_vsx_disassemble_pair(r8, &row8);
            __builtin_vsx_disassemble_pair(r9, &row9);
            __builtin_vsx_disassemble_pair(r10, &row10);
            __builtin_vsx_disassemble_pair(r11, &row11);
            __builtin_vsx_disassemble_pair(r12, &row12);
            __builtin_vsx_disassemble_pair(r13, &row13);
            __builtin_vsx_disassemble_pair(r14, &row14);
            __builtin_vsx_disassemble_pair(r15, &row15);
            __builtin_vsx_disassemble_pair(r16, &row16);

            // First 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[0]),
                            reinterpret_cast<__vector int>(r2[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[0]),
                            reinterpret_cast<__vector int>(r4[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[0] = D0;
            *(VecType *)&Bp[64] = D1;
            *(VecType *)&Bp[128] = D2;
            *(VecType *)&Bp[192] = D3;

            vsum1 = vec_sum4s(D0, vsum1);
            vsum1 = vec_sum4s(D1, vsum1);
            vsum1 = vec_sum4s(D2, vsum1);
            vsum1 = vec_sum4s(D3, vsum1);

            // Next (ldb * 4) 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r5[0]),
                            reinterpret_cast<__vector int>(r6[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r7[0]),
                            reinterpret_cast<__vector int>(r8[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r5[0]),
                            reinterpret_cast<__vector int>(r6[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r7[0]),
                            reinterpret_cast<__vector int>(r8[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[16] = D0;
            *(VecType *)&Bp[80] = D1;
            *(VecType *)&Bp[144] = D2;
            *(VecType *)&Bp[208] = D3;

            vsum2 = vec_sum4s(D0, vsum2);
            vsum2 = vec_sum4s(D1, vsum2);
            vsum2 = vec_sum4s(D2, vsum2);
            vsum2 = vec_sum4s(D3, vsum2);

            // Third (ldb * 8) 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r9[0]),
                            reinterpret_cast<__vector int>(r10[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r11[0]),
                            reinterpret_cast<__vector int>(r12[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r9[0]),
                            reinterpret_cast<__vector int>(r10[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r11[0]),
                            reinterpret_cast<__vector int>(r12[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[32] = D0;
            *(VecType *)&Bp[96] = D1;
            *(VecType *)&Bp[160] = D2;
            *(VecType *)&Bp[224] = D3;

            vsum3 = vec_sum4s(D0, vsum3);
            vsum3 = vec_sum4s(D1, vsum3);
            vsum3 = vec_sum4s(D2, vsum3);
            vsum3 = vec_sum4s(D3, vsum3);

            // Fourth (ldb * 12) 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r13[0]),
                            reinterpret_cast<__vector int>(r14[0])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r15[0]),
                            reinterpret_cast<__vector int>(r16[0])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r13[0]),
                            reinterpret_cast<__vector int>(r14[0])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r15[0]),
                            reinterpret_cast<__vector int>(r16[0])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[48] = D0;
            *(VecType *)&Bp[112] = D1;
            *(VecType *)&Bp[176] = D2;
            *(VecType *)&Bp[240] = D3;

            vsum4 = vec_sum4s(D0, vsum4);
            vsum4 = vec_sum4s(D1, vsum4);
            vsum4 = vec_sum4s(D2, vsum4);
            vsum4 = vec_sum4s(D3, vsum4);

            Bp += 256;

            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r1[1]),
                            reinterpret_cast<__vector int>(r2[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r3[1]),
                            reinterpret_cast<__vector int>(r4[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[0] = D0;
            *(VecType *)&Bp[64] = D1;
            *(VecType *)&Bp[128] = D2;
            *(VecType *)&Bp[192] = D3;

            vsum1 = vec_sum4s(D0, vsum1);
            vsum1 = vec_sum4s(D1, vsum1);
            vsum1 = vec_sum4s(D2, vsum1);
            vsum1 = vec_sum4s(D3, vsum1);

            // Next (ldb * 4) 4 Rows and Second 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r5[1]),
                            reinterpret_cast<__vector int>(r6[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r7[1]),
                            reinterpret_cast<__vector int>(r8[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r5[1]),
                            reinterpret_cast<__vector int>(r6[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r7[1]),
                            reinterpret_cast<__vector int>(r8[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[16] = D0;
            *(VecType *)&Bp[80] = D1;
            *(VecType *)&Bp[144] = D2;
            *(VecType *)&Bp[208] = D3;

            vsum2 = vec_sum4s(D0, vsum2);
            vsum2 = vec_sum4s(D1, vsum2);
            vsum2 = vec_sum4s(D2, vsum2);
            vsum2 = vec_sum4s(D3, vsum2);

            // First 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r9[1]),
                            reinterpret_cast<__vector int>(r10[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r11[1]),
                            reinterpret_cast<__vector int>(r12[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r9[1]),
                            reinterpret_cast<__vector int>(r10[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r11[1]),
                            reinterpret_cast<__vector int>(r12[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[32] = D0;
            *(VecType *)&Bp[96] = D1;
            *(VecType *)&Bp[160] = D2;
            *(VecType *)&Bp[224] = D3;

            vsum3 = vec_sum4s(D0, vsum3);
            vsum3 = vec_sum4s(D1, vsum3);
            vsum3 = vec_sum4s(D2, vsum3);
            vsum3 = vec_sum4s(D3, vsum3);

            // Next (ldb * 4) 4 Rows and First 16 columns
            V0 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r13[1]),
                            reinterpret_cast<__vector int>(r14[1])));
            V1 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(r15[1]),
                            reinterpret_cast<__vector int>(r16[1])));
            V2 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r13[1]),
                            reinterpret_cast<__vector int>(r14[1])));
            V3 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(r15[1]),
                            reinterpret_cast<__vector int>(r16[1])));

            D0 = vec_xxpermdi(V0, V1, 0);
            D1 = vec_xxpermdi(V2, V3, 0);
            D2 = vec_xxpermdi(V0, V1, 3);
            D3 = vec_xxpermdi(V2, V3, 3);

            *(VecType *)&Bp[48] = D0;
            *(VecType *)&Bp[112] = D1;
            *(VecType *)&Bp[176] = D2;
            *(VecType *)&Bp[240] = D3;

            vsum4 = vec_sum4s(D0, vsum4);
            vsum4 = vec_sum4s(D1, vsum4);
            vsum4 = vec_sum4s(D2, vsum4);
            vsum4 = vec_sum4s(D3, vsum4);

            y -= 32;
            b += 32;
            Bp += 256;
        }
        while (y >= 16) {
            // First 4th row and 16 Columns
            VecType b1 = *reinterpret_cast<const VecType *>(&b[0]);
            VecType b2 = *reinterpret_cast<const VecType *>(&b[ldb]);
            VecType b3 = *reinterpret_cast<const VecType *>(&b[ldb * 2]);
            VecType b4 = *reinterpret_cast<const VecType *>(&b[ldb * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row9 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row13 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[0]) = vx_row1;
            *reinterpret_cast<VecType *>(&Bp[64]) = vx_row5;
            *reinterpret_cast<VecType *>(&Bp[128]) = vx_row9;
            *reinterpret_cast<VecType *>(&Bp[192]) = vx_row13;

            vsum1 = vec_sum4s(vx_row1, vsum1);
            vsum1 = vec_sum4s(vx_row5, vsum1);
            vsum1 = vec_sum4s(vx_row9, vsum1);
            vsum1 = vec_sum4s(vx_row13, vsum1);

            // Second 4th Row and 16 Columns
            b1 = *reinterpret_cast<const VecType *>(&b[ldb * 4]);
            b2 = *reinterpret_cast<const VecType *>(&b[ldb * 5]);
            b3 = *reinterpret_cast<const VecType *>(&b[ldb * 6]);
            b4 = *reinterpret_cast<const VecType *>(&b[ldb * 7]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row2 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row6 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row10 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row14 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[16]) = vx_row2;
            *reinterpret_cast<VecType *>(&Bp[80]) = vx_row6;
            *reinterpret_cast<VecType *>(&Bp[144]) = vx_row10;
            *reinterpret_cast<VecType *>(&Bp[208]) = vx_row14;

            vsum2 = vec_sum4s(vx_row2, vsum2);
            vsum2 = vec_sum4s(vx_row6, vsum2);
            vsum2 = vec_sum4s(vx_row10, vsum2);
            vsum2 = vec_sum4s(vx_row14, vsum2);

            b1 = *reinterpret_cast<const VecType *>(&b[ldb * 8]);
            b2 = *reinterpret_cast<const VecType *>(&b[ldb * 9]);
            b3 = *reinterpret_cast<const VecType *>(&b[ldb * 10]);
            b4 = *reinterpret_cast<const VecType *>(&b[ldb * 11]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row3 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row11 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row15 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[32]) = vx_row3;
            *reinterpret_cast<VecType *>(&Bp[96]) = vx_row7;
            *reinterpret_cast<VecType *>(&Bp[160]) = vx_row11;
            *reinterpret_cast<VecType *>(&Bp[224]) = vx_row15;

            vsum3 = vec_sum4s(vx_row3, vsum3);
            vsum3 = vec_sum4s(vx_row7, vsum3);
            vsum3 = vec_sum4s(vx_row11, vsum3);
            vsum3 = vec_sum4s(vx_row15, vsum3);

            b1 = *reinterpret_cast<const VecType *>(&b[ldb * 12]);
            b2 = *reinterpret_cast<const VecType *>(&b[ldb * 13]);
            b3 = *reinterpret_cast<const VecType *>(&b[ldb * 14]);
            b4 = *reinterpret_cast<const VecType *>(&b[ldb * 15]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b1),
                            reinterpret_cast<__vector int>(b2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(b3),
                            reinterpret_cast<__vector int>(b4)));

            VecType vx_row4 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row8 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row12 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row16 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Bp[48]) = vx_row4;
            *reinterpret_cast<VecType *>(&Bp[112]) = vx_row8;
            *reinterpret_cast<VecType *>(&Bp[176]) = vx_row12;
            *reinterpret_cast<VecType *>(&Bp[240]) = vx_row16;

            vsum4 = vec_sum4s(vx_row4, vsum4);
            vsum4 = vec_sum4s(vx_row8, vsum4);
            vsum4 = vec_sum4s(vx_row12, vsum4);
            vsum4 = vec_sum4s(vx_row16, vsum4);

            b += 16;
            Bp += 256;
            y -= 16;
        }
        while (y >= 8) {
            VecType V0, V1, V2, V3;
            VecType D0, D1, D2, D3;
            __vector unsigned char swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            __vector unsigned char swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 3];
            *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 4];
            *(signed long long *)&V2[8] = *(signed long long *)&b[ldb * 5];
            *(signed long long *)&V3[0] = *(signed long long *)&b[ldb * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&b[ldb * 7];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(VecType *)&Bp[0] = D0;
            *(VecType *)&Bp[16] = D1;
            *(VecType *)&Bp[64] = D2;
            *(VecType *)&Bp[80] = D3;

            vsum1 = vec_sum4s(D0, vsum1);
            vsum1 = vec_sum4s(D2, vsum1);
            vsum2 = vec_sum4s(D1, vsum2);
            vsum2 = vec_sum4s(D3, vsum2);

            *(signed long long *)&V0[0] = *(signed long long *)&b[ldb * 8];
            *(signed long long *)&V0[8] = *(signed long long *)&b[ldb * 9];
            *(signed long long *)&V1[0] = *(signed long long *)&b[ldb * 10];
            *(signed long long *)&V1[8] = *(signed long long *)&b[ldb * 11];
            *(signed long long *)&V2[0] = *(signed long long *)&b[ldb * 12];
            *(signed long long *)&V2[8] = *(signed long long *)&b[ldb * 13];
            *(signed long long *)&V3[0] = *(signed long long *)&b[ldb * 14];
            *(signed long long *)&V3[8] = *(signed long long *)&b[ldb * 15];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(VecType *)&Bp[32] = D0;
            *(VecType *)&Bp[48] = D1;
            *(VecType *)&Bp[96] = D2;
            *(VecType *)&Bp[112] = D3;

            vsum3 = vec_sum4s(D0, vsum3);
            vsum3 = vec_sum4s(D2, vsum3);
            vsum4 = vec_sum4s(D1, vsum4);
            vsum4 = vec_sum4s(D3, vsum4);

            Bp += 16 * 8;
            b += 8;
            y -= 8;
        }
        while (y >= 4) {
            int a1 = *reinterpret_cast<const int *>(&b[0]);
            int a2 = *reinterpret_cast<const int *>(&b[ldb * 1]);
            int a3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
            int a4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_row1;
            vsum1 = vec_sum4s(vec_row1, vsum1);

            a1 = *reinterpret_cast<const int *>(&b[ldb * 4]);
            a2 = *reinterpret_cast<const int *>(&b[ldb * 5]);
            a3 = *reinterpret_cast<const int *>(&b[ldb * 6]);
            a4 = *reinterpret_cast<const int *>(&b[ldb * 7]);
            __vector int vec_a1 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a1);
            *reinterpret_cast<VecType *>(&Bp[16]) = vec_row1;
            vsum2 = vec_sum4s(vec_row1, vsum2);

            a1 = *reinterpret_cast<const int *>(&b[ldb * 8]);
            a2 = *reinterpret_cast<const int *>(&b[ldb * 9]);
            a3 = *reinterpret_cast<const int *>(&b[ldb * 10]);
            a4 = *reinterpret_cast<const int *>(&b[ldb * 11]);
            __vector int vec_a2 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a2);
            *reinterpret_cast<VecType *>(&Bp[32]) = vec_row1;
            vsum3 = vec_sum4s(vec_row1, vsum3);

            a1 = *reinterpret_cast<const int *>(&b[ldb * 12]);
            a2 = *reinterpret_cast<const int *>(&b[ldb * 13]);
            a3 = *reinterpret_cast<const int *>(&b[ldb * 14]);
            a4 = *reinterpret_cast<const int *>(&b[ldb * 15]);
            __vector int vec_a3 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a3);
            *reinterpret_cast<VecType *>(&Bp[48]) = vec_row1;
            vsum4 = vec_sum4s(vec_row1, vsum4);

            Bp += 64;
            y -= 4;
            b += 4;
        }
        if (y >= 1 && y <= 3) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail3
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail4
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = b[0];
            vec_tail1[4] = b[ldb];
            vec_tail1[8] = b[ldb * 2];
            vec_tail1[12] = b[ldb * 3];

            vec_tail2[0] = b[ldb * 4];
            vec_tail2[4] = b[ldb * 5];
            vec_tail2[8] = b[ldb * 6];
            vec_tail2[12] = b[ldb * 7];

            vec_tail3[0] = b[ldb * 8];
            vec_tail3[4] = b[ldb * 9];
            vec_tail3[8] = b[ldb * 10];
            vec_tail3[12] = b[ldb * 11];

            vec_tail4[0] = b[ldb * 12];
            vec_tail4[4] = b[ldb * 13];
            vec_tail4[8] = b[ldb * 14];
            vec_tail4[12] = b[ldb * 15];

            if (y == 3) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];

                vec_tail1[2] = b[2];
                vec_tail1[6] = b[ldb + 2];
                vec_tail1[10] = b[ldb * 2 + 2];
                vec_tail1[14] = b[ldb * 3 + 2];

                vec_tail2[2] = b[ldb * 4 + 2];
                vec_tail2[6] = b[ldb * 5 + 2];
                vec_tail2[10] = b[ldb * 6 + 2];
                vec_tail2[14] = b[ldb * 7 + 2];

                vec_tail3[1] = b[ldb * 8 + 1];
                vec_tail3[5] = b[ldb * 9 + 1];
                vec_tail3[9] = b[ldb * 10 + 1];
                vec_tail3[13] = b[ldb * 11 + 1];

                vec_tail3[2] = b[ldb * 8 + 2];
                vec_tail3[6] = b[ldb * 9 + 2];
                vec_tail3[10] = b[ldb * 10 + 2];
                vec_tail3[14] = b[ldb * 11 + 2];

                vec_tail4[1] = b[ldb * 12 + 1];
                vec_tail4[5] = b[ldb * 13 + 1];
                vec_tail4[9] = b[ldb * 14 + 1];
                vec_tail4[13] = b[ldb * 15 + 1];

                vec_tail4[2] = b[ldb * 12 + 2];
                vec_tail4[6] = b[ldb * 13 + 2];
                vec_tail4[10] = b[ldb * 14 + 2];
                vec_tail4[14] = b[ldb * 15 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = b[1];
                vec_tail1[5] = b[ldb + 1];
                vec_tail1[9] = b[ldb * 2 + 1];
                vec_tail1[13] = b[ldb * 3 + 1];

                vec_tail2[1] = b[ldb * 4 + 1];
                vec_tail2[5] = b[ldb * 5 + 1];
                vec_tail2[9] = b[ldb * 6 + 1];
                vec_tail2[13] = b[ldb * 7 + 1];

                vec_tail3[1] = b[ldb * 8 + 1];
                vec_tail3[5] = b[ldb * 9 + 1];
                vec_tail3[9] = b[ldb * 10 + 1];
                vec_tail3[13] = b[ldb * 11 + 1];

                vec_tail4[1] = b[ldb * 12 + 1];
                vec_tail4[5] = b[ldb * 13 + 1];
                vec_tail4[9] = b[ldb * 14 + 1];
                vec_tail4[13] = b[ldb * 15 + 1];
            }
            *reinterpret_cast<VecType *>(&Bp[0]) = vec_tail1;
            *reinterpret_cast<VecType *>(&Bp[16]) = vec_tail2;
            *reinterpret_cast<VecType *>(&Bp[32]) = vec_tail3;
            *reinterpret_cast<VecType *>(&Bp[48]) = vec_tail4;

            vsum1 = vec_sum4s(vec_tail1, vsum1);
            vsum2 = vec_sum4s(vec_tail2, vsum2);
            vsum3 = vec_sum4s(vec_tail3, vsum3);
            vsum4 = vec_sum4s(vec_tail4, vsum4);
            Bp += 64;
        }
        row_sum_eff[0] = vsum1[0];
        row_sum_eff[1] = vsum1[1];
        row_sum_eff[2] = vsum1[2];
        row_sum_eff[3] = vsum1[3];

        row_sum_eff[4] = vsum2[0];
        row_sum_eff[5] = vsum2[1];
        row_sum_eff[6] = vsum2[2];
        row_sum_eff[7] = vsum2[3];

        row_sum_eff[8] = vsum3[0];
        row_sum_eff[9] = vsum3[1];
        row_sum_eff[10] = vsum3[2];
        row_sum_eff[11] = vsum3[3];

        row_sum_eff[12] = vsum4[0];
        row_sum_eff[13] = vsum4[1];
        row_sum_eff[14] = vsum4[2];
        row_sum_eff[15] = vsum4[3];
        row_sum_eff += 16;
        N_dim -= 16;
        B += 16 * ldb;
    }

    if (N_dim > 12 && N_dim < 16) {
        tailBlock16_12xK<__vector signed char>(
                K_dim, N_dim, B, ldb, Bp, row_sum_eff);
    } else if (N_dim >= 1 && N_dim <= 12) {
        tailBlock16_8xK<__vector signed char>(
                K_dim, N_dim, B, ldb, Bp, row_sum_eff);
    }
    return 0;
}

template <typename VecType>
void pack_N16_8bit_V2(int K_Dim, int N_Dim, const int8_t *A, int lda /*  N  */,
        int8_t *Apacked, int32_t *row_sum_eff) {

    int K_block = (K_Dim + 3) & (~3);
    int N_block = (N_Dim + 3) & (~3);

    typedef __vector unsigned char vec_t;
    vec_t mask = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

    while (N_Dim >= 16) {
        const int8_t *a = A;

        size_t y = K_Dim;
        while (y >= 16) {
            //1st 4 Columns
            VecType a1 = *reinterpret_cast<const VecType *>(&a[0]);
            VecType a2 = *reinterpret_cast<const VecType *>(&a[lda]);
            VecType a3 = *reinterpret_cast<const VecType *>(&a[lda * 2]);
            VecType a4 = *reinterpret_cast<const VecType *>(&a[lda * 3]);

            VecType vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            VecType vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));
            VecType vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            VecType vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));

            VecType vx_row1 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row5 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row9 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row13 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Apacked[0]) = vx_row1;
            *reinterpret_cast<VecType *>(&Apacked[64]) = vx_row5;
            *reinterpret_cast<VecType *>(&Apacked[128]) = vx_row9;
            *reinterpret_cast<VecType *>(&Apacked[192]) = vx_row13;

            // 2nd 4 Columns
            a1 = *reinterpret_cast<const VecType *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const VecType *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const VecType *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const VecType *>(&a[lda * 7]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));

            VecType vx_row2 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row6 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row10 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row14 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Apacked[16]) = vx_row2;
            *reinterpret_cast<VecType *>(&Apacked[80]) = vx_row6;
            *reinterpret_cast<VecType *>(&Apacked[144]) = vx_row10;
            *reinterpret_cast<VecType *>(&Apacked[208]) = vx_row14;

            // 3rd 4 Columns
            a1 = *reinterpret_cast<const VecType *>(&a[lda * 8]);
            a2 = *reinterpret_cast<const VecType *>(&a[lda * 9]);
            a3 = *reinterpret_cast<const VecType *>(&a[lda * 10]);
            a4 = *reinterpret_cast<const VecType *>(&a[lda * 11]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));

            VecType vx_row3 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row7 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row11 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row15 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Apacked[32]) = vx_row3;
            *reinterpret_cast<VecType *>(&Apacked[96]) = vx_row7;
            *reinterpret_cast<VecType *>(&Apacked[160]) = vx_row11;
            *reinterpret_cast<VecType *>(&Apacked[224]) = vx_row15;

            // 4th  4 Columns
            a1 = *reinterpret_cast<const VecType *>(&a[lda * 12]);
            a2 = *reinterpret_cast<const VecType *>(&a[lda * 13]);
            a3 = *reinterpret_cast<const VecType *>(&a[lda * 14]);
            a4 = *reinterpret_cast<const VecType *>(&a[lda * 15]);

            vec_even12 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_even34 = reinterpret_cast<VecType>(
                    vec_mergee(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));
            vec_odd12 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a1),
                            reinterpret_cast<__vector int>(a2)));
            vec_odd34 = reinterpret_cast<VecType>(
                    vec_mergeo(reinterpret_cast<__vector int>(a3),
                            reinterpret_cast<__vector int>(a4)));

            VecType vx_row4 = vec_xxpermdi(vec_even12, vec_even34, 0);
            VecType vx_row8 = vec_xxpermdi(vec_odd12, vec_odd34, 0);
            VecType vx_row12 = vec_xxpermdi(vec_even12, vec_even34, 3);
            VecType vx_row16 = vec_xxpermdi(vec_odd12, vec_odd34, 3);

            *reinterpret_cast<VecType *>(&Apacked[48]) = vx_row4;
            *reinterpret_cast<VecType *>(&Apacked[112]) = vx_row8;
            *reinterpret_cast<VecType *>(&Apacked[176]) = vx_row12;
            *reinterpret_cast<VecType *>(&Apacked[240]) = vx_row16;

            y -= 16;
            Apacked += 256;
            a += 16;
        }

        while (y >= 4) {

            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda * 1]);
            int a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda * 3]);
            __vector int vec_a = {a1, a2, a3, a4};
            VecType vec_row1 = reinterpret_cast<VecType>(vec_a);
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_row1;

            // Next 4 Column
            a1 = *reinterpret_cast<const int *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 7]);
            __vector int vec_a1 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a1);
            *reinterpret_cast<VecType *>(&Apacked[16]) = vec_row1;

            // Next 4 Column
            a1 = *reinterpret_cast<const int *>(&a[lda * 8]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 9]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 10]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 11]);
            __vector int vec_a2 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a2);
            *reinterpret_cast<VecType *>(&Apacked[32]) = vec_row1;

            // Next 4 Column
            a1 = *reinterpret_cast<const int *>(&a[lda * 12]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 13]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 14]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 15]);
            __vector int vec_a3 = {a1, a2, a3, a4};
            vec_row1 = reinterpret_cast<VecType>(vec_a3);
            *reinterpret_cast<VecType *>(&Apacked[48]) = vec_row1;

            Apacked += 64;
            a += 4;
            y -= 4;
        }

        if (y <= 3 && y >= 1) {
            VecType vec_tail1
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail2
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail3
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));
            VecType vec_tail4
                    = reinterpret_cast<VecType>(vec_splats(uint8_t(0)));

            vec_tail1[0] = a[0];
            vec_tail1[4] = a[lda];
            vec_tail1[8] = a[lda * 2];
            vec_tail1[12] = a[lda * 3];

            vec_tail2[0] = a[lda * 4];
            vec_tail2[4] = a[lda * 5];
            vec_tail2[8] = a[lda * 6];
            vec_tail2[12] = a[lda * 7];

            vec_tail3[0] = a[lda * 8];
            vec_tail3[4] = a[lda * 9];
            vec_tail3[8] = a[lda * 10];
            vec_tail3[12] = a[lda * 11];

            vec_tail4[0] = a[lda * 12];
            vec_tail4[4] = a[lda * 13];
            vec_tail4[8] = a[lda * 14];
            vec_tail4[12] = a[lda * 15];
            if (y == 3) {
                vec_tail1[1] = a[1];
                vec_tail1[2] = a[2];
                vec_tail1[5] = a[lda + 1];
                vec_tail1[6] = a[lda + 2];

                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[10] = a[lda * 2 + 2];
                vec_tail1[13] = a[lda * 3 + 1];
                vec_tail1[14] = a[lda * 3 + 2];

                // Next 4 rows
                vec_tail2[1] = a[lda * 4 + 1];
                vec_tail2[2] = a[lda * 4 + 2];
                vec_tail2[5] = a[lda * 5 + 1];
                vec_tail2[6] = a[lda * 5 + 2];
                vec_tail2[9] = a[lda * 6 + 1];
                vec_tail2[10] = a[lda * 6 + 2];
                vec_tail2[13] = a[lda * 7 + 1];
                vec_tail2[14] = a[lda * 7 + 2];

                vec_tail3[1] = a[lda * 8 + 1];
                vec_tail3[2] = a[lda * 8 + 2];
                vec_tail3[5] = a[lda * 9 + 1];
                vec_tail3[6] = a[lda * 9 + 2];
                vec_tail3[9] = a[lda * 10 + 1];
                vec_tail3[10] = a[lda * 10 + 2];
                vec_tail3[13] = a[lda * 11 + 1];
                vec_tail3[14] = a[lda * 11 + 2];

                vec_tail4[1] = a[lda * 12 + 1];
                vec_tail4[2] = a[lda * 12 + 2];
                vec_tail4[5] = a[lda * 13 + 1];
                vec_tail4[6] = a[lda * 13 + 2];
                vec_tail4[9] = a[lda * 14 + 1];
                vec_tail4[10] = a[lda * 14 + 2];
                vec_tail4[13] = a[lda * 15 + 1];
                vec_tail4[14] = a[lda * 15 + 2];
            }
            if (y == 2) {
                vec_tail1[1] = a[1];
                vec_tail1[5] = a[lda + 1];
                vec_tail1[9] = a[lda * 2 + 1];
                vec_tail1[13] = a[lda * 3 + 1];

                vec_tail2[1] = a[lda * 4 + 1];
                vec_tail2[5] = a[lda * 5 + 1];
                vec_tail2[9] = a[lda * 6 + 1];
                vec_tail2[13] = a[lda * 7 + 1];

                vec_tail3[1] = a[lda * 8 + 1];
                vec_tail3[5] = a[lda * 9 + 1];
                vec_tail3[9] = a[lda * 10 + 1];
                vec_tail3[13] = a[lda * 11 + 1];

                vec_tail4[1] = a[lda * 12 + 1];
                vec_tail4[5] = a[lda * 13 + 1];
                vec_tail4[9] = a[lda * 14 + 1];
                vec_tail4[13] = a[lda * 15 + 1];
            }
            *reinterpret_cast<VecType *>(&Apacked[0]) = vec_tail1;
            *reinterpret_cast<VecType *>(&Apacked[16]) = vec_tail2;
            *reinterpret_cast<VecType *>(&Apacked[32]) = vec_tail3;
            *reinterpret_cast<VecType *>(&Apacked[48]) = vec_tail4;
            Apacked += 64;
        }
        N_Dim -= 16;
        A += lda * 16;
    }

    if (N_Dim > 12 && N_Dim < 16) {
        tailBlock16_12xK<__vector signed char>(
                K_Dim, N_Dim, A, lda, Apacked, row_sum_eff);
    } else if (N_Dim >= 1 && N_Dim <= 12) {
        tailBlock16_8xK<__vector signed char>(
                K_Dim, N_Dim, A, lda, Apacked, row_sum_eff);
    }
}

inline int pack_N16_8bit(
        dim_t k, dim_t m, const int8_t *a, dim_t lda, int8_t *ap) {
    int32_t i, j;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, m16, k4, k16;
    m16 = (m >> 4) << 4;
    k4 = (k >> 2) << 2;
    k16 = (k >> 4) << 4;
    krows = (k + 3) >> 2;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;

    // MAIN BLOCK
    for (j = 0; j < m16; j += 16) {
        for (i = 0; i < k16; i += 16) {
            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};
            vec_t swizL = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            vec_t swizR = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

            const int8_t *src = &a[lda * j + i];
            int8_t *dest = &ap[j * (krows << 2) + (i << 4)];

            V0 = *(vec_t *)&src[0 * lda];
            V1 = *(vec_t *)&src[1 * lda];
            V2 = *(vec_t *)&src[2 * lda];
            V3 = *(vec_t *)&src[3 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[64] = D1;
            *(vec_t *)&dest[128] = D2;
            *(vec_t *)&dest[192] = D3;

            V0 = *(vec_t *)&src[4 * lda];
            V1 = *(vec_t *)&src[5 * lda];
            V2 = *(vec_t *)&src[6 * lda];
            V3 = *(vec_t *)&src[7 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[16] = D0;
            *(vec_t *)&dest[80] = D1;
            *(vec_t *)&dest[144] = D2;
            *(vec_t *)&dest[208] = D3;

            V0 = *(vec_t *)&src[8 * lda];
            V1 = *(vec_t *)&src[9 * lda];
            V2 = *(vec_t *)&src[10 * lda];
            V3 = *(vec_t *)&src[11 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[32] = D0;
            *(vec_t *)&dest[96] = D1;
            *(vec_t *)&dest[160] = D2;
            *(vec_t *)&dest[224] = D3;

            V0 = *(vec_t *)&src[12 * lda];
            V1 = *(vec_t *)&src[13 * lda];
            V2 = *(vec_t *)&src[14 * lda];
            V3 = *(vec_t *)&src[15 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[48] = D0;
            *(vec_t *)&dest[112] = D1;
            *(vec_t *)&dest[176] = D2;
            *(vec_t *)&dest[240] = D3;
        }
        for (i = k16; i < k4; i += 4) {
            vec_t D0, D1, D2, D3;
            const int8_t *src = &a[lda * j + i];
            int8_t *dest = &ap[j * (krows << 2) + (i << 4)];

            *(int *)&D0[0] = *(int *)&src[0 * lda];
            *(int *)&D0[4] = *(int *)&src[1 * lda];
            *(int *)&D0[8] = *(int *)&src[2 * lda];
            *(int *)&D0[12] = *(int *)&src[3 * lda];
            *(int *)&D1[0] = *(int *)&src[4 * lda];
            *(int *)&D1[4] = *(int *)&src[5 * lda];
            *(int *)&D1[8] = *(int *)&src[6 * lda];
            *(int *)&D1[12] = *(int *)&src[7 * lda];
            *(int *)&D2[0] = *(int *)&src[8 * lda];
            *(int *)&D2[4] = *(int *)&src[9 * lda];
            *(int *)&D2[8] = *(int *)&src[10 * lda];
            *(int *)&D2[12] = *(int *)&src[11 * lda];
            *(int *)&D3[0] = *(int *)&src[12 * lda];
            *(int *)&D3[4] = *(int *)&src[13 * lda];
            *(int *)&D3[8] = *(int *)&src[14 * lda];
            *(int *)&D3[12] = *(int *)&src[15 * lda];

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (j = m16; j < m_cap; ++j) {
        for (i = 0; i < k4; ++i) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;

            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (j < m)
                ap[16 * cell + 4 * moff + koff] = a[lda * j + i];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < m16; ++j) {
        for (i = k4; i < k_cap; ++i) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;

            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (i < k)
                ap[16 * cell + 4 * moff + koff] = a[lda * j + i];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (j = m16; j < m_cap; ++j) {
        for (i = k4; i < k_cap; ++i) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;

            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (j < m && i < k)
                ap[16 * cell + 4 * moff + koff] = a[lda * j + i];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    return 0;
}

template <typename VecType, typename b_type>
inline int pack_T8_8bit_V2_signed(dim_t k, dim_t n, const b_type *b, dim_t ldb,
        uint8_t *bp, bool is_signed) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    krows = (k + 3) >> 2;
    k8 = (k >> 3) << 3;
    n8 = (n >> 3) << 3;

    const uint8_t BitFlipValue = (is_signed ? 0x80 : 0);

    VecType vmask = reinterpret_cast<VecType>(vec_splats(BitFlipValue));

    // MAIN BLOCK
    for (i = 0; i < k8; i += 8) {
        for (j = 0; j < n8; j += 8) {
            VecType V0, V1, V2, V3;
            VecType D01A, D01B, D23A, D23B;
            VecType D0, D1, D2, D3;
            vec_t swizA
                    = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
            vec_t swizB = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                    15, 31};
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};
            uint8_t *dest;

            *(signed long long *)&V0[0]
                    = *(signed long long *)&b[ldb * (i + 0) + j];
            *(signed long long *)&V1[0]
                    = *(signed long long *)&b[ldb * (i + 1) + j];
            *(signed long long *)&V2[0]
                    = *(signed long long *)&b[ldb * (i + 2) + j];
            *(signed long long *)&V3[0]
                    = *(signed long long *)&b[ldb * (i + 3) + j];
            *(signed long long *)&V0[8]
                    = *(signed long long *)&b[ldb * (i + 4) + j];
            *(signed long long *)&V1[8]
                    = *(signed long long *)&b[ldb * (i + 5) + j];
            *(signed long long *)&V2[8]
                    = *(signed long long *)&b[ldb * (i + 6) + j];
            *(signed long long *)&V3[8]
                    = *(signed long long *)&b[ldb * (i + 7) + j];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            dest = &bp[16 * ((j >> 2) * krows + (i >> 1))];

            *(vec_t *)&dest[0]
                    = reinterpret_cast<vec_t>(vec_add(D0, vmask)); //D0;
            *(vec_t *)&dest[16]
                    = reinterpret_cast<vec_t>(vec_add(D1, vmask)); // D1;
            *(vec_t *)&dest[32]
                    = reinterpret_cast<vec_t>(vec_add(D2, vmask)); //D2;
            *(vec_t *)&dest[48]
                    = reinterpret_cast<vec_t>(vec_add(D3, vmask)); //D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (i = 0; i < k8; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j] + BitFlipValue;
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k8; i < k_cap; ++i) {
        for (j = 0; j < n8; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j] + BitFlipValue;
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (i = k8; i < k_cap; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k && j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j] + BitFlipValue;
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    return 0;
}

template <typename b_type>
inline int packB_T8_8bit(dim_t k, dim_t n, const b_type *b, dim_t ldb,
        uint8_t *bp, bool is_signed) {

    if (is_signed) {
        pack_T8_8bit_V2_signed<__vector signed char, b_type>(
                k, n, b, ldb, bp, true);
    } else {
        pack_T8_8bit_V2_signed<__vector unsigned char, b_type>(
                k, n, b, ldb, bp, false);
    }
    return 0;
}

inline int pack_T8_8bit(
        dim_t k, dim_t n, const uint8_t *b, dim_t ldb, uint8_t *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    krows = (k + 3) >> 2;
    k8 = (k >> 3) << 3;
    n8 = (n >> 3) << 3;

    // MAIN BLOCK
    for (i = 0; i < k8; i += 8) {
        for (j = 0; j < n8; j += 8) {
            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
            vec_t swizB = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                    15, 31};
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};
            uint8_t *dest;

            *(signed long long *)&V0[0]
                    = *(signed long long *)&b[ldb * (i + 0) + j];
            *(signed long long *)&V1[0]
                    = *(signed long long *)&b[ldb * (i + 1) + j];
            *(signed long long *)&V2[0]
                    = *(signed long long *)&b[ldb * (i + 2) + j];
            *(signed long long *)&V3[0]
                    = *(signed long long *)&b[ldb * (i + 3) + j];
            *(signed long long *)&V0[8]
                    = *(signed long long *)&b[ldb * (i + 4) + j];
            *(signed long long *)&V1[8]
                    = *(signed long long *)&b[ldb * (i + 5) + j];
            *(signed long long *)&V2[8]
                    = *(signed long long *)&b[ldb * (i + 6) + j];
            *(signed long long *)&V3[8]
                    = *(signed long long *)&b[ldb * (i + 7) + j];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            dest = &bp[16 * ((j >> 2) * krows + (i >> 1))];

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (i = 0; i < k8; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k8; i < k_cap; ++i) {
        for (j = 0; j < n8; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (i = k8; i < k_cap; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k && j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    return 0;
}

typedef __vector int32_t v4si_t __attribute__((aligned(4)));

#define SWIZZLE_4x4 \
    { \
        result_i[0] = vec_perm(result[0], result[1], swizA); \
        result_i[1] = vec_perm(result[0], result[1], swizB); \
        result_i[2] = vec_perm(result[2], result[3], swizA); \
        result_i[3] = vec_perm(result[2], result[3], swizB); \
        result_t[0] = vec_perm(result_i[0], result_i[2], swizC); \
        result_t[1] = vec_perm(result_i[0], result_i[2], swizD); \
        result_t[2] = vec_perm(result_i[1], result_i[3], swizC); \
        result_t[3] = vec_perm(result_i[1], result_i[3], swizD); \
    }

#define SAVE_ACC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), 0); \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), 0); \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), 0);

#define SAVE_ACC1(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), 0); \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), 0); \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), 0);

#define SAVE_ACC_COND(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    if ((n_cap - n) < 3) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), \
                0); \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    if ((n_cap - n) < 2) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), \
                0); \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    if ((n_cap - n) < 1) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), \
                0);

#define SAVE_ACC1_COND(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    if ((n_cap - n) < 3) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), \
                0); \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    if ((n_cap - n) < 2) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), \
                0); \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    if ((n_cap - n) < 1) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), \
                0);

#define SAVE_ACC_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    rowC[0] = result_t[3];

#define SAVE_ACC1_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    rowC[0] = result_t[3];

#define SAVE_ACC_COND_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    if ((n_cap - n) < 3) rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    if ((n_cap - n) < 2) rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    if ((n_cap - n) < 1) rowC[0] = result_t[3];

#define SAVE_ACC1_COND_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    if ((n_cap - n) < 3) rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    if ((n_cap - n) < 2) rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    if ((n_cap - n) < 1) rowC[0] = result_t[3];

#define SET_ACC_ZERO4() \
    __builtin_mma_xxsetaccz(&acc0); \
    __builtin_mma_xxsetaccz(&acc1); \
    __builtin_mma_xxsetaccz(&acc2); \
    __builtin_mma_xxsetaccz(&acc3);

#define SET_ACC_ZERO8() \
    __builtin_mma_xxsetaccz(&acc0); \
    __builtin_mma_xxsetaccz(&acc1); \
    __builtin_mma_xxsetaccz(&acc2); \
    __builtin_mma_xxsetaccz(&acc3); \
    __builtin_mma_xxsetaccz(&acc4); \
    __builtin_mma_xxsetaccz(&acc5); \
    __builtin_mma_xxsetaccz(&acc6); \
    __builtin_mma_xxsetaccz(&acc7);

#define PREFETCH1(x, y) \
    asm volatile("dcbt %0, %1" : : "r"(x), "b"(y) : "memory");

#define MMA __builtin_mma_xvi16ger2pp

inline void gemm_kernel_16bit(dim_t m, dim_t n, dim_t k, float alpha, short *A,
        short *B, int32_t *C, float beta, dim_t ldc) {
    int32_t i;
    int32_t m_cap = (m + 3) & ~3;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    int32_t m_skip;
    int32_t n_skip = (n & 8) != (n_cap & 8);
    int32_t fastpath;
    v4si_t result[4], result_i[4], result_t[4];
    vec_t swizA = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swizB
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swizC = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swizD
            = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    fastpath = ((alpha == 1.0) && (beta == 0.0));

    /* Loop for multiples of 8 */
    i = n_cap >> 3;
    while (i) {
        int32_t j;
        int32_t *CO;
        short *AO;
        CO = C;
        C += ldc << 3;
        AO = A;
        PREFETCH1(A, 128);
        PREFETCH1(A, 256);
        /* Loop for m >= 16. */
        j = m_cap >> 4;
        m_skip = (m >> 4) != (m_cap >> 4);
        while (j) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                MMA(&acc4, rowA[2], rowB[0]);
                MMA(&acc5, rowA[2], rowB[1]);
                MMA(&acc6, rowA[3], rowB[0]);
                MMA(&acc7, rowA[3], rowB[1]);
                rowA += 4;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc3, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc5, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc6, 0);
                        SAVE_ACC1_COND_ABSC(&acc7, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC1_ABSC(&acc7, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc3, 0);
                } else {
                    SAVE_ACC1(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc5, 0);
                } else {
                    SAVE_ACC1(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc6, 0);
                        SAVE_ACC1_COND(&acc7, 0);
                    }
                } else {
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC1(&acc7, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 4);
            BO += (k_cap << 3);
            --j;
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                rowA += 2;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc2, 0);
                        SAVE_ACC1_COND_ABSC(&acc3, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc2, 0);
                        SAVE_ACC1_COND(&acc3, 0);
                    }
                } else {
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC1(&acc3, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 3);
            BO += (k_cap << 3);
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l = 0;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                rowA += 1;
                rowB += 2;
            }

            if (fastpath) {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC_ABSC(&acc0, 0);
                        SAVE_ACC1_COND_ABSC(&acc1, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
            } else {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                            + alpha * result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                            + alpha * result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                            + alpha * result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC(&acc0, 0);
                        SAVE_ACC1_COND(&acc1, 0);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC1(&acc1, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 2);
            BO += (k_cap << 3);
        }

    endloop8:
        B += k_cap << 3;
        i -= 1;
    }

    if (n_cap & 4) {
        int32_t j;
        int32_t *CO;
        short *AO;
        CO = C;
        C += ldc << 2;
        AO = A;
        int32_t n_skip = (n != n_cap);
        /* Loop for m >= 32. */
        m_skip = (m >> 5) != (m_cap >> 5);
        for (j = 0; j < (m_cap >> 5); j++) {
            short *BO = B;
            short *A1 = AO + (16 * k_cap);
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowA1 = (vec_t *)A1;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                MMA(&acc4, rowA1[0], rowB[0]);
                MMA(&acc5, rowA1[1], rowB[0]);
                MMA(&acc6, rowA1[2], rowB[0]);
                MMA(&acc7, rowA1[3], rowB[0]);
                rowA += 4;
                rowA1 += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    SAVE_ACC_COND_ABSC(&acc3, 12);
                    SAVE_ACC_COND_ABSC(&acc4, 16);
                    SAVE_ACC_COND_ABSC(&acc5, 20);
                    SAVE_ACC_COND_ABSC(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc4, 0);
                    SAVE_ACC_ABSC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC_ABSC(&acc7, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    SAVE_ACC_COND(&acc3, 12);
                    SAVE_ACC_COND(&acc4, 16);
                    SAVE_ACC_COND(&acc5, 20);
                    SAVE_ACC_COND(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = beta * CO[0 * ldc + 28 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii]
                                        = beta * CO[1 * ldc + 28 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii]
                                        = beta * CO[2 * ldc + 28 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii]
                                        = beta * CO[3 * ldc + 28 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC(&acc4, 0);
                    SAVE_ACC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC(&acc7, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 5;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 16) != (m_cap & 16);

        if (m_cap & 16) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                rowA += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = beta * CO[0 * ldc + 12 + ii]
                                    + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii]
                                        = beta * CO[1 * ldc + 12 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii]
                                        = beta * CO[2 * ldc + 12 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii]
                                        = beta * CO[3 * ldc + 12 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 4;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                rowA += 2;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc1, 4);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = beta * CO[0 * ldc + 4 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii]
                                        = beta * CO[1 * ldc + 4 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii]
                                        = beta * CO[2 * ldc + 4 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii]
                                        = beta * CO[3 * ldc + 4 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc1, 4);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                }
            }
            CO += 8;
            AO += k_cap << 3;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0;
            __builtin_mma_xxsetaccz(&acc0);
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                rowA += 1;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                }
            } else {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = beta * CO[0 * ldc + ii] + alpha * result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                } else {
                    SAVE_ACC(&acc0, 0);
                }
            }
            CO += 4;
            AO += k_cap << 2;
            BO += k_cap << 2;
        }

    endloop4:
        B += k_cap << 2;
    }
    return;
}

#undef MMA
#define MMA __builtin_mma_xvi8ger4pp
inline void gemm_kernel_8bit(dim_t m, dim_t n, dim_t k, float alpha, int8_t *A,
        uint8_t *B, int32_t *C, float beta, dim_t ldc) {
    int32_t i;
    int32_t m_cap = (m + 3) & ~3;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    int32_t m_skip;
    int32_t n_skip = (n & 8) != (n_cap & 8);
    int32_t fastpath;
    v4si_t result[4], result_i[4], result_t[4];
    vec_t swizA = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swizB
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swizC = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swizD
            = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    fastpath = ((alpha == 1.0) && (beta == 0.0));

    /* Loop for multiples of 8 */
    i = n_cap >> 3;
    while (i) {
        int32_t j;
        int32_t *CO;
        int8_t *AO;
        CO = C;
        C += ldc << 3;
        AO = A;
        PREFETCH1(A, 128);
        PREFETCH1(A, 256);
        /* Loop for m >= 16. */
        j = m_cap >> 4;
        m_skip = (m >> 4) != (m_cap >> 4);
        while (j) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                MMA(&acc4, rowA[2], rowB[0]);
                MMA(&acc5, rowA[2], rowB[1]);
                MMA(&acc6, rowA[3], rowB[0]);
                MMA(&acc7, rowA[3], rowB[1]);
                rowA += 4;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc3, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc5, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc6, 0);
                        SAVE_ACC1_COND_ABSC(&acc7, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC1_ABSC(&acc7, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc3, 0);
                } else {
                    SAVE_ACC1(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc5, 0);
                } else {
                    SAVE_ACC1(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc6, 0);
                        SAVE_ACC1_COND(&acc7, 0);
                    }
                } else {
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC1(&acc7, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 4);
            BO += (k_cap << 3);
            --j;
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                rowA += 2;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc2, 0);
                        SAVE_ACC1_COND_ABSC(&acc3, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc2, 0);
                        SAVE_ACC1_COND(&acc3, 0);
                    }
                } else {
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC1(&acc3, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 3);
            BO += (k_cap << 3);
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l = 0;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                rowA += 1;
                rowB += 2;
            }

            if (fastpath) {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC_ABSC(&acc0, 0);
                        SAVE_ACC1_COND_ABSC(&acc1, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
            } else {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                            + alpha * result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                            + alpha * result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                            + alpha * result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC(&acc0, 0);
                        SAVE_ACC1_COND(&acc1, 0);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC1(&acc1, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 2);
            BO += (k_cap << 3);
        }

    endloop8:
        B += k_cap << 3;
        i -= 1;
    }

    if (n_cap & 4) {
        int32_t j;
        int32_t *CO;
        int8_t *AO;
        CO = C;
        C += ldc << 2;
        AO = A;
        int32_t n_skip = (n != n_cap);
        /* Loop for m >= 32. */
        m_skip = (m >> 5) != (m_cap >> 5);
        for (j = 0; j < (m_cap >> 5); j++) {
            uint8_t *BO = B;
            int8_t *A1 = AO + (16 * k_cap);
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowA1 = (vec_t *)A1;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                MMA(&acc4, rowA1[0], rowB[0]);
                MMA(&acc5, rowA1[1], rowB[0]);
                MMA(&acc6, rowA1[2], rowB[0]);
                MMA(&acc7, rowA1[3], rowB[0]);
                rowA += 4;
                rowA1 += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    SAVE_ACC_COND_ABSC(&acc3, 12);
                    SAVE_ACC_COND_ABSC(&acc4, 16);
                    SAVE_ACC_COND_ABSC(&acc5, 20);
                    SAVE_ACC_COND_ABSC(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc4, 0);
                    SAVE_ACC_ABSC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC_ABSC(&acc7, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    SAVE_ACC_COND(&acc3, 12);
                    SAVE_ACC_COND(&acc4, 16);
                    SAVE_ACC_COND(&acc5, 20);
                    SAVE_ACC_COND(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = beta * CO[0 * ldc + 28 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii]
                                        = beta * CO[1 * ldc + 28 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii]
                                        = beta * CO[2 * ldc + 28 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii]
                                        = beta * CO[3 * ldc + 28 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC(&acc4, 0);
                    SAVE_ACC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC(&acc7, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 5;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 16) != (m_cap & 16);

        if (m_cap & 16) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                rowA += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = beta * CO[0 * ldc + 12 + ii]
                                    + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii]
                                        = beta * CO[1 * ldc + 12 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii]
                                        = beta * CO[2 * ldc + 12 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii]
                                        = beta * CO[3 * ldc + 12 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 4;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                rowA += 2;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc1, 4);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = beta * CO[0 * ldc + 4 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii]
                                        = beta * CO[1 * ldc + 4 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii]
                                        = beta * CO[2 * ldc + 4 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii]
                                        = beta * CO[3 * ldc + 4 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc1, 4);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                }
            }
            CO += 8;
            AO += k_cap << 3;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0;
            __builtin_mma_xxsetaccz(&acc0);
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                rowA += 1;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                }
            } else {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = beta * CO[0 * ldc + ii] + alpha * result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                } else {
                    SAVE_ACC(&acc0, 0);
                }
            }
            CO += 4;
            AO += k_cap << 2;
            BO += k_cap << 2;
        }

    endloop4:
        B += k_cap << 2;
    }
    return;
}

} // namespace impl
} // namespace dnnl

#endif
