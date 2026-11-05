/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GEMMSTONE_SELECTOR_GBR_HPP
#define GEMMSTONE_SELECTOR_GBR_HPP

// The GBR ranker of the G rows (gbr.db); gbr_data.cpp is generated.

#include <cstdint>

#include "gemmstone/config.hpp"
#include "ngen_core.hpp"

GEMMSTONE_NAMESPACE_START

namespace gbr {

// The numeric features, then f_TOKEN (the token in the flags).
constexpr const char *featureNames[] = {"lm", "ln", "lk", "c_wgK", "lchunkA",
        "lchunkB", "copiesA", "copiesB", "lpfA", "lpfB", "ldistA", "ldistB",
        "um", "un", "larea", "tai", "wgM", "wgN", "threads", "wgeff", "lwgs",
        "wgfill", "lkpt", "lkit", "ltraf", "ltcomp", "ltmem", "lslack", "wfrac",
        "gM", "ctzk", "kt_chunk", "f_xaf", "f_st", "f_dm", "f_fx", "f_rr",
        "f_sm", "f_sn", "f_bk0", "f_of", "f_pfaux", "f_pt", "f_bo", "f_hi",
        "f_li"};
constexpr int nFeatures = sizeof(featureNames) / sizeof(*featureNames);

constexpr bool same(const char *a, const char *b) {
    return *a == *b && (!*a || same(a + 1, b + 1));
}

// The engine's limits (64-bit leaf masks); the data static_asserts its trees.
constexpr int maxTrees = 1024, maxLeaves = 64;

struct Scorer {
    uint16_t nTrees;
    uint8_t thrBits, leafBits;
    const uint16_t *thrCount; // per feature: its thresholds in thr
    const double *thr; // sorted by (feature, value)
    const double *scale; // per tree: leaf scale
    const uint64_t *blob; // the packed trees back to back, LSB-first bits
    // The gate: the PCI device ids (CL_DEVICE_ID_INTEL), or with none the core.
    int nDeviceIds;
    const uint32_t *deviceIds;
    ngen::HW hw;
};

// Whether a scorer's gate admits the device.
bool deviceOK(const Scorer &s, uint32_t deviceId, ngen::HW hw);

// The gate: jit M, K multiples (elements); mx: K bytes of the narrower of A, B
extern const int nMult, kMult, mxKBytes;
extern const int mxGroup; // the 2D scale k-group step
extern const int nScorers;
extern const Scorer scorers[]; // a G row's model parameter indexes it

// The problem in gemmstone's m, n (matmul N, M).
struct Shape {
    int64_t m, n, k, batch, eus;
};

struct Row {
    const char *strategy; // the entry's: "aX.. aY.. aB wg MxN[xK] flags.."
    int um, un, wgM, wgN, wgK; // driverInfo's
    int threadsPerEU;
    double esizeA, esizeB; // bytes per element in memory
};

// out[i]: rows[i]'s score, lower = better.
void score(int scorer, const Row *rows, int n, const Shape &shape, double *out);

} // namespace gbr

GEMMSTONE_NAMESPACE_END

#endif
