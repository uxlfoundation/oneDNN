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

// The arithmetic keeps the Python model's evaluation order.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "internal/utils.hpp"
#include "ngen_utils.hpp"
#include "selector/gbr.hpp"

GEMMSTONE_NAMESPACE_START

namespace gbr {

namespace {

using ngen::utils::bsf;

constexpr int firstFlag(int i = 0) { // the numeric features; then f_TOKEN
    return featureNames[i][0] == 'f' && featureNames[i][1] == '_'
            ? i
            : firstFlag(i + 1);
}
constexpr int nNumeric = firstFlag();
static_assert(nFeatures - nNumeric <= 32, "Kernel::flagBits");

// What a row's strategy states; K quantities in bytes.
struct Kernel {
    int64_t um, un, wgM, wgN, wgK, tpe, chunkA, chunkB, copiesA, copiesB, pfA,
            pfB;
    double distA, distB;
    uint32_t flagBits;
};

// A load word "aX<chunk>[x<copies>][+m<pf>@<dist>]" (elements) -> bytes.
const char *parseWord(const char *s, double esize, int64_t &chunk,
        int64_t &copies, int64_t &pf, double &dist) {
    char *e;
    chunk = (int64_t)((double)std::strtol(s + 2, &e, 10) * esize);
    copies = 1;
    pf = 0;
    dist = 0;
    if (*e == 'x') copies = std::strtol(e + 1, &e, 10);
    if (*e == '+') {
        pf = (int64_t)((double)std::strtol(e + 2, &e, 10) * esize);
        dist = (double)std::strtol(e + 1, &e, 10) * esize;
    }
    return e;
}

Kernel parse(const Row &r) {
    Kernel d;
    d.um = r.um;
    d.un = r.un;
    d.tpe = r.threadsPerEU;
    const char *s = parseWord(
            r.strategy, r.esizeA, d.chunkA, d.copiesA, d.pfA, d.distA);
    s = parseWord(s + 1, r.esizeB, d.chunkB, d.copiesB, d.pfB, d.distB);
    d.wgM = r.wgM;
    d.wgN = r.wgN;
    d.wgK = r.wgK;
    s = std::strstr(s, " wg "); // the flags follow its MxN[xK]
    gemm_assert(s != nullptr, "gbr: strategy without wg");
    const char *e = s + 4 + std::strcspn(s + 4, " ");
    d.flagBits = 0;
    for (const char *t = e; *t;) {
        t += *t == ' ';
        size_t n = std::strcspn(t, " ");
        for (int b = 0; b < nFeatures - nNumeric; b++) {
            const char *f = featureNames[nNumeric + b] + 2;
            if (std::strlen(f) == n && !std::strncmp(f, t, n))
                d.flagBits |= 1u << b;
        }
        t += n;
    }
    return d;
}

// C as bf16, as trained (tf32's f32 C included).
constexpr double cBytes = 2;

void features(const Kernel &d, const Shape &p, double esizeB, double *F) {
    int64_t m = p.m, n = p.n, qb = p.batch;
    int64_t qk = (int64_t)std::ceil((double)p.k * esizeB); // K in bytes
    int64_t c = p.eus * d.tpe;
    int64_t threads = d.wgM * d.wgN * d.wgK;
    int64_t tm = div_up(m, d.um), tn = div_up(n, d.un);
    int64_t T = tm * tn * qb;
    double wv = (double)(T * d.wgK) / (double)c;
    int64_t wgs = div_up(tm, d.wgM) * div_up(tn, d.wgN);
    int64_t moved = m * tn + n * tm; // traffic / (qk * qb)
    double traffic = (double)qk * (double)moved * (double)qb;
    double kpt = (double)qk / (double)d.wgK, chunk = (double)d.chunkB;
    double tcomp = (double)((int64_t)std::ceil(wv) * d.um * d.un) * kpt;
    int64_t gM = div_up(tn, d.wgN); // wg rows along n: passes over the weights
    double wvg = (double)(wgs * d.wgM * d.wgN * d.wgK * qb) / (double)c;
    int i = 0;
    F[i++] = std::log2((double)n); // lm, ln: matmul M, N
    F[i++] = std::log2((double)m);
    F[i++] = std::log2((double)qk);
    F[i++] = (double)d.wgK;
    F[i++] = std::log2((double)d.chunkA);
    F[i++] = std::log2((double)d.chunkB);
    F[i++] = (double)d.copiesA;
    F[i++] = (double)d.copiesB;
    F[i++] = std::log2((double)(d.pfA + 1));
    F[i++] = std::log2((double)(d.pfB + 1));
    F[i++] = std::log2(d.distA + 1);
    F[i++] = std::log2(d.distB + 1);
    F[i++] = (double)d.um;
    F[i++] = (double)d.un;
    F[i++] = std::log2((double)(d.um * d.un));
    F[i++] = (double)(d.um * d.un) / (double)(d.um + d.un);
    F[i++] = (double)d.wgM;
    F[i++] = (double)d.wgN;
    F[i++] = (double)threads;
    F[i++] = (double)T / (double)(wgs * d.wgM * d.wgN);
    F[i++] = std::log2((double)(wgs * qb));
    F[i++] = std::min(1.0, (double)tn / (double)d.wgN)
            * std::min(1.0, (double)tm / (double)d.wgM);
    F[i++] = std::log2(kpt);
    F[i++] = std::log2(std::max(1.0, kpt / chunk));
    F[i++] = std::log2((double)moved / (double)(m + n));
    F[i++] = std::log2(tcomp);
    F[i++] = std::log2(
            traffic + cBytes * (double)n * (double)m * (double)qb);
    F[i++] = std::log2(tcomp
            / ((double)n * (double)m * (double)qk * (double)qb / (double)c));
    F[i++] = wvg - std::floor(wvg);
    F[i++] = (double)gM;
    F[i++] = (double)std::min(9, bsf((uint64_t)qk));
    F[i++] = std::fmod((double)qk, chunk) / chunk;
    for (int b = 0; b < nFeatures - nNumeric; b++)
        F[i++] = (double)((d.flagBits >> b) & 1);
}

// Trees: features bucketed once per row, then QuickScorer leaf masks.
struct Split {
    uint16_t bucket; // j: the node fails for rows with bucket(x) > j
    uint16_t tree;
    uint64_t keep; // the leaves still reachable when it fails
};

struct FeatureSplits {
    const double *edge, *edgeEnd; // sorted thresholds, in Scorer::thr
    std::vector<uint16_t> tree; // by bucket
    std::vector<uint64_t> keep;
    std::vector<uint32_t> fails; // per bucket b: the splits with bucket < b
};

struct Forest {
    std::vector<std::vector<Split>> splits; // per feature (decode only)
    std::vector<FeatureSplits> byFeature;
    std::vector<uint32_t> leafFirst; // per tree: its first leaf in leaves
    std::vector<double> leaves; // left to right, tree by tree
};

// Decodes one subtree of tree t; returns its leaf count.
int build(Forest &f, const Scorer &s, uint32_t &i, int t, int firstLeaf,
        const std::vector<uint8_t> &feat, const uint32_t *first, double scale) {
    auto read = [&](int n) {
        uint32_t v = 0;
        for (int j = 0; j < n; j++, i++)
            v |= (uint32_t)((s.blob[i >> 6] >> (i & 63)) & 1) << j;
        return v;
    };
    if (!read(1)) {
        uint32_t q = read(s.leafBits);
        int64_t v = (q >> (s.leafBits - 1))
                ? (int64_t)q - ((int64_t)1 << s.leafBits)
                : (int64_t)q;
        f.leaves.push_back((double)v * scale);
        return 1;
    }
    uint32_t k = read(s.thrBits);
    int nl = build(f, s, i, t, firstLeaf, feat, first, scale);
    uint64_t left = (nl >= 64 ? ~0ull : ((1ull << nl) - 1)) << firstLeaf;
    f.splits[feat[k]].push_back(
            {(uint16_t)(k - first[feat[k]]), (uint16_t)t, ~left});
    return nl + build(f, s, i, t, firstLeaf + nl, feat, first, scale);
}

Forest decode(const Scorer &s) {
    Forest f;
    f.splits.resize(nFeatures);
    // feature j: thr[first[j], first[j + 1])
    uint32_t first[nFeatures + 1] = {};
    for (int j = 0; j < nFeatures; j++)
        first[j + 1] = first[j] + s.thrCount[j];
    std::vector<uint8_t> feat(first[nFeatures]);
    for (int j = 0; j < nFeatures; j++)
        std::fill(feat.begin() + first[j], feat.begin() + first[j + 1], j);
    uint32_t i = 0;
    for (int t = 0; t < s.nTrees; t++) {
        f.leafFirst.push_back((uint32_t)f.leaves.size());
        build(f, s, i, t, 0, feat, first, s.scale[t]);
    }
    f.byFeature.resize(nFeatures);
    for (int j = 0; j < nFeatures; j++) {
        auto &v = f.splits[j];
        std::stable_sort(
                v.begin(), v.end(), [](const Split &a, const Split &b) {
            return a.bucket < b.bucket;
        });
        auto &out = f.byFeature[j];
        out.edge = s.thr + first[j];
        out.edgeEnd = s.thr + first[j + 1];
        out.fails.assign(s.thrCount[j] + 1, 0);
        for (const auto &sp : v) {
            out.tree.push_back(sp.tree);
            out.keep.push_back(sp.keep);
            out.fails[sp.bucket + 1]++;
        }
        for (size_t b = 1; b < out.fails.size(); b++)
            out.fails[b] += out.fails[b - 1];
    }
    f.splits.clear();
    return f;
}

// The scores of a call's rows; the trees add in the model's order.
void predict(const Forest &f, const double *X, size_t nrows, double *out) {
    size_t nTrees = f.leafFirst.size();
    std::vector<uint16_t> B(nrows * nFeatures);
    uint16_t bmin[nFeatures];
    for (int j = 0; j < nFeatures; j++) {
        const auto &e = f.byFeature[j];
        bmin[j] = 0xffff;
        for (size_t r = 0; r < nrows; r++) {
            auto b = (uint16_t)(std::lower_bound(e.edge, e.edgeEnd,
                                        X[r * nFeatures + j])
                    - e.edge);
            B[r * nFeatures + j] = b;
            bmin[j] = std::min(bmin[j], b);
        }
    }
    uint64_t base[maxTrees], mask[maxTrees];
    for (size_t t = 0; t < nTrees; t++)
        base[t] = ~0ull;
    for (int j = 0; j < nFeatures; j++) {
        const auto &fs = f.byFeature[j];
        for (uint32_t k = 0; k < fs.fails[bmin[j]]; k++)
            base[fs.tree[k]] &= fs.keep[k];
    }
    for (size_t r = 0; r < nrows; r++) {
        memcpy(mask, base, nTrees * sizeof(uint64_t));
        for (int j = 0; j < nFeatures; j++) {
            const auto &fs = f.byFeature[j];
            uint32_t k1 = fs.fails[B[r * nFeatures + j]];
            for (uint32_t k = fs.fails[bmin[j]]; k < k1; k++)
                mask[fs.tree[k]] &= fs.keep[k];
        }
        double total = 0.0;
        for (size_t t = 0; t < nTrees; t++)
            total += f.leaves[f.leafFirst[t] + bsf(mask[t])];
        out[r] = total;
    }
}

const std::vector<Forest> &forests() {
    static const std::vector<Forest> all = [] {
        std::vector<Forest> v;
        for (int i = 0; i < nScorers; i++)
            v.push_back(decode(scorers[i]));
        return v;
    }();
    return all;
}

} // namespace

bool deviceOK(const Scorer &s, uint32_t deviceId, ngen::HW hw) {
    if (s.nDeviceIds == 0) return s.hw == hw;
    return std::find(s.deviceIds, s.deviceIds + s.nDeviceIds, deviceId)
            != s.deviceIds + s.nDeviceIds;
}

void score(int scorer, const Row *rows, int n, const Shape &shape, double *out) {
    if (n == 0) return;
    std::vector<double> X(size_t(n) * nFeatures);
    for (int i = 0; i < n; i++)
        features(parse(rows[i]), shape, rows[i].esizeB, &X[i * nFeatures]);
    predict(forests()[scorer], X.data(), n, out);
    for (int i = 0; i < n; i++)
        out[i] = -out[i];
}

} // namespace gbr

GEMMSTONE_NAMESPACE_END
