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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "selector/gbr_catalog.hpp"

GEMMSTONE_NAMESPACE_START

namespace gbr {

namespace {

// Element-count field v of e0-byte elements on e-byte elements.
int kEl(long v, double e0, double e) {
    return std::max(1, int(v * e0 / e));
}

int chunkEl(long v, double e0, double e) { // a k-load: at least 32 bytes
    return int(std::max(32., v * e0) / e);
}

// "aX[/U]<chunk>..[+P<pf>[,stride]@<dist>..]": the chunk; pf and dist with a dist only.
std::string word(const std::string &w, double e0, double e) {
    const char *s = w.c_str(), *c = s + ((w.size() > 2 && w[2] == '/') ? 4 : 2);
    char *j;
    long chunk = std::strtol(c, &j, 10);
    std::string out = std::string(s, c) + std::to_string(chunkEl(chunk, e0, e));
    const char *p = std::strchr(j, '+');
    if (!p || p[1] == '\0' || p[1] == '+') return out + j;
    char *b, *a, *d;
    long pf = std::strtol(p + 2, &b, 10);
    a = b;
    if (*a == ',') std::strtol(a + 1, &a, 10);
    if (b == p + 2 || *a != '@') return out + j;
    long dist = std::strtol(a + 1, &d, 10);
    return out + std::string(j, p + 2 - j) + std::to_string(int(pf * e0 / e))
            + std::string(b, a + 1) + std::to_string(kEl(dist, e0, e)) + d;
}

// The strategy on e-byte elements: the A, B words, then the first sb/b, k, pk, ki.
std::string retype(const std::string &s, double e0, double e) {
    std::string out;
    bool barrier = false, unrollK = false, kPad = false, kChunk = false;
    size_t i = 0;
    for (int t = 0; i < s.size(); t++) {
        size_t j = s.find(' ', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        auto at = [&](size_t n, bool &done) {
            if (done || tok.size() <= n || !std::isdigit((unsigned char)tok[n]))
                return;
            char *d;
            long v = std::strtol(tok.c_str() + n, &d, 10);
            tok = tok.substr(0, n) + std::to_string(kEl(v, e0, e)) + d;
            done = true;
        };
        if (t < 2)
            tok = word(tok, e0, e);
        else if (tok.compare(0, 2, "sb") == 0) at(2, barrier);
        else if (tok.compare(0, 2, "pk") == 0) at(2, kPad);
        else if (tok.compare(0, 2, "ki") == 0) at(2, kChunk);
        else if (tok[0] == 'b') at(1, barrier);
        else if (tok[0] == 'k') at(1, unrollK);
        out += (t ? " " : "") + tok;
        i = j + 1;
    }
    return out;
}

double bytes(const char *prec) {
    return charToType(prec[0]).bits() / 8.0;
}

std::string key(const kcatalog::Entry &e) {
    return std::to_string(e.driverInfo.unroll[LoopM]) + " "
            + std::to_string(e.driverInfo.unroll[LoopN]) + " 1 0 " + e.strategy;
}

constexpr const char *derived[] = {"Q", "E", "T", "O"};

} // namespace

ExpandedCatalog::ExpandedCatalog(const kcatalog::Catalog &db) : catalog_(db) {
    for (int i = 0; i < db.entryCount; i++) {
        auto e = db.entries[i];
        int mask = e.model.paramCount > 1 ? int(e.model.params[1]) : 0;
        e.model.paramCount = 1;
        entries_.push_back(e);
        double e0 = bytes(e.selector.precisions[0]);
        for (int b = 0; b < int(sizeof(derived) / sizeof(*derived)); b++) {
            if (!(mask >> b & 1)) continue;
            auto d = e;
            double eb = bytes(derived[b]);
            d.selector.precisions[0] = d.selector.precisions[1] = derived[b];
            strategies_.push_back(retype(e.strategy, e0, eb));
            d.strategy = strategies_.back().c_str();
            d.driverInfo.unroll[LoopK] = int(e.driverInfo.unroll[LoopK] * e0 / eb);
            if (*derived[b] == 'T') d.driverInfo.alignment[2] = 4; // f32 C
            entries_.push_back(d);
        }
    }
    std::vector<std::string> keys;
    std::vector<size_t> order(entries_.size());
    for (size_t i = 0; i < entries_.size(); i++) {
        keys.push_back(key(entries_[i]));
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
        auto &x = entries_[i], &y = entries_[j];
        if (x.selector < y.selector) return true;
        if (y.selector < x.selector) return false;
        return keys[i] != keys[j] ? keys[i] < keys[j] : x.model.params[0] < y.model.params[0];
    });
    std::vector<kcatalog::Entry> sorted;
    sorted.reserve(order.size());
    for (auto i : order)
        sorted.push_back(entries_[i]);
    entries_ = std::move(sorted);
    catalog_.entries = entries_.data();
    catalog_.entryCount = int(entries_.size());
}

} // namespace gbr

GEMMSTONE_NAMESPACE_END
