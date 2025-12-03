/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_PROBLEM_HPP
#define GPU_INTEL_JIT_IR_PROBLEM_HPP

#include <string>
#include <vector>

#include "gpu/intel/jit/dsl/tensor.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

enum class tensor_kind_t {
    undef,
    src,
    wei,
    dst,
    bias,
    a,
    b,
    c,
};

std::string to_string(tensor_kind_t tensor);

using pvar_t = dsl::idx_t;

namespace pvars {
extern pvar_t g;
extern pvar_t ic;
extern pvar_t id;
extern pvar_t ih;
extern pvar_t iw;
extern pvar_t kd;
extern pvar_t kh;
extern pvar_t kw;
extern pvar_t mb;
extern pvar_t oc;
extern pvar_t od;
extern pvar_t oh;
extern pvar_t ow;
extern pvar_t sd;
extern pvar_t sh;
extern pvar_t sw;
extern pvar_t dd;
extern pvar_t dh;
extern pvar_t dw;
extern pvar_t pd;
extern pvar_t ph;
extern pvar_t pw;
extern pvar_t b;
extern pvar_t m;
extern pvar_t n;
extern pvar_t k;
} // namespace pvars

template <typename T>
using pvar_map_t = dsl::idx_map_t<T>;
using tile_t = dsl::tile_t;
using coord_t = dsl::coord_t;
using icoord_t = dsl::icoord_t;

struct tile_coord_t {
    tile_t tile;
    coord_t coord;
    bool is_valid;

    tile_coord_t(bool is_valid = true) : is_valid(is_valid) {}
    tile_coord_t(const tile_t &tile, const coord_t &coord = {})
        : tile(tile), coord(coord), is_valid(true) {}
    bool is_invalid() const { return !is_valid; }
    size_t size() const { return tile.size(); }
    dim_t elems() const { return tile.elems(); }
    bool has_zero_coord() const {
        for (auto &d : coord) {
            if (!is_zero(coord.at(d))) return false;
        }
        return true;
    }

    tile_coord_t sub(
            const tile_t &sub_tile, const coord_t &sub_coord = {}) const {
        coord_t new_coord = coord;
        for (auto &d : sub_coord)
            new_coord[d] += sub_coord[d];
        return tile_coord_t(sub_tile, new_coord);
    }

    tile_coord_t sub(const tile_coord_t &sub_tile_coord) const {
        return sub(sub_tile_coord.tile, sub_tile_coord.coord);
    }

    std::string str() const {
        if (is_invalid()) return "tile: (invalid)";

        ostringstream_t oss;
        oss << "tile: " << tile.str();
        if (!has_zero_coord()) oss << " coord: " << coord.str();
        return oss.str();
    }

    XE_DEFINE_DUMP()

    static tile_coord_t invalid() { return tile_coord_t(false); }
};

template <typename F>
void for_each(const tile_t &tile, const F &f) {
    ir_utils::for_each(tile.values(),
            [&](const std::vector<dim_t> &idxs) { f(icoord_t(idxs)); });
}

bool is_input_spatial(const pvar_t &pvar);
bool is_output_spatial(const pvar_t &pvar);
bool is_kernel_spatial(const pvar_t &pvar);
bool is_dilation(const pvar_t &pvar);
bool is_padding(const pvar_t &pvar);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::gpu::intel::jit::pvar_t> {
    size_t operator()(const dnnl::impl::gpu::intel::jit::pvar_t &pvar) const {
        return pvar.get_hash();
    }
};
} // namespace std

#endif
