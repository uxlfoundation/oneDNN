/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_REORDER_JIT_CONFIG_HPP
#define GPU_INTEL_REORDER_JIT_CONFIG_HPP

#include "gpu/intel/jit/ir/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {
namespace jit {

using namespace intel::jit;
extern pvar_t pvars[];

// Parameters for kernel generation.
class config_t : public prim_config_t {
public:
    std::string str() const override {
        ostringstream_t ss;
        ss << src_layout().user().str() << " -> " << dst_layout().user().str();
        return ss.str();
    }

    tile_t shape(bool pad) const override { return {}; }

    const std::vector<pvar_t> &index_dims() const override {
        static const std::vector<pvar_t> null {};
        return null;
    }

    int pad_block(const pvar_t &d) const override { return 0; }

    int simd() const { return options().simd(); }
    compute::nd_range_t nd_range() const;
    const std::vector<tile_t> &tiles() const { return tiles_; }
    const std::array<tile_t, 3> &grid() const { return grid_; }

    config_t(const dsl::kernel::options_t &ec, layout_t src, layout_t dst);

private:
    std::vector<tile_t> tiles_;
    std::array<tile_t, 3> grid_;
};

} // namespace jit
} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
