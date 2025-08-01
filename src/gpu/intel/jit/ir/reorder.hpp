/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_REORDER_HPP
#define GPU_INTEL_JIT_IR_REORDER_HPP

#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/reorder/normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Implements reorder between GRF buffers in given layouts. Conversion between
// data types is supported.
class reorder_t : public func_impl_t {
public:
    IR_DECL_TYPE(reorder_t)

    static func_t make(layout_t src_layout, layout_t dst_layout) {
        reorder::normalize(src_layout, dst_layout);
        return func_t(
                new reorder_t(std::move(src_layout), std::move(dst_layout)));
    }

    std::string str() const override {
        ostringstream_t oss;
        oss << "reorder[" << src_layout << ", " << dst_layout << "]";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst_buf, 0)
    IR_DEFINE_ARG_GET(src_buf, 1)

    layout_t src_layout;
    layout_t dst_layout;

private:
    reorder_t(layout_t src_layout, layout_t dst_layout)
        : func_impl_t(_type_info())
        , src_layout(std::move(src_layout))
        , dst_layout(std::move(dst_layout)) {}
};

inline stmt_t create_reorder_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf) {
    gpu_assert(src.ndims() == dst.ndims()) << "Layouts are incompatible.";
    gpu_assert(src.elems() == dst.elems()) << "Layouts are incompatible.";
    auto func = reorder_t::make(src, dst);
    return func.call({dst_buf, src_buf});
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
