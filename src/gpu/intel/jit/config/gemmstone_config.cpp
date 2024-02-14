/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/jit/config/gemmstone_config.hpp"
#include "gemmstone/problem.hpp"

namespace gemmstone {

BinaryOp PostOpsProblem::toBinaryOp(const PostOps::entry_t &e) {
    using namespace dnnl::impl;
    switch (e.as_binary().alg) {
        case alg_kind::binary_add: return BinaryOp::Add;
        case alg_kind::binary_sub: return BinaryOp::Sub;
        case alg_kind::binary_mul: return BinaryOp::Mul;
        case alg_kind::binary_div: return BinaryOp::Div;
        case alg_kind::binary_min: return BinaryOp::Min;
        case alg_kind::binary_max: return BinaryOp::Max;
        case alg_kind::binary_prelu: return BinaryOp::Prelu;
        default: gpu_error_not_expected();
    }
    return BinaryOp::Add;
}

} // namespace gemmstone
