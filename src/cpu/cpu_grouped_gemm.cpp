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

#include "common/impl_list_item.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/ref_grouped_gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::grouped_gemm_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),
// clang-format off
constexpr impl_list_item_t cpu_grouped_gemm_impl_list[] = REG_GROUPED_GEMM_P({
        INSTANCE(ref_grouped_gemm_t)
        nullptr,
});
// clang-format on
#undef INSTANCE
} // namespace

const impl_list_item_t *
cpu_engine_impl_list_t::get_grouped_gemm_implementation_list() {
    return cpu_grouped_gemm_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
