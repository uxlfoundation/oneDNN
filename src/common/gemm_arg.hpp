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

#ifndef COMMON_GEMM_ARG_HPP
#define COMMON_GEMM_ARG_HPP

namespace dnnl {
namespace impl {
namespace gemm_arg {

constexpr int A = 96;
constexpr int B = 97;
constexpr int C = 98;

inline bool is_valid(int arg) { return arg == A || arg == B || arg == C; }

} // namespace gemm_arg
} // namespace impl
} // namespace dnnl

#endif
