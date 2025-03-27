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

#ifndef GEMMSTONE_GUARD_PRE_CONFIG_HPP
#define GEMMSTONE_GUARD_PRE_CONFIG_HPP
// This header is for exposing structures necessary for use in config.hpp

#include "internal/namespace_start.hxx"

// Binary operations.
enum class BinaryOp {
    Add, Sub, Mul, Div,
    Min, Max,
    Prelu,
    ScaleSub    /* internal use only */
};

#include "internal/namespace_end.hxx"
#endif
