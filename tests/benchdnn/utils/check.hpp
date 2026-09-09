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

#ifndef UTILS_CHECK_HPP
#define UTILS_CHECK_HPP

#include <iostream>

enum class runtime_kind_t : unsigned {
    undefined,
    cpu,
    gpu,
    all,
};

extern runtime_kind_t default_runtime_kind;
extern runtime_kind_t check_ref_impl;

std::ostream &operator<<(std::ostream &, runtime_kind_t);

struct res_t;
// Checks if unexpected reference implementation was hit.
int check_ref_impl_hit(res_t *res);

#endif
