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

#ifndef UTILS_PRB_HPP
#define UTILS_PRB_HPP

#include "dnn_types.hpp"
#include "utils/impl_filter.hpp"
#include "utils/res.hpp"
#include "utils/wrapper.hpp"

#include <cassert>
#include <string>

// A base class for all driver-specific `prb_t` problem descriptors. It holds
// the members that are common across **every** driver and exposes a polymorphic
// interface for all functions to rely on it.
struct base_prb_t {
    base_prb_t() = default;
    base_prb_t(dir_t dir, bool inplace, const attr_t &attr,
            const impl_filter_t &impl_filter)
        : dir(dir), inplace(inplace), attr(attr), impl_filter(impl_filter) {}

    virtual ~base_prb_t() = default;

    dir_t dir = FLAG_FWD;
    bool inplace = false;
    attr_t attr;
    impl_filter_t impl_filter;

    // Use to construct memory descriptor when dimensions are runtime since such
    // memory descriptors can't be used directly from a query call and memory
    // objects can't be constructed from such descriptors.
    // Drivers supporting runtime dimensions must override this call.
    virtual benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    const char *str() const { return repro.c_str(); }

protected:
    // Collects driver-specific reproducer information into a single line. Each
    // driver provides its own implementation and must call it last in the ctor
    // to assign the result to `repro`.
    virtual std::string set_repro_line() = 0;

    std::string repro;
};

#endif
