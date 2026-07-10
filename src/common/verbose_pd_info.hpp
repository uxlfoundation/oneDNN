/*******************************************************************************
* Copyright 2026 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#ifndef COMMON_VERBOSE_PD_INFO_HPP
#define COMMON_VERBOSE_PD_INFO_HPP

#include "common/c_types_map.hpp"

#include <mutex>
#include <string>

namespace dnnl {
namespace impl {

struct primitive_desc_t;

/// A container for primitive desc verbose string.
struct pd_info_t {
    pd_info_t() = default;
    pd_info_t(const pd_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
    pd_info_t &operator=(const pd_info_t &rhs) {
        is_initialized_ = rhs.is_initialized_;
        str_ = rhs.str_;
        return *this;
    }
    ~pd_info_t() = default;

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(engine_t *engine, const primitive_desc_t *pd);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true; // no verbose -> info is always ready
#else
    bool is_initialized_ = false;
#endif

    // Alas, `std::once_flag` cannot be manually set and/or copied (in terms of
    // its state). Hence, when `pd_info_t` is copied the `initialization_flag_`
    // is always reset. To avoid re-initialization we use an extra
    // `is_initialized_` flag, that should be checked before calling `init()`.
    std::once_flag initialization_flag_;
};

// Enum to define which dims member of memory::desc to be dumped.
enum class dims_type_t {
    undef,
    dims,
    strides,
};

std::string md2fmt_str(
        const char *name, const memory_desc_t *md, format_kind_t user_format);
std::string md2dim_str(
        const memory_desc_t *md, dims_type_t dims_type = dims_type_t::dims);
std::string arg2str(int arg);
// Returns a verbose string of dimensions or descriptor from src, wei, and/or
// dst memory descs. Can be called externally to provide info about actual
// values of runtime dimensions.
std::string rt_dims2fmt_str(primitive_kind_t prim_kind,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *dst_md);
// Returns a verbose string of all supported by a primitive memory descriptors.
// Can be called externally to provide info about actual tag and stride values
// of runtime dimensions.
std::string rt_mds2str(primitive_kind_t prim_kind, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bia_md,
        const memory_desc_t *dst_md);
// Returns a verbose string for primitive attributes. Used in ukernel API.
std::string attr2str(const primitive_attr_t *attr);

std::string md2fmt_tag_str(const memory_desc_t *md);

} // namespace impl
} // namespace dnnl

#endif
