/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef COMMON_REORDER_PD_HPP
#define COMMON_REORDER_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cache_hit_types.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "primitive_attr.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_iface.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define VDISPATCH_REORDER(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, reorder, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_REORDER_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, reorder, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

#define VDISPATCH_REORDER_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, reorder, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

// NOLINTBEGIN(google-default-arguments)
struct reorder_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::reorder;

    using base_class = reorder_pd_t;
    using hint_class = reorder_pd_t;

    const reorder_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_FROM) return arg_usage_t::input;

        if (arg == DNNL_ARG_TO) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_FROM: return src_md(0);
            case DNNL_ARG_TO: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override { return 1; }

    float beta() const {
        const int sum_idx = attr()->post_ops_.find(primitive_kind::sum);
        return sum_idx == -1 ? 0 : attr()->post_ops_.entry_[sum_idx].sum.scale;
    }

protected:
    reorder_desc_t desc_;
    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    reorder_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const reorder_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<reorder_desc_t>(adesc))
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc) {}
};
// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
