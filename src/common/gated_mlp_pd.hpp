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

#ifndef COMMON_GATED_MLP_PD_HPP
#define COMMON_GATED_MLP_PD_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_desc.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define DNNL_ARG_WEIGHTS_GATE DNNL_ARG_WEIGHTS_0
#define DNNL_ARG_WEIGHTS_UP DNNL_ARG_WEIGHTS_1
#define DNNL_ARG_WEIGHTS_DOWN DNNL_ARG_WEIGHTS_2

#define VDISPATCH_GATED_MLP(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, gated_mlp, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_GATED_MLP_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, gated_mlp, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

static inline gated_mlp_desc_t create_gated_mlp_desc(
        const memory_desc_t *src_md, const memory_desc_t *w_gate_md,
        const memory_desc_t *w_up_md, const memory_desc_t *w_down_md,
        const memory_desc_t *dst_md, const alg_kind_t activation) {
    auto desc = gated_mlp_desc_t();
    desc.primitive_kind = primitive_kind::gated_mlp;
    desc.src_desc = *src_md;
    desc.w_gate_desc = *w_gate_md;
    desc.w_up_desc = *w_up_md;
    desc.w_down_desc = *w_down_md;
    desc.dst_desc = *dst_md;
    desc.activation = activation;
    return desc;
}

// NOLINTBEGIN(google-default-arguments)
struct gated_mlp_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::gated_mlp;
    using base_class = gated_mlp_pd_t;
    using hint_class = gated_mlp_pd_t;

    const gated_mlp_desc_t *desc() const { return &desc_; }
    dim_t MB() const { return arg_md(DNNL_ARG_SRC)->dims[0]; }
    dim_t IC() const { return arg_md(DNNL_ARG_SRC)->dims[1]; }
    dim_t OC() const { return arg_md(DNNL_ARG_WEIGHTS_GATE)->dims[1]; }
    alg_kind_t activation() const { return desc_.activation; }

    int n_outputs() const override { return 1; }
    int n_inputs() const override {
        return int(all_idxs().size()) - n_outputs();
    }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_DST) return arg_usage_t::output;
        const auto &idxs = all_idxs();
        if (std::find(idxs.begin(), idxs.end(), arg) != idxs.end())
            return arg_usage_t::input;
        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        return (index == 0) ? src0_md() : &glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return w_gate_md();
            case 1: return w_up_md();
            case 2: return w_down_md();
            default: return &glob_zero_md;
        }
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        return (index == 0) ? dst0_md() : &glob_zero_md;
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0, user_input);
            case DNNL_ARG_WEIGHTS_GATE: return weights_md(0, user_input);
            case DNNL_ARG_WEIGHTS_UP: return weights_md(1, user_input);
            case DNNL_ARG_WEIGHTS_DOWN: return weights_md(2, user_input);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg, user_input);
        }
    }

protected:
    gated_mlp_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<gated_mlp_desc_t>(adesc)) {}

    bool pd_ok() const {
        if (!utils::one_of(activation(), alg_kind::eltwise_gelu_erf,
                    alg_kind::eltwise_gelu_tanh, alg_kind::eltwise_swish))
            return false;
        const auto &idxs = all_idxs();
        const auto dims = all_dims(MB(), IC(), OC());
        for (int i = 0; i < int(dims.size()); i++) {
            const auto *md = arg_md(idxs[i]);
            if ((md->dims[0] != dims[i][0]) || (md->dims[1] != dims[i][1])
                    || (md->ndims != 2))
                return false;
        }
        return true;
    }

    bool set_default_format(const memory_desc_t *md) {
        return !memory_desc_wrapper(md).format_any();
    }

    bool set_default_formats() {
        bool ok = true;
        for (auto idx : all_idxs())
            ok &= set_default_format(arg_md(idx));
        ok &= attr_.post_ops_.set_default_formats(arg_md(DNNL_ARG_DST))
                == status::success;
        return ok;
    }

private:
    gated_mlp_desc_t desc_;

    const memory_desc_t *src0_md() const { return &desc_.src_desc; }
    const memory_desc_t *w_gate_md() const { return &desc_.w_gate_desc; }
    const memory_desc_t *w_up_md() const { return &desc_.w_up_desc; }
    const memory_desc_t *w_down_md() const { return &desc_.w_down_desc; }
    const memory_desc_t *dst0_md() const { return &desc_.dst_desc; }

    static const std::vector<int> &all_idxs() {
        static const std::vector<int> idx {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS_GATE,
                DNNL_ARG_WEIGHTS_UP, DNNL_ARG_WEIGHTS_DOWN, DNNL_ARG_DST};
        return idx;
    }

    static std::vector<std::array<dim_t, DNNL_MAX_NDIMS>> all_dims(
            dim_t mb, dim_t ic, dim_t oc) {
        return std::vector<std::array<dim_t, DNNL_MAX_NDIMS>> {
                {mb, ic}, // DNNL_ARG_SRC
                {ic, oc}, // DNNL_ARG_WEIGHTS_GATE
                {ic, oc}, // DNNL_ARG_WEIGHTS_UP
                {oc, ic}, // DNNL_ARG_WEIGHTS_DOWN
                {mb, ic}, // DNNL_ARG_DST
        };
    }

    static int group_size(
            const quant_entries_t &q, const memory_desc_t &desc, int arg) {
        dim_t out = utils::array_product(desc.dims, desc.ndims);
        if (q.has_default_groups(arg)) {
            for (int idx : mask_iterator(q.get_mask(arg)))
                out /= desc.dims[idx];
        } else {
            for (int idx : mask_iterator(q.get_mask(arg)))
                out /= (desc.dims[idx] / q.get_group(arg, idx));
        }
        return static_cast<int>(out);
    }
};
// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif
