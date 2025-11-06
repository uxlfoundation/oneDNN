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

#ifndef COMMON_GROUPED_GEMM_PD_HPP
#define COMMON_GROUPED_GEMM_PD_HPP

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

#include "utils.hpp"

#include "primitive_hashing.hpp"

#define VDISPATCH_GROUPED_GEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, grouped_gemm, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_GROUPED_GEMM_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, grouped_gemm, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

// NOLINTBEGIN(google-default-arguments)
struct grouped_gemm_pd_t : public primitive_desc_t {
    const grouped_gemm_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg >= DNNL_ARG_MULTIPLE_SRC
                && arg < DNNL_ARG_MULTIPLE_SRC + n_inputs())
            return arg_usage_t::input;

        if (arg >= DNNL_ARG_MULTIPLE_WEIGHTS
                && arg < DNNL_ARG_MULTIPLE_WEIGHTS + n_inputs())
            return arg_usage_t::input;

        if (arg >= DNNL_ARG_MULTIPLE_BIAS
                && arg < DNNL_ARG_MULTIPLE_BIAS + n_inputs())
            return arg_usage_t::input;

        if (arg >= DNNL_ARG_MULTIPLE_DST
                && arg < DNNL_ARG_MULTIPLE_DST + n_inputs())
            return arg_usage_t::output;

        // TODO: figure out if needs anything else
        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false /** ??? */) const override {
        int src_index = arg - DNNL_ARG_MULTIPLE_SRC;
        if (src_index >= 0 && src_index < group_size_) return src_md(src_index);

        int weight_index = arg - DNNL_ARG_MULTIPLE_WEIGHTS;
        if (weight_index >= 0 && weight_index < group_size_)
            return wei_md(weight_index);

        int bias_index = arg - DNNL_ARG_MULTIPLE_BIAS;
        if (bias_index >= 0 && bias_index < group_size_)
            return bias_md(bias_index);

        int dst_index = arg - DNNL_ARG_MULTIPLE_DST;
        if (dst_index >= 0 && dst_index < group_size_)
            return dst_md(dst_index, user_input); /** ??? */

        return primitive_desc_t::arg_md(arg);
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index < n_inputs())
            return user_input ? desc()->src_mds[index] : &src_mds_[index];
        return &glob_zero_md;
    }
    const memory_desc_t *wei_md(int index = 0, bool user_input = false) const {
        if (index < group_size_)
            return user_input ? desc()->wei_mds[index] : &wei_mds_[index];
        return &glob_zero_md;
    }
    const memory_desc_t *bias_md(int index = 0, bool user_input = false) const {
        // Check if bias exists (bias_mds_ vector is not empty) and index is valid
        if (!bias_mds_.empty() && index < group_size_)
            return user_input ? desc()->bias_mds[index] : &bias_mds_[index];
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index < group_size_)
            return user_input ? desc()->dst_mds[index] : &dst_mds_[index];
        return &glob_zero_md;
    }

    int n_inputs() const override {
        return group_size_ * (bias_mds_.size() ? 3 : 2);
    }
    int n_outputs() const override { return group_size_; }

protected:
    int group_size_;
    std::vector<memory_desc_t> src_mds_, wei_mds_, bias_mds_, dst_mds_;

    grouped_gemm_desc_t desc_;

    grouped_gemm_pd_t(const primitive_attr_t *attr, int group_size,
            const memory_desc_t *const *src_mds,
            const memory_desc_t *const *wei_mds,
            const memory_desc_t *const *bias_mds,
            const memory_desc_t *const *dst_mds)
        : primitive_desc_t(attr, primitive_kind::grouped_gemm)
        , group_size_(group_size) {
        src_mds_.reserve(group_size_);
        for (int i = 0; i < group_size_; ++i) {
            src_mds_.push_back(*src_mds[i]);
            wei_mds_.push_back(*wei_mds[i]);
            if (bias_mds) bias_mds_.push_back(*bias_mds[i]);
            dst_mds_.push_back(*dst_mds[i]);
        }

        init_desc();
    }

    grouped_gemm_pd_t(const grouped_gemm_pd_t &other)
        : primitive_desc_t(other)
        , group_size_(other.group_size_)
        , src_mds_(other.src_mds_)
        , wei_mds_(other.wei_mds_)
        , bias_mds_(other.bias_mds_)
        , dst_mds_(other.dst_mds_) {
        init_desc();
    }

    grouped_gemm_pd_t &operator=(const grouped_gemm_pd_t &other) {
        DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
        group_size_ = other.group_size_;
        src_mds_ = other.src_mds_;
        wei_mds_ = other.wei_mds_;
        bias_mds_ = other.bias_mds_;
        dst_mds_ = other.dst_mds_;

        init_desc();
        return *this;
    }

    status_t init(engine_t *engine) {
        // Validate scales attribute
        const auto &scales = attr()->scales_;
        if (!scales.has_default_values()) {
            std::vector<int> supported_args;
            for (int i = 0; i < group_size_; ++i) {
                supported_args.push_back(DNNL_ARG_MULTIPLE_SRC + i);
                supported_args.push_back(DNNL_ARG_MULTIPLE_WEIGHTS + i);
                supported_args.push_back(DNNL_ARG_MULTIPLE_DST + i);
            }

            VDISPATCH_GROUPED_GEMM(scales.has_default_values(supported_args),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Check for allowed masks
            for (int i = 0; i < group_size_; ++i) {
                // Source: per-tensor (mask == 0) or row-wise (mask == 1)
                if (!scales.has_default_values(DNNL_ARG_MULTIPLE_SRC + i)) {
                    int mask = scales.get_mask(DNNL_ARG_MULTIPLE_SRC + i);
                    VDISPATCH_GROUPED_GEMM(mask == 0 || mask == 1,
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                }

                // Weights: per-tensor (mask == 0) or column-wise (mask == 2)
                if (!scales.has_default_values(DNNL_ARG_MULTIPLE_WEIGHTS + i)) {
                    int mask = scales.get_mask(DNNL_ARG_MULTIPLE_WEIGHTS + i);
                    VDISPATCH_GROUPED_GEMM(mask == 0 || mask == 2,
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                }

                // Destination: per-tensor (mask == 0)
                if (!scales.has_default_values(DNNL_ARG_MULTIPLE_DST + i)) {
                    int mask = scales.get_mask(DNNL_ARG_MULTIPLE_DST + i);
                    VDISPATCH_GROUPED_GEMM(
                            mask == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }
        }

        return status::success;
    }

    status_t set_default_params() {
        // TODO: figure out
        return status::unimplemented;
    }

private:
    void init_desc() {
        desc_ = grouped_gemm_desc_t();
        desc_.primitive_kind = primitive_kind::grouped_gemm;
        desc_.group_size = group_size_;
        for (int i = 0; i < group_size_; ++i) {
            desc_.src_mds.push_back(&src_mds_[i]);
            desc_.wei_mds.push_back(&wei_mds_[i]);
            if (bias_mds_.size()) desc_.bias_mds.push_back(&bias_mds_[i]);
            desc_.dst_mds.push_back(&dst_mds_[i]);
        }
    }
};
// NOLINTEND(google-default-arguments)

#define DECLARE_GROUPED_GEMM_PD_t(impl_name, ...) \
    static status_t create(grouped_gemm_pd_t **grouped_gemm_pd, \
            dnnl::impl::engine_t *engine, const primitive_attr_t *attr, \
            int group_size, const memory_desc_t *const *src_mds, \
            const memory_desc_t *const *wei_mds, \
            const memory_desc_t *const *bias_mds, \
            const memory_desc_t *const *dst_mds) { \
        using namespace status; \
        auto _pd = make_unique_pd<pd_t>( \
                attr, group_size, src_mds, wei_mds, bias_mds, dst_mds); \
        if (_pd == nullptr) return out_of_memory; \
        CHECK(_pd->init(engine)); \
        CHECK(_pd->init_scratchpad_md()); \
        return safe_ptr_assign(*grouped_gemm_pd, _pd.release()); \
    } \
    status_t create_primitive( \
            std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> \
                    &primitive, \
            dnnl::impl::engine_t *engine, const cache_blob_t &cache_blob, \
            bool force_create_from_blob) const override { \
        return primitive_t::create_primitive_common<__VA_ARGS__, pd_t>( \
                primitive, this, engine, false, cache_blob, \
                force_create_from_blob); \
    } \
    pd_t *clone() const override { \
        auto new_pd = utils::make_unique<pd_t>(*this); \
        if (!new_pd->is_initialized()) return nullptr; \
        return new_pd.release(); \
    } \
    const char *name() const override { return impl_name; }

#define DECLARE_GROUPED_GEMM_PD_T(impl_name, ...) \
    DECLARE_GROUPED_GEMM_PD_t(impl_name, __VA_ARGS__)

} // namespace impl
} // namespace dnnl

#endif
