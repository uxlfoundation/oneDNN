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

#include "common/impl_list_item.hpp"

namespace dnnl {
namespace impl {

pk_dt_impl_key_t::pk_dt_impl_key_t(prop_kind_t kind, data_type_t src_dt,
        data_type_t wei_dt, data_type_t dst_dt)
    : kind_(kind) {
    using namespace data_type;
    using namespace prop_kind;

    const bool is_fwd
            = utils::one_of(kind_, forward_training, forward_inference);
    const bool is_bwd_d = kind_ == backward_data;
    const bool is_bwd_w = kind_ == backward_weights;

    // src_dt
    if (is_fwd) {
        if (utils::one_of(src_dt, s8, u8)) {
            s_dt_ += "xi8";
        } else if (utils::one_of(src_dt, f8_e5m2, f8_e4m3)) {
            s_dt_ += "xf8";
        } else {
            s_dt_ += dnnl_dt2str(src_dt);
        }
    } else if (is_bwd_d) {
        s_dt_ += "*";
    } else if (is_bwd_w) {
        if (utils::one_of(src_dt, f8_e5m2, f8_e4m3)) {
            s_dt_ += "xf8";
        } else {
            s_dt_ += dnnl_dt2str(src_dt);
        }
    } else {
        assert(!"unexpected");
    }
    s_dt_ += ":";

    // wei_dt
    if (is_fwd) {
        if (utils::one_of(wei_dt, f8_e5m2, f8_e4m3)) {
            s_dt_ += "xf8";
        } else if (src_dt == f32 && utils::one_of(wei_dt, f32, bf16, f16)) {
            s_dt_ += "xf";
        } else {
            s_dt_ += dnnl_dt2str(wei_dt);
        }
    } else if (is_bwd_d) {
        if (utils::one_of(wei_dt, f8_e5m2, f8_e4m3)) {
            s_dt_ += "xf8";
        } else if (dst_dt == f32 && utils::one_of(wei_dt, f32, bf16, f16)) {
            s_dt_ += "xf";
        } else {
            s_dt_ += dnnl_dt2str(wei_dt);
        }
    } else if (is_bwd_w) {
        s_dt_ += "*";
    } else {
        assert(!"unexpected");
    }
    s_dt_ += ":";

    // dst_dt
    if (is_fwd) {
        s_dt_ += "*";
    } else if (is_bwd_d || is_bwd_w) {
        if (utils::one_of(dst_dt, s8, u8)) {
            s_dt_ += "xi8";
        } else if (utils::one_of(dst_dt, f8_e5m2, f8_e4m3)) {
            s_dt_ += "xf8";
        } else {
            s_dt_ += dnnl_dt2str(dst_dt);
        }
    } else {
        assert(!"unexpected");
    }
}

} // namespace impl
} // namespace dnnl
