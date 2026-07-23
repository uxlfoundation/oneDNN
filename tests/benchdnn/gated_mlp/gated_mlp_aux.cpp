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

#include <sstream>
#include <string.h>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "gated_mlp/gated_mlp.hpp"

namespace gated_mlp {

dnnl_alg_kind_t str2activation(const char *str) {
    if (!strcmp(str, "swish")) return dnnl_eltwise_swish;
    if (!strcmp(str, "gelu_erf")) return dnnl_eltwise_gelu_erf;
    if (!strcmp(str, "gelu_tanh")) return dnnl_eltwise_gelu_tanh;
    assert(!"unknown activation");
    return dnnl_eltwise_swish;
}

const char *activation2str(dnnl_alg_kind_t act) {
    switch (act) {
        case dnnl_eltwise_swish: return "swish";
        case dnnl_eltwise_gelu_erf: return "gelu_erf";
        case dnnl_eltwise_gelu_tanh: return "gelu_tanh";
        default: assert(!"unknown activation"); return "unknown";
    }
}

dnnl_data_type_t prb_t::get_dt(data_kind_t data_kind) const {
    switch (data_kind) {
        case SRC: return src_dt();
        case WEI: return w_gate_dt(); // All weights share WEI kind.
        case DST: return dst_dt();
        default: assert(!"unexpected"); return dnnl_data_type_undef;
    }
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> prb_t::get_md(int arg) const {
    switch (arg) {
        case DNNL_ARG_SRC:
            return dnn_mem_t::init_md(ndims, src_dims.data(), src_dt(), stag);
        case DNNL_ARG_WEIGHTS_GATE:
            return dnn_mem_t::init_md(
                    ndims, w_gate_dims.data(), w_gate_dt(), wtag);
        case DNNL_ARG_WEIGHTS_UP:
            return dnn_mem_t::init_md(ndims, w_up_dims.data(), w_up_dt(), wtag);
        case DNNL_ARG_WEIGHTS_DOWN:
            return dnn_mem_t::init_md(
                    ndims, w_down_dims.data(), w_down_dt(), wtag);
        case DNNL_ARG_DST:
            return dnn_mem_t::init_md(ndims, dst_dims.data(), dst_dt(), dtag);
        default:
            assert(!"unsupported arg");
            return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }
}

std::string prb_t::set_repro_line() {
    dnnl::impl::stringstream_t s;
    dump_global_params(s);
    settings_t def;

    bool has_default_dts = true;
    for (const auto &i_dt : dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || !has_default_dts) s << "--dt=" << dt << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";
    if (canonical || wtag != def.wtag[0]) s << "--wtag=" << wtag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || activation != def.activation[0])
        s << "--activation=" << activation2str(activation) << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";
    if (canonical || !impl_filter.is_def() || !global_impl_filter.is_def())
        s << impl_filter;

    s << static_cast<const prb_dims_t &>(*this);

    return s.str();
}

} // namespace gated_mlp
