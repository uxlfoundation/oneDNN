/*******************************************************************************
* Copyright 2017-2025 Intel Corporation
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

#include "deconv/deconv.hpp"

namespace deconv {

cfg_t::cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds) {
    output_data_kind_ = (prb->dir & FLAG_FWD) ? DST
            : (prb->dir & FLAG_WEI)           ? WEI
                                              : SRC;
    for (const auto kind : kinds) {
        auto orig_data_type = prb->get_dt(kind);
        auto data_type = deduce_cfg_data_type(orig_data_type, prb->attr, kind);
        cfg_entry_.emplace(kind,
                cfg_entry_t {
                        kind, orig_data_type, data_type, get_cfg_map(kind)});
    }

    adjust_ranges();
    print_fill_cfg_verbose();
}

// Adjust density based on accumulation chain.
float cfg_t::get_density(const cfg_t::density_args_t &density_args) const {
    float density = 1.f;
    std::string safe_n_acc_str = "N/A";

    const data_kind_t allowed_non_dense_kind
            = output_data_kind_ == DST ? SRC : DST;

    if (density_args.data_kind == allowed_non_dense_kind) {
        int64_t safe_n_acc = get_safe_n_acc();
        safe_n_acc_str = std::to_string(safe_n_acc);

        // Bump density for some empiric value for int8 validation to hit
        // saturation bound.
        float safe_density = (float)safe_n_acc / density_args.n_acc;
        if (is_int8()) safe_density *= 3.f;
        density = MIN2(density, safe_density);
    }

    BENCHDNN_PRINT(6, "[FILL_CFG][%s] n_acc=%lld safe_n_acc=%s; density=%f\n",
            data_kind2str(density_args.data_kind),
            (long long)density_args.n_acc, safe_n_acc_str.c_str(), density);

    return density;
}

cfg_t::cfg_entry_t::cfg_map_t cfg_t::get_cfg_map(data_kind_t kind) const {
    static const cfg_t::cfg_entry_t::cfg_map_t src_cfg_map = {
            {{dnnl_f64}, {-32, 32}},
            {{dnnl_f32}, {-32, 32}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_f8_e5m2}, {-4, 4}},
            {{dnnl_f8_e4m3}, {-4, 4}},
            {{dnnl_s16}, {-1024, 1024}},
            {{dnnl_u16}, {0, 2048}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t wei_cfg_map = {
            {{dnnl_f64}, {-32, 32}},
            {{dnnl_f32}, {-32, 32}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-2, 2}},
            {{dnnl_f8_e5m2}, {-2, 2}},
            {{dnnl_f8_e4m3}, {-2, 2}},
            {{dnnl_s16}, {-1024, 1024}},
            {{dnnl_s8}, {-4, 4}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t bia_cfg_map = {
            {{dnnl_f64}, {-8, 8}},
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-8, 8}},
            {{dnnl_f8_e5m2}, {-8, 8}},
            {{dnnl_f8_e4m3}, {-8, 8}},
            {{dnnl_s16}, {-2048, 2048}},
            {{dnnl_u16}, {0, 2048}},
            {{dnnl_s8}, {-8, 8}},
            {{dnnl_u8}, {0, 8}},
            {{dnnl_s32}, {-8, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t dst_cfg_map = {
            {{dnnl_f64}, {-8, 8}},
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_f8_e5m2}, {-4, 4}},
            {{dnnl_f8_e4m3}, {-4, 4}},
            {{dnnl_s16}, {-1024, 1024}},
            {{dnnl_u16}, {0, 2048}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 160}},
            {{dnnl_s32}, {-128, 128}},
    };

    switch (kind) {
        case SRC: return src_cfg_map;
        case WEI: return wei_cfg_map;
        case BIA: return bia_cfg_map;
        case DST: return dst_cfg_map;
        default: assert(!"unsupported data kind"); break;
    }
    static cfg_t::cfg_entry_t::cfg_map_t dummy;
    return dummy;
}

} // namespace deconv
