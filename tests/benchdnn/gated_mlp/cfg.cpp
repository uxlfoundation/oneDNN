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

#include "gated_mlp/gated_mlp.hpp"

namespace gated_mlp {

cfg_t::cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds)
    : base_cfg_t {prb->attr.acc_mode} {
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

// Adjust density based on accumulation chain length.
float cfg_t::get_density(const cfg_t::density_args_t &density_args) const {
    float density = 1.f;
    if (density_args.data_kind != SRC) return density;

    const int64_t safe_n_acc = get_safe_n_acc();
    float safe_density = (float)safe_n_acc / density_args.n_acc;
    density = MIN2(density, safe_density);

    BENCHDNN_PRINT(6, "%s safe_n_acc=%d density=%f\n", "[FILL_CFG]",
            (int)safe_n_acc, density);

    return density;
}

cfg_t::cfg_entry_t::cfg_map_t cfg_t::get_cfg_map(data_kind_t kind) const {
    // Tight ranges for chained pipeline. SRC is further scaled in fill.
    static const cfg_t::cfg_entry_t::cfg_map_t src_cfg_map = {
            {{dnnl_f32}, {-1, 1}},
            {{dnnl_bf16}, {-1, 1}},
            {{dnnl_f16}, {-1, 1}},
            {{dnnl_s8}, {-1, 1}},
            {{dnnl_u8}, {0, 2}},
            {{dnnl_s4}, {-1, 1}},
            {{dnnl_u4}, {0, 2}},
    };

    // Weight tensor ranges, small to control accumulation magnitude.
    static const cfg_t::cfg_entry_t::cfg_map_t wei_cfg_map = {
            {{dnnl_f32}, {-1, 1}},
            {{dnnl_bf16}, {-1, 1}},
            {{dnnl_f16}, {-1, 1}},
            {{dnnl_s8}, {-1, 1}},
            {{dnnl_u8}, {0, 2}},
            {{dnnl_s4}, {-1, 1}},
            {{dnnl_u4}, {0, 2}},
    };

    // Destination tensor ranges.
    static const cfg_t::cfg_entry_t::cfg_map_t dst_cfg_map = {
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_s32}, {-8, 8}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 8}},
            {{dnnl_s4}, {-2, 2}},
            {{dnnl_u4}, {0, 4}},
    };

    switch (kind) {
        case SRC: return src_cfg_map;
        case WEI: return wei_cfg_map;
        case DST: return dst_cfg_map;
        default: assert(!"unsupported data kind"); break;
    }
    static cfg_t::cfg_entry_t::cfg_map_t dummy;
    return dummy;
}

} // namespace gated_mlp
