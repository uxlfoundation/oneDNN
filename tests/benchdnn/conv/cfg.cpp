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

#include "conv/conv.hpp"

namespace conv {

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

    adjust_ranges_for_safe_n_acc();

    // Use wider dst to test proper u8 loads.
    const bool is_int8_and_wide_dst = this->get_dt(SRC) == dnnl_u8
            && dnnl_data_type_size(this->get_dt(WEI)) == 1
            && dnnl_data_type_size(this->get_dt(DST)) >= 4;
    if (is_int8_and_wide_dst) { set_range_max(SRC, 160); }

    // For s8s8 weights have to be even to comply with adjust_scale of 0.5f.
    // Divide the range by factor of two here, and multiply values by factor
    // of two when do filling.
    const bool is_s8s8
            = this->get_dt(SRC) == dnnl_s8 && this->get_dt(WEI) == dnnl_s8;
    if (is_s8s8) {
        set_range_min(WEI, -2);
        set_range_max(WEI, 2);
    }

    // Keep legacy filling for Wino.
    if (prb->alg == WINO) {
        if (prb->dt[0] == dnnl_f32) {
            set_range_min(SRC, -16);
            set_range_max(SRC, 128);
            set_range_min(WEI, 2);
            set_range_max(WEI, 64);
        } else if (prb->dt[0] == dnnl_f16) {
            set_range_min(SRC, -2);
            set_range_max(SRC, 16);
            set_range_min(WEI, 1);
            set_range_max(WEI, 6);
        } else {
            assert(!"unsupported data type for Wino.");
        }
    }

    BENCHDNN_PRINT(6,
            "[FILL_CFG] SRC_%s=[%d;%d]; WEI_%s=[%d;%d]; DST_%s=[%d;%d];\n",
            dt2str(this->get_dt(SRC)), get_range_min(SRC), get_range_max(SRC),
            dt2str(this->get_dt(WEI)), get_range_min(WEI), get_range_max(WEI),
            dt2str(this->get_dt(DST)), get_range_min(DST), get_range_max(DST));
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
        // It seems that reduced precision values are accumulated on Nvidia HW
        // with atomics since false positive may or may not occur. To remove
        // the possibility of false positive, need to put more zeroes to reduce
        // the range of output value to stay precise.
        if (is_nvidia_gpu() && get_dt(density_args.data_kind) == dnnl_bf16)
            density /= 2.f;
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
            {{dnnl_f4_e2m1}, {0, 1}},
            {{dnnl_f4_e3m0}, {0, 1}},
            {{dnnl_f8_e5m2}, {-4, 4}},
            {{dnnl_f8_e4m3}, {-4, 4}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t wei_cfg_map = {
            {{dnnl_f64}, {-32, 32}},
            {{dnnl_f32}, {-32, 32}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-2, 2}},
            {{dnnl_f4_e2m1}, {-1, 1}},
            {{dnnl_f4_e3m0}, {-1, 1}},
            {{dnnl_f8_e5m2}, {-2, 2}},
            {{dnnl_f8_e4m3}, {-2, 2}},
            {{dnnl_s8}, {-4, 4}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t bia_cfg_map = {
            {{dnnl_f64}, {-8, 8}},
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-8, 8}},
            {{dnnl_f8_e5m2}, {-8, 8}},
            {{dnnl_f8_e4m3}, {-8, 8}},
            {{dnnl_s8}, {-8, 8}},
            {{dnnl_u8}, {0, 8}},
            {{dnnl_s32}, {-8, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t dst_cfg_map = {
            {{dnnl_f64}, {-8, 8}},
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_f4_e2m1}, {-2, 2}},
            {{dnnl_f4_e3m0}, {-2, 2}},
            {{dnnl_f8_e5m2}, {-4, 4}},
            {{dnnl_f8_e4m3}, {-4, 4}},
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

} // namespace conv
