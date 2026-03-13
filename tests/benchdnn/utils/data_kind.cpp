/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <cassert>
#include <vector>

#include "oneapi/dnnl/dnnl_types.h"

#include "common.hpp"
#include "utils/data_kind.hpp"

struct data_kind_entry_t {
    std::vector<int> exec_args;
    std::string label;
};

const std::map<data_kind_t, data_kind_entry_t> &data_kind_table() {
    static const std::map<data_kind_t, data_kind_entry_t> data_kind_table_ {
            // Important implementation detail:
            // `arg` to `kind` conversion is 2-to-1, and transparent.
            // Since `kind` to `arg` conversion is 1-to-2, it is done according
            // to comparison logic. To maintain it easier, first element of arg
            // vector is the one that corresponts to the argument expected in
            // comparison.
            {SRC, {{DNNL_ARG_DIFF_SRC, DNNL_ARG_SRC}, "SRC"}},
            {SRC_1, {{DNNL_ARG_DIFF_SRC_1, DNNL_ARG_SRC_1}, "SRC_ADD"}},
            {SRC_2, {{DNNL_ARG_DIFF_SRC_2, DNNL_ARG_SRC_2}, "SRC_2"}},
            {SRC_ITER,
                    {{DNNL_ARG_DIFF_SRC_ITER, DNNL_ARG_SRC_ITER}, "SRC_ITER"}},
            {SRC_ITER_C,
                    {{DNNL_ARG_DIFF_SRC_ITER_C, DNNL_ARG_SRC_ITER_C},
                            "SRC_ITER_C"}},
            {WEI, {{DNNL_ARG_DIFF_WEIGHTS, DNNL_ARG_WEIGHTS}, "WEI"}},
            {WEI_ITER,
                    {{DNNL_ARG_DIFF_WEIGHTS_ITER, DNNL_ARG_WEIGHTS_ITER},
                            "WEI_ITER"}},
            {BIA, {{DNNL_ARG_DIFF_BIAS, DNNL_ARG_BIAS}, "BIA"}},
            {DST, {{DNNL_ARG_DST, DNNL_ARG_DIFF_DST}, "DST"}},
            {DST_ITER,
                    {{DNNL_ARG_DST_ITER, DNNL_ARG_DIFF_DST_ITER}, "DST_ITER"}},
            {DST_ITER_C,
                    {{DNNL_ARG_DST_ITER_C, DNNL_ARG_DIFF_DST_ITER_C},
                            "DST_ITER_C"}},
            {MEAN, {{DNNL_ARG_MEAN}, "MEAN"}},
            {VAR, {{DNNL_ARG_VARIANCE}, "VAR"}},
            {SC, {{DNNL_ARG_DIFF_SCALE, DNNL_ARG_SCALE}, "SC"}},
            {SH, {{DNNL_ARG_DIFF_SHIFT, DNNL_ARG_SHIFT}, "SH"}},
            {AUGRU_ATTENTION,
                    {{DNNL_ARG_DIFF_AUGRU_ATTENTION, DNNL_ARG_AUGRU_ATTENTION},
                            "AUGRU_ATTENTION"}},
            {WEI_PEEPHOLE,
                    {{DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE,
                             DNNL_ARG_WEIGHTS_PEEPHOLE},
                            "WEI_PEEPHOLE"}},
            {WEI_PROJECTION,
                    {{DNNL_ARG_DIFF_WEIGHTS_PROJECTION,
                             DNNL_ARG_WEIGHTS_PROJECTION},
                            "WEI_PROJECTION"}},
            {DROPOUT_MASK, {{DNNL_ARG_ATTR_DROPOUT_MASK}, "DROPOUT_MASK"}},
            {DST_SCALES, {{DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST}, "DST_SCALES"}},
            {SDPA_STATS, {{DNNL_ARG_DST_1}, "SDPA_STATS"}},
            {DAT_TOTAL, {{DNNL_ARG_UNDEF}, "incorrect data kind"}},
    };
    return data_kind_table_;
}

data_kind_t exec_arg2data_kind(int arg) {
    for (const auto &e : data_kind_table()) {
        for (const auto &a : e.second.exec_args) {
            if (a == arg) return e.first;
        }
    }

    int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    bool is_post_ops_arg = (arg & post_ops_range);
    bool is_dw_post_op = (arg & DNNL_ARG_ATTR_POST_OP_DW);
    bool is_scales_arg = (arg & DNNL_ARG_ATTR_SCALES);
    bool is_zero_point_arg = (arg & DNNL_ARG_ATTR_ZERO_POINTS);
    bool is_dropout_arg = (arg & DNNL_ARG_ATTR_DROPOUT_PROBABILITY)
            || (arg & DNNL_ARG_ATTR_DROPOUT_MASK)
            || (arg & DNNL_ARG_ATTR_DROPOUT_SEED)
            || (arg & DNNL_ARG_ATTR_DROPOUT_OFFSET);
    if (!is_post_ops_arg && !is_dw_post_op && !is_scales_arg
            && !is_zero_point_arg && !is_dropout_arg)
        BENCHDNN_PRINT(0, "Error: arg \'%d\' was not recognized\n", arg);

    return DAT_TOTAL;
}

int data_kind2exec_arg(data_kind_t dk) {
    for (const auto &e : data_kind_table()) {
        // See `data_kind_table()` comment. It explains why `0` index is taken.
        if (e.first == dk) return e.second.exec_args[0];
    }

    BENCHDNN_PRINT(0, "Error: data_kind \'%s\' was not recognized\n",
            data_kind2str(dk));
    return DNNL_ARG_UNDEF;
}

const char *data_kind2str(data_kind_t dk) {
    for (const auto &e : data_kind_table()) {
        if (e.first == dk) return e.second.label.c_str();
    }

    return data_kind_table().at(DAT_TOTAL).label.c_str();
}
