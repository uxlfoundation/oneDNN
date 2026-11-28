/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_axis : s.axis)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto prb = std::make_shared<prb_t>(s.prb_dims, i_dir, i_sdt, i_ddt,
                i_stag, i_dtag, i_strides, i_alg, i_axis, s.has_stats[0], i_mb,
                i_inplace, i_attr, i_ctx_init, i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    for_(const auto &i_scales : s.scales)
    for (const auto &e : i_scales.scales) {
        if (!e.second.has_single_element()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: scales support only a single element input.");
            return FAIL;
        }
    }

    static constexpr int n_inputs = 2;

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `strides` option expects two inputs in format "
                    "`[SRC]:[DST]` (colon must present).");
            return FAIL;
        }
        for (int i = 0; i < n_inputs; i++) {
            if (!i_strides[i].empty()
                    && i_strides[i].size()
                            != static_cast<size_t>(s.prb_dims.ndims)) {
                BENCHDNN_PRINT(0, "%s\n",
                        "ERROR: number of dimensions in the `strides` option "
                        "doesn't match the number of dimensions in the "
                        "problem.");
                return FAIL;
            }
        }
    }

    for_(const auto &i_strides : s.strides)
    for_(const auto &i_stag : s.stag)
    for (const auto &i_dtag : s.dtag) {
        const bool no_stride_with_tag
                = IMPLICATION(i_stag != def.stag[0], i_strides[0].empty())
                && IMPLICATION(i_dtag != def.dtag[0], i_strides[1].empty());
        if (!no_stride_with_tag) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: both `strides` and `tag` knobs can't be used "
                    "with either of `src`, or `dst` tensors.");
            return FAIL;
        }
    }

    for_(const auto &i_strides : s.strides)
    for (auto i_inplace : s.inplace) {
        if (i_inplace && !i_strides[1].empty()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `dst` strides can't be used with `inplace=true`, "
                    "only `src` strides are allowed.");
            return FAIL;
        }
    }

    return OK;
}

int bench(int argc, char **argv) {
    driver_name = "softmax";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_dt(s.sdt, def.sdt, argv[0], "sdt")
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_axis(s.axis, def.axis, argv[0])
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_dims(s.prb_dims, argv[0]);

            SAFE(verify_input(s, def), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}
} // namespace softmax
