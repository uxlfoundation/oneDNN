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

#include <stdio.h>
#include <stdlib.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "gated_mlp/gated_mlp.hpp"

namespace gated_mlp {

TASK_EXECUTOR_DECL_TYPES;

void check_correctness(
        const settings_t &s, driver_task_executor_t &task_executor) {
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_activation : s.activation)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        auto prb = std::make_shared<prb_t>(s.prb_dims, i_dt, i_stag, i_wtag,
                i_dtag, i_activation, i_attr, i_ctx_init, i_ctx_exe,
                s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    static constexpr int n_inputs = 5; // src, w_gate, w_up, w_down, dst dt

    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            BENCHDNN_PRINT(0,
                    "ERROR: gated_mlp driver: `dt` option expects either a "
                    "single input or five inputs in SRC, W_GATE, W_UP, "
                    "W_DOWN, DST order. Current size is: \"%ld\"\n",
                    (long)i_dt.size());
            SAFE_V(FAIL);
        }
    }

    static constexpr int n_dims = 3; // MB, IC, OC
    if (s.prb_dims.ndims != n_dims) {
        BENCHDNN_PRINT(0,
                "ERROR: Expected number of dims is `%d` (MBxICxOC), "
                "provided `%d`.\n",
                n_dims, s.prb_dims.ndims);
        SAFE_V(FAIL);
    }
    return OK;
}

static const std::string help_activation
        = "ACTIVATION    (Default: `swish`)\n    Specifies the gated "
          "activation function.\n    `swish`, `gelu_erf`, `gelu_tanh`.\n";

int bench(int argc, char **argv) {
    driver_name = "gated_mlp";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_vector_option(s.activation, def.activation,
                        str2activation, argv[0], "activation", help_activation)
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_dims(s.prb_dims, argv[0]);

            SAFE(verify_input(s), WARN);

            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace gated_mlp
