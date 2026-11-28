/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "prelu/prelu.hpp"

namespace prelu {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        auto prb = std::make_shared<prb_t>(s.prb_vdims, i_dir, i_sdt, i_stag,
                i_strides, i_attr, i_ctx_init, i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    // Expect exactly two inputs for problem dimensions.
    static constexpr int n_inputs = 2;

    if (s.prb_vdims.n_inputs() != n_inputs) {
        BENCHDNN_PRINT(0, "%s\n",
                "Error: input tensors were specified in wrong format. "
                "Please use NxNxNxNxN:MxMxMxMxM as a problem description "
                "format.");
    }

    for (const auto &i_sdt : s.sdt) {
        if (i_sdt.size() != 1 && i_sdt.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input data types were specified in wrong format. "
                    "Please use --sdt=X:X format.");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_stag : s.stag) {
        if (i_stag.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input format tags were specified in wrong format. "
                    "Please use --stag=X:X format.");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != STRIDES_SIZE) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `strides` option expects three inputs in format "
                    "`[SRC]:[WEI]:[DST]` (two colons must present).");
            SAFE_V(FAIL);
        }
        const int strides_ndims[STRIDES_SIZE]
                = {s.prb_vdims.ndims, s.prb_vdims.ndims, s.prb_vdims.ndims};
        for (int i = 0; i < STRIDES_SIZE; i++) {
            if (!i_strides[i].empty()
                    && i_strides[i].size()
                            != static_cast<size_t>(strides_ndims[i])) {
                BENCHDNN_PRINT(0, "%s\n",
                        "ERROR: number of dimensions in the `strides` option "
                        "doesn't match the number of dimensions in the "
                        "problem.");
                SAFE_V(FAIL);
            }
        }
    }

    for_(const auto &i_strides : s.strides)
    for (const auto &i_stag : s.stag) {
        const bool no_stride_with_tag = IMPLICATION(i_stag[0] != def.stag[0][0],
                                                i_strides[STRIDES_SRC].empty())
                && IMPLICATION(i_stag[1] != def.stag[0][1],
                        i_strides[STRIDES_WEI].empty());
        if (!no_stride_with_tag) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: both `strides` and `stag` knobs can't be used "
                    "with either of `src`, or `wei` tensors.");
            SAFE_V(FAIL);
        }
    }

    return OK;
}

int bench(int argc, char **argv) {
    driver_name = "prelu";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.sdt, def.sdt, argv[0])
                || parse_multi_tag(s.stag, def.stag, argv[0])
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s, def), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}
} // namespace prelu
