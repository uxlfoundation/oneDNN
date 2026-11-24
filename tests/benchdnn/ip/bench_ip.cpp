/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "ip/ip.hpp"

namespace ip {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_bia_dt_ : s.bia_dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (const auto &i_mb : s.mb) {
        auto i_bia_dt = i_bia_dt_;
        if (i_dir & FLAG_BIA) {
            if (i_bia_dt != dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Warning: `--dir=FWD_B,BWD_WB` options are "
                        "incompatible with `--bia-dt` option. To specify a "
                        "bias data type, use `--dir=FWD_D,FWD_I,BWD_W` values "
                        "intead.");
            }
            // The f32 data type should be used as the default for bias with
            // directions that include a bias.
            i_bia_dt = dnnl_f32;
        }

        auto prb = std::make_shared<prb_t>(s.desc, i_dir, i_dt, i_bia_dt,
                i_stag, i_wtag, i_dtag, i_strides, i_mb, i_attr, i_ctx_init,
                i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    static constexpr int n_inputs = 3;
    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s%d.\n",
                    "ERROR: `dt` option expects either a single input or three "
                    "inputs in SRC, WEI, DST order. Current size is: ",
                    static_cast<int>(i_dt.size()));
            return FAIL;
        }
    }

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != STRIDES_SIZE) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `strides` option expects three inputs in format "
                    "`[SRC]:[WEI]:[DST]` (two colons must present).");
            return FAIL;
        }
        const int strides_ndims[STRIDES_SIZE] = {s.desc.ndims, s.desc.ndims, 2};
        for (int i = 0; i < STRIDES_SIZE; i++) {
            if (!i_strides[i].empty()
                    && i_strides[i].size()
                            != static_cast<size_t>(strides_ndims[i])) {
                BENCHDNN_PRINT(0, "%s\n",
                        "ERROR: number of dimensions in the `strides` option "
                        "doesn't match the number of dimensions in the "
                        "problem.");
                return FAIL;
            }
        }
    }

    for (const auto &i_strides : s.strides) {
        const bool strided_input = !i_strides[STRIDES_SRC].empty()
                || !i_strides[STRIDES_WEI].empty()
                || !i_strides[STRIDES_DST].empty();
        if (!strided_input) continue;

        for_(const auto &i_stag : s.stag)
        for_(const auto &i_wtag : s.wtag)
        for (const auto &i_dtag : s.dtag) {
            const bool no_stride_with_tag
                    = IMPLICATION(i_stag != def.stag[0],
                              i_strides[STRIDES_SRC].empty())
                    && IMPLICATION(i_wtag != def.wtag[0],
                            i_strides[STRIDES_WEI].empty())
                    && IMPLICATION(i_dtag != def.dtag[0],
                            i_strides[STRIDES_DST].empty());

            if (!no_stride_with_tag) {
                BENCHDNN_PRINT(0, "%s\n",
                        "ERROR: both `strides` and `tag` knobs can not be used "
                        "with either of `src`, `wei`, and `dst` tensors.");
                return FAIL;
            }
        }
    }

    return OK;
}

int bench(int argc, char **argv) {
    driver_name = "ip";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia-dt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);

            SAFE(verify_input(s, def), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace ip
