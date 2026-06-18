/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
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

#include "cpu/rv64/jit_uni_reduction.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t jit_uni_reduction_t::init(engine_t *engine) {
    UNUSED(engine);
    kernel_.reset(new jit_uni_reduction_kernel_t(pd()->get_conf()));
    CHECK(kernel_->create_kernel());
    return status::success;
}

status_t jit_uni_reduction_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const auto &conf = pd()->get_conf();
    const dim_t idle_size = conf.idle_size;
    const dim_t reduce_size = conf.reduce_size;
    const size_t src_dt_size = conf.src_dt_size;
    const size_t dst_dt_size = conf.dst_dt_size;

    parallel_nd(idle_size, [= COMPAT_THIS_CAPTURE](dim_t i) {
        const dim_t src_off = i * reduce_size * src_dt_size;
        const dim_t dst_off = i * dst_dt_size;

        jit_uni_reduction_args_t args;
        args.src = src + src_off;
        args.dst = dst + dst_off;
        args.reduce_size = reduce_size;
        (*kernel_)(&args);
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
