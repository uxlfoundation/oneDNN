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

#ifndef GPU_GPU_MATMUL_PD_HPP
#define GPU_GPU_MATMUL_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_matmul_pd_t : public matmul_pd_t {
    using matmul_pd_t::matmul_pd_t;

    bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
            const std::vector<int> &supported_qmodes
            = {quantization_mode::static_sazp}) const override {
        if (!batch_groups_ok()) return false;
        return matmul_pd_t::attr_scales_ok(supported_args, supported_qmodes);
    }

    bool has_blocks() {
        for (auto md : {&src_md_, &weights_md_, &bias_md_, &dst_md_}) {
            memory_desc_wrapper mdw(md);
            if (mdw.is_blocking_desc()) {
                if (mdw.blocking_desc().inner_nblks != 0) { return true; }
            }
        }
        return false;
    }

protected:
    // 3D (batch) groups for scales and zero points are not supported on GPU.
    bool batch_groups_ok() const {
        const auto &scales = attr()->scales_;
        for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (!scales.has_default_values(arg)
                    && scales.get(arg).get_group(2) > 1)
                return false;
        }
        const auto &zps = attr()->zero_points_;
        for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (!zps.has_default_values(arg) && zps.get(arg).get_group(2) > 1)
                return false;
        }
        return true;
    }
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
