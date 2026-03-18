/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_BRGEMM_JIT_BRGEMM_KERNEL_HPP
#define CPU_RV64_BRGEMM_JIT_BRGEMM_KERNEL_HPP

#include "cpu/rv64/brgemm/brgemm_types.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_brgemm_kernel_t;

struct brgemm_kernel_common_t : public brgemm_kernel_t {
    brgemm_kernel_common_t(const brgemm_desc_t &brg);
    ~brgemm_kernel_common_t() override;

    status_t create_kernel() override;
    void operator()(brgemm_kernel_params_t *) const override;
    const brgemm_desc_t &get_brg() const override { return brg_; }

private:
    brgemm_desc_t brg_;
    jit_brgemm_kernel_t *jit_kernel_ = nullptr;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_BRGEMM_JIT_BRGEMM_KERNEL_HPP
