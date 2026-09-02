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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

namespace dnnl {

TEST(legacy_cpu_isa_compat_test, PublicValues) {
    EXPECT_EQ(static_cast<unsigned>(dnnl_cpu_isa_sse41), 0x1u);
    EXPECT_EQ(static_cast<unsigned>(dnnl_cpu_isa_avx), 0x3u);
    EXPECT_EQ(static_cast<unsigned>(cpu_isa::sse41), 0x1u);
    EXPECT_EQ(static_cast<unsigned>(cpu_isa::avx), 0x3u);

    EXPECT_STREQ(dnnl_cpu_isa2str(dnnl_cpu_isa_sse41), "cpu_isa_sse41");
    EXPECT_STREQ(dnnl_cpu_isa2str(dnnl_cpu_isa_avx), "cpu_isa_avx");
}

#if DNNL_X64 && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
TEST(legacy_cpu_isa_compat_test, SetterUsesReferenceDispatch) {
    ASSERT_EQ(set_max_cpu_isa(cpu_isa::sse41), status::success);
    EXPECT_EQ(get_effective_cpu_isa(), cpu_isa::isa_default);
}
#endif

} // namespace dnnl
