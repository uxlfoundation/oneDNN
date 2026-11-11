/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/gemm/jit/gen_kernel_db.hpp"

namespace gemmstone {
#define _CATALOG_ gemm_catalog
#include "selector/db/kernel.db"
#undef _CATALOG_
// The per-device LUTs (lut_<PCI device id>.db); kkit's lut.py edits both lists.
// lut.py: catalogs
#define _CATALOG_ gemm_lut_674f
#include "selector/db/lut_674f.db"
#undef _CATALOG_
// lut.py: end
} // namespace gemmstone

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

gemmstone::kcatalog::Catalog catalog() {
    return gemmstone::gemm_catalog;
}

namespace {
struct lut_t {
    uint32_t device_id;
    const char *name;
    gemmstone::kcatalog::Catalog catalog;
};
const lut_t luts[] = {
        // lut.py: table
        {0x674f, "lut_674f.db", gemmstone::gemm_lut_674f},
        // lut.py: end
};
} // namespace

const gemmstone::kcatalog::Catalog *lut_catalog(uint32_t device_id) {
    for (const auto &l : luts)
        if (l.device_id == device_id) return &l.catalog;
    return nullptr;
}

std::string catalog_ref(const gemmstone::kcatalog::Entry *entry) {
    auto in = [&](const char *name, const gemmstone::kcatalog::Catalog &c) {
        return entry >= c.entries && entry < c.entries + c.entryCount
                ? std::string(name) + ":" + std::to_string(entry - c.entries)
                : std::string();
    };
    std::string r = in("kernel.db", catalog());
    for (const auto &l : luts)
        r += in(l.name, l.catalog);
    return r.empty() ? "?" : r;
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
