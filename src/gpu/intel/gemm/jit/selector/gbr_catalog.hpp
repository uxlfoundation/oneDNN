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

#ifndef GEMMSTONE_SELECTOR_GBR_CATALOG_HPP
#define GEMMSTONE_SELECTOR_GBR_CATALOG_HPP

// The G catalog: gbr.db's rows plus the rows their masks derive.

#include <deque>
#include <string>
#include <vector>

#include "gemmstone/kernel_catalog.hpp"

GEMMSTONE_NAMESPACE_START

namespace gbr {

// A stored H row {'G', 2, {scorer, mask}}: mask bits derive Q/E/T/O rows
// (the strategy on that element size, K quantities kept in bytes).
class ExpandedCatalog {
public:
    explicit ExpandedCatalog(const kcatalog::Catalog &db);
    ExpandedCatalog(const ExpandedCatalog &) = delete;
    ExpandedCatalog &operator=(const ExpandedCatalog &) = delete;
    // db with the expanded entries; valid while this object lives.
    const kcatalog::Catalog &catalog() const { return catalog_; }

private:
    kcatalog::Catalog catalog_;
    // kcatalog::Selector order; ties by "um un 1 0 strategy", then scorer.
    std::vector<kcatalog::Entry> entries_;
    std::deque<std::string> strategies_;
};

} // namespace gbr

GEMMSTONE_NAMESPACE_END

#endif
