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

#ifndef COMMON_REORDER_PRIMITIVE_DESC_IFACE_HPP
#define COMMON_REORDER_PRIMITIVE_DESC_IFACE_HPP

#include "common/primitive_desc_iface.hpp"
#include "common/primitive_iface.hpp"

namespace dnnl {
namespace impl {

struct reorder_primitive_desc_iface_t : public dnnl_primitive_desc {
    reorder_primitive_desc_iface_t(const std::shared_ptr<primitive_desc_t> &pd,
            const engine_t *engine, engine_t *src_engine, engine_t *dst_engine)
        : dnnl_primitive_desc(pd, engine)
        , src_engine_(src_engine)
        , dst_engine_(dst_engine)
        , scratchpad_engine_(nullptr) {}

    dnnl::impl::engine_t *src_engine() const override { return src_engine_; }
    dnnl::impl::engine_t *dst_engine() const override { return dst_engine_; }

    dnnl::impl::engine_t *scratchpad_engine() const override {
        return scratchpad_engine_;
    }

    dnnl::impl::status_t query(
            dnnl::impl::query_t what, int idx, void *result) const override {
        auto status = dnnl::impl::status::success;
        switch (what) {
            case dnnl::impl::query::reorder_src_engine:
                *(dnnl::impl::engine_t **)result = src_engine();
                break;
            case dnnl::impl::query::reorder_dst_engine:
                *(dnnl::impl::engine_t **)result = dst_engine();
                break;
            default: status = dnnl_primitive_desc::query(what, idx, result);
        }
        return status;
    }

    status_t create_primitive_iface(
            std::pair<primitive_iface_t *, cache_state_t> &primitive_iface,
            const cache_blob_t &cache_blob) const override {
        // Step 1: create impl::primitive_t or get it from primitive cache
        std::pair<std::shared_ptr<primitive_t>, cache_state_t> p;
        // Top level primitive can be fetched from the primitive cache since
        // it's fetching it faster.
        constexpr bool force_create_from_blob = false;
        auto status = pd_->create_primitive(
                p, engine(), cache_blob, force_create_from_blob);
        if (status != status::success) return status;
        // Step 2: create primitive_iface_t, init and return it to user
        primitive_iface_t *p_iface = nullptr;
        CHECK(safe_ptr_assign(p_iface,
                new primitive_iface_t(
                        p.first, engine(), src_engine_, dst_engine_)));
        status = p_iface->init();
        if (status != status::success) {
            p_iface->release();
            return status;
        }
        primitive_iface = std::make_pair(p_iface, p.second);
        return status::success;
    }

private:
    dnnl::impl::engine_t *src_engine_;
    dnnl::impl::engine_t *dst_engine_;
    dnnl::impl::engine_t *scratchpad_engine_;
};

} // namespace impl
} // namespace dnnl

#endif
