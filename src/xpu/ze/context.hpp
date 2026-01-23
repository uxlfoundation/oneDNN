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

#ifndef XPU_ZE_CONTEXT_HPP
#define XPU_ZE_CONTEXT_HPP

#include "xpu/context.hpp"

#include "xpu/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

struct event_t : public xpu::event_t {
    event_t() = default;
    event_t(const event_t &) = default;
    event_t(const ze_event_handle_t &event) { events_.emplace_back(event); }
    event_t(const std::vector<ze_event_handle_t> &event) : events_(event) {}
    event_t(std::vector<ze_event_handle_t> &&event)
        : events_(std::move(event)) {}
    event_t(ze_event_handle_t &&event) { events_.emplace_back(event); }
    ~event_t() override = default;

    event_t &operator=(event_t &&other) {
        std::swap(events_, other.events_);
        return *this;
    }
    event_t &operator=(const event_t &other) {
        events_ = other.events_;
        return *this;
    }

    const ze_event_handle_t &operator[](size_t i) const { return events_[i]; }
    ze_event_handle_t &operator[](size_t i) { return events_[i]; }
    size_t size() const { return events_.size(); }

    static event_t &from(xpu::event_t &event) {
        return *utils::downcast<event_t *>(&event);
    }
    static const event_t &from(const xpu::event_t &event) {
        return *utils::downcast<const event_t *>(&event);
    }
    std::unique_ptr<xpu::event_t> clone() const override {
        return std::unique_ptr<xpu::event_t>(new event_t(*this));
    }
    void append(const xpu::event_t &event) {
        auto &other = *utils::downcast<const event_t *>(&event);
        events_.insert(
                events_.end(), other.events_.begin(), other.events_.end());
    }

    std::vector<ze_event_handle_t> events_;
};

class context_t final : public xpu::context_t {
public:
    context_t() = default;
    ~context_t() override = default;

    context_t &operator=(const context_t &other) {
        events_ = other.events_;
        return *this;
    }
    void set_deps(std::vector<ze_event_handle_t> &&event) {
        events_ = event_t(event);
    }
    void set_deps(event_t &&events) { events_ = std::move(events); }

    xpu::event_t &get_deps() override { return events_; }
    const xpu::event_t &get_deps() const override { return events_; }
    void append_deps(const xpu::event_t &event) override {
        events_.append(event);
    }

private:
    event_t events_;
};

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_CONTEXT_HPP
