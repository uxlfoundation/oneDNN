/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "common/primitive_exec_types.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc.hpp"

#include <cassert>

namespace dnnl {
namespace impl {

memory_arg_t::memory_arg_t(memory_t *mem, bool is_const)
    : mem_(mem), is_const_(is_const) {
    if (mem_) mem_->retain();
}

memory_arg_t::~memory_arg_t() {
    if (mem_) mem_->release();
}

memory_arg_t::memory_arg_t(const memory_arg_t &other)
    : mem_(other.mem_), is_const_(other.is_const_) {
    if (mem_) mem_->retain();
}

memory_arg_t &memory_arg_t::operator=(const memory_arg_t &other) {
    if (this == &other) return *this;

    if (mem_) memory_deleter_t {}(mem_);

    mem_ = other.mem_;
    is_const_ = other.is_const_;
    if (mem_) mem_->retain();

    return *this;
}

memory_arg_t::memory_arg_t(memory_arg_t &&other)
    : mem_(other.mem_), is_const_(other.is_const_) {
    other.mem_ = nullptr;
    other.is_const_ = false;
}

memory_arg_t &memory_arg_t::operator=(memory_arg_t &&other) {
    if (this == &other) {
        assert(!"self move assign is not expected");
        return *this;
    }

    if (mem_) memory_deleter_t {}(mem_);

    mem_ = other.mem_;
    is_const_ = other.is_const_;

    other.mem_ = nullptr;
    other.is_const_ = false;

    return *this;
}

status_t cvt_primitive_args(const primitive_desc_t *pd, int nargs,
        const dnnl_exec_arg_t *c_args, exec_args_t &args) {
    using namespace status;

    if (!IMPLICATION(nargs > 0, c_args != nullptr)) return invalid_arguments;

    // TODO: better put extra_* in primitive_desc
    int n_inputs = 0, extra_inputs = 0;
    int n_outputs = 0, extra_outputs = 0;

    for (int i = 0; i < nargs; ++i) {
        int arg = c_args[i].arg;
        auto *mem = c_args[i].memory;

        // allows dummy arguments
        if (mem == nullptr) continue;

        VCONDCHECK(primitive, exec, check, primitive, args.count(arg) == 0,
                invalid_arguments,
                "The same argument kind %d is passed multiple times", arg);

        switch (pd->arg_usage(arg)) {
            case primitive_desc_t::arg_usage_t::input:
                args[arg] = {mem, true};
                n_inputs++;
                extra_inputs += (arg & DNNL_ARG_ATTR_ZERO_POINTS)
                        || (arg & DNNL_ARG_ATTR_SCALES)
                        || (arg & DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS)
                        // 1x1 + dw conv fusion
                        || (arg
                                == (DNNL_ARG_ATTR_POST_OP_DW
                                        | DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC))
                        || (arg
                                == (DNNL_ARG_ATTR_POST_OP_DW
                                        | DNNL_ARG_ATTR_SCALES
                                        | DNNL_ARG_WEIGHTS))
                        || (arg
                                == (DNNL_ARG_ATTR_POST_OP_DW
                                        | DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
                        || (arg == DNNL_ARG_ATTR_DROPOUT_PROBABILITY)
                        || (arg == DNNL_ARG_ATTR_DROPOUT_SEED)
                        || (arg == DNNL_ARG_ATTR_ROUNDING_SEED);
                break;
            case primitive_desc_t::arg_usage_t::output:
                args[arg] = {mem, false};
                n_outputs++;
                extra_outputs += (arg == DNNL_ARG_SCRATCHPAD)
                        || (arg == DNNL_ARG_ATTR_DROPOUT_MASK)
                        || (arg & DNNL_ARG_ATTR_SCALES);
                break;
            case primitive_desc_t::arg_usage_t::unused:
                VINFO(primitive, exec, check, primitive,
                        "unused primitive execution argument (%d)", arg);
                break;
        }
    }

    VCONDCHECK(primitive, exec, check, primitive,
            (n_inputs == pd->n_inputs() + extra_inputs), invalid_arguments,
            "bad number of inputs (expected %d got %d)",
            pd->n_inputs() + extra_inputs, n_inputs);
    VCONDCHECK(primitive, exec, check, primitive,
            (n_outputs == pd->n_outputs() + extra_outputs), invalid_arguments,
            "bad number of outputs (expected %d got %d)",
            pd->n_outputs() + extra_outputs, n_outputs);

    return success;
}

exec_ctx_t::exec_ctx_t(stream_t *stream, exec_args_t &&args)
    : impl_(new exec_ctx_impl_t(stream, std::move(args))) {}

exec_ctx_t::exec_ctx_t(const exec_ctx_t &other, exec_args_t &&args)
    : impl_(new exec_ctx_impl_t(other.stream(), std::move(args),
            other.get_memory_mapping(), other.get_resource_mapper())) {}

void exec_ctx_t::set_memory_mapping(void *handle, void *host_ptr) {
    impl_->set_memory_mapping(handle, host_ptr);
}
void exec_ctx_t::set_resource_mapper(const resource_mapper_t *resource_mapper) {
    impl_->set_resource_mapper(resource_mapper);
}
void exec_ctx_t::set_scratchpad_grantor(
        const memory_tracking::grantor_t *scratchpad_grantor) {
    impl_->set_scratchpad_grantor(scratchpad_grantor);
}

stream_t *exec_ctx_t::stream() const {
    return impl_->stream();
}
const exec_args_t &exec_ctx_t::args() const {
    return impl_->args();
}
const std::unordered_map<void *, void *> &
exec_ctx_t::get_memory_mapping() const {
    return impl_->get_memory_mapping();
}
const resource_mapper_t *exec_ctx_t::get_resource_mapper() const {
    return impl_->get_resource_mapper();
}
const memory_tracking::grantor_t &exec_ctx_t::get_scratchpad_grantor() const {
    return impl_->get_scratchpad_grantor();
}

memory_t *exec_ctx_t::input(int arg) const {
    return impl_->input(arg);
}
memory_t *exec_ctx_t::output(int arg) const {
    return impl_->output(arg);
}
memory_t *exec_ctx_t::memory(int arg) const {
    return impl_->memory(arg);
}

status_t exec_ctx_t::zero_pad_output(int arg) const {
    memory_t *mem = this->output(arg);
    if (mem == nullptr) return status::success;

    return mem->zero_pad(*this);
}

void *exec_ctx_t::host_ptr(
        int arg, bool do_zeropad, status_t *status, int index) const {
    status_t local_status = status::success;
    if (status) *status = local_status;

    if (impl_->args().count(arg) != 1) return nullptr;

    auto *mem = args().at(arg).mem();
    if (do_zeropad) local_status = mem->zero_pad(*this);
    if (status) *status = local_status;

    auto *mem_storage = mem->memory_storage(index);
    return impl_->host_ptr(mem_storage);
}

void *exec_ctx_t::host_ptr(
        const memory_storage_t *mem_storage, bool require_host_ptr) const {
    return impl_->host_ptr(mem_storage, require_host_ptr);
}

void *exec_ctx_t::map_memory_storage(
        const memory_storage_t *storage, stream_t *stream, size_t size) const {
    return impl_->map_memory_storage(storage, stream, size);
}
void exec_ctx_t::unmap_memory_storage(const memory_storage_t *storage,
        void *mapped_ptr, stream_t *stream) const {
    return impl_->unmap_memory_storage(storage, mapped_ptr, stream);
}

memory_desc_wrapper exec_ctx_t::memory_mdw(
        int arg, const memory_desc_t *md_from_primitive_desc) const {
    return impl_->memory_mdw(arg, md_from_primitive_desc);
}

exec_ctx_impl_t::~exec_ctx_impl_t() = default;

void exec_ctx_impl_t::set_memory_mapping(void *handle, void *host_ptr) {
    assert(memory_mapping_.count(handle) == 0);
    memory_mapping_.insert({handle, host_ptr});
}

void exec_ctx_impl_t::set_scratchpad_grantor(
        const memory_tracking::grantor_t *scratchpad_grantor) {
    scratchpad_grantor_.reset(scratchpad_grantor);
}

const memory_tracking::grantor_t &
exec_ctx_impl_t::get_scratchpad_grantor() const {
    assert(scratchpad_grantor_);
    return *scratchpad_grantor_;
}

memory_t *exec_ctx_impl_t::input(int arg) const {
    if (args_.count(arg) != 1) return nullptr;
    const auto &ma = args_.at(arg);
    assert(ma.is_const());
    return ma.mem();
}

memory_t *exec_ctx_impl_t::output(int arg) const {
    if (args_.count(arg) != 1) return nullptr;
    const auto &ma = args_.at(arg);
    assert(!ma.is_const());
    return ma.mem();
}

memory_t *exec_ctx_impl_t::memory(int arg) const {
    assert(args_.count(arg) == 1);
    const auto &ma = args_.at(arg);
    assert(!ma.is_const());
    return ma.mem();
}

void *exec_ctx_impl_t::host_ptr(
        const memory_storage_t *mem_storage, bool require_host_ptr) const {
    if (!mem_storage || mem_storage->is_null()) return nullptr;

    void *handle = mem_storage->root_storage()->data_handle();
    void *base_ptr = nullptr;
    if (memory_mapping_.count(handle) > 0) {
        base_ptr = memory_mapping_.at(handle);
        base_ptr = reinterpret_cast<char *>(base_ptr)
                + mem_storage->base_offset();
    } else {
        base_ptr = require_host_ptr ? nullptr : handle;
    }
    return base_ptr;
}

void *exec_ctx_impl_t::map_memory_storage(
        const memory_storage_t *storage, stream_t *stream, size_t size) const {
    if (!storage || storage->is_null()) return nullptr;

    if (memory_mapping_.count(storage->data_handle()) > 0) {
        return host_ptr(storage);
    }

    void *mapped_ptr;
    status_t status = storage->map_data(&mapped_ptr, stream, size);
    assert(status == status::success);
    MAYBE_UNUSED(status);
    return mapped_ptr;
}

void exec_ctx_impl_t::unmap_memory_storage(const memory_storage_t *storage,
        void *mapped_ptr, stream_t *stream) const {
    if (!storage || storage->is_null()
            || memory_mapping_.count(storage->data_handle()) > 0)
        return;

    status_t status = storage->unmap_data(mapped_ptr, stream);
    assert(status == status::success);
    MAYBE_UNUSED(status);
}

memory_desc_wrapper exec_ctx_impl_t::memory_mdw(
        int arg, const memory_desc_t *md_from_primitive_desc) const {
    if (md_from_primitive_desc) {
        memory_desc_wrapper mdw_from_primitive_desc(md_from_primitive_desc);
        if (!mdw_from_primitive_desc.has_runtime_dims_or_strides())
            return mdw_from_primitive_desc;
    }
    if (args_.count(arg) != 1) return memory_desc_wrapper(&glob_zero_md);
    return memory_desc_wrapper(args_.at(arg).mem()->md());
}

exec_ctx_impl_t::exec_ctx_impl_t(stream_t *stream, exec_args_t &&args)
    : stream_(stream), args_(std::move(args)) {}

exec_ctx_impl_t::exec_ctx_impl_t(stream_t *stream, exec_args_t &&args,
        const std::unordered_map<void *, void *> &memory_mapping,
        const resource_mapper_t *resource_mapper)
    : stream_(stream)
    , args_(std::move(args))
    , memory_mapping_(memory_mapping)
    , resource_mapper_(resource_mapper) {}

} // namespace impl
} // namespace dnnl
