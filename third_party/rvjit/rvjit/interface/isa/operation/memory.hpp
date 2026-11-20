/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_MEMORY_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_MEMORY_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/expression.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace memory {

/// Memory operation direction
enum class dir_t { load, store };

/// Vector memory addressing mode
enum class vec_addr_t {
    unit, // Consecutive elements from the base
    stride, // Elements a register-held byte stride apart
    index, // Elements at the byte offsets a vector register holds
};

/// Generic memory operation description
///
/// @details Describes the data's form and type and the addressing mode; the
///     data register, the address and the stride or index register are the
///     arguments of the expression emitting it (`args_t`)
struct op_t : public typed_t {
    using operand_t = rvjit::operand_t;

    operand_t data;
    dir_t dir = dir_t::load;
    vec_addr_t vector_mode = vec_addr_t::unit;
    predicate_t predicate = predicate_t::unmasked;

    op_t() = default;

    /// A load or store of `dt` data in the given register form
    ///
    /// @param form The storage category of the data
    /// @param dir The direction of the operation
    /// @param dt The data type
    op_t(rvjit::form_t form, dir_t dir, type_t dt) : dir(dir) {
        data.form = form;
        data.set_type(dt);
    }

    /// A load of `dt` data in the given register form
    ///
    /// @param form The storage category of the data
    /// @param dt The data type
    static op_t load(rvjit::form_t form, type_t dt) {
        return op_t(form, dir_t::load, dt);
    }

    /// A store of `dt` data in the given register form
    ///
    /// @param form The storage category of the data
    /// @param dt The data type
    static op_t store(rvjit::form_t form, type_t dt) {
        return op_t(form, dir_t::store, dt);
    }

    bool is_scalar() const override { return data.is_scalar(); }
    bool is_vector() const override { return data.is_vector(); }
    bool is_int() const override { return data.is_int(); }
    bool is_float() const override { return data.is_float(); }
    bool is_signed() const override { return data.is_signed(); }

    /// Sets the data to the scalar form
    op_t &scalar() {
        data.form = rvjit::form_t::scalar;
        vector_mode = vec_addr_t::unit;
        return *this;
    }

    /// Sets the data to the vector form with unit-stride addressing
    ///
    /// @param vm The vector mask policy
    op_t &vector(predicate_t vm = predicate_t::unmasked) {
        data.form = rvjit::form_t::vector;
        vector_mode = vec_addr_t::unit;
        predicate = vm;
        return *this;
    }

    /// Sets the vector addressing to a register-held byte stride
    ///
    /// @note The expression is called with the register holding the stride
    op_t &strided() {
        vector_mode = vec_addr_t::stride;
        return *this;
    }

    /// Sets the vector addressing to the byte offsets of a vector register
    ///
    /// @note The expression is called with the register holding the offsets
    op_t &indexed() {
        vector_mode = vec_addr_t::index;
        return *this;
    }

    /// Sets the direction to load
    op_t &load() {
        dir = dir_t::load;
        return *this;
    }

    /// Sets the direction to store
    op_t &store() {
        dir = dir_t::store;
        return *this;
    }

    /// Sets the data type
    ///
    /// @param dt The data type
    op_t &as_type(type_t dt) {
        data.set_type(dt);
        return *this;
    }
};

/// The registers a memory expression is emitted on
struct args_t {
    name_t data;
    pointer_t addr;
    name_t args;

    /// Names the data register, the address and the addressing register
    ///
    /// @param data The register loaded or stored
    /// @param addr The memory address
    /// @param args The register holding the stride of a strided access or the
    ///     offsets of an indexed one, `no_name` otherwise
    args_t(name_t data, const pointer_t &addr, name_t args = no_name)
        : data(data), addr(addr), args(args) {}
};

using expression_t = operation::expression_t<op_t, args_t>;
using expression_impl_t = operation::expression_impl_t<op_t, args_t>;

} // namespace memory
} // namespace rvjit

namespace rvjit {
namespace operation {

/// Memory expressions
///
/// @details A factory of `memory::expression_t`: each method binds an
///     operation to the ISA logic of its direction, and the expression is
///     then called with the registers, e.g. `load(form, dt)(data, addr)`
///
/// @note An implementation overrides the protected accessors of the
///     directions it implements; the others are never valid
struct memory_t {
    static constexpr const char *NAME = "rvjit::operation::memory";

    virtual ~memory_t() = default;

    /// Load expression, loading `addr` into `data`
    ///
    /// @param op The operation, whose direction becomes load
    memory::expression_t load(const memory::op_t &op) const {
        memory::op_t o = op;
        return memory::expression_t(load_(), o.load());
    }

    /// Load expression of a data form and type
    ///
    /// @param form The storage category of the data
    /// @param dt The data type
    /// @param mode The vector addressing mode
    memory::expression_t load(form_t form, type_t dt,
            memory::vec_addr_t mode = memory::vec_addr_t::unit) const {
        memory::op_t o = memory::op_t::load(form, dt);
        o.vector_mode = mode;
        return load(o);
    }

    /// Store expression, storing `data` into `addr`
    ///
    /// @param op The operation, whose direction becomes store
    memory::expression_t store(const memory::op_t &op) const {
        memory::op_t o = op;
        return memory::expression_t(store_(), o.store());
    }

    /// Store expression of a data form and type
    ///
    /// @param form The storage category of the data
    /// @param dt The data type
    /// @param mode The vector addressing mode
    memory::expression_t store(form_t form, type_t dt,
            memory::vec_addr_t mode = memory::vec_addr_t::unit) const {
        memory::op_t o = memory::op_t::store(form, dt);
        o.vector_mode = mode;
        return store(o);
    }

    /// The load or store expression of `op`, as its direction says
    ///
    /// @param op The operation
    memory::expression_t of(const memory::op_t &op) const {
        return op.dir == memory::dir_t::load ? load(op) : store(op);
    }

protected:
    /// The logic of `load`
    virtual const memory::expression_impl_t &load_() const {
        return unimplemented_();
    }

    /// The logic of `store`
    virtual const memory::expression_impl_t &store_() const {
        return unimplemented_();
    }

private:
    static const memory::expression_impl_t &unimplemented_() {
        static const memory::expression_impl_t none;
        return none;
    }
};

/// Assembles a `memory_t` from expression implementations
///
/// @details Each implementation derives from `memory::expression_impl_t` and
///     is constructible from `emitter_t`
template <typename LoadOp, typename StoreOp>
struct memory_compose_t : public memory_t {
    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    explicit memory_compose_t(emitter_t e) : load_impl_(e), store_impl_(e) {}

protected:
    const memory::expression_impl_t &load_() const override {
        return load_impl_;
    }
    const memory::expression_impl_t &store_() const override {
        return store_impl_;
    }

    LoadOp load_impl_;
    StoreOp store_impl_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
