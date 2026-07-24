/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_UTILS_JIT_IO_HELPER_HPP
#define CPU_AARCH64_UTILS_JIT_IO_HELPER_HPP

#include "common/c_types_map.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#include "xbyak_aarch64/xbyak_aarch64_reg.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace io {

template <typename TReg>
class saturation_conf_t {
public:
    // Force user to be explicit
    saturation_conf_t() = delete;
    saturation_conf_t(const TReg &lbound_vec, const TReg &ubound_vec)
        : lbound_vec(lbound_vec), ubound_vec(ubound_vec) {}

    const TReg lbound_vec;
    const TReg ubound_vec;
};

struct tail_conf_t {
    // Force the user to be explicit
    tail_conf_t() = delete;
    tail_conf_t(size_t tail_size, const Xbyak_aarch64::PReg &pred)
        : tail_size(tail_size), pred(pred) {}

    const size_t tail_size;
    const Xbyak_aarch64::PReg pred;
};

/*
 * The goal of the io_helper is to abstract away the load and store instructions
 * and their variants for different ISAs, and data-types.
 *
 * The intended use-case is the following:
 *     1. Your data is stored in some run-time determined data-type
 *     2. You want to load it as F32s into an SIMD register for computation
 *        (possibly with a tail)
 *     3. You want to store it back into memory as some run-time determined data-type
 *        (possibly with saturation, and/or a tail)
 *
 * The implementation asks for registers to hold tail masks, or saturation data
 * (as part of tail_conf_t and saturation_conf_t respectively), but you only pay
 * for what you use, so these registers are only written to if you do a load or
 * store that requires them to be initialised. Until then, you may re-use them
 * in your kernels.
 *
 * The temporary general purpose register may be corrupted at any time and
 * should be reserved exclusively for the io_helper.
 */
template <cpu_isa_t isa>
class jit_io_helper_t {
public:
    using TReg = typename cpu_isa_traits<isa>::TReg;

    jit_io_helper_t(jit_generator_t *host,
            const saturation_conf_t<TReg> &sat_conf,
            const tail_conf_t &tail_conf, const Xbyak_aarch64::XReg &temp_xreg);

    static bool is_supported_dt(data_type_t dt);

    void broadcast_load(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr);

    void load(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr, bool is_tail = false);

    void store(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &dst_addr, bool is_tail = false);

    void gather_load(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &base_addr, const TReg &offsets_vec,
            const bool is_tail = false);

private:
    // Saturation readiness is tied to the data-type, and may need to be
    // re-initialised if we store a data-type we haven't prepared before. We
    // only keep one data-type prepared at a time because this should fit most
    // use-cases.
    struct saturation_ready_t {
        data_type_t dt = data_type::undef;
        bool is_ready = false;
    };

    // Tail and saturation state are initialised lazily on the first load/store
    // that needs them.
    void lazy_init_saturation(data_type_t dt);
    void lazy_init_tail(bool is_tail);

    void broadcast_load_word(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr) const;
    void load_words(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr, bool is_tail) const;
    void gather_load_words(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &base_addr, const TReg &offsets_vec,
            const bool is_tail = false) const;
    void store_words(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &dst_addr, bool is_tail) const;

    void broadcast_load_halfword(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr) const;
    void load_halfwords(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr, bool is_tail) const;
    void gather_load_halfwords(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &base_addr, const TReg &offsets_vec,
            const bool is_tail = false) const;
    void store_halfwords(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &dst_addr, bool is_tail) const;

    void broadcast_load_byte(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr) const;
    void load_bytes(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &src_addr, bool is_tail) const;
    void gather_load_bytes(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &base_addr, const TReg &offsets_vec,
            const bool is_tail = false) const;
    void store_bytes(data_type_t dt, const TReg &vec,
            const Xbyak_aarch64::XReg &dst_addr, bool is_tail) const;

    void saturate(data_type_t dt, const TReg &vec) const;
    void extend_xf16_to_f32(data_type_t dt, const TReg &vec) const;
    void narrow_f32_to_xf16(data_type_t dt, const TReg &vec) const;

    Xbyak_aarch64::_PReg get_pred(bool is_tail) const;

    template <size_t lane_size, bool is_load>
    void gen_asimd_load_store(const Xbyak_aarch64::VReg &vec,
            const Xbyak_aarch64::XReg &addr, bool is_tail) const;

    jit_generator_t *host_ = nullptr;
    const saturation_conf_t<TReg> sat_conf_;
    const tail_conf_t tail_conf_;

    saturation_ready_t saturation_init_state_ {};
    // asimd needs no further tail preparation. SVE will be lazily initialised
    // later.
    bool is_tail_ready_ = (isa == asimd);

    const Xbyak_aarch64::XReg temp_xreg_;
};

} // namespace io
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_UTILS_JIT_IO_HELPER_HPP
