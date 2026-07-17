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

#include <cstdint>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/utils/jit_io_helper.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace io {
using namespace Xbyak_aarch64;

namespace {
constexpr int word_size = sizeof(uint32_t);
constexpr int half_size = sizeof(uint16_t);
constexpr int byte_size = sizeof(uint8_t);

// TODO: With C++17's `if constexpr` we could delete these helpers and replace
// statements like `if (isa == sve)` with `if constexpr (isa == sve)`.
VReg to_vreg(const Reg &v) {
    assert(v.isVRegVec());
    return VReg(v.getIdx());
}

ZReg to_zreg(const Reg &v) {
    assert(v.isZReg());
    return ZReg(v.getIdx());
}

// The extend and narrow functions are only needed for asimd because SVE has
// specialised instructions for extending or narrowing during stores.
void asimd_extend_i8_to_s32(
        jit_generator_t *host, data_type_t dt, const VReg &vec) {
    assert(dt == data_type::s8 || dt == data_type::u8);

    if (dt == data_type::s8) {
        host->sxtl(vec.h8, vec.b8);
        host->sxtl(vec.s4, vec.h4);
    } else {
        host->uxtl(vec.h8, vec.b8);
        host->uxtl(vec.s4, vec.h4);
    }
}

void asimd_narrow_s32_to_i8(
        jit_generator_t *host, data_type_t dt, const VReg &vec) {
    assert(dt == data_type::s8 || dt == data_type::u8);

    if (dt == data_type::s8) {
        host->sqxtn(vec.h4, vec.s4);
        host->sqxtn(vec.b8, vec.h8);
    } else {
        host->uqxtn(vec.h4, vec.s4);
        host->uqxtn(vec.b8, vec.h8);
    }
}
} // namespace

template <cpu_isa_t isa>
jit_io_helper_t<isa>::jit_io_helper_t(jit_generator_t *host,
        const saturation_conf_t<TReg> &sat_conf, const tail_conf_t &tail_conf,
        const XReg &temp_xreg)
    : host_(host)
    , sat_conf_(sat_conf)
    , tail_conf_(tail_conf)
    , temp_xreg_(temp_xreg) {
    static_assert(isa == asimd || isa == sve,
            "isa template parameter must be asimd or plain sve "
            "(no specific vector lengths)");

    assert(host != nullptr);

    saturation_init_state_.dt = data_type::undef;
    saturation_init_state_.is_ready = false;
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::lazy_init_saturation(data_type_t dt) {
    if (!types::is_integral_dt(dt)
            || (saturation_init_state_.dt == dt
                    && saturation_init_state_.is_ready)) {
        return;
    }

    host_->init_saturate_f32(sat_conf_.lbound_vec, sat_conf_.ubound_vec,
            temp_xreg_, data_type::f32, dt, true);

    saturation_init_state_.dt = dt;
    saturation_init_state_.is_ready = true;
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::lazy_init_tail(bool is_tail) {
    if (!is_tail || is_tail_ready_) { return; }

    if (isa == sve) {
        host_->set_preg(tail_conf_.pred.s, tail_conf_.tail_size, temp_xreg_);
    }

    is_tail_ready_ = true;
}

template <cpu_isa_t isa>
bool jit_io_helper_t<isa>::is_supported_dt(data_type_t dt) {
    const bool dt_is_implemented
            = utils::one_of(dt, data_type::f32, data_type::s32, data_type::f16,
                    data_type::bf16, data_type::s8, data_type::u8);

    // TODO: emulate bf16 on unsupported platforms?
    return dt_is_implemented && platform::has_data_type_support(dt);
}

template <cpu_isa_t isa>
Xbyak_aarch64::_PReg jit_io_helper_t<isa>::get_pred(bool is_tail) const {
    return (is_tail ? tail_conf_.pred : host_->P_ALL_ONE) / Xbyak_aarch64::T_z;
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::saturate(data_type_t dt, const TReg &vec) const {
    if (!types::is_integral_dt(dt)) { return; }

    host_->saturate_f32(vec, TReg(sat_conf_.lbound_vec.getIdx()),
            TReg(sat_conf_.ubound_vec.getIdx()), data_type::s8,
            host_->P_ALL_ONE, true);
    host_->uni_frinti(vec.s, vec.s);
    host_->uni_fcvtzs(vec.s, vec.s);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::extend_xf16_to_f32(
        data_type_t dt, const TReg &vec) const {
    assert(dt == data_type::f16 || dt == data_type::bf16);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        if (dt == data_type::f16) {
            host_->fcvt(sve_vec.s, host_->P_ALL_ONE, sve_vec.h);
        } else {
            host_->lsl(sve_vec.s, host_->P_ALL_ONE, 16);
        }
    } else {
        const auto &asimd_vec = to_vreg(vec);
        if (dt == data_type::f16) {
            host_->fcvtl(asimd_vec.s4, asimd_vec.h4);
        } else {
            host_->shll(asimd_vec.s4, asimd_vec.h4, 16);
        }
    }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::narrow_f32_to_xf16(
        data_type_t dt, const TReg &vec) const {
    assert(dt == data_type::f16 || dt == data_type::bf16);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        if (dt == data_type::f16) {
            host_->fcvt(sve_vec.h, host_->P_ALL_ONE, sve_vec.s);
        } else {
            host_->bfcvt(sve_vec.h, host_->P_ALL_ONE, sve_vec.s);
        }
    } else {
        const auto &asimd_vec = to_vreg(vec);
        if (dt == data_type::f16) {
            host_->fcvtn(asimd_vec.h4, asimd_vec.s4);
        } else {
            host_->bfcvtn(asimd_vec.h4, asimd_vec.s4);
        }
    }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::load(
        data_type_t dt, const TReg &vec, const XReg &src_addr, bool is_tail) {
    assert(is_supported_dt(dt));

    lazy_init_tail(is_tail);

    switch (types::data_type_size(dt)) {
        case word_size: load_words(dt, vec, src_addr, is_tail); break;
        case half_size: load_halfwords(dt, vec, src_addr, is_tail); break;
        case byte_size: load_bytes(dt, vec, src_addr, is_tail); break;
        default: assert(!"unreachable");
    }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::gather_load(data_type_t dt, const TReg &vec,
        const Xbyak_aarch64::XReg &base_addr, const TReg &offsets_vec,
        const bool is_tail) {
    assert(is_supported_dt(dt));

    lazy_init_tail(is_tail);

    switch (types::data_type_size(dt)) {
        case word_size:
            gather_load_words(dt, vec, base_addr, offsets_vec, is_tail);
            break;
        case half_size:
            gather_load_halfwords(dt, vec, base_addr, offsets_vec, is_tail);
            break;
        case byte_size:
            gather_load_bytes(dt, vec, base_addr, offsets_vec, is_tail);
            break;
        default: assert(!"unreachable");
    }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::store(
        data_type_t dt, const TReg &vec, const XReg &dst_addr, bool is_tail) {
    assert(is_supported_dt(dt));

    lazy_init_saturation(dt);
    lazy_init_tail(is_tail);

    switch (types::data_type_size(dt)) {
        case word_size: store_words(dt, vec, dst_addr, is_tail); break;
        case half_size: store_halfwords(dt, vec, dst_addr, is_tail); break;
        case byte_size: store_bytes(dt, vec, dst_addr, is_tail); break;
        default: assert(!"unreachable");
    }
}

// This function is meaningless on SVE and should never be implemented or called
template <>
template <size_t lane_size, bool is_load>
void jit_io_helper_t<sve>::gen_asimd_load_store(
        const VReg &, const XReg &, bool is_tail) const {
    assert(!"Do not call this on SVE");
}

// This function generates efficient combinations of floating-point scalar and
// vector loads to simulate masked loads on asimd
template <>
template <size_t lane_size, bool is_load>
void jit_io_helper_t<asimd>::gen_asimd_load_store(
        const VReg &vec, const XReg &addr, bool is_tail) const {
    // We define three major types: the "Base" type, which is the element size
    // (word, halfword, byte) of the type T we are trying to load, a "Larger"
    // type, which is the element size "one larger" than Base, and a "Full"
    // type, which is one larger than "Larger"
    // For example:
    // if T = s8  --> Base = byte --> Larger = halfword   --> Full = word
    // if T = f32 --> Base = word --> Larger = doubleword --> Full = quadword
    //
    // Similarly each Base and Larger can be either a treated as a
    // Scalar floating-point register (BReg, HReg, SReg, ...) or a
    // Vector register (VReg16B, VReg8H, VReg4S, ...)
    //
    // Since we are loading with the goal of converting to f32, there are only
    // three tail cases (and one full-load case):
    // A general load of type T in asimd can always be broken down into:
    // 1) a destructive Scalar floating-point load of the Base size
    // 2) a destructive Scalar floating-point load of the Larger size
    // 3) a destructive Scalar floating-point load of the Larger size
    //    + a constructive vector lane-load of the Base size (into lane index 2)
    // 4) a non-tail destructive floating-point load of the Full size
    //
    // For example:
    //
    // if T = f16 --> Base = halfword --> Larger = word --> Full = doubleword
    // tail = 1 --> 1 HReg load
    // tail = 2 --> 1 SReg load
    // tail = 3 --> 1 SReg load + 1 VReg8H[2] lane-load
    // non-tail --> 1 DReg load
    //
    // Stores work exactly the same, except for the underlying instruction used.
    using VectorBase =
            typename std::conditional<lane_size == byte_size, VReg16B,
                    typename std::conditional<lane_size == half_size, VReg8H,
                            VReg4S>::type>::type;
    using ScalarBase = typename std::conditional<lane_size == byte_size, BReg,
            typename std::conditional<lane_size == half_size, HReg,
                    SReg>::type>::type;
    using ScalarLarger = typename std::conditional<lane_size == byte_size, HReg,
            typename std::conditional<lane_size == half_size, SReg,
                    DReg>::type>::type;
    using ScalarFull = typename std::conditional<lane_size == byte_size, SReg,
            typename std::conditional<lane_size == half_size, DReg,
                    QReg>::type>::type;

    const VectorBase vec_base(vec.getIdx());
    const ScalarBase scalar_base(vec.getIdx());
    const ScalarLarger scalar_large(vec.getIdx());
    const ScalarFull scalar_full(vec.getIdx());

    if (is_load) {
        if (!is_tail) { return host_->ldr(scalar_full, ptr(addr)); }

        switch (tail_conf_.tail_size) {
            case 0: break;
            case 1: host_->ldr(scalar_base, ptr(addr)); break;
            case 2: host_->ldr(scalar_large, ptr(addr)); break;
            case 3:
                host_->ldr(scalar_large, ptr(addr));
                host_->add(temp_xreg_, addr, 2 * lane_size);
                host_->ld1(vec_base[2], ptr(temp_xreg_));
                break;
            default: assert(!"unreachable");
        }
    } else {
        if (!is_tail) { return host_->str(scalar_full, ptr(addr)); }

        switch (tail_conf_.tail_size) {
            case 0: break;
            case 1: host_->str(scalar_base, ptr(addr)); break;
            case 2: host_->str(scalar_large, ptr(addr)); break;
            case 3:
                host_->str(scalar_large, ptr(addr));
                host_->add(temp_xreg_, addr, 2 * lane_size);
                host_->st1(vec_base[2], ptr(temp_xreg_));
                break;
            default: assert(!"unreachable");
        }
    }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::broadcast_load(
        data_type_t dt, const TReg &vec, const Xbyak_aarch64::XReg &src_addr) {
    assert(is_supported_dt(dt));

    switch (types::data_type_size(dt)) {
        case word_size: broadcast_load_word(dt, vec, src_addr); break;
        case half_size: broadcast_load_halfword(dt, vec, src_addr); break;
        case byte_size: broadcast_load_byte(dt, vec, src_addr); break;
        default: assert(!"unreachable");
    }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::broadcast_load_word(data_type_t dt, const TReg &vec,
        const Xbyak_aarch64::XReg &src_addr) const {
    assert(dt == data_type::f32 || dt == data_type::s32);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        host_->ld1rw(sve_vec.s, host_->P_ALL_ONE, ptr(src_addr));
    } else {
        const auto &asimd_vec = to_vreg(vec);
        host_->ld1r(asimd_vec.s4, ptr(src_addr));
    }

    if (dt == data_type::s32) { host_->uni_scvtf(vec.s, vec.s); }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::load_words(data_type_t dt, const TReg &vec,
        const XReg &src_addr, bool is_tail) const {
    assert(dt == data_type::f32 || dt == data_type::s32);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        host_->ld1w(sve_vec.s, get_pred(is_tail), ptr(src_addr));
    } else {
        const auto &asimd_vec = to_vreg(vec);
        gen_asimd_load_store<word_size, true>(asimd_vec, src_addr, is_tail);
    }

    if (dt == data_type::s32) { host_->uni_scvtf(vec.s, vec.s); }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::gather_load_words(data_type_t dt, const TReg &vec,
        const XReg &base_addr, const TReg &offsets_vec,
        const bool is_tail) const {
    assert(dt == data_type::f32 || dt == data_type::s32);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        const auto &sve_offsets = to_zreg(offsets_vec);
        host_->ld1w(sve_vec.s, get_pred(is_tail),
                ptr(base_addr, sve_offsets.s, SXTW));
    } else {
        const auto &asimd_vec = to_vreg(vec);
        const auto &asimd_offsets = to_vreg(offsets_vec);
        const int num_lanes_to_load = is_tail ? tail_conf_.tail_size : 4;

        for (int i = 0; i < num_lanes_to_load; ++i) {
            host_->mov(WReg(temp_xreg_.getIdx()), asimd_offsets.s[i]);
            host_->add(temp_xreg_, base_addr, temp_xreg_);
            host_->ld1(asimd_vec.s[i], ptr(temp_xreg_));
        }
    }

    if (dt == data_type::s32) { host_->uni_scvtf(vec.s, vec.s); }
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::store_words(data_type_t dt, const TReg &vec,
        const XReg &dst_addr, bool is_tail) const {
    assert(dt == data_type::f32 || dt == data_type::s32);

    saturate(dt, vec);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        host_->st1w(sve_vec.s, get_pred(is_tail), ptr(dst_addr));
        return;
    }

    const auto &asimd_vec = to_vreg(vec);
    gen_asimd_load_store<word_size, false>(asimd_vec, dst_addr, is_tail);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::broadcast_load_halfword(data_type_t dt,
        const TReg &vec, const Xbyak_aarch64::XReg &src_addr) const {
    assert(dt == data_type::f16 || dt == data_type::bf16);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        host_->ld1rh(sve_vec.s, host_->P_ALL_ONE, ptr(src_addr));
    } else {
        const auto &asimd_vec = to_vreg(vec);
        host_->ld1r(asimd_vec.h8, ptr(src_addr));
    }

    extend_xf16_to_f32(dt, vec);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::load_halfwords(data_type_t dt, const TReg &vec,
        const XReg &src_addr, bool is_tail) const {
    assert(dt == data_type::f16 || dt == data_type::bf16);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);

        host_->ld1h(sve_vec.s, get_pred(is_tail), ptr(src_addr));

        extend_xf16_to_f32(dt, vec);
        return;
    }

    const auto &asimd_vec = to_vreg(vec);
    gen_asimd_load_store<half_size, true>(asimd_vec, src_addr, is_tail);

    extend_xf16_to_f32(dt, vec);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::gather_load_halfwords(data_type_t dt,
        const TReg &vec, const XReg &base_addr, const TReg &offsets_vec,
        const bool is_tail) const {
    assert(dt == data_type::f16 || dt == data_type::bf16);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        const auto &sve_offsets = to_zreg(offsets_vec);
        host_->ld1h(sve_vec.s, get_pred(is_tail),
                ptr(base_addr, sve_offsets.s, SXTW));
    } else {
        const auto &asimd_vec = to_vreg(vec);
        const auto &asimd_offsets = to_vreg(offsets_vec);
        const int num_lanes_to_load = is_tail ? tail_conf_.tail_size
                                              : simd_elems(data_type::f32, isa);

        for (int i = 0; i < num_lanes_to_load; ++i) {
            host_->mov(WReg(temp_xreg_.getIdx()), asimd_offsets.s[i]);
            host_->add(temp_xreg_, base_addr, temp_xreg_);
            host_->ld1(asimd_vec.h[i], ptr(temp_xreg_));
        }
    }

    extend_xf16_to_f32(dt, vec);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::store_halfwords(data_type_t dt, const TReg &vec,
        const XReg &dst_addr, bool is_tail) const {
    assert(dt == data_type::f16 || dt == data_type::bf16);

    narrow_f32_to_xf16(dt, vec);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        host_->st1h(sve_vec.s, get_pred(is_tail), ptr(dst_addr));
        return;
    }

    const auto &asimd_vec = to_vreg(vec);

    gen_asimd_load_store<half_size, false>(asimd_vec, dst_addr, is_tail);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::broadcast_load_byte(data_type_t dt, const TReg &vec,
        const Xbyak_aarch64::XReg &src_addr) const {
    assert(dt == data_type::s8 || dt == data_type::u8);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        if (dt == data_type::s8) {
            host_->ld1rsb(sve_vec.s, host_->P_ALL_ONE, ptr(src_addr));
        } else {
            host_->ld1rb(sve_vec.s, host_->P_ALL_ONE, ptr(src_addr));
        }
    } else {
        const auto &asimd_vec = to_vreg(vec);
        host_->ld1r(asimd_vec.b16, ptr(src_addr));
        asimd_extend_i8_to_s32(host_, dt, asimd_vec);
    }

    host_->uni_scvtf(vec.s, vec.s);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::load_bytes(data_type_t dt, const TReg &vec,
        const XReg &src_addr, bool is_tail) const {
    assert(dt == data_type::s8 || dt == data_type::u8);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        if (dt == data_type::s8) {
            host_->ld1sb(sve_vec.s, get_pred(is_tail), ptr(src_addr));
        } else {
            host_->ld1b(sve_vec.s, get_pred(is_tail), ptr(src_addr));
        }
    } else {
        const auto &asimd_vec = to_vreg(vec);
        gen_asimd_load_store<byte_size, true>(asimd_vec, src_addr, is_tail);
        asimd_extend_i8_to_s32(host_, dt, asimd_vec);
    }

    host_->uni_scvtf(vec.s, vec.s);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::gather_load_bytes(data_type_t dt, const TReg &vec,
        const XReg &base_addr, const TReg &offsets_vec,
        const bool is_tail) const {
    assert(dt == data_type::s8 || dt == data_type::u8);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        const auto &sve_offsets = to_zreg(offsets_vec);
        if (dt == data_type::s8) {
            host_->ld1sb(sve_vec.s, get_pred(is_tail),
                    ptr(base_addr, sve_offsets.s, SXTW));
        } else {
            host_->ld1b(sve_vec.s, get_pred(is_tail),
                    ptr(base_addr, sve_offsets.s, SXTW));
        }
    } else {
        const auto &asimd_vec = to_vreg(vec);
        const auto &asimd_offsets = to_vreg(offsets_vec);
        const int num_lanes_to_load = is_tail ? tail_conf_.tail_size : 4;

        for (int i = 0; i < num_lanes_to_load; ++i) {
            host_->mov(WReg(temp_xreg_.getIdx()), asimd_offsets.s[i]);
            host_->add(temp_xreg_, base_addr, temp_xreg_);
            host_->ld1(asimd_vec.b[i], ptr(temp_xreg_));
        }

        asimd_extend_i8_to_s32(host_, dt, asimd_vec);
    }

    host_->uni_scvtf(vec.s, vec.s);
}

template <cpu_isa_t isa>
void jit_io_helper_t<isa>::store_bytes(data_type_t dt, const TReg &vec,
        const XReg &dst_addr, bool is_tail) const {
    assert(dt == data_type::s8 || dt == data_type::u8);

    saturate(dt, vec);

    if (isa == sve) {
        const auto &sve_vec = to_zreg(vec);
        host_->st1b(sve_vec.s, get_pred(is_tail), ptr(dst_addr));
        return;
    }

    const auto &asimd_vec = to_vreg(vec);
    asimd_narrow_s32_to_i8(host_, dt, asimd_vec);
    gen_asimd_load_store<byte_size, false>(asimd_vec, dst_addr, is_tail);
}

template class jit_io_helper_t<asimd>;
template class jit_io_helper_t<sve>;

} // namespace io
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
