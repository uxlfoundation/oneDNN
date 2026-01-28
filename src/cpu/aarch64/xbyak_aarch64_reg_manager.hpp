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
#ifndef CPU_AARCH64_XBYAK_AARCH64_REG_MANAGER_HPP
#define CPU_AARCH64_XBYAK_AARCH64_REG_MANAGER_HPP

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>
#include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h"
#include <type_traits>

namespace Xbyak_aarch64 {

// Static definitions for different types of registers in relation to which family they belong to.
enum class RegFamily { GP, FP, Pred };

template <class RegT>
struct reg_family;

template <>
struct reg_family<WReg> {
    static constexpr RegFamily value = RegFamily::GP;
};
template <>
struct reg_family<XReg> {
    static constexpr RegFamily value = RegFamily::GP;
};

template <>
struct reg_family<BReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<HReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<SReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<DReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<QReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg8B> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg16B> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg4H> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg8H> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg2S> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg4S> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg1D> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<VReg2D> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<ZReg> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<ZRegB> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<ZRegH> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<ZRegS> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<ZRegD> {
    static constexpr RegFamily value = RegFamily::FP;
};
template <>
struct reg_family<ZRegQ> {
    static constexpr RegFamily value = RegFamily::FP;
};

template <>
struct reg_family<PReg> {
    static constexpr RegFamily value = RegFamily::Pred;
};
template <>
struct reg_family<PRegB> {
    static constexpr RegFamily value = RegFamily::Pred;
};
template <>
struct reg_family<PRegH> {
    static constexpr RegFamily value = RegFamily::Pred;
};
template <>
struct reg_family<PRegS> {
    static constexpr RegFamily value = RegFamily::Pred;
};
template <>
struct reg_family<PRegD> {
    static constexpr RegFamily value = RegFamily::Pred;
};

class RegPoolManager {
public:
    // Usage:
    // XReg x_eg = rm.alloc<XReg>(); // Allocate next available X register (freed by user)
    // rm.free(x_eg);                // Free x_eg

    // register allocation method - accepts int to specify an unused reg, or no arg to get next free reg
    template <class RegT>
    RegT alloc() {
        switch (reg_family<RegT>::value) {
            case RegFamily::GP: {
                const int idx = next_gp_idx();
                gp_reg(idx);
                return RegT(idx);
            }
            case RegFamily::FP: {
                const int idx = next_fp_idx();
                fp_reg(idx);
                return RegT(idx);
            }
            case RegFamily::Pred: {
                const int idx = next_pred_idx();
                pred_reg(idx);
                return RegT(idx);
            }
            default: throw std::runtime_error("Unknown register family");
        }
    }
    template <class RegT>
    RegT alloc(int idx) {
        switch (reg_family<RegT>::value) {
            case RegFamily::GP: gp_reg(idx); return RegT(idx);
            case RegFamily::FP: fp_reg(idx); return RegT(idx);
            case RegFamily::Pred: pred_reg(idx); return RegT(idx);
            default: throw std::runtime_error("Unknown register family");
        }
    }

    // takes register object and moves it from in use to free set
    template <class RegT>
    void free(RegT reg) {
        const int idx = reg.getIdx();
        switch (reg_family<RegT>::value) {
            case RegFamily::GP: release_gp(idx); break;
            case RegFamily::FP: release_fp(idx); break;
            case RegFamily::Pred: release_pred(idx); break;
            default: throw std::runtime_error("Unknown register family");
        }
    }

    // getter methods - return vectors of register indices representing a set
    std::vector<int> get_free_gps() const {
        return make_index_vector(free_gp_regs);
    }
    std::vector<int> get_in_use_gps() const {
        return make_index_vector(in_use_gp);
    }
    std::vector<int> get_preserved_gps() const {
        return make_index_vector(preserved_gp);
    }
    std::vector<int> get_used_gps() const { return make_index_vector(used_gp); }

    std::vector<int> get_free_fps() const {
        return make_index_vector(free_fp_regs);
    }
    std::vector<int> get_in_use_fps() const {
        return make_index_vector(in_use_fp);
    }
    std::vector<int> get_preserved_fps() const {
        return make_index_vector(preserved_fp);
    }
    std::vector<int> get_used_fps() const { return make_index_vector(used_fp); }

    std::vector<int> get_free_preds() const {
        return make_index_vector(free_pred_regs);
    }
    std::vector<int> get_in_use_preds() const {
        return make_index_vector(in_use_pred);
    }
    std::vector<int> get_preserved_preds() const {
        return make_index_vector(preserved_pred);
    }
    std::vector<int> get_used_preds() const {
        return make_index_vector(used_pred);
    }

    // member function - add a special register to the free pool of general registers
    void add_to_gp_pool(const XReg &reg) { add_to_gp_pool(reg.getIdx()); }
    void add_to_gp_pool(int idx) {
        if ((idx < 0 || idx > 31))
            throw std::runtime_error("Register index out of range");
        const bool in_free = free_gp_regs.count(idx) != 0;
        const bool in_preserved = preserved_gp.count(idx) != 0;
        const bool in_use = in_use_gp.count(idx) != 0;
        if (in_free || in_preserved || in_use)
            throw std::runtime_error("Register already tracked");
        free_gp_regs.insert(idx);
    }

    // helper function - returns true if a register object is currently in the used set of registers
    template <class RegT>
    bool reg_in_use(const RegT &reg) const {
        return reg_in_use_idx(reg.getIdx(), reg_family<RegT>::value);
    }

    // helper functions - returns true if an index in a given family is in use
    bool gp_idx_in_use(int reg_idx) const {
        return reg_in_use_idx(reg_idx, RegFamily::GP);
    }
    bool fp_idx_in_use(int reg_idx) const {
        return reg_in_use_idx(reg_idx, RegFamily::FP);
    }
    bool pred_idx_in_use(int reg_idx) const {
        return reg_in_use_idx(reg_idx, RegFamily::Pred);
    }

    // scoped register handling with RAII
    // usage: XReg reg11 = rm.alloc<XReg>(11);
    //        auto scoped = rm.makeScoped(reg11);
    // or
    //        auto scoped_reg = rm.makeScoped(rm.alloc<XReg>());
    // reg11 & scoped_reg will free at end of scope when guards' dtors called.
    template <class Reg>
    class Scoped {
    public:
        explicit Scoped(RegPoolManager &rm, Reg r)
            // pointer to allocator, allocate scoped reg at construction & track, unowned = nullptr
            : rm_(&rm), reg_(r) {
            validate_scoped_reg(rm_, reg_);
        }

        // if object is owner of scoped reg and goes out of scope, deallocate
        ~Scoped() {
            if (!rm_) return;
            rm_->free(reg_);
        }

        // disable copy - scoped regs are move only to avoid ownership/double free issues as per RAII
        Scoped(const Scoped &) = delete;
        Scoped &operator=(const Scoped &) = delete;

        // move constructor - used when scoped regs initialised from rvalue (incl. std::move)
        Scoped(Scoped &&other) noexcept : rm_(other.rm_), reg_(other.reg_) {
            other.rm_ = nullptr; // set previous owner to no longer own
        }

        // expose underlying register for implicit use in JIT helpers
        operator const Reg &() const noexcept { return reg_; }
        const Reg &get() const noexcept { return reg_; }

    private:
        RegPoolManager *rm_
                = nullptr; // pointer to allocator, initialised as nullptr
        Reg reg_ {};
    };

    // helper factory - calls Scoped ctor
    template <class Reg>
    inline Scoped<Reg> makeScoped(Reg r) & {
        return Scoped<Reg>(*this, r);
    }

    // helper methods to return special registers as per AArch64 calling convention
    // NOTE: indirect results register (x8) has been added to the call clobbered general pool
    // intra-procedure-call register 0 : x16
    inline XReg _ipc_reg_0() {
        used_gp.insert(16);
        return XReg(16);
    }
    // intra-procedure-call register 1 : x17
    inline XReg _ipc_reg_1() {
        used_gp.insert(17);
        return XReg(17);
    }
    // platform register : x18
    inline XReg _platform_reg() {
        used_gp.insert(18);
        return XReg(18);
    }
    // frame pointer : x29
    inline XReg _frame_pointer() {
        used_gp.insert(29);
        return XReg(29);
    }
    // link register : x30
    inline XReg _link_reg() {
        used_gp.insert(30);
        return XReg(30);
    }

private:
    // helper method - converts members of set to vector
    static inline std::vector<int> make_index_vector(const std::set<int> &set) {
        std::vector<int> indices;
        indices.reserve(set.size());
        for (int idx : set)
            indices.emplace_back(idx);
        return indices;
    }

    // helper method - checks reg in use before scoping
    template <class RegT>
    static void validate_scoped_reg(RegPoolManager *rm, RegT reg) {
        const RegFamily family = reg_family<RegT>::value;
        if (!rm->reg_in_use(reg))
            throw std::runtime_error(scoped_reg_error(family));
    }

    // helper switch case for error messages
    static const char *scoped_reg_error(RegFamily family) noexcept {
        switch (family) {
            case RegFamily::GP:
                return "Cannot create GP scoped reg for a register that is not "
                       "in use";
            case RegFamily::FP:
                return "Cannot create FP scoped reg for a register that is not "
                       "in use";
            case RegFamily::Pred:
                return "Cannot create predicate scoped reg for a register that "
                       "is not in use";
            default: return "Cannot create scoped reg for unknown family";
        }
    }

    // helper method - checks if a register index for a given family is currently in use
    bool reg_in_use_idx(int idx, RegFamily family) const {
        switch (family) {
            case RegFamily::GP:
                if (idx < 0 || idx > 31) {
                    throw std::runtime_error("GP register index out of range");
                }
                return in_use_gp.find(idx) != in_use_gp.end();
            case RegFamily::FP:
                if (idx < 0 || idx > 31) {
                    throw std::runtime_error("FP register index out of range");
                }
                return in_use_fp.find(idx) != in_use_fp.end();
            case RegFamily::Pred:
                if (idx < 0 || idx > 15) {
                    throw std::runtime_error(
                            "Predicate register index out of range");
                }
                return in_use_pred.find(idx) != in_use_pred.end();
            default: throw std::runtime_error("Unknown register family");
        }
    }

    // helper method - finds next free register for a given family
    int next_gp_idx() const {
        if (!free_gp_regs.empty()) return *free_gp_regs.begin();
        if (!preserved_gp.empty()) return *preserved_gp.begin();
        throw std::runtime_error("No free GP registers available");
    }
    int next_fp_idx() const {
        if (!free_fp_regs.empty()) return *free_fp_regs.begin();
        if (!preserved_fp.empty()) return *preserved_fp.begin();
        throw std::runtime_error("No free FP registers available");
    }
    int next_pred_idx() const {
        if (!free_pred_regs.empty()) return *free_pred_regs.begin();
        if (!preserved_pred.empty()) return *preserved_pred.begin();
        throw std::runtime_error("No free predicate registers available");
    }

    // tracking for in-use indices for a given register family
    void gp_reg(int idx) {
        if (reg_in_use_idx(idx, RegFamily::GP))
            throw std::runtime_error("Specified GP register currently in use");
        auto it = free_gp_regs.find(idx);
        auto pres_it = preserved_gp.find(idx);
        if (it != free_gp_regs.end()) {
            in_use_gp.insert(idx);
            free_gp_regs.erase(it);
            used_gp.insert(idx);
        } else if (pres_it != preserved_gp.end()) {
            in_use_gp.insert(idx);
            preserved_gp.erase(pres_it);
            used_gp.insert(idx);
        } else {
            throw std::runtime_error(
                    "Requested register not in free/preserved pools.");
        }
    }
    void fp_reg(int idx) {
        if (reg_in_use_idx(idx, RegFamily::FP))
            throw std::runtime_error("Specified FP register currently in use");
        auto it = free_fp_regs.find(idx);
        auto pres_it = preserved_fp.find(idx);
        if (it != free_fp_regs.end()) {
            in_use_fp.insert(idx);
            free_fp_regs.erase(it);
            used_fp.insert(idx);
        } else if (pres_it != preserved_fp.end()) {
            in_use_fp.insert(idx);
            preserved_fp.erase(pres_it);
            used_fp.insert(idx);
        } else {
            throw std::runtime_error(
                    "Requested register not in free/preserved/in-use sets.");
        }
    }
    void pred_reg(int idx) {
        if (reg_in_use_idx(idx, RegFamily::Pred))
            throw std::runtime_error(
                    "Specified Pred register currently in use");
        auto it = free_pred_regs.find(idx);
        auto pres_it = preserved_pred.find(idx);
        if (it != free_pred_regs.end()) {
            in_use_pred.insert(idx);
            free_pred_regs.erase(it);
            used_pred.insert(idx);
        } else if (pres_it != preserved_pred.end()) {
            in_use_pred.insert(idx);
            preserved_pred.erase(pres_it);
            used_pred.insert(idx);
        } else {
            throw std::runtime_error(
                    "Requested predicate register not in free/preserved "
                    "pools.");
        }
    }

    // member function - moves given index from in-use set to free set for given family
    void release_gp(int idx) {
        if ((idx < 0 || idx > 31))
            throw std::runtime_error("Register index out of range");
        auto it = in_use_gp.find(idx);
        if (it == in_use_gp.end())
            throw std::runtime_error("GP register not in use");
        in_use_gp.erase(it);
        free_gp_regs.insert(idx);
    }
    void release_fp(int idx) {
        if ((idx < 0 || idx > 31))
            throw std::runtime_error("Register index out of range");
        auto it = in_use_fp.find(idx);
        if (it == in_use_fp.end())
            throw std::runtime_error("FP register not in use");
        in_use_fp.erase(it);
        free_fp_regs.insert(idx);
    }
    void release_pred(int idx) {
        if ((idx < 0 || idx > 15))
            throw std::runtime_error("Predicate register index out of range");
        auto it = in_use_pred.find(idx);
        if (it == in_use_pred.end())
            throw std::runtime_error("Predicate register not in use");
        in_use_pred.erase(it);
        free_pred_regs.insert(idx);
    }

    // set of call clobbered registers: 0 - 15
    static const std::set<int> &base_free_gp() {
        static const std::set<int> s {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return s;
    }
    // set of call preserved registers: 19 - 28
    static const std::set<int> &base_preserved_gp() {
        static const std::set<int> s {19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
        return s;
    }

    std::set<int> used_gp;
    std::set<int> in_use_gp;
    std::set<int> free_gp_regs = base_free_gp();
    std::set<int> preserved_gp = base_preserved_gp();

    // set of call clobbered FP registers: 0 - 7; 16 - 31
    static const std::set<int> &base_free_fp() {
        static const std::set<int> s {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
        return s;
    }
    // set of call-preserved FP registers: 8 - 15
    static const std::set<int> &base_preserved_fp() {
        static const std::set<int> s {8, 9, 10, 11, 12, 13, 14, 15};
        return s;
    }

    std::set<int> used_fp;
    std::set<int> in_use_fp;
    std::set<int> free_fp_regs = base_free_fp();
    std::set<int> preserved_fp = base_preserved_fp();

    // set of call clobbered predicate registers: 0 - 3
    static const std::set<int> &base_free_pred() {
        static const std::set<int> s {0, 1, 2, 3};
        return s;
    }
    // set of call preserved predicate registers: 4 - 15
    static const std::set<int> &base_preserved_pred() {
        static const std::set<int> s {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return s;
    }

    std::set<int> used_pred;
    std::set<int> in_use_pred;
    std::set<int> free_pred_regs = base_free_pred();
    std::set<int> preserved_pred = base_preserved_pred();
};
} // namespace Xbyak_aarch64

#endif
