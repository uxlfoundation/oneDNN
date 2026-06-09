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

#include "xbyak_aarch64_reg.h"
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Please see the Procedure Call Standard https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst for more details.

namespace Xbyak_aarch64 {

// Static definitions for different types of registers in relation to which family they belong to.
enum class RegFamily { gp, vec, pred, unsupported };

template <class RegT> struct is_gp_reg {
  static constexpr bool value = std::is_base_of<RReg, RegT>::value;
};

template <class RegT> struct is_vec_reg {
  static constexpr bool value =
      std::is_base_of<VRegSc, RegT>::value || (std::is_base_of<VRegVec, RegT>::value && !std::is_base_of<VRegElem, RegT>::value && !std::is_base_of<VRegList, RegT>::value) || (std::is_base_of<_ZReg, RegT>::value && !std::is_base_of<ZRegElem, RegT>::value && !std::is_base_of<ZRegList, RegT>::value);
};

template <class RegT> struct is_pred_reg {
  static constexpr bool value = std::is_base_of<_PReg, RegT>::value;
};

// helper trait determines if operator- is defined for a given register type
template <class RegT> struct is_list_family_reg {
  static constexpr bool value = (std::is_base_of<VRegVec, RegT>::value && !std::is_base_of<VRegElem, RegT>::value && !std::is_base_of<VRegList, RegT>::value) || (std::is_base_of<_ZReg, RegT>::value && !std::is_base_of<ZRegElem, RegT>::value && !std::is_base_of<ZRegList, RegT>::value);
};

template <class RegT> struct reg_family {
  static constexpr RegFamily value = is_gp_reg<RegT>::value ? RegFamily::gp : (is_vec_reg<RegT>::value ? RegFamily::vec : (is_pred_reg<RegT>::value ? RegFamily::pred : RegFamily::unsupported));
};

class RegPoolManager {
public:
  // Usage:
  // XReg x_eg = rm.alloc<XReg>(); // Allocate next available X register (freed by user)
  // rm.free(x_eg);                // Free x_eg

  // register allocation method - accepts int to specify an unused reg, or no arg to get next free reg
  template <class RegT> RegT alloc() {
    switch (reg_family<RegT>::value) {
    case RegFamily::gp: {
      const int idx = next_gp_idx();
      gp_reg(idx);
      return RegT(idx);
    }
    case RegFamily::vec: {
      const int idx = next_vec_idx();
      vec_reg(idx);
      return RegT(idx);
    }
    case RegFamily::pred: {
      const int idx = next_pred_idx();
      pred_reg(idx);
      return RegT(idx);
    }
    default:
      throw Error(ERR_RM_ILLEGAL_REG_FAMILY);
    }
  }
  template <class RegT> RegT alloc(int idx) {
    switch (reg_family<RegT>::value) {
    case RegFamily::gp:
      gp_reg(idx);
      return RegT(idx);
    case RegFamily::vec:
      vec_reg(idx);
      return RegT(idx);
    case RegFamily::pred:
      pred_reg(idx);
      return RegT(idx);
    default:
      throw Error(ERR_RM_ILLEGAL_REG_FAMILY);
    }
  }

  // takes register object and moves it from live to free set
  template <class RegT> void free(RegT reg) {
    const int idx = reg.getIdx();
    switch (reg_family<RegT>::value) {
    case RegFamily::gp:
      release_gp(idx);
      break;
    case RegFamily::vec:
      release_vec(idx);
      break;
    case RegFamily::pred:
      release_pred(idx);
      break;
    default:
      throw Error(ERR_RM_ILLEGAL_REG_FAMILY);
    }
  }

  // getter methods - return vectors of register indices representing a set
  // NOTE: the "used" sets track any register index that the manager has allocated at any point; the "live" sets track registers that are currently allocated by the manager.
  // The format for the getter methods are as follows:
  //
  //  get_{register type}_{register set}_{register family}
  //  get_<all/volatile/non-volatile>_<live/free/preserved/used>_<gp/vec/pred>
  //
  // E.g. get_volatile_live_gp() returns a vector of indices of general purpose registers that are currently in use and are volatile (call clobbered) as per the AArch64 calling convention.

  std::vector<int> get_all_live_gp() const { return make_index_vector(live_gp); }
  std::vector<int> get_all_live_vec() const { return make_index_vector(live_vec); }
  std::vector<int> get_all_live_pred() const { return make_index_vector(live_pred); }
  std::vector<int> get_all_free_gp() const { return make_index_vector(free_gp_regs); }
  std::vector<int> get_all_free_vec() const { return make_index_vector(free_vec_regs); }
  std::vector<int> get_all_free_pred() const { return make_index_vector(free_pred_regs); }
  std::vector<int> get_all_preserved_gp() const { return make_index_vector(preserved_gp); }
  std::vector<int> get_all_preserved_vec() const { return make_index_vector(preserved_vec); }
  std::vector<int> get_all_preserved_pred() const { return make_index_vector(preserved_pred); }
  std::vector<int> get_all_used_gp() const { return make_index_vector(used_gp); }
  std::vector<int> get_all_used_vec() const { return make_index_vector(used_vec); }
  std::vector<int> get_all_used_pred() const { return make_index_vector(used_pred); }
  std::vector<int> get_volatile_live_gp() const { return cond_index_vector(live_gp, base_free_gp()); }
  std::vector<int> get_volatile_live_vec() const { return cond_index_vector(live_vec, base_free_vec()); }
  std::vector<int> get_volatile_live_pred() const { return cond_index_vector(live_pred, base_free_pred()); }
  std::vector<int> get_volatile_free_gp() const { return cond_index_vector(free_gp_regs, base_free_gp()); }
  std::vector<int> get_volatile_free_vec() const { return cond_index_vector(free_vec_regs, base_free_vec()); }
  std::vector<int> get_volatile_free_pred() const { return cond_index_vector(free_pred_regs, base_free_pred()); }
  std::vector<int> get_volatile_preserved_gp() const { return cond_index_vector(preserved_gp, base_free_gp()); }
  std::vector<int> get_volatile_preserved_vec() const { return cond_index_vector(preserved_vec, base_free_vec()); }
  std::vector<int> get_volatile_preserved_pred() const { return cond_index_vector(preserved_pred, base_free_pred()); }
  std::vector<int> get_volatile_used_gp() const { return cond_index_vector(used_gp, base_free_gp()); }
  std::vector<int> get_volatile_used_vec() const { return cond_index_vector(used_vec, base_free_vec()); }
  std::vector<int> get_volatile_used_pred() const { return cond_index_vector(used_pred, base_free_pred()); }
  std::vector<int> get_non_volatile_live_gp() const { return cond_index_vector(live_gp, base_preserved_gp()); }
  std::vector<int> get_non_volatile_live_vec() const { return cond_index_vector(live_vec, base_preserved_vec()); }
  std::vector<int> get_non_volatile_live_pred() const { return cond_index_vector(live_pred, base_preserved_pred()); }
  std::vector<int> get_non_volatile_free_gp() const { return cond_index_vector(free_gp_regs, base_preserved_gp()); }
  std::vector<int> get_non_volatile_free_vec() const { return cond_index_vector(free_vec_regs, base_preserved_vec()); }
  std::vector<int> get_non_volatile_free_pred() const { return cond_index_vector(free_pred_regs, base_preserved_pred()); }
  std::vector<int> get_non_volatile_preserved_gp() const { return cond_index_vector(preserved_gp, base_preserved_gp()); }
  std::vector<int> get_non_volatile_preserved_vec() const { return cond_index_vector(preserved_vec, base_preserved_vec()); }
  std::vector<int> get_non_volatile_preserved_pred() const { return cond_index_vector(preserved_pred, base_preserved_pred()); }
  std::vector<int> get_non_volatile_used_gp() const { return cond_index_vector(used_gp, base_preserved_gp()); }
  std::vector<int> get_non_volatile_used_vec() const { return cond_index_vector(used_vec, base_preserved_vec()); }
  std::vector<int> get_non_volatile_used_pred() const { return cond_index_vector(used_pred, base_preserved_pred()); }

  // clear methods - remove indices from sets for a given family
  // The format for the clear methods are as follows:
  //
  //  clear_{register set}_{register family}
  //  clear_<live/free/preserved/used>_<gp/vec/pred>
  //
  // E.g. clear_free_vec() would remove any indices for vector registers that are currently in the free_vec_regs set.
  // NOTE: clear is not a safe operation to call unless you are sure that the indices being cleared are not currently in use by the manager.
  void clear_live_gp() { live_gp.clear(); }
  void clear_live_vec() { live_vec.clear(); }
  void clear_live_pred() { live_pred.clear(); }
  void clear_free_gp() { free_gp_regs.clear(); }
  void clear_free_vec() { free_vec_regs.clear(); }
  void clear_free_pred() { free_pred_regs.clear(); }
  void clear_preserved_gp() { preserved_gp.clear(); }
  void clear_preserved_vec() { preserved_vec.clear(); }
  void clear_preserved_pred() { preserved_pred.clear(); }
  void clear_used_gp() { used_gp.clear(); }
  void clear_used_vec() { used_vec.clear(); }
  void clear_used_pred() { used_pred.clear(); }

  // member function - add a special or reserved register to the free pool of registers
  void add_to_gp_pool(const XReg &reg) { add_to_gp_pool(reg.getIdx()); }
  void add_to_gp_pool(int idx) {
    if ((idx < 0 || idx > 31))
      throw Error(ERR_ILLEGAL_REG_IDX);
    const bool in_free = free_gp_regs.count(idx) != 0;
    const bool in_preserved = preserved_gp.count(idx) != 0;
    const bool in_use = live_gp.count(idx) != 0;
    if (in_free || in_preserved || in_use)
      throw Error(ERR_RM_REG_ALREADY_TRACKED);
    free_gp_regs.insert(idx);
  }
  void add_to_vec_pool(int idx) {
    if ((idx < 0 || idx > 31))
      throw Error(ERR_ILLEGAL_REG_IDX);
    const bool in_free = free_vec_regs.count(idx) != 0;
    const bool in_preserved = preserved_vec.count(idx) != 0;
    const bool in_use = live_vec.count(idx) != 0;
    if (in_free || in_preserved || in_use)
      throw Error(ERR_RM_REG_ALREADY_TRACKED);
    free_vec_regs.insert(idx);
  }
  void add_to_pred_pool(int idx) {
    if ((idx < 0 || idx > 15))
      throw Error(ERR_ILLEGAL_REG_IDX);
    const bool in_free = free_pred_regs.count(idx) != 0;
    const bool in_preserved = preserved_pred.count(idx) != 0;
    const bool in_use = live_pred.count(idx) != 0;
    if (in_free || in_preserved || in_use)
      throw Error(ERR_RM_REG_ALREADY_TRACKED);
    free_pred_regs.insert(idx);
  }
  // helper function - returns true if a register object is currently in the used set of registers
  template <class RegT> bool reg_in_use(const RegT &reg) const { return reg_in_use_idx(reg.getIdx(), reg_family<RegT>::value); }

  // reserve an index for a register family to prevent it from being touched by the register manager. Use add_to_*_pool to re-add a reserved register.
  // this is preferable if the register is being used outside the managers control, or if the user doesn't want to track the register in the used set e.g. if the register is preserved and in use, but has already been saved to stack
  void reserve_gp(const int idx) {
    if (live_gp.count(idx) != 0) {
      throw Error(ERR_RM_REG_IN_USE);
    } else if (free_gp_regs.count(idx) != 0) {
      free_gp_regs.erase(idx);
      return;
    } else if (preserved_gp.count(idx) != 0) {
      preserved_gp.erase(idx);
      return;
    } else {
      throw Error(ERR_RM_REG_NOT_TRACKED);
    }
  }
  void reserve_vec(const int idx) {
    if (live_vec.count(idx) != 0) {
      throw Error(ERR_RM_REG_IN_USE);
    } else if (free_vec_regs.count(idx) != 0) {
      free_vec_regs.erase(idx);
      return;
    } else if (preserved_vec.count(idx) != 0) {
      preserved_vec.erase(idx);
      return;
    } else {
      throw Error(ERR_RM_REG_NOT_TRACKED);
    }
  }
  void reserve_pred(const int idx) {
    if (live_pred.count(idx) != 0) {
      throw Error(ERR_RM_REG_IN_USE);
    } else if (free_pred_regs.count(idx) != 0) {
      free_pred_regs.erase(idx);
      return;
    } else if (preserved_pred.count(idx) != 0) {
      preserved_pred.erase(idx);
      return;
    } else {
      throw Error(ERR_RM_REG_NOT_TRACKED);
    }
  }
  void reserve(const Reg &reg) {
    switch (reg_family<Reg>::value) {
    case RegFamily::gp:
      reserve_gp(reg.getIdx());
      break;
    case RegFamily::vec:
      reserve_vec(reg.getIdx());
      break;
    case RegFamily::pred:
      reserve_pred(reg.getIdx());
      break;
    default:
      throw Error(ERR_RM_ILLEGAL_REG_FAMILY);
    }
  }
  // helper functions - returns true if an index in a given family is in use
  bool gp_idx_in_use(int reg_idx) const { return reg_in_use_idx(reg_idx, RegFamily::gp); }
  bool vec_idx_in_use(int reg_idx) const { return reg_in_use_idx(reg_idx, RegFamily::vec); }
  bool pred_idx_in_use(int reg_idx) const { return reg_in_use_idx(reg_idx, RegFamily::pred); }

  // scoped register handling with RAII
  // usage: XReg reg11 = rm.alloc<XReg>(11);
  //        auto scoped = rm.makeScoped(reg11);
  // or
  //        auto scoped_reg = rm.allocScoped<XReg>();
  // reg11 & scoped_reg will free at end of scope when guards' dtors called.
  template <class Reg> class Scoped {
  public:
    explicit Scoped(RegPoolManager &rm, Reg r)
        // pointer to allocator, allocate scoped reg at construction & track, unowned = nullptr
        : rm_(&rm), reg_(r), generation_(rm.generation_) {
      validate_scoped_reg(rm_, reg_);
    }

    // if object is owner of scoped reg and goes out of scope, deallocate
    // if object is not owner (e.g. moved from), or manager has been reset (generation mismatch), do not deallocate
    ~Scoped() {
      if (!rm_)
        return;
      if (generation_ != rm_->generation_)
        return;
      rm_->free(reg_);
    }

    // disable copy - scoped regs are move only to avoid ownership/double free issues as per RAII
    Scoped(const Scoped &) = delete;
    Scoped &operator=(const Scoped &) = delete;

    // move constructor - used when scoped regs initialised from rvalue (incl. std::move)
    Scoped(Scoped &&other) noexcept : rm_(other.rm_), reg_(other.reg_), generation_(other.generation_) {
      other.rm_ = nullptr; // set previous owner to no longer own
    }

    // implicit conversion operator - does not mutate Scoped objects
    // allows Scoped<Reg> to be used in place of Reg in most contexts, but requires user to be explicit if they wish to break RAII
    operator const Reg &() const {
      if (!rm_ || generation_ != rm_->generation_)
        throw Error(ERR_RM_BAD_SCOPE);
      return reg_;
    }
    // explicitly expose underlying register
    const Reg &get() const {
      if (!rm_ || generation_ != rm_->generation_)
        throw Error(ERR_RM_BAD_SCOPE);
      return reg_;
    }

    // generic forward accessors to underlying register object
    uint32_t getIdx() const { return get().getIdx(); }
    uint32_t getBit() const { return get().getBit(); }
    bool isRReg() const { return get().isRReg(); }
    bool isVRegSc() const { return get().isVRegSc(); }
    bool isVRegVec() const { return get().isVRegVec(); }
    bool isZReg() const { return get().isZReg(); }
    bool isPRegZ() const { return get().isPRegZ(); }
    bool isPRegM() const { return get().isPRegM(); }

    // predicate register specific operator/ forwarding
    template <class RegT = Reg> typename std::enable_if<std::is_same<RegT, PReg>::value, decltype(std::declval<const RegT &>() / T_z)>::type operator/(PredType t) const { return get() / t; }

    // predicate isM()/isZ() helper forwarding
    template <class RegT = Reg> typename std::enable_if<std::is_base_of<_PReg, RegT>::value, bool>::type isM() const { return get().isM(); }
    template <class RegT = Reg> typename std::enable_if<std::is_base_of<_PReg, RegT>::value, bool>::type isZ() const { return get().isZ(); }

    // vector register specific getLane() forwarding
    template <class RegT = Reg> typename std::enable_if<std::is_base_of<VRegVec, RegT>::value, uint32_t>::type getLane() const { return get().getLane(); }

    // vector register index operator[] forwarding
    template <class RegT = Reg> typename std::enable_if<is_list_family_reg<RegT>::value, decltype(std::declval<const RegT &>()[0])>::type operator[](uint32_t i) const { return get()[i]; }

    // reg list construction operator forwarding for ScopedReg-Reg and ScopedReg-ScopedReg. Reg-ScopedReg sits at namespace scope.
    template <class RegT = Reg> typename std::enable_if<is_list_family_reg<RegT>::value, decltype(std::declval<const RegT &>() - std::declval<const RegT &>())>::type operator-(const Reg &other) const { return get() - other; }
    template <class RegT = Reg> typename std::enable_if<is_list_family_reg<RegT>::value, decltype(std::declval<const RegT &>() - std::declval<const RegT &>())>::type operator-(const Scoped<Reg> &other) const { return get() - other.get(); }

  private:
    RegPoolManager *rm_ = nullptr; // pointer to allocator, initialised as nullptr
    Reg reg_{};
    std::size_t generation_ = 0;
  };

  // helper factory - calls Scoped ctor
  // must be called on a tracked register object in the in_use set of its respective family.
  template <class Reg> inline Scoped<Reg> makeScoped(Reg r) & { return Scoped<Reg>(*this, r); }

  // streamlined helper factory for allocating and scoping in one call
  template <class RegT> inline Scoped<RegT> allocScoped() & { return makeScoped(alloc<RegT>()); }
  template <class RegT> inline Scoped<RegT> allocScoped(int n) & { return makeScoped(alloc<RegT>(n)); }
  // helper for allocating and scoping a raw, non-live register in one call
  // constraints on the underlying register are the same - the raw register must represent a valid & free register index for a given family
  // DANGER: when the scoped guard goes out of scope, the raw register will still be directly accessible, but the manager will consider the index to be free
  template <class RegT> inline Scoped<RegT> allocScoped(RegT r) & { return makeScoped(alloc<RegT>(r.getIdx())); }

  // scoped reg_in_use overload
  template <class RegT> bool reg_in_use(const Scoped<RegT> &reg) const { return reg_in_use(reg.get()); }

  // helper methods to return special registers as per AArch64 calling convention
  // NOTE: indirect results register (x8) has been added to the call clobbered general pool
  // intra-procedure-call register 0 : x16
  inline XReg ipc_reg_0() {
    used_gp.insert(16);
    return XReg(16);
  }
  // intra-procedure-call register 1 : x17
  inline XReg ipc_reg_1() {
    used_gp.insert(17);
    return XReg(17);
  }
  // platform register : x18
  inline XReg platform_reg() {
    used_gp.insert(18);
    return XReg(18);
  }
  // frame pointer : x29
  inline XReg frame_pointer() {
    used_gp.insert(29);
    return XReg(29);
  }
  // link register : x30
  inline XReg link_reg() {
    used_gp.insert(30);
    return XReg(30);
  }

  // hard reset
  void reset_manager() {
    ++generation_;
    live_gp.clear();
    live_vec.clear();
    live_pred.clear();
    free_gp_regs = base_free_gp();
    free_vec_regs = base_free_vec();
    free_pred_regs = base_free_pred();
    preserved_gp = base_preserved_gp();
    preserved_vec = base_preserved_vec();
    preserved_pred = base_preserved_pred();
    used_gp.clear();
    used_vec.clear();
    used_pred.clear();
  }

private:
  // helper method - converts members of set to vector of indices
  static inline std::vector<int> make_index_vector(const std::set<int> &set) {
    std::vector<int> indices;
    indices.reserve(set.size());
    for (int idx : set)
      indices.emplace_back(idx);
    return indices;
  }

  // helper method - returns a vector of indices for an intersection of two sets
  static inline std::vector<int> cond_index_vector(const std::set<int> &set, const std::set<int> &condition_set) {
    std::vector<int> indices;
    indices.reserve(set.size());
    for (int idx : set)
      if (condition_set.count(idx))
        indices.emplace_back(idx);
    return indices;
  }

  // helper method - checks reg in live_* set before scoping
  template <class RegT> static void validate_scoped_reg(RegPoolManager *rm, RegT reg) {
    if (!rm->reg_in_use(reg))
      throw Error(ERR_RM_REG_NOT_IN_USE);
  }

  // helper method - checks if a register index for a given family is currently in use
  bool reg_in_use_idx(int idx, RegFamily family) const {
    switch (family) {
    case RegFamily::gp:
      if (idx < 0 || idx > 31) {
        throw Error(ERR_ILLEGAL_REG_IDX);
      }
      return live_gp.count(idx);
    case RegFamily::vec:
      if (idx < 0 || idx > 31) {
        throw Error(ERR_ILLEGAL_REG_IDX);
      }
      return live_vec.count(idx);
    case RegFamily::pred:
      if (idx < 0 || idx > 15) {
        throw Error(ERR_ILLEGAL_REG_IDX);
      }
      return live_pred.count(idx);
    default:
      throw Error(ERR_RM_ILLEGAL_REG_FAMILY);
    }
  }

  // helper method - finds next free register for a given family
  int next_gp_idx() const {
    if (!free_gp_regs.empty())
      return *free_gp_regs.begin();
    if (!preserved_gp.empty())
      return *preserved_gp.begin();
    throw Error(ERR_RM_NO_UNALLOCATED_REG);
  }
  int next_vec_idx() const {
    if (!free_vec_regs.empty())
      return *free_vec_regs.begin();
    if (!preserved_vec.empty())
      return *preserved_vec.begin();
    throw Error(ERR_RM_NO_UNALLOCATED_REG);
  }
  int next_pred_idx() const {
    if (!free_pred_regs.empty())
      return *free_pred_regs.begin();
    if (!preserved_pred.empty())
      return *preserved_pred.begin();
    throw Error(ERR_RM_NO_UNALLOCATED_REG);
  }

  // tracking method for in-use indices for a given register family
  void gp_reg(int idx) {
    if (reg_in_use_idx(idx, RegFamily::gp))
      throw Error(ERR_RM_REG_IN_USE);
    auto it = free_gp_regs.find(idx);
    auto pres_it = preserved_gp.find(idx);
    if (it != free_gp_regs.end()) {
      live_gp.insert(idx);
      free_gp_regs.erase(it);
      used_gp.insert(idx);
    } else if (pres_it != preserved_gp.end()) {
      live_gp.insert(idx);
      preserved_gp.erase(pres_it);
      used_gp.insert(idx);
    } else {
      throw Error(ERR_RM_REG_NOT_TRACKED);
    }
  }
  void vec_reg(int idx) {
    if (reg_in_use_idx(idx, RegFamily::vec))
      throw Error(ERR_RM_REG_IN_USE);
    auto it = free_vec_regs.find(idx);
    auto pres_it = preserved_vec.find(idx);
    if (it != free_vec_regs.end()) {
      live_vec.insert(idx);
      free_vec_regs.erase(it);
      used_vec.insert(idx);
    } else if (pres_it != preserved_vec.end()) {
      live_vec.insert(idx);
      preserved_vec.erase(pres_it);
      used_vec.insert(idx);
    } else {
      throw Error(ERR_RM_REG_NOT_TRACKED);
    }
  }
  void pred_reg(int idx) {
    if (reg_in_use_idx(idx, RegFamily::pred))
      throw Error(ERR_RM_REG_IN_USE);
    auto it = free_pred_regs.find(idx);
    auto pres_it = preserved_pred.find(idx);
    if (it != free_pred_regs.end()) {
      live_pred.insert(idx);
      free_pred_regs.erase(it);
      used_pred.insert(idx);
    } else if (pres_it != preserved_pred.end()) {
      live_pred.insert(idx);
      preserved_pred.erase(pres_it);
      used_pred.insert(idx);
    } else {
      throw Error(ERR_RM_REG_NOT_TRACKED);
    }
  }

  // member function - moves given index from in-use set to free set for given family
  void release_gp(int idx) {
    if ((idx < 0 || idx > 31))
      throw Error(ERR_ILLEGAL_REG_IDX);
    auto it = live_gp.find(idx);
    if (it == live_gp.end())
      throw Error(ERR_RM_REG_NOT_IN_USE);
    live_gp.erase(it);
    free_gp_regs.insert(idx);
  }
  void release_vec(int idx) {
    if ((idx < 0 || idx > 31))
      throw Error(ERR_ILLEGAL_REG_IDX);
    auto it = live_vec.find(idx);
    if (it == live_vec.end())
      throw Error(ERR_RM_REG_NOT_IN_USE);
    live_vec.erase(it);
    free_vec_regs.insert(idx);
  }
  void release_pred(int idx) {
    if ((idx < 0 || idx > 15))
      throw Error(ERR_ILLEGAL_REG_IDX);
    auto it = live_pred.find(idx);
    if (it == live_pred.end())
      throw Error(ERR_RM_REG_NOT_IN_USE);
    live_pred.erase(it);
    free_pred_regs.insert(idx);
  }

  // Please see the Procedure Call Standard https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst for more details on calling conventions.
  // Each register family (gp, vec, pred) has 4 tracking sets (used, whether the the register manager has allocated the register at any point; live, for the registers currently allocated by the manager; free_*_regs, which tracks untracked call clobbered registers,
  // as well as call preserved registers that have been allocated previously; preserved_*, which tracks call preserved registers that have not been allocated by the manager) and two ground truth sets (base_free/call clobbered; base_preserved/call preserved).
  // set of call clobbered registers: 0 - 15
  static const std::set<int> &base_free_gp() {
    static const std::set<int> s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    return s;
  }
  // set of call preserved registers: 19 - 28
  static const std::set<int> &base_preserved_gp() {
    static const std::set<int> s{19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
    return s;
  }

  std::set<int> used_gp;
  std::set<int> live_gp;
  std::set<int> free_gp_regs = base_free_gp();
  std::set<int> preserved_gp = base_preserved_gp();

  // set of call clobbered vec registers: 0 - 7; 16 - 31
  static const std::set<int> &base_free_vec() {
    static const std::set<int> s{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    return s;
  }
  // set of call-preserved vec registers: 8 - 15
  static const std::set<int> &base_preserved_vec() {
    static const std::set<int> s{8, 9, 10, 11, 12, 13, 14, 15};
    return s;
  }

  std::set<int> used_vec;
  std::set<int> live_vec;
  std::set<int> free_vec_regs = base_free_vec();
  std::set<int> preserved_vec = base_preserved_vec();

  // set of call clobbered predicate registers: 0 - 3
  static const std::set<int> &base_free_pred() {
    static const std::set<int> s{0, 1, 2, 3};
    return s;
  }
  // set of call preserved predicate registers: 4 - 15
  static const std::set<int> &base_preserved_pred() {
    static const std::set<int> s{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    return s;
  }

  std::set<int> used_pred;
  std::set<int> live_pred;
  std::set<int> free_pred_regs = base_free_pred();
  std::set<int> preserved_pred = base_preserved_pred();

  std::size_t generation_ = 0;
};
// for list construction operator for RegT-Scoped<RegT>
template <class RegT> typename std::enable_if<is_list_family_reg<RegT>::value, decltype(std::declval<const RegT &>() - std::declval<const RegT &>())>::type operator-(const RegT &lhs, const RegPoolManager::Scoped<RegT> &rhs) { return lhs - rhs.get(); }
} // namespace Xbyak_aarch64

#endif
