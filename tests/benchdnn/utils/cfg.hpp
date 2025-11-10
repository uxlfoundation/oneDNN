/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef UTILS_CFG_HPP
#define UTILS_CFG_HPP

#include <algorithm>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_types.h"

#include "common.hpp"

class data_type_hash_t {
public:
    size_t operator()(dnnl_data_type_t dt) const {
        return std::hash<int>()(static_cast<int>(dt));
    }
};

class data_kind_hash_t {
public:
    size_t operator()(data_kind_t data_kind) const {
        return std::hash<int>()(static_cast<int>(data_kind));
    }
};

// `base_cfg_t` class is a base class to define configurations across drivers.
// Parent `cfg_t` object specifies a constructor to initialize `cfg_entry_`
// member. This is needed since only `prb` object has necessary information.
// See below for public interfaces to use and modify the `cfg` object.
struct base_cfg_t {
    struct cfg_entry_t {
        // Supplies min and max ranges for filling for a given data type.
        struct cfg_range_t {
            int range_min;
            int range_max;
        };

        using cfg_map_t = std::unordered_map<dnnl_data_type_t, cfg_range_t,
                data_type_hash_t>;

        // Constructor takes:
        // * Data kind it was created for (this is used only for accessing
        //   correspondent cfg_entry_t elements).
        // * Original data type, as final data type may be altered by
        //   fpmath-mode value or different from dst_dt sum_dt value.
        // * Final data type to set cfg for.
        // * Initial `cfg_map` map of ranges for each data type. It is copied
        //   to provide an ability to adjust final values required in certain
        //   scenarios.
        cfg_entry_t(data_kind_t dk, dnnl_data_type_t orig_dt,
                dnnl_data_type_t dt, const cfg_map_t &cfg_map)
            : data_kind_(dk)
            , orig_data_type_(orig_dt)
            , data_type_(dt)
            , cfg_map_(cfg_map) {}

        int get_range_min() const { return get_cfg_range().range_min; }
        int get_range_max() const { return get_cfg_range().range_max; }
        int get_range_abs_max() const {
            return std::max(abs(get_range_min()), abs(get_range_max()));
        }

        void set_range_min(int new_value) {
            auto it = cfg_map_.find(data_type_);
            if (it == cfg_map_.end()) {
                assert(!"entry was not found!");
                return;
            }
            (*it).second.range_min = new_value;
        }
        void set_range_max(int new_value) {
            auto it = cfg_map_.find(data_type_);
            if (it == cfg_map_.end()) {
                assert(!"entry was not found!");
                return;
            }
            (*it).second.range_max = new_value;
        }

        dnnl_data_type_t get_orig_dt() const { return orig_data_type_; }
        dnnl_data_type_t get_dt() const { return data_type_; }
        data_kind_t get_dk() const { return data_kind_; }

    private:
        data_kind_t data_kind_; // For searching elements in base_cfg_t.
        dnnl_data_type_t orig_data_type_;
        dnnl_data_type_t data_type_;
        cfg_map_t cfg_map_;

        const cfg_range_t &get_cfg_range() const {
            const auto it = cfg_map_.find(data_type_);
            if (it != cfg_map_.end()) return (*it).second;
            assert(!"unexpected");
            static cfg_range_t dummy;
            return dummy;
        }
    };

    int get_range_min(data_kind_t dk) const {
        return cfg_entry_.at(dk).get_range_min();
    }
    int get_range_max(data_kind_t dk) const {
        return cfg_entry_.at(dk).get_range_max();
    }

    dnnl_data_type_t get_orig_dt(data_kind_t dk) const {
        return cfg_entry_.at(dk).get_orig_dt();
    }
    dnnl_data_type_t get_dt(data_kind_t dk) const {
        return cfg_entry_.at(dk).get_dt();
    }
    // Returns a valid data type if swap is needed and `undef` otherwise.
    dnnl_data_type_t get_swapped_dt(data_kind_t dk) const {
        const bool swap_dt = dk == DST && get_orig_dt(dk) != get_dt(dk);
        return swap_dt ? get_dt(dk) : dnnl_data_type_undef;
    }

    // This type allows to differentiate density in filling functions by certain
    // criteria. Members used in each driver may be different.
    struct density_args_t {
        // Data kind like SRC, WEI, DST, etc.
        data_kind_t data_kind;
        // Number of accumulators in the chain. Longer chains to be more sparse.
        int64_t n_acc;
    };

    // Base config has to know a map of ranges to have its interfaces working.
    virtual cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const = 0;

    // Density definition may be modified by each parent as necessary.
    virtual float get_density(const density_args_t &density_args) const {
        return 1.f;
    }

protected:
    std::unordered_map<data_kind_t, cfg_entry_t, data_kind_hash_t> cfg_entry_;
    data_kind_t output_data_kind_ = DST; // Assume FWD by default.

    int64_t get_safe_digits() const {
        return MIN2(digits_dt(cfg_entry_.at(output_data_kind_).get_dt()),
                digits_dt(dnnl_f32));
    }

    bool is_int8(data_kind_t dk = WEI) const {
        // This function is designed to bump density for int8 cases to trigger
        // saturation and rounding. However, with s32 output data type bumped
        // density can exceed safe f32 reference output value and lead to
        // mismatching results. Thus, remove x8s8s32 configuration from int8
        // definition.
        return dnnl_data_type_size(cfg_entry_.at(dk).get_dt()) == 1
                && cfg_entry_.at(output_data_kind_).get_dt() != dnnl_s32;
    }

    // Find the number of accumulators safe to use with the following equations:
    // Integer value can be expressed exactly with floating-point is
    // MAX_DIGITS = MIN2(std::numeric_limit::digits(dst_dt),
    //                   std::numeric_limit::digits(f32));
    // PREC = (1 << MAX_DIGITS);
    // SUM_1_N(VALUES) <= PREC;   This should hold to get precise answer.
    // SUM_1_N(VALUES) <= N_ACC * MAX_VALUE <= PREC;  It's a top estimate
    // MAX_VALUE = MAX_VAL_SRC * MAX_VAL_WEI;
    // SAFE_N_ACC <= PREC / MAX_VALUE;
    //
    // However, there're very low precision data types with `MAX_DIGITS` being
    // very low. For such types, just return a predefined number of safe
    // elements to avoid adjusting every parent config and to let all the
    // spectrum of values to appear in the output to test conversions to those
    // data types.
    //
    // `check_n_acc` == false will let non-postive value to be returned. This
    // is needed to let `adjust_ranges_for_safe_n_acc` modify iteratively cfg
    // values until ranges will fit the output range nicely.
    int64_t get_safe_n_acc(const std::vector<data_kind_t> &kinds = {SRC, WEI},
            bool check_n_acc = true) const {
        const int64_t safe_digits = get_safe_digits();
        if (safe_digits <= 0) {
            BENCHDNN_PRINT(
                    0, "%s\n", "[CFG] Error: safe digits is non-positive.");
            SAFE_V(FAIL);
        } else if (safe_digits <= 2) {
            // fp4 category.
            return 1;
        } else if (safe_digits <= 4) {
            // fp8/int4 category. Hopefully, int4 won't ever make it to DST.
            return 2;
        }

        int64_t max_value = 1;
        for (auto k : kinds) {
            const auto &cfg_entry = cfg_entry_.at(k);
            max_value *= cfg_entry.get_range_abs_max();
        }
        const int64_t safe_n_acc = (1LL << safe_digits) / max_value;
        if (check_n_acc && safe_n_acc <= 0) {
            BENCHDNN_PRINT(0,
                    "[CFG] Error: there's no safe accumulation chain for a "
                    "given data type combination. Safe_digits=%d, "
                    "max_value=%d.\n",
                    static_cast<int>(safe_digits), static_cast<int>(max_value));
            SAFE_V(FAIL);
        }
        return safe_n_acc;
    }

    // Modification of ranges has to happen at construction stage.
    void set_range_min(data_kind_t dk, int new_value) {
        cfg_entry_.at(dk).set_range_min(new_value);
    }
    void set_range_max(data_kind_t dk, int new_value) {
        cfg_entry_.at(dk).set_range_max(new_value);
    }

    void print_fill_cfg_verbose(
            const std::vector<data_kind_t> &kinds = {SRC, WEI, DST}) const {
        constexpr int verbose_level = 6;
        if (verbose < verbose_level) return;

        std::string s("[FILL_CFG]");
        for (const auto &k : kinds) {
            s += " " + std::string(data_kind2str(k)) + "_" + dt2str(get_dt(k))
                    + "=[" + std::to_string(get_range_min(k)) + ";"
                    + std::to_string(get_range_max(k)) + "];";
        }
        BENCHDNN_PRINT(verbose_level, "%s\n", s.c_str());
    }

    // Use larger values for SRC to test proper u8 loads when DST data type
    // allows it.
    void adjust_ranges_for_u8_load() {
        const bool is_int8_and_wide_dst = get_dt(SRC) == dnnl_u8
                && dnnl_data_type_size(get_dt(WEI)) == 1
                && dnnl_data_type_size(get_dt(DST)) >= 4;
        if (is_int8_and_wide_dst) { set_range_max(SRC, 160); }
    }

    // For s8s8 weights have to be even to comply with adjust_scale of 0.5f.
    // Divide the range by factor of two here, and multiply values by factor
    // of two when do filling.
    void adjust_ranges_for_s8s8() {
        const bool is_s8s8 = get_dt(SRC) == dnnl_s8 && get_dt(WEI) == dnnl_s8;
        if (is_s8s8) {
            set_range_min(WEI, -2);
            set_range_max(WEI, 2);
        }
    }

    // u8 weights will have big value set for kernels to distinguish s8 loads
    // from u8 since those use different instructions. However, when DST data
    // type is narrow, big value will make the library to saturate which leads
    // to false positive results. Adjust big value for that case.
    // This adjustment is somewhat opposite to `adjust_ranges_for_u8_load`.
    void adjust_ranges_for_u8u8() {
        const bool is_u8u8 = get_dt(SRC) == dnnl_u8 && get_dt(WEI) == dnnl_u8;
        if (is_u8u8) {
            // Adjust SRC entry to fit for u8u8 to fit any dst data type.
            set_range_max(SRC, 1);

            // Any data type but 4-bit wide in DST is not enough to catch
            // incorrect loads. Leave filling to fit the basic validation
            // purpose.
            const bool is_wide_dst = dnnl_data_type_size(get_dt(DST)) >= 4;
            if (!is_wide_dst) set_range_max(WEI, 8);
        }
    }

    // Configuration like f32:f32:s8 may trigger an assert `safe_n_acc <= 0`.
    // It can be solved only at construction stage by adjusting min and max
    // range values. It's not expected to be triggered for regular configs since
    // output data type most of time is of same or larger width.
    // The idea behind reduction is to reduce ranges of SRC and WEI step by step
    // by a factor of two until at least a single result of multiplication fits
    // the output data type range.
    void adjust_ranges_for_safe_n_acc() {
        int64_t safe_n_acc
                = get_safe_n_acc({SRC, WEI}, /* check_n_acc = */ false);
        data_kind_t cur_kind = SRC;
        while (safe_n_acc < 1) {
            set_range_min(cur_kind, get_range_min(cur_kind) / 2);
            set_range_max(cur_kind, get_range_max(cur_kind) / 2);
            int64_t max_value = static_cast<int64_t>(
                                        cfg_entry_.at(SRC).get_range_abs_max())
                    * cfg_entry_.at(WEI).get_range_abs_max();
            safe_n_acc = (1LL << get_safe_digits()) / max_value;
            cur_kind = cur_kind == SRC ? WEI : SRC;
        }
    }

    // A collection of functions that adjust data filling ranges to fit
    // numerical properties or requested data types.
    void adjust_ranges() {
        adjust_ranges_for_u8_load();
        adjust_ranges_for_s8s8();
        adjust_ranges_for_u8u8();
        adjust_ranges_for_safe_n_acc();
    }
};

#endif
