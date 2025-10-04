/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_CONV_UTILS_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_CONV_UTILS_HPP

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace amx_utils {

using namespace dnnl::impl::utils;

struct spatial_features_3d_t {

    spatial_features_3d_t(const jit_conv_conf_t &jcp)
        : input_size_(jcp.id)
        , filter_size_(jcp.kd)
        , dilate_(jcp.dilate_d + 1)
        , stride_(jcp.stride_d)
        , init_pad_(jcp.f_pad)
        , end_pad_(jcp.back_pad)
        , is_fast_path_(dilate_ == 1 && stride_ == 1)
        , compute_extended_features_(!(is_fast_path_ || dilate_ != 1))
        , filter_(0)
        , lower_offset_(0)
        , output_offset_(0)
        , init_overflow_(0)
        , end_overflow_(0) {}

    inline int get_init_overflow(const int in) const {
        if (is_fast_path_)
            return nstl::max(0, filter_size_ - 1 - in - init_pad_);
        if (dilate_ != 1)
            return div_up(
                    nstl::max(0, (filter_size_ - 1) * dilate_ - in - init_pad_),
                    dilate_);
        return nstl::max(0, (filter_size_ - 1 - in - init_pad_) / stride_);
    }

    inline int get_end_overflow(const int in) const {
        if (is_fast_path_)
            return nstl::max(0, filter_size_ - input_size_ + in - end_pad_);
        if (dilate_ != 1)
            return div_up(nstl::max(0,
                                  (filter_size_ - 1) * dilate_ + 1 - input_size_
                                          + in - end_pad_),
                    dilate_);
        return nstl::max(
                0, (filter_size_ - input_size_ + in - end_pad_) / stride_);
    }

    void update_params(const int in) {

        init_overflow_ = get_init_overflow(in);
        end_overflow_ = get_end_overflow(in);

        // overflow_kd_hi
        const int overflow_filter_hi_ = compute_extended_features_
                ? filter_size_ - 1
                        - nstl::modulo(input_size_ - 1 + end_pad_ - in, stride_)
                : 0;
        // overflow_kd_lo
        const int overflow_filter_lo_
                = compute_extended_features_ ? (in + init_pad_) % stride_ : 0;

        filter_ = compute_extended_features_
                ? (overflow_filter_hi_ - overflow_filter_lo_) / stride_ + 1
                : filter_size_;

        lower_offset_ = compute_extended_features_
                ? overflow_filter_lo_ + end_overflow_ * stride_
                : end_overflow_;

        output_offset_ = compute_extended_features_
                ? (in + init_pad_ - lower_offset_) / stride_
                : in + init_pad_ - end_overflow_ * dilate_;
    }

    inline int get_filter_padding() const {
        return filter_ - init_overflow_ - end_overflow_;
    }

    inline int get_lower_offset() const { return lower_offset_; }

    inline int get_output_offset() const { return output_offset_; }

private:
    int input_size_;
    int filter_size_;
    int dilate_;
    int stride_;
    int init_pad_; // f_pad
    int end_pad_; // back_pad
    bool is_fast_path_; // 'dilate_ == 1 && stride_ == 1'
    bool compute_extended_features_; // eq. '(!is_fast_path_) && dilate_ == 1'

    int filter_;
    int lower_offset_; // d_lo
    int output_offset_; // d_oj

    int init_overflow_; // d_t_overflow
    int end_overflow_; // d_b_overflow
};

} // namespace amx_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
