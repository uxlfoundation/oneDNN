/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022, 2025-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_REORDER_JIT_UNI_REORDER_KERNEL_HPP
#define CPU_AARCH64_REORDER_JIT_UNI_REORDER_KERNEL_HPP

#include <cassert>

#include "common/c_types_map.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/reorder/jit_uni_reorder_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
/* kernel */
class jit_uni_reorder_kernel_t : public jit_generator_t {
public:
    struct desc_t {
        int id;
        prb_t prb;
        // TODO: Add kernel name field here?
        // const char* name;
    };

    struct call_param_t {
        const void *in = nullptr;
        void *out = nullptr;
        const float *src_scales = nullptr;
        const float *dst_scales = nullptr;
        int32_t src_zp = 0;
        int32_t dst_zp = 0;
        int32_t *compensation_scratch = nullptr;
    };

    // The additional structure is needed because
    // using a data structure with tail processing
    // data for non-tail cases reduces kernel
    // performance. This is because there is too
    // much data that has to be transferred to the kernel.
    struct tail_call_param_t {
        call_param_t base_params;
        int64_t curr_data_chunks[DNNL_MAX_NDIMS] = {-1};
        int64_t zeroing_data = static_cast<int64_t>(false);
        int64_t skip_kernel_execution = static_cast<int64_t>(false);
    };

    jit_uni_reorder_kernel_t(const desc_t &desc);
    ~jit_uni_reorder_kernel_t() override = default;

    /** inits kernel descriptor:
     *      desc            -- kernel descriptor (output)
     *      prb             -- problem descriptor (input)
     *      ndims_ker_max   -- limit the maximum number of dimensions kernel
     *                         will process (optional, 0 -- no limitation) */
    static status_t desc_init(
            desc_t &desc, const prb_t &prb, int ndims_ker_max = 0);

    /** selects kernel for the problem described in desc */
    static jit_uni_reorder_kernel_t *create_handle(const desc_t &desc);

    void operator()(const call_param_t *c) const {
        return jit_generator_t::operator()(c);
    }

    void operator()(const tail_call_param_t *c) const {
        return jit_generator_t::operator()(c);
    }

    /** Minimal reasonable/desirable kernel size.
    * The constant might be used to determine how a problem should be split
    * between kernel and threading driver. */
    static constexpr size_t ker_prb_size_min = 64;

protected:
    using XReg = Xbyak_aarch64::XReg;
    using WReg = Xbyak_aarch64::WReg;
    using ZReg = Xbyak_aarch64::ZReg;
    using ZRegS = Xbyak_aarch64::ZRegS;
    using VReg = Xbyak_aarch64::VReg;
    using VReg4S = Xbyak_aarch64::VReg4S;
    using PReg = Xbyak_aarch64::PReg;

    enum class scale_arg_t { NONE, SRC, DST };

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct impl_desc_t {
        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 0;
        int tail_len_unroll = 0;
        int len_unroll = 0;
    };

    static bool interim_f32_needed(const prb_t &prb, bool compensation_needed);

    // Common data
    XReg o_addr(int o_off, bool with_type_multiplier = true);
    XReg src_s_addr(int s_off);
    XReg dst_s_addr(int s_off);
    XReg c_addr(int c_off);
    XReg data_chunk_addr(int node_id);
    void zero_dst_memory(const int bytes_to_zeroing);

    // Common dt conversion utils
    void cvt_z_s32_f32(const size_t startIdx, const size_t regNum);
    void cvt_v_s32_f32(const size_t startIdx, const size_t regNum);
    void cvt_z_f32_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_f32_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_f32_bf16(const size_t startIdx, const size_t regNum);
    void cvt_v_bf16_fp32(const size_t startIdx, const size_t regNum);
    void cvt_v_f16_f32(const size_t startIdx, const size_t regNum);
    void cvt_v_f32_f16(const size_t startIdx, const size_t regNum);
    void cvt_z_s8_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_s8_s32(const size_t startIdx, const size_t regNum);
    void cvt_z_s8_f32(const size_t startIdx, const size_t regNum);
    void cvt_v_s8_f32(const size_t startIdx, const size_t regNum);
    void cvt_z_b_s(const size_t startIdx, const size_t regNum);
    void cvt_v_b_s(const size_t startIdx, const size_t regNum);
    void cvt_z_u8_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_u8_s32(const size_t startIdx, const size_t regNum);
    void cvt_z_s32_s8(const size_t startIdx, const size_t regNum);
    void cvt_v_s32_s8(const size_t startIdx, const size_t regNum);
    void cvt_z_u8_s8(const size_t startIdx, const size_t regNum);
    void cvt_v_u8_s8(const size_t startIdx, const size_t regNum);
    void cvt_z_u32_u8(const size_t startIdx, const size_t regNum);
    void cvt_v_u32_u8(const size_t startIdx, const size_t regNum);
    void cvt_z_s32_u8(const size_t startIdx, const size_t regNum);
    void cvt_v_s32_u8(const size_t startIdx, const size_t regNum);
    void cvt_z_s8_u8(const size_t startIdx, const size_t regNum);
    void cvt_v_s8_u8(const size_t startIdx, const size_t regNum);

    // Common scaffolding (create loops, set up tail handling, etc.)
    static bool impl_desc_init(const prb_t &prb, impl_desc_t *desc);
    void generate() override;
    bool impl();
    void create_loops(const impl_desc_t &desc,
            const std::array<const XReg, 3> &reg_cnt, int jit_loop);
    void check_if_this_is_last_chunk(const XReg reg_curr_chunk, int node_id);
    void finalize_tail_loop(int i_step, int o_step, int s_step, int c_step,
            const int curr_node_id);
    void inject_kernel_body(const impl_desc_t &desc);
    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int prev_c_off, int &i_off, int &o_off, int &s_off, int &c_off,
            int step_size = 1);
    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1);

    // This is where it gets specific. This function is overriden by each
    // specialised kernel to produce its unrolled code.
    virtual void compute(int ndims, int len_unroll, bool tail_processing) = 0;

    const desc_t desc_;
    const prb_t &prb_;
    bool compensation_needed_;

    static constexpr int64_t with_tail_info_ = static_cast<int64_t>(true);
    static constexpr int64_t without_tail_info_ = static_cast<int64_t>(false);

    int itype_sz_;
    int otype_sz_;
    int stype_sz_;

    const cpu_isa_t isa_;

    const XReg reg_ptr_in_ = x6;
    const XReg reg_ptr_out_ = x2;
    const XReg reg_ptr_src_scales_ = x1;
    const XReg reg_ptr_dst_scales_ = x12;
    const XReg reg_ptr_comp_ = x3;
    const WReg reg_scale_adjust_ = w5;

    const XReg reg_off_in_ = x8;
    const XReg reg_off_out_ = x9;
    const XReg reg_off_comp_ = x11;

    /* X_TMP is required to set address to
     x_tmp_vec(X_TMP_0 - X_TMP_4). */
    XReg X_TMP = x20;

    VReg4S xmm_src_scales_ = v15.s;
    VReg4S xmm_dst_scales_ = v11.s;
    VReg4S xmm_zero_ = v14.s;
    ZRegS ymm_zero_ = z14.s;
    VReg4S xmm_tmp_ = v12.s;
    const VReg4S xmm_src_zp_ = v9.s;
    const VReg4S xmm_dst_zp_ = v10.s;
    const VReg4S xmm_compensation = v8.s;
    VReg4S xmm_saturation_ubound_ = v12.s;
    ZRegS ymm_saturation_ubound_ = z12.s;

    /* Note: x22 - x28 are already used as temporal registgers
       in jit_generator.hpp.
       x_ptr_(in|out|scale|comp)_off keeps (base + offset) address. */
    XReg x_ptr_in_off = reg_ptr_in_;
    XReg x_ptr_out_off = reg_ptr_out_;
    XReg x_ptr_comp_off = reg_ptr_comp_;
    XReg x_ptr_src_scale_off = x19;
    XReg x_ptr_dst_scale_off = x29;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_lsb_256 = p7;
    PReg p_lsb_128 = p6;
    PReg p_lsb_64 = p4;
    PReg p_tmp0 = p5;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    VReg v_tmp0 = v20;
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};

class generic_kernel_t : public jit_uni_reorder_kernel_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(generic_kernel_t)

    using jit_uni_reorder_kernel_t::jit_uni_reorder_kernel_t;

    static bool applicable(const prb_t &prb) {
        using namespace data_type;

        bool bf16_ok
                = (mayiuse_bf16() && (prb.itype == bf16) && (prb.otype == bf16)
                          && !interim_f32_needed(prb, false) && prb.beta == 0.f)
                || (prb.itype != bf16 && prb.otype != bf16)
                || (prb.itype == f32 && prb.otype == bf16 && mayiuse_bf16()
                        && prb.beta == 0.f)
                || (prb.itype == bf16 && prb.otype == f32 && mayiuse_bf16()
                        && prb.beta == 0.f);

        bool is_f16 = (prb.itype == f16 || prb.otype == f16);
        bool f16_ok = (prb.itype == f32 && prb.otype == f16 && prb.beta == 0.f)
                || (prb.itype == f16 && prb.otype == f32 && prb.beta == 0.f)
                || (prb.itype == f16 && prb.otype == f16 && prb.beta == 0.f);

        bool ok = true && prb.ndims > 0
                && utils::one_of(
                        prb.itype, f32, f16, bf16, s32, data_type::s8, u8)
                && utils::one_of(
                        prb.otype, f32, f16, bf16, s32, data_type::s8, u8)
                && utils::everyone_is(
                        0, prb.ioff, prb.ooff) /* do we need this? */
                && utils::one_of(prb.beta, 0.f, 1.f) /* anything else? */
                && impl_desc_init(prb, nullptr) && prb_has_small_strides(prb)
                && bf16_ok && IMPLICATION(is_f16, f16_ok);

        return ok;
    }

private:
    void compute(int ndims, int len, bool tail_processing) override;
    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off, const int *c_off,
            const int *zero_padding, const bool tail_processing);
};

class tr8x8_kernel_t : public jit_uni_reorder_kernel_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(tr8x8_kernel_t)

    using jit_uni_reorder_kernel_t::jit_uni_reorder_kernel_t;

    static bool applicable(const prb_t &prb) {
        using namespace data_type;

        static constexpr int desirable_node_size = 8;
        static constexpr int desirable_stride = 1;

        // This process relies on swapping the two innermost dimensions.
        // Therefore, the input stride in the second node and output stride in
        // first node have to be equal to 1.
        bool ok = mayiuse(sve_256) && prb.ndims >= 2
                && ((utils::one_of(prb.itype, u8, data_type::s8, s32, f32)
                        && utils::one_of(
                                prb.otype, u8, data_type::s8, s32, f32)))
                && utils::everyone_is(desirable_node_size, prb.n(0), prb.n(1))
                && utils::everyone_is(desirable_stride, prb.os(0), prb.is(1))
                && !prb.is_tail_present
                && prb.src_scale_type == scale_type_t::NONE
                && prb.dst_scale_type == scale_type_t::NONE && prb.beta == 0.f
                && !(prb.req_s8s8_comp || prb.req_asymmetric_comp);

        return ok;
    }

private:
    void compute(int ndims, int len, bool) override;
    void tr8x8_core(int i_off, int o_off);
};

// TODO: Combine this with tr8x8_kernel_t?
class tr4x8_kernel_t : public jit_uni_reorder_kernel_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(tr4x8_kernel_t)

    using jit_uni_reorder_kernel_t::jit_uni_reorder_kernel_t;

    static bool applicable(const prb_t &prb) {
        using namespace data_type;

        // The kernel is specialised for f32 -> bf16 reorders.
        //
        // This process relies on swapping the two innermost dimensions.
        // Therefore, the input stride in the second node and output stride in
        // first node have to be equal to 1.
        return mayiuse(sve_256) && prb.ndims >= 2
                && (prb.itype == f32 && prb.otype == bf16) && prb.n(0) == 4
                && prb.n(1) == 8 && utils::everyone_is(1, prb.os(0), prb.is(1))
                && !prb.is_tail_present
                && prb.src_scale_type == scale_type_t::NONE
                && prb.dst_scale_type == scale_type_t::NONE && prb.beta == 0.f
                && !(prb.req_s8s8_comp || prb.req_asymmetric_comp);
    }

private:
    void compute(int ndims, int len, bool) override;
    void tr4x8_core(int i_off, int o_off);
};

/* TODO: add trans_t class */

// Seperate class for no unroll/threading burden
struct jit_single_blk_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)
    using XReg = Xbyak_aarch64::XReg;
    using ZRegS = Xbyak_aarch64::ZRegS;
    using ZReg = Xbyak_aarch64::ZReg;
    using PReg = Xbyak_aarch64::PReg;
    using VReg = Xbyak_aarch64::VReg;

    static bool applicable(const prb_t &p);

    jit_single_blk_kernel_t(const prb_t &prb);

    void generate() override;

    void gen_loadu(const ZRegS ymm, const XReg &addr, int size);

    void gen_storeu(const XReg &addr, const ZRegS ymm, int size);

    void gen_maskloadu(
            const ZRegS ymm, const XReg &addr, const PReg mask, int size);

    void gen_maskstoreu(
            const XReg &addr, const ZRegS ymm, const PReg mask, int size);

    // Register allocation xmm0~11
    void gen_transpose_8x8();

    void gen_transpose_4x4();

    // keep order nchw -> nChw()C
    // or nChw()C -> nchw
    void gen_setmask(int mask);

    void gen_tr4x4(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);
    void gen_ker4x4(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);

    // TODO: Mark parameter with type information
    // XXX: !
    // offset in byte offset
    // stride in element number
    //
    // Gen specific 8x8 transform respect to certain tail condition
    void gen_tr8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);

    // tail: 0 ~ 8
    // support: either in_tail or out_tail is not 8, but not both
    void gen_ker8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);

    void gen_ker16x16_in_8x8(
            int i_off, int o_off, int input_stride, int output_stride);

    // tail can be 1 ~ 16, using sve2 for now
    void gen_ker16x16_in_8x8(int i_off, int o_off, int input_stride,
            int output_stride, int in_tail, int out_tail);

    void gen_ker32x32_in_16x16(
            int i_off, int o_off, int input_stride, int output_stride);

    void gen_ker32x32_in_16x16(int i_off, int o_off, int input_stride,
            int output_stride, int in_tail, int out_tail);

    void gen_ker64x64_in_32x32(
            int i_off, int o_off, int input_stride, int output_stride);

    void gen_ker64x64_in_32x32(int i_off, int o_off, int input_stride,
            int output_stride, int in_tail, int out_tail);

private:
    // 6 ~ 12
    constexpr static int xmm_save_start_from = 6;
    constexpr static int xmm_width = 16;

    void preamble();

    void postamble();

    const prb_t &prb_;

    int itype_sz_;
    int otype_sz_;
    int block_sz;

    XReg reg_ptr_in_ = abi_param1;
    XReg reg_ptr_out_ = abi_param2;
    XReg reg_ptr_tail = abi_param3;

    /* Because the callee-saved registers are not restored blk_reorder,
     the temporary registers (x9-x15) must be assigned.
     Must be selected from the temporary registers (x9-x15). */
    XReg x_addr = x10;
    XReg x_tmp_0 = x11;
    XReg x_tmp_1 = x12;

    /* Avoid P_TMP(p7) in jit_generator.hpp. */
    PReg p_lsb_256 = p6;
    PReg p_mask = p5;
    PReg p_tmp1 = p4;
    PReg p_tmp2 = p3;

    ZRegS ymm_tmp = z0.s;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    VReg v_tmp0 = v20;
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
