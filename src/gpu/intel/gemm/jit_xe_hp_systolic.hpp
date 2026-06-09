/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_JIT_XE_HP_SYSTOLIC_HPP
#define GPU_INTEL_GEMM_JIT_XE_HP_SYSTOLIC_HPP

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "gemmstone/driver_info.hpp"
#include "gemmstone/problem.hpp"
#include "gpu/intel/gemm/jit/pd.hpp"
#include "gpu/intel/gemm/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

struct xe_hp_systolic_t : public gemm::primitive_t {
    struct pd_t : public jit::pd_t {
        using jit::pd_t::pd_t;

        DECLARE_COMMON_PD_T("jit:xe_hp:gemm:any", xe_hp_systolic_t);

        status_t init(impl::engine_t *engine);
        void init_scratchpad();

        bool use_nocopy();
        bool use_nocopy_xehpg(data_type_t dt, unsigned ld_align);
        status_t set_default_formats(data_type_t dt);

        bool packed_a() const { return packed_a_; }
        bool packed_b() const { return packed_b_; }
        bool packed_c() const { return packed_c_; }

        static int64_t nice_ld(int64_t ld, int sz, bool get_max = false) {
            const auto align = 32;
            const auto no_align = 64;

            auto new_ld = (ld * sz + align - 1) & ~(align - 1);
            if (get_max || (new_ld & (no_align - 1)) == 0) new_ld += align;

            return new_ld / sz;
        }

        int64_t get_ld_packed(int64_t k, bool get_max = false) const {
            auto a_sz = types::data_type_size(cfg().a_type());

            int unroll_k = int(32 / a_sz);
            auto ld = utils::rnd_up(k, unroll_k);
            if (with_ab_zero_points()) ld += unroll_k;

            return nice_ld(ld, int(a_sz), get_max);
        }

        int64_t max_ld_packed(int64_t k) const {
            return get_ld_packed(k, true);
        }

        dim_t lda_packed(int64_t k) const {
            return packed_a() ? a_packed_stride_ / unroll_m()
                              : get_ld_packed(k);
        }
        dim_t ldb_packed(int64_t k) const {
            return packed_b() ? b_packed_stride_ / unroll_n()
                              : get_ld_packed(k);
        }
        dim_t ldc_packed() const {
            return packed_c() ? c_packed_stride_ / unroll_n() : 0;
        }

        // C-offset kind from the active C-side mask:
        // 'F' full / 'R' row / 'C' column / 'M' both / 'N' none.
        char co_kind() const {
            int m = -1;
            if (cfg().with_c_zero_points() || cfg().with_bias())
                m = cfg().cmask;
            switch (m) {
                case 0: return 'F';
                case (1 << 1): return 'R';
                case (1 << 0): return 'C';
                case 3: return 'M';
                default: return 'N';
            }
        }

        bool with_batch() const { return cfg().problem.batchDims >= 1; }
        // Int-only datatype config; reads desc() since it is used pre-cfg-seed.
        bool dt_int_ok() const {
            using namespace data_type;
            const auto &d = desc();
            return utils::one_of(d->a_type(), u8, s8)
                    && utils::one_of(d->b_type(), u8, s8)
                    && utils::one_of(d->c_type(), s32, f32, s8, u8, f16);
        }
        bool with_ab_zero_points() const {
            return cfg().with_a_zero_points() || cfg().with_b_zero_points();
        }

        bool allow_k_blocking() const {
            const auto &po = cfg().problem.postOps;
            return (acc_type_ == cfg().c_type())
                    && IMPLICATION(po.len() > 0, po[0].is_sum());
        }

        int unroll_m() const { return unroll_m_; }
        int unroll_n() const { return unroll_n_; }
        bool alt() const { return alt_; }

        // Shared kernel geometry, snapshotted from selection in init().
        const gemmstone::CommonDriverInfo &driver_info() const {
            return driver_info_;
        }

        status_t query(query_t what, int idx, void *result) const override {
            switch ((int)what) {
                case (int)query::preferred_gpu_grf_per_thread: {
                    *(int *)result = 256;
                    break;
                }
                default: return gemm::pd_t::query(what, idx, result);
            }
            return status::success;
        }

        const compute::device_info_t *dev_info_ = nullptr;

    private:
        bool any_prepacked_ = false;
        bool packed_a_ = false, packed_b_ = false, packed_c_ = false;
        int unroll_m_ = 0;
        int unroll_n_ = 0;
        bool alt_ = false;
        // Prepacked input leading-dim strides (raw, pre-/unroll), seeded in init.
        dim_t a_packed_stride_ = 0, b_packed_stride_ = 0, c_packed_stride_ = 0;
        data_type_t acc_type_ = data_type::undef;
        gemmstone::CommonDriverInfo driver_info_;
    };

    status_t init(impl::engine_t *engine) override;

public:
    xe_hp_systolic_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t init_compute(impl::engine_t *engine);

    bool enable_mn_blocking() const;
    std::tuple<int64_t, int64_t, int64_t> get_blocking() const;

    status_t launch_clear_sum(const exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_copy(const exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &src, int64_t offset_src, int64_t ld_src,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_compute(const exec_ctx_t &ctx, int32_t m, int32_t n,
            int32_t k, const memory_storage_t &ap, int64_t offset_a,
            int32_t lda, const memory_storage_t &bp, int64_t offset_b,
            int32_t ldb, const memory_storage_t &c, int64_t offset_c,
            int32_t ldc, float alpha, float beta, const memory_storage_t *ao,
            const memory_storage_t *bo, const memory_storage_t &co,
            int64_t offset_co, int po_count, const memory_storage_t **po_src,
            int64_t *offset_po_src, bool first_k_block, bool last_k_block,
            int32_t batch, int32_t stride_a, int32_t stride_b,
            int32_t stride_c) const;

    static const int A_PACKED_ = 0;
    static const int B_PACKED_ = 1;

    compute::kernel_t kernel_[2][2]; // [first_k_block][last_k_block]
    compute::kernel_t copy_kernel_[2][2]; // [trans][clear_sum]

    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;
    int eu_count_ = 0;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
