#ifndef CPU_RV64_RVV_CONVOLUTION_HPP
#define CPU_RV64_RVV_CONVOLUTION_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_convolution_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace data_type;
            using namespace format_tag;

            const data_type_t sdt = src_md()->data_type;
            const data_type_t wdt = weights_md()->data_type;
            const data_type_t bdt
                    = with_bias() ? weights_md(1)->data_type : data_type::undef;
            const data_type_t ddt = dst_md()->data_type;

            const bool types_ok = (sdt == f32 && wdt == f32)
                    || (sdt == s8 && wdt == s8) || (sdt == u8 && wdt == s8);
            const bool dst_ok = utils::one_of(ddt, f32, s8, u8, s32);

            VDISPATCH_CONV(attr()->scales_.has_default_values(),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_CONV(rvv_postops_t::post_ops_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
            VDISPATCH_CONV(G() == 1, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(KDD() == 0 && KDH() == 0 && KDW() == 0,
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_CONV(types_ok && dst_ok, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(platform::has_data_type_support(sdt)
                            && platform::has_data_type_support(wdt)
                            && platform::has_data_type_support(ddt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(IMPLICATION(with_bias(), bdt == ddt),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);

            // Accept plain NCHW/NHWC layouts only
            src_is_nhwc_ = memory_desc_matches_tag(*src_md(), nhwc);
            dst_is_nhwc_ = memory_desc_matches_tag(*dst_md(), nhwc);
            const bool src_plain_ok
                    = src_is_nhwc_ || memory_desc_matches_tag(*src_md(), nchw);
            const bool dst_plain_ok
                    = dst_is_nhwc_ || memory_desc_matches_tag(*dst_md(), nchw);
            const bool wei_ok = memory_desc_matches_tag(*weights_md(), oihw);
            VDISPATCH_CONV(src_plain_ok && wei_ok && dst_plain_ok,
                    VERBOSE_UNSUPPORTED_TAG);

            src_dt_ = sdt;
            wei_dt_ = wdt;
            dst_dt_ = ddt;
            OC_ = OC();
            IC_ = IC();
            KH_ = KH();
            KW_ = KW();
            MB_ = MB();
            IH_ = IH();
            IW_ = IW();
            OH_ = OH();
            OW_ = OW();

            // Sizes for prepacked/reordered buffers
            wei_pack_elems_ = static_cast<size_t>(OC_) * KH_ * KW_ * IC_;
            src_reorder_elems_ = static_cast<size_t>(MB_) * IH_ * IW_ * IC_;
            dst_reorder_elems_ = static_cast<size_t>(MB_) * OH_ * OW_ * OC_;

            // Book scratchpad: packed weights, optional src/dst reorder buffers
            init_scratchpad();

            return status::success;
        }

        void init_scratchpad() {
            using namespace memory_tracking;
            registrar_t scratchpad(scratchpad_registry());
            scratchpad.book(names::key_conv_permuted_weights, wei_pack_elems_,
                    types::data_type_size(wei_dt_));
            if (!src_is_nhwc_)
                scratchpad.book(names::key_conv_tr_src, src_reorder_elems_,
                        types::data_type_size(src_dt_));
            if (!dst_is_nhwc_)
                scratchpad.book(names::key_conv_ncsp_dst, dst_reorder_elems_,
                        types::data_type_size(dst_dt_));
        }

        // cached flags and sizes
        bool src_is_nhwc_ = false;
        bool dst_is_nhwc_ = false;
        dim_t MB_ = 0, IC_ = 0, OC_ = 0, IH_ = 0, IW_ = 0, OH_ = 0, OW_ = 0,
              KH_ = 0, KW_ = 0;
        size_t wei_pack_elems_ = 0;
        size_t src_reorder_elems_ = 0;
        size_t dst_reorder_elems_ = 0;
        data_type_t src_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
    };

    inline rvv_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
