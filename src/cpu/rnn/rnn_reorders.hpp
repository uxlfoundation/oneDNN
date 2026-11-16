/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_RNN_RNN_REORDERS_HPP
#define CPU_RNN_RNN_REORDERS_HPP

#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/gemm/gemm_pack.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct rnn_data_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_data_reorder", rnn_data_reorder_t);

        status_t init(const engine_t *engine, const engine_t *src_engine,
                const engine_t *dst_engine) {
            using namespace format_tag;
            using namespace status;
            const memory_desc_wrapper id(src_md()), od(dst_md());

            bool args_ok = impl::is_dense_format_kind({src_md(), dst_md()});
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            PD_CHECK_ARG(id.data_type() == data_type::f32);
            PD_CHECK_ARG(utils::one_of(
                    od.data_type(), data_type::s8, data_type::u8));
            PD_CHECK_ARG(utils::one_of(id.ndims(), 3, 4));
            PD_CHECK_ARG(!id.has_runtime_dims_or_strides());
            auto skip_mask = primitive_attr_t::skip_mask_t::rnn_data_qparams
                    | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                    | primitive_attr_t::skip_mask_t::
                            rnn_weights_projection_qparams;
            PD_CHECK_ARG(attr()->has_default_values(skip_mask));
            PD_CHECK_ARG(IMPLICATION(id.ndims() == 3,
                    id.matches_tag(tnc) && od.matches_tag(tnc)));
            PD_CHECK_ARG(IMPLICATION(id.ndims() == 4,
                    id.matches_tag(ldnc) && od.matches_tag(ldnc)));
#undef PD_CHECK_ARG
            if (!args_ok) return invalid_arguments;

            CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));
            return status::success;
        }
    };

    rnn_data_reorder_t(const pd_t *apd) : primitive_t(apd) {}

private:
    bool is_dense() const;
    template <typename out_data_t>
    status_t execute_dense(out_data_t *output, const float *input,
            const float scale, const float shift) const;
    template <typename out_data_t>
    status_t execute_generic(out_data_t *output, const float *input,
            float scale, float shift) const;
    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct rnn_weights_reorder_s8_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        using gemm_pack_f = dnnl_status_t (*)(const char *identifier,
                const char *transa, const char *transb, const dim_t *M,
                const dim_t *N, const dim_t *K, const dim_t *lda,
                const dim_t *ldb, const void *src, void *dst);

        DECLARE_COMMON_PD_T("rnn_weights_reorder_s8", rnn_weights_reorder_s8_t);

        status_t init(const engine_t *engine, const engine_t *src_engine,
                const engine_t *dst_engine) {
            using namespace format_tag;
            using namespace rnn_packed_format;
            using namespace status;
            const memory_desc_wrapper id(src_md()), od(dst_md());

            bool args_ok = impl::is_dense_format_kind({src_md(), dst_md()});
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            // Fast checks
            PD_CHECK_ARG(utils::one_of(
                    id.data_type(), data_type::f32, data_type::s8));
            PD_CHECK_ARG(od.data_type() == data_type::s8);
            PD_CHECK_ARG(od.format_kind() == format_kind::rnn_packed);
            PD_CHECK_ARG(utils::one_of(
                    od.rnn_packed_desc().format, ldigo_p, ldio_p));
            PD_CHECK_ARG(od.ndims() == id.ndims());
            // TODO: we have to skip projection qparam even for regular lstm
            // as we use the same attr for regular weights and projection
            auto skip_mask = primitive_attr_t::skip_mask_t::rnn_data_qparams
                    | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                    | primitive_attr_t::skip_mask_t::
                            rnn_weights_projection_qparams;
            PD_CHECK_ARG(attr()->has_default_values(skip_mask));
            if (!args_ok) return invalid_arguments;

            // Slower checks
            PD_CHECK_ARG(id.is_dense());
            if (!args_ok) return invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(ldigo, ldgoi, ldio, ldoi);
            if (itag == format_tag::undef) return invalid_arguments;

            // TODO: add support for layer and direction dimensions
            // weights_layer and weights_iter
            if (id.ndims() == 5
                    && !utils::one_of(
                            attr()->rnn_weights_qparams_.mask_, 0, 24))
                return unimplemented;
            // weights_projection
            if (id.ndims() == 4
                    && !utils::one_of(
                            attr()->rnn_weights_projection_qparams_.mask_, 0,
                            8))
                return unimplemented;
#undef PD_CHECK_ARG

            itag_ = itag;

            CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            const bool is_s8s8 = dst_md()->extra.flags
                    & memory_extra_flags::rnn_s8s8_compensation;
            gemm_pack = is_s8s8 ? &gemm_s8s8s32_pack : &gemm_s8u8s32_pack;

            return status::success;
        }

        format_tag_t itag_ = format_tag::undef;
        format_tag_t otag_ = format_tag::undef;
        size_t thr_scratch_comp_sz_ = 0;
        int nthr_; // To not exceed the limit in execute used for set up.
        gemm_pack_f gemm_pack;

    private:
        void init_scratchpad() {
            using namespace format_tag;

            const memory_desc_wrapper id(src_md());
            const size_t nelems = id.nelems();
            const auto &dims = id.dims();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            const size_t quantization_size = nelems;
            // we do not use GO directly, as this can cause false
            // sharing when parallelizing on I (2 threads writing to
            // the same cache line)
            thr_scratch_comp_sz_ = itag_ == ldigo ? dims[3] * dims[4] : dims[3];
            thr_scratch_comp_sz_ = utils::rnd_up(thr_scratch_comp_sz_, 16);
            size_t reduction_size = 0;
            if (utils::one_of(itag_, ldigo, ldio))
                reduction_size = nthr_ * thr_scratch_comp_sz_;

            scratchpad.template book<int8_t>(
                    key_reorder_rnn_weights_quantization, quantization_size);
            scratchpad.template book<int32_t>(
                    key_reorder_rnn_weights_reduction, reduction_size);
        }
    };

    rnn_weights_reorder_s8_t(const pd_t *apd) : primitive_t(apd) {}

private:
    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct rnn_weights_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        format_tag_t itag_;

        status_t init(const engine_t *engine, const engine_t *src_engine,
                const engine_t *dst_engine) {
            using namespace format_tag;
            using namespace rnn_packed_format;
            using namespace status;

            const memory_desc_wrapper id(src_md()), od(dst_md());
            bool args_ok = impl::is_dense_format_kind({src_md(), dst_md()});
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            const data_type_t idt = id.data_type();
            const data_type_t odt = od.data_type();
            PD_CHECK_ARG((idt == data_type::f32
                                 && utils::one_of(
                                         odt, data_type::f32, data_type::bf16))
                    || (idt == data_type::bf16 && odt == data_type::bf16));
            PD_CHECK_ARG(od.format_kind() == format_kind::rnn_packed);
            PD_CHECK_ARG(utils::one_of(
                    od.rnn_packed_desc().format, ldigo_p, ldgoi_p, ldio_p));
            PD_CHECK_ARG(attr()->has_default_values());
#undef PD_CHECK_ARG
            if (!args_ok) return invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(ldigo, ldgoi, ldio, ldoi);
            if (itag == format_tag::undef) return invalid_arguments;

            itag_ = itag;

            CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace format_tag;
            using namespace rnn_packed_format;

            const memory_desc_wrapper id(src_md());
            const memory_desc_wrapper od(dst_md());
            const rnn_packed_desc_t &rnn_pdata = od.rnn_packed_desc();

            format_tag_t itag = id.matches_one_of_tag(ldigo, ldgoi, ldio);
            const bool layout_cross_case
                    = (itag == ldigo && rnn_pdata.format == ldgoi_p)
                    || (itag == ldgoi && rnn_pdata.format == ldigo_p)
                    || (itag == ldio && rnn_pdata.format == ldio_p),
                    dt_cross_case = id.data_type() == data_type::f32
                    && od.data_type() == data_type::bf16;
            const size_t sz = id.nelems();
            const size_t odt_size = od.data_type_size();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_reorder_rnn_weights_transposition,
                    layout_cross_case ? sz : 0, odt_size);
            scratchpad.book(key_reorder_rnn_weights_xf16_cvt,
                    dt_cross_case ? sz : 0, odt_size);
        }
    };

    rnn_weights_reorder_t(const pd_t *apd) : primitive_t(apd) {}

private:
    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct rnn_brgemm_weights_reorder_s8_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_brgemm_weights_reorder_s8_t",
                rnn_brgemm_weights_reorder_s8_t);

        format_tag_t itag_;
        format_tag_t otag_;
        int nthr_; // To not exceed the limit in execute used for set up.
        size_t thr_scratch_comp_sz_ = 0;

        status_t init(const engine_t *engine, const engine_t *src_engine,
                const engine_t *dst_engine) {
            using namespace status;
            using namespace format_tag;
            using namespace memory_extra_flags;

            const memory_desc_wrapper id(src_md()), od(dst_md());

            const bool args_ok
                    = impl::is_dense_format_kind({src_md(), dst_md()})
                    && utils::one_of(
                            id.data_type(), data_type::f32, data_type::s8)
                    && od.data_type() == data_type::s8 && id.is_dense();
            if (!args_ok) return invalid_arguments;

            const auto skip_mask
                    = primitive_attr_t::skip_mask_t::rnn_data_qparams
                    | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                    | primitive_attr_t::skip_mask_t::
                            rnn_weights_projection_qparams;
            if (!attr()->has_default_values(skip_mask))
                return invalid_arguments;

            // TODO: add support for layer and direction dimensions
            // weights_layer and weights_iter
            if (id.ndims() == 5
                    && !utils::one_of(
                            attr()->rnn_weights_qparams_.mask_, 0, 24))
                return unimplemented;
            // weights_projection
            if (id.ndims() == 4
                    && !utils::one_of(
                            attr()->rnn_weights_projection_qparams_.mask_, 0,
                            8))
                return unimplemented;

            // Check the proper memory desc has been passed to u8s8 and s8s8
            const bool check_u8s8 = (od.extra().flags & rnn_u8s8_compensation)
                    && od.extra().compensation_mask
                            == ((id.ndims() == 5) ? 27 /* 11011 */
                                                  : 13 /* 1101 */);
            const bool check_s8s8 = od.extra().flags & rnn_s8s8_compensation
                    && od.extra().compensation_mask == 0;
            if (!(check_u8s8 || check_s8s8)) return invalid_arguments;

            itag_ = format_tag::undef;

            format_tag_t otag, itag;

            itag = id.matches_one_of_tag(ldigo, ldio);
            otag = od.matches_one_of_tag(
                    ldgOI64o4i, ldgOI32o4i, ldgOI16o4i, ldOI32o4i, ldOI16o4i);
            if (itag != format_tag::undef && otag != format_tag::undef) {
                itag_ = itag;
                otag_ = otag;
            } else {
                return invalid_arguments;
            }

            CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace format_tag;

            const memory_desc_wrapper id(src_md());
            const size_t nelems = id.nelems();
            const auto &dims = id.dims();
            const auto ndims = id.ndims();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            const size_t quantization_size = nelems;
            // we do not use GO directly, as this can cause false
            // sharing when parallelizing on I (2 threads writing to
            // the same cache line)
            thr_scratch_comp_sz_ = (ndims == 5) ? dims[3] * dims[4] : dims[3];
            thr_scratch_comp_sz_ = utils::rnd_up(thr_scratch_comp_sz_, 16);
            const size_t reduction_size = nthr_ * thr_scratch_comp_sz_;

            scratchpad.template book<int8_t>(
                    key_reorder_rnn_weights_quantization, quantization_size);
            scratchpad.template book<int32_t>(
                    key_reorder_rnn_weights_reduction, reduction_size);
        }
    };

    rnn_brgemm_weights_reorder_s8_t(const pd_t *apd) : primitive_t(apd) {}

private:
    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
