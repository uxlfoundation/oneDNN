#ifndef CPU_RV64_RVV_MATMUL_HPP
#define CPU_RV64_RVV_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

template <data_type_t d_type>
struct rvv_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GCV", rvv_matmul_t)

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const memory_desc_wrapper src_mdw(src_md(0));
            const memory_desc_wrapper weights_mdw(weights_md(0));
            const memory_desc_wrapper dst_mdw(dst_md(0));
            const memory_desc_wrapper bias_mdw = bias_md_;

            if (has_zero_dim_memory() || src_mdw.has_runtime_dims_or_strides()
                    || weights_mdw.has_runtime_dims_or_strides()
                    || dst_mdw.has_runtime_dims_or_strides()
                    || bias_mdw.has_runtime_dims_or_strides())
                return status::unimplemented;

            const bool types_ok = src_mdw.data_type() == d_type
                    && weights_mdw.data_type() == d_type
                    && dst_mdw.data_type() == d_type
                    && desc()->accum_data_type == d_type;
            if (!types_ok) return status::unimplemented;

            if (!attr()->scales_.has_default_values())
                return status::unimplemented;

            if (attr()->post_ops_.len() != 0) {
                rvv_postops_t po_handler(attr()->post_ops_);
                if (!po_handler.has_postops()) return status::unimplemented;
            }

            if (!set_default_formats()) return status::unimplemented;

            if (!check_layouts(src_mdw, weights_mdw, dst_mdw))
                return status::unimplemented;

            const auto wei_ndims = weights_mdw.ndims();
            for (int i = 0; i < wei_ndims - 2; ++i) {
                if (src_mdw.dims()[i] != weights_mdw.dims()[i]
                        && weights_mdw.dims()[i] != 1)
                    return status::unimplemented;
            }

            if (!check_bias(dst_mdw, bias_mdw)) return status::unimplemented;

            if (!set_default_formats()) return status::unimplemented;

            return status::success;
        }

        bool is_row_major(const memory_desc_wrapper &mdw) const {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;

            const auto &strides = mdw.blocking_desc().strides;
            if (strides[ndims - 1] != 1) return false;

            dim_t expected_stride = mdw.dims()[ndims - 1];
            for (int d = ndims - 2; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= mdw.dims()[d];
            }
            return true;
        }

        bool is_col_major(const memory_desc_wrapper &mdw) const {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;

            const auto &strides = mdw.blocking_desc().strides;
            const auto &dims = mdw.dims();

            if (strides[ndims - 2] != 1) return false;
            if (strides[ndims - 1] != dims[ndims - 2]) return false;

            dim_t expected_stride = dims[ndims - 2] * dims[ndims - 1];
            for (int d = ndims - 3; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= dims[d];
            }
            return true;
        }

        bool check_layouts(const memory_desc_wrapper &src_mdw,
                const memory_desc_wrapper &wei_mdw,
                const memory_desc_wrapper &dst_mdw) const {
            if (!is_row_major(src_mdw) || !is_row_major(dst_mdw)) return false;

            if (!is_row_major(wei_mdw) && !is_col_major(wei_mdw)) return false;

            return true;
        }

        bool check_bias(const memory_desc_wrapper &dst_mdw,
                const memory_desc_wrapper &bias_mdw) const {
            if (bias_mdw.is_zero()) return true;

            if (bias_mdw.data_type() != d_type) return false;

            const int dst_ndims = dst_mdw.ndims();
            const int bias_ndims = bias_mdw.ndims();
            if (bias_ndims > dst_ndims) return false;

            const auto *dst_dims = dst_mdw.dims();
            const auto *bias_dims = bias_mdw.dims();

            for (int d = 1; d <= bias_ndims; ++d) {
                const dim_t bias_dim = bias_dims[bias_ndims - d];
                const dim_t dst_dim = dst_dims[dst_ndims - d];
                if (bias_dim != 1 && bias_dim != dst_dim) return false;
            }
            return true;
        }
    };

    rvv_matmul_t(const pd_t *apd);
    status_t execute(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_MATMUL_HPP