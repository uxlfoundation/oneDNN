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

#ifndef CPU_X64_IP_CONVOLUTION_HPP
#define CPU_X64_IP_CONVOLUTION_HPP

#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"
#include "cpu/gemm/gemm.hpp"

#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct ip_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        pd_t(const pd_t &other)
            : cpu_convolution_fwd_pd_t(other)
            , ip_pd_(other.ip_pd_->clone())
            , name_(other.name_) {}

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_fwd_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                ip_pd_ = *it;
                const bool has_no_compensation
                        = ip_pd_->weights_md()->extra.flags
                        == memory_extra_flags::none;
                if (has_no_compensation) {
                    // avoid gemm implementation
                    std::string impl_name(ip_pd_->name());
                    if (std::string::npos != impl_name.find(GEMM_IMPL_STR))
                        continue;
                    // avoid reference implementation
                    if (std::string::npos != impl_name.find("ref:any"))
                        continue;
                    return status::success;
                }
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine);

        std::shared_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:any+";

        void init_name() {
            const std::string ips(ip_pd_->name());
            const std::string prefix = "x64:";
            const size_t pos = ips.find(prefix);
            name_.append(ips, pos + prefix.length(), std::string::npos);
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd);
    };

    ip_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

struct ip_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        pd_t(const pd_t &other)
            : cpu_convolution_bwd_data_pd_t(other)
            , ip_pd_(other.ip_pd_->clone())
            , name_(other.name_) {}

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_bwd_data_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            while (++it != it.end()) {
                ip_pd_ = *it;
                const bool has_no_compensation
                        = ip_pd_->weights_md()->extra.flags
                        == memory_extra_flags::none;
                if (has_no_compensation) {
                    // avoid gemm implementation
                    std::string impl_name(ip_pd_->name());
                    if (std::string::npos != impl_name.find(GEMM_IMPL_STR))
                        continue;
                    // avoid reference implementation
                    if (std::string::npos != impl_name.find("ref:any"))
                        continue;
                    return status::success;
                }
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine);

        std::shared_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:any";

        void init_name() {
            name_.append("+");
            name_.append(ip_pd_->name());
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd);
    };

    ip_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

struct ip_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        pd_t(const pd_t &other)
            : cpu_convolution_bwd_weights_pd_t(other)
            , ip_pd_(other.ip_pd_->clone())
            , name_(other.name_) {}

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_bwd_weights_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                ip_pd_ = *it;
                const bool has_no_compensation
                        = ip_pd_->weights_md()->extra.flags
                        == memory_extra_flags::none;
                if (has_no_compensation) {
                    // avoid gemm implementation
                    std::string impl_name(ip_pd_->name());
                    if (std::string::npos != impl_name.find(GEMM_IMPL_STR))
                        continue;
                    // avoid reference implementation
                    if (std::string::npos != impl_name.find("ref:any"))
                        continue;
                    return status::success;
                }
            }
            return status::unimplemented;
        }
        status_t init(engine_t *engine);

        std::shared_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:any";

        void init_name() {
            name_.append("+");
            name_.append(ip_pd_->name());
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd);
    };

    ip_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
