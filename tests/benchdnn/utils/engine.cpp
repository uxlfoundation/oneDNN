/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "utils/engine.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "oneapi/dnnl/dnnl_ze.hpp"
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "tests/test_thread_decl.hpp"
#endif

// Engine kind used to run oneDNN primitives for testing
dnnl_engine_kind_t engine_tgt_kind = dnnl_cpu;
// Engine index used to run oneDNN primitives for testing
size_t engine_index = 0;

const engine_t &get_test_engine() {
    static const engine_t instance(engine_tgt_kind);
    assert((engine_tgt_kind == dnnl_cpu && instance.is_cpu())
            || (engine_tgt_kind == dnnl_gpu && instance.is_gpu()));
    return instance;
}

const engine_t &get_cpu_engine() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    // In case of lacking CPU engine, just re-use testing one.
    return get_test_engine();
#else
    static const engine_t instance(dnnl_cpu);
    return instance;
#endif
}

namespace {
void maybe_print_cpu_engine_error_message() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    fprintf(stderr,
            "ERROR: can't create CPU engine. Possible reasons for this error:\n"
            "- Incorrect SYCL_DEVICE_FILTER. The filter must be either unset "
            "or include 'opencl:cpu' devices.\n"
            "- Missing TBB library which is required for OpenCL CPU runtime. "
            "Check that TBB library is available in the system.\n"
            "- Missing OpenCL CPU runtime or other issues with OpenCL CPU "
            "runtime. Check that output from `sycl-ls` or `clinfo -l` commands "
            "include any CPU devices.\n");
#endif
}
} // namespace

engine_t::engine_t(dnnl_engine_kind_t engine_kind) {
    size_t idx = engine_kind == dnnl_cpu ? 0 : engine_index;
    try {
        engine_ = dnnl::engine(
                static_cast<dnnl::engine::kind>(engine_kind), idx);
    } catch (const dnnl::error &e) {
        if (engine_kind == dnnl_cpu) maybe_print_cpu_engine_error_message();
        DNN_SAFE_V(e.status);
    }
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) {
        if (has_bench_mode_bit(mode_bit_t::corr)) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: the modifier to disable host memory usage "
                    "cannot be used for correctness testing.");
            DNN_SAFE_V(dnnl_invalid_arguments);
        }
    }
    // Temporary workaround.
    // TODO: make mode-modifier=M by default for perf on GPU.
    // TODO-TODO: make mode-modifier=M by default (including CPU).
    if (has_bench_mode_bit(mode_bit_t::perf) && engine_kind == dnnl_gpu) {
        bench_mode_modifier |= mode_modifier_t::no_ref_memory;
    }
}

engine_t::engine_t(dnnl_engine_t engine) : engine_(engine, /* weak = */ true) {}

engine_t::engine_t(const dnnl::engine &engine) : engine_(engine) {}

engine_t::engine_t(const engine_t &other, bool recreate_on_copy) {
    if (!recreate_on_copy) {
        engine_ = other.engine_;
        return;
    }

    if (other.is_cpu()) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        engine_ = dnnl::sycl_interop::make_engine(
                dnnl::sycl_interop::get_device(other.engine_),
                dnnl::sycl_interop::get_context(other.engine_));
#else
        engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
#endif
    } else if (other.is_gpu()) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        engine_ = dnnl::ocl_interop::make_engine(
                dnnl::ocl_interop::get_device(other.engine_),
                dnnl::ocl_interop::get_context(other.engine_));
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        engine_ = dnnl::sycl_interop::make_engine(
                dnnl::sycl_interop::get_device(other.engine_),
                dnnl::sycl_interop::get_context(other.engine_));
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
        engine_ = dnnl::ze_interop::make_engine(
                dnnl::ze_interop::get_driver(other.engine_),
                dnnl::ze_interop::get_device(other.engine_),
                dnnl::ze_interop::get_context(other.engine_));
#else
        assert(!"unsupported GPU runtime");
#endif
    } else {
        assert(!"unsupported engine kind");
    }
}

dnnl::engine::kind engine_t::get_kind() const {
    // An empty engine (e.g., a host scalar memory carries no engine) returns
    // `any` kind.
    if (engine_.get(/* allow_empty = */ true) == nullptr)
        return dnnl::engine::kind::any;
    return engine_.get_kind();
}

bool engine_t::is_cpu() const {
    return get_kind() == dnnl::engine::kind::cpu;
}

bool engine_t::is_gpu() const {
    return get_kind() == dnnl::engine::kind::gpu;
}

bool is_cpu(const engine_t &engine) {
    return engine.is_cpu();
}

bool is_gpu(const engine_t &engine) {
    return engine.is_gpu();
}

bool is_async(const engine_t &engine) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (is_cpu(engine))
        return dnnl::testing::get_threadpool()->get_flags()
                & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
#else
    if (is_cpu(engine)) return DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL;
#endif
    // all supported GPU runtimes are async
    return is_gpu(engine);
}

bool is_sycl_engine(const engine_t &engine) {
    if (is_cpu(engine)) return DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL;
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL;
    return false;
}

bool is_opencl_engine(const engine_t &engine) {
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL;
    return false;
}

bool is_ze_engine(const engine_t &engine) {
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE;
    return false;
}

bool is_nvidia_gpu(const engine_t &engine) {
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if (!is_gpu(engine)) return false;
    constexpr int nvidia_vendor_id = 0x10DE;
    auto eng = dnnl::engine(engine, true);
    auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<::sycl::info::device::vendor_id>();
    return eng_vendor_id == nvidia_vendor_id;
#endif
    return false;
}

bool is_amd_gpu(const engine_t &engine) {
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
    if (!is_gpu(engine)) return false;
    constexpr int amd_vendor_id = 0x1002;
    auto eng = dnnl::engine(engine, true);
    auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<::sycl::info::device::vendor_id>();
    return eng_vendor_id == amd_vendor_id;
#endif
    return false;
}

bool is_generic_gpu(const engine_t &engine) {
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR == DNNL_VENDOR_GENERIC
    return is_gpu(engine);
#endif

    return false;
}
