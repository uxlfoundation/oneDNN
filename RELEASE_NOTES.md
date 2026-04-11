# Performance Optimizations
## Intel 64/AMD64 Processors
* Improved performance on future Intel Core Ultra processors with Intel AVX10.2 instruction set support (code name Nova Lake). These optimizations are now enabled by default on compatible processors.
* Improved performance on future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (code name Diamond Rapids). These optimizations are now enabled by default on compatible processors.
* Improved performance of `fp8` and `int8` matmul with transposed source on processors with Intel AMX instruction set support.
* Improved performance of `bf16` and `f16` matmul with transposed source on processors with Intel AVX2 instruction set support.

## Intel Graphics
* Introduced initial performance optimizations for future integrated GPUs based on Xe3p-LPG architecture.
* Introduced initial performance optimizations for future discrete GPUs based on Xe3p-XPC architecture. This is a preview functionality not recommended for production use.
* Improved `f16` matmul performance on Intel Arc Graphics for Intel Core Ultra processor Series 3 (formerly Panther Lake).
* Improved performance of matmul with host-side scalar arguments.
* Improved matmul performance for cases with small M/N and large K.
* Improved SDPA forward and backpropagation subgraph performance with Graph API.

## AArch64 Processors
* Improved `f16` and `f32` softmax performance across Arm Neoverse cores.
* Improved eltwise performance on Arm Neoverse N1 cores.
* Improved matmul and convolution performance on Arm Neoverse V2 cores.
* Improved performance of multiple primitives by quering processor cache sizes.

## RISC-V Processors
* Improved `f32` matmul, inner product, convolution, softmax and layer normalization primitives performance on processors with `V` extension support.
* Improved `f16` softmax primitive performance on processors with `Zvfh` extension support.

# Functionality
## Functional API
* **[experimental]** Introduced [grouped memory format] and [grouped matmul support] to improve performance of AI models based on Mixture-of-Experts (MoE) architecture. This is an experimental feature that requires opt-in with [`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`] build option. Optimized version of this functionality is implemented for Intel GPUs.
* **[experimental]** Extended grouped matmul with optional execution-time hint [`DNNL_ARG_HINT_MAX_GROUP_SIZE`] to communicate the maximum size of the group across the variable dimension for the execution call.

[grouped memory format]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_grouped_mem.html
[grouped matmul support]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_matmul.html#grouped-gemm-support
[`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_experimental.html#onednn-experimental-grouped-memory
[`DNNL_ARG_HINT_MAX_GROUP_SIZE`]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_matmul.html#execution-hints

## Graph API
* Introduced [`Dropout`] operation. Extended supported fusion patterns to enable fusion of `Dropout` with `Matmul`, `Softmax`, and elementwise operations.

[`Dropout`]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_op_dropout.html

# Usability
## Common
* Extended information about primitive execution available in VTune Profiler with the same level of details as reported by oneDNN [verbose mode]. This feature requires VTune Profiler 2025.7 or later.

[verbose mode]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_verbose.html

## Intel Graphics
* **[experimental]** Introduced support for Level Zero runtime on Intel GPUs. New functionality includes [Level Zero interoperability API] and build knob `ONEDNN_GPU_RUNTIME=ZE`.

[Level Zero interoperability API]: https://uxlfoundation.github.io/oneDNN/v3.12/group_dnnl_api_ze_interop.html

## AArch64 Processors
* Reduced memory usage of certain convolutions on Arm Neoverse V1/V2 cores.
* Fixed a bug causing high-memory usage and crashes in convolution with certain post-ops.

# Validation
* Extended benchdnn with support for integer masks in quantization attributes.
* Improved consistency of benchdnn performance results when data compression is enabled by default on Intel Graphics.

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated
  and will be removed in future releases. If you are using this API consider switching to [matmul primitive].
* `f4_e3m0` data type is deprecated and will be removed in future releases.

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.12/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_matmul.html

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Alexandre de Limas Santana @alexandrelimassantana, Andrei (Andrey) Khropov @andrey-khropov, Andrei Hutu @Anndrey24, Fadi Arafeh @fadara01, George Nash @georgen117, Kamil Wieloch @kwieloch-intel, Kasture Deeksha, MarkVeerasingam @MarkVeerasingam, Nikhil Gupta @nikhil-arm, @pmanczak, @vishwascm, and Xia Zhuozhao @xiazhuozhao.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.12/MAINTAINERS.md
