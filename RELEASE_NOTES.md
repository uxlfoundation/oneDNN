# Performance Optimizations

## Intel 64/AMD64 Processors
* Introduced initial support for AI Compute Extensions (ACE) instructions. This functionality is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_ACE`.
* Improved performance of future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (codename Diamond Rapids).
* Improved performance of `fp8` matmul with block-wise weights on future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (codename Diamond Rapids).
* Improved performance of `bf16` and `f16` matmul with relaxed accumulation mode on Intel Xeon processors with Intel AMX instruction set support.
* Improved performance of strided deconvolution on Intel Xeon Scalable processors (formerly Sapphire Rapids).
* Improved performance of floating-point [Scaled Dot Product Attention (SDPA)] training forward propagation subgraph with Graph API.

[Scaled Dot Product Attention (SDPA)]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_graph_sdpa.html
## Intel Graphics
* Improved performance of future discrete GPUs based on Xe3p-XPC architecture (codename Crescent Island).
* Improved performance of future integrated GPUs based on Xe3p-LPG architecture (codename Nova Lake P).
* Improved performance of Scaled Dot Product Attention (SDPA) training forward and backpropagation subgraphs.
* Improved performance of Scaled Dot Product Attention (SDPA) subgraphs with asymmetric head sizes.
* Improved performance of grouped matmul with small group sizes.
* Improved performance of `f8` matmul with source and weights scales mask `3`.
* Improved performance of `f16` and `s8` matmul with `u4` weights and `N=1`.
* Reduced convolution and deconvolution primitives creation time.

## AArch64 Processors
TBD

## RISC-V Processors
TBD

# Functionality

## Functional API
* Introduced [`binary_mul_inplace`] algorithm for binary post-op. Unlike `binary_mul` new algorithm uses matmul destination tensor as one of the inputs. Optimized version is available in matmul on Intel GPUs.
* **[experimental]** Extended eltwise post-ops support in grouped matmul with all supported algorithms. Optimized implementation is available on Intel GPUs.
* **[experimental]** Extended grouped matmul with support for backpropagation cases (2D grouped by 3D dense and 2D grouped by 2D grouped) covering `f32` , `f16` and `bf16` data types. Optimized implementation is available for Intel GPUs. This is an experimental feature that requires opt-in with [`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`] build option. 

[`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_experimental.html#onednn-experimental-grouped-memory

[`binary_mul_inplace`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_post_ops.html#in-place-binary-post-ops

# Usability

## Common
* Updated `mxfp8` downconversion implementations to saturate instead of overflowing. New behavior is consistent with OCP MX specification and aligned with preferred behavior in PyTorch.

## Intel 64/AMD64 processors
* Cleaned up implicit narrowing conversions and removed suppression of MSVC compiler warning C4244. 

## Intel Graphics
* **[experimental]** Refactoring verbose profiling implementation for Level Zero runtime to avoid spurious synchronizations.
* **[experimental]** Introduced support for concurrent primitive execution with the Level Zero runtime on Intel GPUs.
* **[experimental]** Introduced support for verbose profiling based on sycl_ext_oneapi_profiling_tag SYCL extension. This is an experimental feature that requires opt-in with [`ONEDNN_EXPERIMENTAL_ENABLE_SYCL_PROFILING_TAG=ON`] build option.

[`ONEDNN_EXPERIMENTAL_ENABLE_SYCL_PROFILING_TAG=ON`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_experimental.html#onednn-experimental-enable-sycl-profiling-tag

# Validation
* Updated benchdnn `smoke` and `CI` test sets for matmul using parameter space sampling approach.
* **[experimental]** Extended benchdnn `--grouped` knob with `balanced`, `hot`, and `decode` strategies for offset generation to generate MoE-style group distributions in grouped matmul validation.

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated and will be removed in future releases. If you are using this API consider switching to [matmul primitive].

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.14/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_matmul.html

# Breaking changes
* Removed optimizations for Intel Iris Xe MAX Graphics and Intel Graphics included with 11th-14th generation Intel Core processors. oneDNN remains functional on these platforms and dispatches generic OpenCL implementation.
* Removed optimizations for processors with Intel SSE4.1 and Intel AVX instruction sets. oneDNN remains functional on these platforms and dispatches generic C++ implementation.
* Removed optimizations for `tf32` `fpmath_mode` in matmul on future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (codename Diamond Rapids).

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Abhishek Kumar @abhishek-iitmadras, Aditya Singh @adityasingh2400, Akihiro Tabuchi @Akihiro-Tabuchi, AragornOfKebroyd @AragornOfKebroyd, Aron Xu @happyaron, @AyushSinghBaiswar, Codrut Irimie @CodrutIrimieARM, Crefeda Rodrigues @cfRod, elimor01 @MorelElian, Emilio Cota @cota, Ishita Shreya @ishita-shreya, Kamil Jackiewicz @kjackiew, Kamil Wieloch @kwieloch-intel, Keerthana KT @Keerthana-64, Léandre LE DUC @leduclean, Leon Kennedy @leoken01, Megha Sangtani @megha-sangtani, Mohammed Bilgrami @mohbil01, Nikhil Gupta @nikhil-arm, PiotrReiterIntel @PiotrReiterIntel, Puneet Matharu @puneetmatharu, @rinatrap, Thiago Macieira @thiagomacieira, @Tiwari-Avanish, Udit Kumar Agarwal @uditagarwal97, @velonica0, Wang hongyan @ww8191201-coder, and @xinghai-zh.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.14/MAINTAINERS.md
