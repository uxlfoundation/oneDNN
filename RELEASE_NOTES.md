# Performance Optimizations
## Intel 64/AMD64 Processors
* Improved performance on future Intel Core Ultra processors with Intel AVX10.2 instruction set support (codename Nova Lake).
* Improved performance of `f8` [quantized Scaled Dot Product Attention (SDPA)] subgraph with Graph API.
* 

[quantized Scaled Dot Product Attention (SDPA)]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_graph_sdpa_quantized.html

## Intel Graphics
* Improved performance for future integrated GPUs based on Xe3p-LPG architecture (codename Nova Lake P).
* Improved `u8`/`s8` convolution performance on Intel Arc B-series graphics.
* Improved `f16` and `u8`/`s8` matmul performance with `u8`/`s8` and `u4`/`s4` weights in non-transposed layout.

## AArch64 Processors
* Improved `u8/s8` matmuls with `u8/s8/f16/s32` outputs
* Improved `u8/s8` convolutions with `bf16/f32` outputs
* Improved `u8/s8` lnorm performance
* Improved performance of convolution training on platforms with 128-bit SVE
* Improved performance of pooling on platforms with 128-bit SVE
* Improved performance of `bf16` inner-product
* Improved multi-threaded bnorm performance
* Improved binary operator, and post-op performance
* Improved performance of `gelu_erf` activations

## RISC-V Processors
* Improved `f32` convolution, matmul, inner product, binary, eltwise, pooling, batch normalization, and group normalization primitive performance on processors with `V` extension support.
* Improved `f16` matmul, binary, eltwise, pooling, softmax, and layer normalization primitive performance on processors with `Zvfh` extension support.
* Improved `bf16` matmul primitive performance on processors with `Zvfbfwma` extension support.

# Functionality

## Functional API
* **[experimental]** Introduced support for eltwise and binary post-ops in matmul with grouped memory. Optimized implementation is available on Intel GPUs.
* **[experimental]** Extended grouped matmul with NVFP4 quantization scheme, including support for `f4_e2m1` tensors with `f8_e4m3` grouped scales and per-group binary post-op to implement global `fp32` scale. This is an experimental feature that requires opt-in with [`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`] build option.

[`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_experimental.html#onednn-experimental-grouped-memory

## Graph API
* Introduced support for device-side seed, offset, and probability arguments for `Dropout` operation.

[`Dropout`]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_op_dropout.html

# Usability

## Common
* Introduced [user-managed scratchpad] support in Graph API.

[user-managed scratchpad]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_graph_scratchpad.html

## Intel Graphics
* Refactored verbose profiling on Intel GPUs to avoid spurious synchronizations with SYCL or OpenCL runtimes. The new implementation reports device time instead of host time and is compatible with SYCL Graph record/replay mode. 
* Reduced memory consumption of [Gated MLP subgraph] with Graph API.
* Enabled interoperability with SYCL Graph native recording mode for Intel GPUs.
* Introduced `ONEDNN_ZE_INCLUDE_DIR` and `ONEDNN_OCL_INCLUDE_DIR` build knobs to use Level Zero or OpenCL headers from a user-defined location instead of the vendored headers.
* **[experimental]** Introduced support for [persistent cache] with Level Zero runtime on GPU. Level Zero support is experimental.

## AArch64 Processors
* Fixed a correctness issue with leaky ReLU with alpha > 1
* Fixed an issue where convolutions could be accumulated in a lower precision than intended
* Reduced baseline stack-space usage across all operators

[Gated MLP subgraph]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_graph_gated_mlp.html
[persistent cache]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_persistent_cache.html

# Validation
* Extended SYCL Graph validation mode in benchdnn with support for native recording mode. This mode is enabled using `--execution-mode=native_graph` knob.
* Enabled SYCL recording mode validation `--execution-mode=graph` for benchdnn `--graph` driver.
* Introduced benchdnn knob `--mode=S` to improve performance validation speed in simulation or emulation environments.
* Improved GPU performance reporting for `--mode=F` by stabilizing measurement methodology and reducing inaccuracies caused by cache effects and run-to-run variability.

# Deprecated Functionality
* The [BLAS-like API], including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32`, is deprecated
  and will be removed in future releases. If you are using this API, consider switching to the [matmul primitive].
* `f4_e3m0` data type is deprecated and will be removed in future releases.
* Optimizations for Intel Iris Xe MAX Graphics and Intel Graphics included with 11th-14th Generation Intel Core Processors are deprecated and will be removed in future releases.


[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.13/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_matmul.html

# Breaking Changes
* The minimum version of Arm® Compute Library is now v53.1.0

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Alexandre de Limas Santana @alexandrelimassantana, Andrei Hutu @Anndrey24, Anna Sztukowska @asztukow, @bhanuprasad14, Fadi Arafeh @fadara01, George Nash @georgen117, Georgii Zagoruiko @AstonMartin-one-77, Henry Gardiner @henry-gar, Kamil Wieloch @kwieloch-intel, Keanu Czirjak @keanucz, Michał Patronik @mikita12, Qize Li @Ga1axy0, Rohan @Rohanjames1997, velonica0 @velonica0 and Xiuchuan Zhai @azhai219.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.13/MAINTAINERS.md
