# Performance Optimizations
## Intel 64/AMD64 Processors
* Improved `fp32` matmul performance with `fp4` compressed weights.
* Improved `fp32` matmul performance for cases when one of the tensors has a trivial dimension on processors with Intel AVX-512 instruction set support.

## Intel Graphics
* Improved `fp16`/`bf16` matmul performance for large tensor cases on Intel Graphics for Intel Core Ultra processor Series 3 (formerly Panther Lake).
* Improved matmul performance for cases with 4-byte alignment on Intel GPUs based on Xe2 architecture.
* Improved performance of `fp16`/`bf16` matmul with `mxfp4` weights.
* Improved convolution performance with host-side scalar scales and zero points.

## AArch64 Processors
* Improved performance of `bf16` matmul.
* Improved performance of `bf16/int8` convolutions.
* Improved matmul performance for cases when one of the tensor has a trivial dimension.
* Improved performance of `s8/u8` eltwise post-ops on Arm(R) Neoverse(TM) V1 processors.
* Improved `f16` and `bf16` eltwise performance with `abs`, `relu`, `square`, `sqrt`, `clip`, and `clip_v2` algorithms.
* Improved eltwise `exp` algorithm performance on Arm(R) Neoverse(TM) N1 processors.
* Improved reorder primitive performance.

## RISC-V Processors
* Improved `f32` matmul, inner product, convolution, softmax, batch normalization, layer normalization, and group normalization primitives performance.
* Improved eltwise and binary primitives performance.
* Improved `f32` and `fp16` pooling primitive performance.
* Improved `fp32` to `u8` reorder primitive performance.

# Functionality
## Common

## Functional API
* Introduced destination tensor [dynamic quantization] in matmul primitive following Open Compute Microscaling (MX) formats specification. See [MXFP8 matmul tutorial] for quick introduction into MX-capabilities in oneDNN.
* Introduced support for NVFP4 quantization scheme. The changes include support for `fp8_e4m3` grouped scales and dynamic quantization support for destination tensor with NVFP4-specific formula for scales computation.
* Introduced support for dropout as a primitive attribute for matmul, softmax and eltwise primitives.

[dynamic quantization]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_attributes_quantization.html#dynamic-quantization
[MXFP8 matmul tutorial]: https://uxlfoundation.github.io/oneDNN/v3.11/page_mxfp_matmul_cpp.html#doxid-mxfp-matmul-cpp

## Graph API 
* Introduced support for [RMS Normalization] operation. 
* Introduced support for output gradient of attention mask for SDPA and GQA training.

[RMS Normalization]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_op_rmsnorm.html

## Intel Graphics
* Introduced support for convolution with `u8` weights.
* Introduced support for 2D grouped scales in `fp8` matmul.

## Intel 64/AMD64 Processors
* Introduced support for different data types of source and destination in pooling forward propagation.

## AArch64 Processors
* Added limited support for the BRGEMM Microkernel API
* Added limited support for Windows on Arm builds with MSVC

# Usability
## Common
* Extended [quantization attributes] documentation to cover all quantization schemes supported by the library.
* Added [matmul fp8 quantization] example demonstrating use of matmul primitive with `fp8` source, destination, and weights.
* Enabled `ONEDNN_ENABLE_GRAPH_DUMP` knob by default.

## Intel 64/AMD64 Processors
* Extended oneDNN [threadpool runtime] with an option to support asynchronous execution and updated all CPU implementations accordingly. This extension makes oneDNN compatible with OpenXLA "thunk" runtime.
* Introduced [`ONEDNN_SAFE_RBP`] build knob that instructs x64 implementations to preserve value of `rbp` register for tools that rely on stack unwinding. This option may have visible performance impact on some workloads.

## AArch64 Processors
* Fixed a potential overflow on AArch64 builds with Arm Compute Library.
* Significantly reduced memory consumption of convolution primitive with large spatial filters during primitive creation.

## Intel Graphics
* Removed build time dependency on OpenCL runtime in SYCL build configuration.

[quantization attributes]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_attributes_quantization.html
[matmul fp8 quantization]: https://uxlfoundation.github.io/oneDNN/v3.11/page_matmul_f8_quantization_cpp.html
[threadpool runtime]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_build_options.html#threadpool
[verbose mode]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_verbose.html
[`ONEDNN_SAFE_RBP`]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_build_options.html#onednn-safe-rbp

# Validation

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated
  and will be removed in future releases. If you are using this API consider switching to [matmul primitive].

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.11/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_matmul.html

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Andrei Hutu @Anndrey24, Anna Sztukowska @asztukow, Arseniy Obolenskiy @aobolensk, Avanish Tiwari @Tiwari-Avanish, czekun @ZackyLake, Deeksha Kasture @kasturedeeksha, Fadi Arafeh @fadara01, Gassan Salama @gassan-arm, Henry Gardiner @henry-gar, @jstachowintel, Keanu Czirjak @keanucz, Krishna Sai @krishnasai-mcw, Murray Steele @murste01, Narendra Bagria @narenbagria, Joseph Kuo @PershingSquare, @pmanczak, @vishwascm, Yejing Lai @Yejing-Lai, 夏卓昭 @xiazhuozhao

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.11/MAINTAINERS.md
