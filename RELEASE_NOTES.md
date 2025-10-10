# Performance Optimizations
## Intel Architecture Processors
* Improved performance on future Intel Xeon processors with Intel AVX 10.2 and Intel AMX instruction sets support.
  This functionality is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_512_AMX_2`.
* Improved performance on future Intel Core processors with Intel AVX 10.2 instruction set support. This functionality is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_512`.
* Improved matmul performance with `int4` weights and per-channel zero-points.
* Improved performance of `int8` matmul and inner product primitives with `fp16` destination.
* Improved performance of subgraphs containing sequence of multiple binary ops with Graph API.

## Intel Graphics Products
*Improve GEMM performance for small batch size on Intel Core Ultra processors (Series 2) (formerly Lunar Lake).
* Improved matmul performance for Qwen2-7B shapes on Intel Arc graphics (formerly DG2) and Intel Arc Graphics for Intel Core Ultra processors (formerly Arrow Lake-H).
* Improved `int8` matmul performance with `int4` weights and per-tensor zero-points.
* Improved `bf16` matmul performance with `fp8` weights.
* Graph API optimizations:
  * Improved Scaled Dot Product Attention (SDPA) subgraph performance when relaxed accumulation mode is enabled on Intel Core Ultra processors (formerly Meteor Lake).
  * Improved performance of [Grouped Query Attention (GQA)] subgraphs for training forward and backward propagation.
  * Improved SDPA and GQA subgraphs performance when using host-side scalars.
  * Improved performance of GQA subgraph for 2nd token scenarios.
  * Improved performance of subgraphs containing sequence of multiple binary ops.

[Grouped Query Attention (GQA)]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_graph_gqa.html#gqa-for-training-forward-propagation

## AArch64-based Processors
TBD

# Functionality
## Functional API

* Introduced [host-side scalar memory objects]. This functionality allows passing host-side scalars instead of device memory objects when using oneDNN with OpenCL or SYCL runtimes. Host-side scalars are currently supported in matmul and convolution primitives on Intel GPUs.

[host-side scalar memory objects]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_host_side_scalars.html
* Introduced support for pre-computed reductions in matmul primitive. This functionality is intended to improve performance in case of `int8` activations and `int8` weights with zero-point.

## Graph API 

* Introduced [`host_scalar` property] for logical tensors. This functionality allows passing host-side scalars instead of device memory objects when using oneDNN with OpenCL or SYCL runtimes.
* Introduced [accumulation mode attribute] support in `Matmul` op. This attribute allows relaxing `fp32` accumulation requirements to achieve performance benefits on some platforms. 

[`host_scalar` property]: https://uxlfoundation.github.io/oneDNN/v3.10/enum_dnnl_graph_logical_tensor_property_type.html
[accumulation mode]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_op_matmul.html

## Intel Graphics Products
* Introduced support for `fp4` weights in matmul primitive.
* Introduced support for grouped quantization with group size 16 in matmul with int8 compressed weights on Intel GPUs.
* Introduced support group size16 `int8` for decompressed weight with regular weights decompression.

## Intel Architecture Processors
* Introduced `fp4` weights support for `fp32` matmul and convolution  for future Intel Xeon processors with Intel AVX 10.2/AVX512 instruction sets support.

# Usability
* Extended diagnostics available in verbose mode for primitive descriptor creation issues.
* Extended dispatch diagnostics in verbose mode output for primitives implementations on Intel GPUs.

# Validation

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated and will be removed in future releases. If you are using this API consider switching to [matmul primitive].

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.10/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_matmul.html

# Breaking Changes

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Andrei Hutu @Anndrey24, 
Anna Sztukowska @asztukow, Arseniy Obolenskiy @aobolensk, Avanish Tiwari @Tiwari-Avanish, Daniel Kuts @apach301, Daniel Whittaker @danwhittaker-arm, Deeksha Kasture @kasturedeeksha, George Nash @georgen117, Henry Gardiner @henry-gar, Kasture Deeksha, Keanu Czirjak @keanucz, Krishna Sai @krishnasai-mcw, Marek Michalowski @michalowski-arm, Sheldon Robinson @sheldonrobinson, @Shreyas-fuj, Viktoriia Gvozdeva @vgvozdeva, Xiang1 Guo, Yejing Lai @Yejing-Lai, Yonghao Gu, Yusuf Butt @UseTheForce007, Zhibo Li @zhili03, @almayne, @co63oc, @focusunsink, @gassan-arm, @jstachowintel, @pmanczak, @puneetmatharu, @raistefintel, @vishwascm, @vyevtyus, @zhangfeiv0, @zhangjian29, and @xiazhuozhao.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.10/MAINTAINERS.md
