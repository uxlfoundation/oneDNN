# Performance Optimizations
## Intel Architecture Processors
* Improved performance on future Intel Xeon processors with Intel AVX 10.2 and Intel AMX instruction sets support.
  This functionality is not dispatched by default and requires opt-in with environment
  variable `ONEDNN_MAX_CPU_ISA=AVX10_2_512_AMX_2`.
* Improved performance on future Intel Core processors with Intel AVX 10.2 instruction set support. This functionality
  is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_512`.
* Improved performance of matmul primitive on processors with Intel AMX support.
* Improved performance of `f32` matmul primitive for GEMV cases on on processors with Intel AVX2 instruction
  set support.
* Improved matmul performance with `int4` and `int8` compressed weights and per-channel zero-points.
* Improved `f32` matmul performance with `int4` and `int8` compressed weights on processors with Intel AVX2 and
  Intel AVX512 instruction set support.
* Improved `bf16` matmul performance with `int4` and `int8` compressed weights on processors with Intel AVX512,
  Intel DL Boost and bfloat16 instruction set support.
* Improved performance of `int8` convolution primitive when using zero points.
* Improved performance of `int8` matmul and inner product primitives with `fp16` destination.
* Improved performance of `f32` and `bf16` convolution primitive with `int8` destination.
* Improved performance of RNN primitive on processors with Intel AVX2 instruction set support when using OpenMP runtime.
* Improved performance of subgraphs containing sequence of multiple binary ops with Graph API.

## Intel Graphics Products
* Improve GEMM performance for small batch size on Intel Core Ultra processors (Series 2) (formerly Lunar Lake).
* Improved matmul performance for Qwen2-7B shapes on Intel Arc graphics (formerly Alchemist) and
  Intel Arc Graphics for Intel Core Ultra processors (formerly Arrow Lake-H).
* Improved `int8` matmul performance with `int4` weights and per-tensor zero-points.
* Improved `bf16` matmul performance with `fp8` weights.
* Graph API optimizations:
  * Improved [Scaled Dot Product Attention (SDPA)] subgraph performance for inference when relaxed accumulation mode
    is enabled on Intel Core Ultra processors (formerly Meteor Lake).
  * Improved SDPA and GQA subgraphs performance when using host-side scalars.
  * Improved performance of GQA subgraph for 2nd token scenarios.
  * Improved performance of subgraphs containing sequence of multiple binary ops.
  * Improved performance of [Grouped Query Attention (GQA)] subgraphs for training forward and backward propagation.

[Grouped Query Attention (GQA)]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_graph_gqa.html#gqa-for-training-forward-propagation
[Scaled Dot Product Attention (SDPA)]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_graph_sdpa.html

## AArch64-based Processors
* Improved reorder primitive performance.
* Improved `bf16` convolutions performance.
* Improved convolutions performance on CPUs with 128-bit SVE support.
* Improved eltwise primitive performance on Arm(R) Neoverse(TM) N1 processor.

# Functionality
## Functional API
* Introduced [host-side scalar memory objects]. This functionality allows passing host-side scalars instead of device
  memory objects when using oneDNN with OpenCL or SYCL runtimes. Host-side scalars are currently supported in matmul
  and convolution primitives on Intel GPUs.
* Introduced support for pre-computed reductions in matmul primitive. This functionality is intended to improve
 performance in case of `int8` activations and `int8` weights with zero-point.

[host-side scalar memory objects]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_host_side_scalars.html

## Graph API
* Introduced [`host_scalar` property] for logical tensors. This functionality allows passing host-side scalars instead
  of device memory objects when using oneDNN with OpenCL or SYCL runtimes. Host-side scalars are currently supported to
  define attention scale, sequence length, and the negative infinity value in SDPA/GQA subgraphs.
* Introduced [accumulation mode attribute] support in `Matmul` op. This attribute allows relaxing `fp32` accumulation
  requirements to achieve performance benefits on some platforms.

[`host_scalar` property]: https://uxlfoundation.github.io/oneDNN/v3.10/enum_dnnl_graph_logical_tensor_property_type.html
[accumulation mode attribute]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_op_matmul.html

## Intel Graphics Products
* Introduced support for `fp4` weights in matmul primitive.
* Introduced support for weight scales and zero-points with group size 16 in matmul with compressed weights.

## Intel Architecture Processors
* Introduced `fp4` weights support for `fp32` matmul and convolution  for future Intel Xeon processors with
  Intel AVX10.2 instruction set support.

# Usability
* Extended diagnostics available in verbose mode for primitive descriptor creation issues.
* Extended dispatch diagnostics in verbose mode output for primitives implementations on Intel GPUs.

# Known Limitations
* Convolution primitive may require excessive amount of scratchpad memory for shapes with large input width value on Intel CPUs.
* `bf16` convolution primitive has a performance regression on Intel Arc B-series graphics.
* Reduction primitive may produce incorrect results for tensors exceeding 4 GB on Intel Arc graphics (formerly DG2) and Intel Arc Graphics for Intel Core Ultra processors (formerly Arrow Lake-H).
* Concat primitive may produce incorrect results for certain shapes on Intel Arc A-series GPUs.
* `fp16` matmul primitive has a performance regression on Intel GPUs based on Xe2 architecture.
* `f32` matmul primitive may sporadically produce incorrect results on Intel Arc B-series graphics.
* `int8` inner product primitive with tensors exceeding 4 Gb in size may produce incorrect results on Intel Datacenter GPU Max series.
* `bf16` layer normalization backpropagation may produce incorrect results on Intel Datacenter GPU Max Series.	

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated
  and will be removed in future releases. If you are using this API consider switching to [matmul primitive].

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.10/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_matmul.html

# Breaking Changes
## AArch64-based Processors
* Bumped the minimum required [Arm(R) Compute Library](https://github.com/ARM-software/ComputeLibrary) version to 52.4.0

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Andrei Hutu @Anndrey24,
Anna Sztukowska @asztukow, Arseniy Obolenskiy @aobolensk, Avanish Tiwari @Tiwari-Avanish, Daniel Kuts @apach301,
Daniel Whittaker @danwhittaker-arm, Deeksha Kasture @kasturedeeksha, George Nash @georgen117,
Henry Gardiner @henry-gar, Keanu Czirjak @keanucz, Krishna Sai @krishnasai-mcw,
Marek Michalowski @michalowski-arm, Sheldon Robinson @sheldonrobinson, @Shreyas-fuj, Viktoriia Gvozdeva @vgvozdeva,
Xiang1 Guo, Yejing Lai @Yejing-Lai, Yonghao Gu, Yusuf Butt @UseTheForce007, Zhibo Li @zhili03, @almayne, @co63oc,
@focusunsink, @gassan-arm, @jstachowintel, @pmanczak, @puneetmatharu, @raistefintel, @vishwascm, @vyevtyus, @zhangfeiv0,
@zhangjian29, and @xiazhuozhao.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.10/MAINTAINERS.md
