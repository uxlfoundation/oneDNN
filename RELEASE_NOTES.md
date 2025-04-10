# Performance Optimizations
## Intel Architecture Processors
* Improved matmul and inner product primitives performance on processors with Intel AMX instruction set support.
* Improved performance of convolution and inner product primitives on processors with Intel AVX2 instruction set support.
* Improved performance of `int8` convolution support with zero points.
* Improved `fp32` convolution performance with `fp16` and `bf16` compressed weights on processors with Intel AVX2 or Intel AVX-512 instruction set support.
* Improved `fp16`/`bf16` depthwise convolution performance with `fp32` bias or `sum` post-ops or dilation.
* Improved `bf16` pooling backpropagation performance.
* Improved binary post-ops performance with `per_w` broadcast.

## Intel Graphics Products
* Improved performance on Intel Arc graphics for future Intel Core Ultra processors (code name Panther Lake).
* Improved convolution performance on:
  * Intel Arc Graphics for Intel Core Ultra processor series 2 (formerly Lunar Lake).
  * Intel Arc B-series discrete graphics (formerly Battlemage).
* Improved `int8` matmul performance with zero-points support for source and weight tensors.
* Improved `f4_e2m1` and `f4_e3m0` matmul and reorder performance.
* Improved performance of the following subgraphs with Graph API:
  * [Scaled Dot Product Attention (SDPA)] with `int4` and `int8` [compressed key and value].
  * `fp16`/`bf16` SDPA with `fp32` intermediate data types. Using `fp32` intermediate data types is recommended.
  * SDPA with head size 512 and 576.
  * [Grouped Query Attention (GQA)] with 5D input tensors.

[Scaled Dot Product Attention (SDPA)]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_graph_sdpa.html
[compressed key and value]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_graph_sdpa_compressed_kv.html
[Grouped Query Attention (GQA)]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_graph_gqa.html

## AArch64-based Processors
* Improved `fp16` reorder performance.
* Improved `int8` matmul performance.
* Improved `bf16` inner product forward propagation performance with Arm Compute Library (ACL).
* Improved `bf16` eltwise performance.
* Improved convolution performance on processors with SVE support with ACL.

# Functionality

## Common
* Extended Graph API [`Softmax`] operation to support `inf_as_zero` mode. This functionality enables SDPA subgraph compliant with Pytorch [Safe Softmax] semantics.

[`Softmax`]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_op_softmax.html
[Safe Softmax]: https://github.com/pytorch/pytorch/issues/55056

## Intel Architecture Processors

* Introduced support for `f32` convolution with `fp16` compressed weights.
* Enabled `int8`/`int4` compressed weights support in matmul primitive.

## Intel Graphics Products
* Introduced select algorithm support in [binary primitive].
* Introduced support for `f4_e2m1` and `f4_e3m0` data types in convolution primitive.
* Introduced support for the [GenIndex] operation in Graph API.

[binary primitive]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_binary.html
[GenIndex]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_op_genindex.html

## Generic GPU Vendor
* Introduced support for:
  * Vanilla RNN forward propagation.
  * Inner product backpropagation.
  * Group normalization.
* Improved accuracy of inner product primitive with sum post-ops for large shapes.

## NVIDIA GPUs
* Introduced Graph API support.

# Usability

* Added support for group normalization primitive with [`ONEDNN_ENABLE_PRIMITIVE`] build option.
* Enabled support for ROCm 6 on AMD GPUs.
* Improved CMake integration for oneDNN installation with Nvidia backend enabled.
* Reduced memory footprint for matmul primitive when using ACL.

[`ONEDNN_ENABLE_PRIMITIVE`]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_build_options.html#onednn-enable-primitive

# Validation
* Added benchdnn option [`--execution-mode`] to test oneDNN functionality with SYCL Graph record/execute mode.
* Extended benchdnn option [`--cold-cache`] with support for cold TLB mode.
* Added benchdnn option `--bia-dt` to control bias data type for matmul, inner product, convolution, and deconvolution primitives.
* Extended syntax of benchdnn `--dt` option in [Graph API driver] to manage data types of individual tensors in a pattern.

[`--execution-mode`]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.8/tests/benchdnn/doc/knobs_common.md#--execution-mode
[`--cold-cache`]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.8/tests/benchdnn/doc/knob_cold_cache.md
[Graph API driver]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.8/tests/benchdnn/doc/driver_graph.md

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated and will be removed in future releases. If you are using this API consider switching to [matmul primitive].

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.8/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_matmul.html

# Breaking Changes
* Removed the experimental [Graph Compiler] backend for Graph API. 

[Graph Compiler]: https://uxlfoundation.github.io/oneDNN/v3.7/dev_guide_graph_compiler.html

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Aditya Tewari @aditew01, Alexander Simonov @asimonov1, Denis @redradist, Dmitriy Ovchinnikov @inteldimitrius, Eliezer Weissmann @eliezerweissmann, Hubert Maciak @hmaciak, Ilya Lavrenov @ilya-lavrenov, James McGregor @Jmc18134, @jstachowintel, Marek Michalowski @michalowski-arm, Maria Zhukova @mzhukova, Orel Yehuda @yehudaorel, Ravi Pushkar @rpushkarr, Renato Barros Arantes @renato-arantes, @Shreyas-fuj, Shu Chen @shu1chen, Viktoriia Gvozdeva @vgvozdeva, Yair Obodovsky @yair-obodovsky, and @zhangfeiv0.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.8/MAINTAINERS.md
