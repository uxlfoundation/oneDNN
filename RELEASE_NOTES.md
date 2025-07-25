# Performance Optimizations
## Intel Architecture Processors
* Introduced initial support for future Intel Xeon processors with Intel AVX 10.2 and Intel AMX instruction sets support.
  This functionality is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_512_AMX_2`.
* Introduced initial support for future Intel Core processors with Intel AVX 10.2 instruction set support. This functionality is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_512`.
* Improved initialization time for convolution primitive when a large number of threads is used by introducing a new thread partition estimation and adjusting several blocking parameters.
* Improved performance of `fp8` convolution primitive with scales and `bf16` output
* Improved performance of matmul primitive with post-ops on processors with Intel AMX support
* Improved performance of RNN primitive for LBR_GRU and VANILLA_LSTM cell types on processors with Intel AVX2 instruction set support
* Improved performance of the following subgraphs with Graph API:
    * [Scaled Dot Product Attention (SDPA)] with implicit causal mask.
    * [Grouped Query Attention (GQA)] flavor specific for GEMMA models.

[Scaled Dot Product Attention (SDPA)]: https://uxlfoundation.github.io/oneDNN/v3.9/dev_guide_graph_sdpa.html
[Grouped Query Attention (GQA)]: https://uxlfoundation.github.io/oneDNN/v3.9/dev_guide_graph_gqa.html

## Intel Graphics Products
* Improved performance on Intel GPUs based on Xe3 architecture.
* Improved matmul performance for Intel Arc Graphics for Intel Core Ultra processors (Series 2) (formerly Lunar Lake).
* Improved RNN primitive performance with LBR_GRU cell type.
* Improved `int8` convolution performance with plain weights and trivial filter.
* Improved convolution performance with `NCHW` activations with 1x1 filter and unit strides.
* Improved `fp32` softmax performance.
* Improved performance of reorder when used with USM host memory.
* Improved performance of the following subgraphs with Graph API:
    * SDPA with implicit causal mask.
    * SDPA with bottom-right implicit causal mask.
    * `fp32` SDPA.
    * `fp16` SDPA on Intel GPUs without Intel XMX cores.

## AArch64-based Processors
* Improved `int8` convolution performance.
* Improved `bf16` depthwise convolution performance.
* Improved `f16` matmul performance with Arm Compute Library (ACL).

# Functionality
## Functional API
* Introduced [Root Mean Square Normalization (RMSNorm) mode] for layer normalization primitive. This functionality is optimized for Intel CPUs and Intel GPUs.
* Sparse memory objects and sparse matmul are promoted to production status.

[Root Mean Square Normalization (RMSNorm) mode]: https://uxlfoundation.github.io/oneDNN/v3.9/dev_guide_layer_normalization.html#root-mean-square-normalization-mode

## Graph API
* Introduced support for tanh approximation in [`GELU`] operation.
* Extended Graph API [`Softmax`] operation to support optional `stats` output.
* Introduced support for SDPA training forward propagation and backpropagation.

[`GELU`]: https://uxlfoundation.github.io/oneDNN/v3.9/dev_guide_op_gelu.html
[`SoftMax`]: https://uxlfoundation.github.io/oneDNN/v3.9/dev_guide_op_softmax.html

## Microkernel API
* Introduced support for `fp8` data type.

## Intel Architecture Processors
* Introduced support for select algorithm in binary post-op.
* Introduced source, destination, and weight scales support in `fp8` convolution and deconvolution primitives.

## Intel Graphics Products
* Introduced support for select algorithm in binary primitive.

## Generic GPU Vendor
* Introduced support for RNN Vanilla backward propagation.

# Usability
* Enabled build with `-Wundef` compiler flag.
* [Experimental] Introduced support for kernel compilation with [SYCL kernel compiler] extension.

[SYCL kernel compiler]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc

# Validation
* Improved benchdnn performance by optimizing input data filling and testing results comparison steps. 

# Known Limitations

# Deprecated Functionality
* [BLAS-like API] including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions is deprecated and will be removed in future releases. If you are using this API consider switching to [matmul primitive].

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.8/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_matmul.html

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Aditya Tewari @aditew01, Alexander Simonov @asimonov1, @Anallear, Anna Sztukowska @asztukow, Avanish Tiwari @Tiwari-Avanish, Dmitriy Ovchinnikov @inteldimitrius, Kasture Deeksha, Krishna Sai @krishnasai-mcw, Manaal @manaalmj, Marek Michalowski @michalowski-arm, Orel Yehuda @yehudaorel, Ruqiu Cao @rcao8, Tsao Zhong @CaoZhongZ, Viktoriia Gvozdeva @vgvozdeva, Yair Obodovsky @yair-obodovsky, Ye Tao @taoye9, Yuanyuan Chen @cyyever, @gausah-arm, @karmeh01, @pmanczak, and @zhangfeiv0. We would also like to thank everyone who asked questions and reported issues.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.9/MAINTAINERS.md
