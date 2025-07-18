# Generic GPU support

## General information

Support for a generic GPU is implemented with generic SYCL kernels. The feature
is disabled by default. Users must enable it at build time with the CMake option
`DNNL_GPU_VENDOR=GENERIC`. The target GPUs can be used via oneDNN engine
abstraction. The engine should be created using `dnnl::engine::kind::gpu` engine
kind or the user can provide `sycl::device` objects that correspond to the
target GPUs.

## Limitations
* Supported target devices: Intel and NVIDIA GPUs
* Known issue with matmul, RNN, inner product and softmax primitive on Intel
 devices, due to kernel argument size limit. 

## Pre-requisites
* Intel GPUs
    * [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
* NVIDIA GPUs
    * [oneAPI DPC++ Compiler with support for CUDA](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda)
      or [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.xvbgvc) with [NVIDIA plugin](https://developer.codeplay.com/products/oneapi/nvidia/home)

NOTE: The Intel GPU is the default target and therefore the SYCL kernels are
always compiled at least for the default target. If the compiler also supports
NVIDIA GPUs then the SYCL kernels will also be compiled for NVIDIA GPUs.

IMPORTANT: If there are multiple GPUs in the system it is the user's
responsibility to ensure that the correct SYCL device representing the target
GPU is selected at runtime. The environment variable `ONEAPI_DEVICE_SELECTOR`
may be used to restrict the set of devices that can be used. For example, if
there are Intel and NVIDIA GPUs in the system and the goal is to use the NVIDIA
one, the environment variable can be set to `cuda:*`.


# Supported Primitives

General limitations:

* Currently blocked formats are not supported by any implementations unless
  explicitly listed
* There's a limit of maximum 5 post-ops for the implementations
* The maximum supported size of any dimension of any input/output tensor of a
    primitive is `INT32_MAX`

## Batch Normalization

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`
* Supported data types:
    * Forward direction: `f32`, `bf16`, `f16`, `s8`
    * Backward direction: `f32`, `bf16`, `f16`

## Binary

* Supported formats: plain formats, `Ab32a`, `aBc32b`
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`, `s32`

## Convolution

The implementation supports forward data, backward data, and backward weights
directions.

* Supported input/output formats: plain formats
* Supported weights formats: `goiw`, `goihw`, `goidhw`, `oiw`, `oihw`, `oidhw`
* Supported data types: `f32`, `bf16`, `f16`, `s32`, `s8`, `u8`
* Limitations
    * Some very large problem sizes currently return `unimplemented` due to an
      issue with long execution times

## Concat

* Supported formats: plain formats
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `s32`

## Deconvolution

The implementation supports forward and backward data and backward weights
directions.

* Supported input/output formats: plain formats
* Supported weights formats: `goiw`, `goihw`, `goidhw`, `oiw`, `oihw`, `oidhw`
* Supported data types: `f32`, `bf16`, `f16`, `s32`, `s8`, `u8`
* Limitations
    * Some problems with large input/output tensors currently return `unimplemented`
      due to an issue with long execution times

## Eltwise

The implementation supports both forward and backward directions.

* Supported algorithms: `abs`, `clip`, `clip_v2`, `elu`, `exp`, `gelu_erf`,
`gelu_tanh`, `hardsigmoid`, `hardswish`, `linear`, `log`, `logistic`, `mish`,
`pow`, `relu`, `round`, `soft_relu`, `sqrt`, `square`,`swish` and `tanh`
* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`, `N`
* Supported data types: `f32`, `bf16`, `f16`, `s32`, `s8`, `u8`

## Group Normalization
* Supported Direction:  Both forward and backward directions are supported.
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`, `s32`
* Support Data layouts: All data layouts are supported.


## Inner Product

The implementation supports both forward and backward directions.

* Supported formats: All plain formats are supported.
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`, `s32`
* Supported post-ops: All post-ops mentioned in the specification are supported.
Note: The backward pass does not support post-ops. You should not use post-ops in the forward pass during training.

## Layer Normalization

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`
* Supported input/output data types for forward direction: `f32`, `bf16`, `f16`,
  `s8`, `u8`
* Supported input/output data types for backward direction: `f32`, `bf16`
* Supported scale/shift data types: `f32`, `bf16`, `f16`

## LRN

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`
* Supported data types: `f32`, `bf16`, `f16`

## Matmul

* Supported formats: plain formats
* Supported input/output data types: `f32`, `bf16`, `f16`, `s8`, `u8`, `s32`
* Limitations
    * Runtime dims is not supported
    * PReLU post-op is not supported

## Pooling

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`
* Supported data types for forward direction: `f32`, `bf16`, `f16`, `s8`, `u8`
* Supported data types for backward direction: `f32`, `bf16`, `f16`

## PReLU

The implementation supports both forward and backward propagations.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`
* Supported data types `f32`, `f16`, `bf16`, `s8` and `u8` data types

## Reorder

* Supported formats: plain formats
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`

## Resampling

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`
* Supported data types: `f32`, `bf16`, `f16`, `s32`, `s8`, `u8`

## Softmax/LogSoftmax

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`
* Supported data types for forward direction: `f32`, `bf16`, `f16`, `s8`, `u8`
* Supported data types for backward direction: `f32`, `bf16`, `f16`

## Shuffle

The implementation supports both forward and backward propagations.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`
* Forward pass supports `f32`, `f16`, `bf16` and `s8` data types.
* Backward pass supports `f32` and `bf16` data types.

## Sum

* Supported formats: plain formats with up to 7 dimensions
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`

## Reduction

* Supported formats: plain formats with up to 6 dimensions
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`

## RNN

The implementation supports only Vanilla RNN cell kind for both forward and backward propagations.

* Supported formats: `ldigo`, `ldgoi`
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`
* Supported direction: `left2right`, `right2left`, `concat`, `sum`
