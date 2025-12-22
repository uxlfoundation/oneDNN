# Support Root Mean Square Normalization in Graph API
=====================================================================

## Motivation

Root Mean Square Normalization (RMSNorm)[[#1]][1] has emerged as a key component
in modern large language models (LLMs) and transformer architectures. Compare to
Layer Normalization, RMSNorm is more efficient in normalizing inputs by using
only the root mean square statistic, eliminating the need for mean subtraction
and variance computation.

RMSNorm has been widely adopted in state-of-the-art models including:
- LLaMA, LLaMA 2, and LLaMA 3 (Meta)
- Gemma (Google)
- Various other transformer-based architectures

This RFC proposes adding the RMSNormalization operation to the oneDNN Graph API.
This addition enables mapping RMSNorm to the oneDNN Graph, allowing fusion with
other operations for enhanced backend optimizations. This enhancement aims to
boost performance for LLM inference and training workloads.

## Mathematical Formulation

RMSNorm normalizes the input using the root mean square:

  - Root Mean Square (RMS):

    ```math
    RMS(x) = {\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}}
    ```

  - Normalization:

    ```math
    y_i = (x_i / RMS(x)) * γ_i
    ```

Where:
- `x` is the input tensor
- `n` is the normalization dimension size
- `ε` (epsilon) is a small constant for numerical stability
- `γ` (gamma/weight) is the learnable scale parameter
- `y` is the output tensor

Key differences from LayerNorm:
- No mean subtraction (β/bias parameter)
- No variance computation
- Comparable accuracy in LLM training

## Operations Used in Frameworks and Toolkits

### RMS Normalization Forward

| Framework | PyTorch                                    | TensorFlow/Keras        | ONNX                     | OpenVINO                | DNNL                             |
| --------- | ------------------------------------------ | ----------------------- | ------------------------ | ----------------------- | -------------------------------- |
| op        | rms_norm[[#2]][2]                          | RMSNormalization[[#3]][3] | RMSNormalization[[#4]][4] | RMS[[#5]][5]            | rms_normalization_forward[[#6]][6] |
| input     | input                                      | inputs                  | X                        | data                    | src                              |
| input     | weight                                     | scale                   | scale                    | gamma                   | gamma                            |
| input     |                                            |                         |                          |                         | beta                             |
| attribute | normalized_shape / eps                     | axis, epsilon           | axis, epsilon, stash_type | epsilon                 | epsilon                          |
| output    | output                                     | outputs                 | Y                        | output                  | dst                              |

**Framework Implementation Notes:**

- **PyTorch**: Native `rms_norm` operation defined in ATen. Computes root mean
  square normalization over specified dimensions. The operation is defined in
  `native_functions.yaml` and is widely used in LLM implementations including
  HuggingFace Transformers.

- **TensorFlow/Keras**: Native `RMSNormalization` layer in Keras. Normalizes
  inputs using root mean square over specified axis. Supports learnable scale
  parameter and configurable epsilon for numerical stability.

- **ONNX**: Dedicated `RMSNormalization` operation (opset 23) for root mean square
  layer normalization. Computation: `Y = X / sqrt(ReduceMean(X^2, normalized_axes) + epsilon) * scale`,
  where `normalized_axes = [axis, ..., rank of X - 1]`. Key attributes:
  - `axis` (default=-1): First normalization dimension （-1 is last dimension）
  - `epsilon` (default=1e-05): Small constant for numerical stability
  - `stash_type` (default=1): Floating-point precision for computation (1=FP32)

- **OpenVINO**: Dedicated `RMS` operation for root mean square normalization.
  Computes `output = data / sqrt(ReduceMean(data^2, axes) + epsilon) * gamma`.
  Supports `compute_type` attribute for specifying accumulation precision.

- **DNNL**: Primitive support added (refer to primitive RFC).

### RMS Normalization Backward

| Framework | PyTorch (custom)         | DNNL                           | oneDNN Graph API               |
|-----------|--------------------------|--------------------------------|--------------------------------|
| op        | rms_norm_backward        | rms_normalization_backward     | RMSNormBackward                |
| input     | grad_out                 | src                            | src                            |
| input     | input                    | diff_dst                       | diff_dst                       |
| input     | rstd                     | variance                       | variance (optional)            |
| input     | weight (optional)        |                                |                                |
| input     |                          |                                |                                |
| attribute | normalized_shape         |                                | axis                           |
| attribute | eps                      | epsilon                        | epsilon                        |
| output    | grad_input               | diff_src                       | diff_src                       |
| output    | grad_weight (optional)   |                                |                                |

## Proposal

### Option 1: Add RMSNorm as a dedicated operation (Recommended)

#### Forward Operation

| RMSNorm            | oneDNN Graph API   |
| ------------------ | ------------------ |
| input              | src                |
| input              | gamma (optional)   |
| attribute          | axis               |
| attribute          | epsilon            |
| output             | dst                |

**Operation Specification:**

- **src** (required): Input tensor of shape `[N, ...]` where normalization is
  performed over dimensions `[axis, ..., rank-1]`.

- **gamma** (optional): Scale parameter that must be broadcastable to the input
  tensor. If not provided, the output is not scaled (equivalent to gamma=1).

- **axis** (required): The first normalization dimension. If rank(src) is r,
  axis' allowed range is [-r, r). Negative value means counting dimensions from
  the back. For example, `axis=-1` normalizes over the last dimension only.

- **epsilon** (optional, default=1e-5): Small constant added to denominator for
  numerical stability.

- **dst** (required): Output tensor with the same shape as src.

**Semantic Notes:**

1. Gamma parameter is optional. If not provided, only normalization is performed
   without scaling (equivalent to gamma=1).
2. No beta/bias parameter (distinguishes from LayerNorm).
3. Aligned with most frameworks (PyTorch, TensorFlow, ONNX) which do not expose
   intermediate statistics in the forward pass.

**Example Usage:**

```cpp
// Create RMSNorm op for transformer layer
// Input: [batch=32, seq_len=128, hidden_dim=768]
// Normalize over last dimension
auto rms_norm_op = graph::op_t(graph::op::kind::RMSNorm);
rms_norm_op.set_attr<int64_t>(graph::op_attr::axis, -1);
rms_norm_op.set_attr<float>(graph::op_attr::epsilon, 1e-5f);
```

#### Backward Operation

| Operation              | oneDNN Graph API      |
|------------------------|-----------------------|
| op                     | RMSNormBackward       |
| input                  | src                   |
| input                  | diff_dst              |
| input                  | variance (optional)  |
| attribute              | axis                  |
| attribute              | epsilon               |
| output                 | diff_src              |

**Operation Specification:**

- **src** (required): Original input tensor (forward input).

- **diff_dst** (required): Gradient with respect to forward output.

- **variance** (optional): Cached variance, which is the mean square. If
  provided, avoids recomputation. If not provided, will be recomputed from src.

- **axis** (required): Same as forward pass. The first normalization dimension.

- **epsilon** (optional, default=1e-5): Same as forward pass.

- **diff_src** (required): Gradient with respect to input.

**Optimization Note:**

When `variance` is not provided, it will be recomputed from `src` during the
backward pass. Caching `variance` from forward to backward is a common
framework-level optimization, but not required at the Graph API level to
maintain consistency with forward operation design.

**Pros:**

1. Clear semantics with one-to-one mapping to framework operations for both
   forward and backward.
2. Simpler parameter set than LayerNorm (no mean, no beta) in both directions.
3. Distinguishes RMSNorm from LayerNorm at the API level, making fusion patterns
   explicit.
4. Easier for frameworks to integrate both forward and backward passes.
5. Separate operations avoid complex conditional logic in implementation.

**Cons:**

1. Adds two normalization operations to maintain (forward and backward).
2. Requires separate fusion patterns from LayerNorm.

### Option 2: Extend LayerNorm with a flag to indicate RMSNorm behavior

#### Forward Operation

| LayerNorm          | oneDNN Graph API   |
| ------------------ | ------------------ |
| input              | src                |
| input              | gamma (optional)   |
| input              | beta (optional)    |
| attribute          | keep_stats         |
| attribute          | begin_norm_axis    |
| attribute          | use_affine         |
| attribute          | epsilon            |
| attribute          | rms_norm           |
| output             | dst                |
| output             | mean (optional)    |
| output             | variance (optional)|

Add a new attribute `rms_norm` (default=false) to LayerNorm. When
`rms_norm=true`, the operation performs RMS normalization (no mean subtraction,
no variance computation). The `beta` parameter is ignored and `mean`/`variance`
outputs are not generated when `rms_norm=true`.

#### Backward Operation

| Operation              | oneDNN Graph API      |
|------------------------|-----------------------|
| op                     | LayerNormBackward     |
| input                  | src                   |
| input                  | diff_dst              |
| input                  | mean (optional)       |
| input                  | variance (optional)   |
| input                  | gamma (optional)      |
| attribute              | begin_norm_axis       |
| attribute              | use_affine            |
| attribute              | epsilon               |
| attribute              | rms_norm              |
| output                 | diff_src              |
| output                 | diff_gamma (optional) |
| output                 | diff_beta (optional)  |

Add a new attribute `rms_norm` (default=false) to LayerNormBackward. When
`rms_norm=true`, the operation computes gradients for RMS normalization (no
mean/variance/gamma inputs needed). The `diff_gamma` and `diff_beta` outputs are
not generated when `rms_norm=true`.

**Pros:**

1. Reuses existing LayerNorm infrastructure for both forward and backward.
2. Fewer operations to maintain (single operation covers both LayerNorm and
   RMSNorm).
3. Consistent approach across forward and backward passes.

**Cons:**

1. Overloaded semantics - LayerNorm becomes ambiguous for both forward and
   backward.
2. Framework integration is less intuitive (need to set flags in both
   directions).
3. More complex implementation with conditional logic in both forward and
   backward.
4. Mean and variance inputs become conditionally required in backward pass.
5. Different computation paths based on `rms_norm` flag add complexity to both
   operations.

## Decision

**Choose Option 1** - Add RMSNorm as dedicated forward and backward operations.

**Reasoning:**

1. **Clear Semantics**: RMSNorm has fundamentally different semantics from
   LayerNorm (no mean subtraction, no bias). Dedicated ops make this explicit
   for both forward and backward passes.

2. **Framework Adoption**: All major LLM frameworks treat RMSNorm as a distinct
   operation. PyTorch, OpenVINO, and ONNX all have dedicated RMSNorm
   implementations for both forward and backward passes.

3. **Growing Usage**: RMSNorm is becoming the standard for new LLMs (LLaMA
   family, Mistral, Gemma). This trend justifies first-class operations.

4. **Simpler API**: Fewer attributes and inputs than overloading LayerNorm.
   No optional mean/variance outputs needed in either direction.

5. **Easier Maintenance**: Separate operations avoid complex conditional logic
   in both forward and backward implementations.

## References
1. [Root Mean Square Layer Normalization paper][1]
2. [PyTorch rms_norm implementation][2]
3. [Keras RMSNormalization layer][3]
4. [ONNX RMSNormalization operator][4]
5. [OpenVINO RMS operation][5]
6. [DNNL RMSNorm primitive RFC][6]

[1]: https://arxiv.org/abs/1910.07467
[2]: https://github.com/pytorch/pytorch/blob/0b4dd08e047bda63e1e8dc78f52bcda51562caa5/aten/src/ATen/native/native_functions.yaml#L3348
[3]: https://github.com/keras-team/keras/blob/v3.9.2/keras/src/layers/normalization/rms_normalization.py
[4]: https://onnx.ai/onnx/operators/onnx__RMSNormalization.html#rmsnormalization-23
[5]: https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/internal/rms.html
[6]: ../../rfcs/20250410-rms-norm/README.md
