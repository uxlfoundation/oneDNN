# Support Dropout Operation in Graph API

## Background

Dropout [[#1]][1] is a regularization technique widely adopted in deep learning
models, including transformers and convolutional neural networks. It helps
prevent overfitting by randomly zeroing out elements of the input tensor during
training and scaling the outputs to maintain the expected sum. During inference,
Dropout is typically bypassed, and the input is passed through unchanged.

Mathematically, Dropout is defined as follows:

**Forward (Training)**

Given input tensor $X$ and dropout probability $p$:

For each element $i$:

> $$
> M_i \sim \text{Bernoulli}(1-p)
> $$
> $$
> Y_i = \frac{X_i \cdot M_i}{1-p}
> $$

where $M$ is the mask tensor (same shape as $X$), and $Y$ is the output tensor.

**Forward (Inference)**

Dropout is typically bypassed during inference:

> $$
> Y_i = X_i
> $$

**Backward**

Given gradient tensor $dY$, mask tensor $M$, and dropout probability $p$:

> $$
> dX_i = \frac{dY_i \cdot M_i}{1-p}
> $$

where $dX$ is the gradient with respect to the input.

## Adoption in Frameworks and Libraries

**PyTorch**

troch.nn.Dropout [[#2]][2] takes `p` (probability) and `inplace` as attributes,
with one input and one output. Internally, the random number generator used to
generate the dropout mask differs by device:

- CPU: Uses torch.Generator backed by the C++ Mersenne Twister engine [[#3]][3],
  which consumes a 64-bit seed.
- CUDA / XPU: Uses a 64-bit Philox RNG state (seed + offset) obtained from the
  device generator [[#4]][4][[#5]][5]. The offset is advanced after each kernel
  launch to carve out disjoint subsequences of random numbers.

**TensorFlow**

tf.keras.layers.Dropout [[#6]][6] takes `rate` (probability), an optional `noise_shape`
(shape for randomly generated keep/drop flags) and an optional seed as
attributes, with one input and one output. Internally, the random number
generator produces a 64-bit pair (key, counter) from the seed and passes
them to a selected RNG algorithm to generate the mask.
Supported algorithms [[#7]][7] include Philox, ThreeFry, or AUTO_SELECT
(which lets TensorFlow choose the best algorithm for the current device;
AUTO_SELECT may pick an algorithm other than Philox or ThreeFry).

**ONNX**

ONNX's Dropout operation [[#8]][8] takes `data`, `ratio` (probability) and `training_mode`
as inputs, and `seed` as an attribute. Outputs include the dropped tensor and
the dropout mask. ONNX does not specify the algorithm used to generate random numbers.

**cuDNN**

cuDNN Graph library does not provide a dedicated Dropout operation. Instead, it
introduces an RNG operation, which can be used to implement Dropout with two
pointwise Multiply operations.
The RNG operation takes `seed`, `offset` as inputs, supports various
distributions (Bernoulli, Uniform, Normal) by setting attributes, and outputs
the generated tensor. The RNG is based on the Philox algorithm.

**oneDNN**

In oneDNN primitive, Dropout is implemented as a binary post-op applied to
output tensor before other post-ops. It uses the Philox algorithm and takes
`seed` (s32) and `probability` (f32) as inputs, and outputs a `mask` (u8).

## Proposal

Based on the investigation above, we propose to define the Dropout operation in
the oneDNN Graph API as follows:

**Operation Kind:**

- `Dropout` (C++), `dnnl_graph_op_dropout` (C)

**Inputs:**

- `src`: input tensor (`f32` / `bf16` / `f16`)
- `seed`: random seed (`uint64_t`, single value)
- `offset`: random offset (`uint64_t`, single value)
- `p`: dropout probability (`f32`, single value)

**Outputs:**

- `dst`: output tensor (`f32` / `bf16` / `f16`)
- `mask`: mask tensor (`u8`, optional)

**Notes:**

- Dropout will use the Philox algorithm for random number generation.
- Data types for `src`, `p`, `dst` and `mask` are consistent with the
  current oneDNN primitive design.
- The `mask` output is optional, allowing flexibility for scenarios where
  the mask is not needed.
- The `offset` input is introduced to ensure reproducibility and correct
  partitioning across devices and threads, following best practices from PyTorch
  and cuDNN.
- Data types for `seed` and `offset` are aligned to PyTorch to ensure
  compatibility and support large models.
- A dedicated DropoutBackward operation is not defined, as the backward
  computation shares the same mathematical formula as the forward pass.

**Example Usage:**

```cpp

using namespace dnnl::graph;

graph g = graph(engine::kind::gpu);

logical_tensor src = logical_tensor(ID_SRC, data_type::f16, {256, 1000});
logical_tensor seed = logical_tensor(ID_SEED, data_type::u64, {1});
logical_tensor offset = logical_tensor(ID_SEED, data_type::u64, {1});
logical_tensor p = logical_tensor(ID_SEED, data_type::f32, {1});
logical_tensor dst = logical_tensor(ID_DST, data_type::f16, {256, 1000});
logical_tensor mask = logical_tensor(ID_MASK, data_type::u8, {256, 1000});

op dropout = op(ID_DROPOUT, op::kind::Dropout, "dropout");
dropout.add_input(src);
dropout.add_input(seed);
dropout.add_input(offset);
dropout.add_input(p);
dropout.add_output(dst);
dropout.add_output(mask);

g.add_op(dropout);
g.finalize();

```

**Open Question:**

- The probability `p` is currently defined as an input to align with the oneDNN
  primitive design and to facilitate integration between the graph API and
  primitive API. Defining `p` as an attribute is also possible and may be
  considered as an open.
- Treating `seed`, `offset`, and `p` as host scalars is recommended, as this
  approach optimizes performance and usability for single-value inputs.

## References

1. Improving neural networks by preventing co-adaptation of feature detectors, https://arxiv.org/abs/1207.0580
2. Dropout operation in PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
3. CPU Dropout random number generator in PyTorch, https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp#L94
4. CUDA Dropout random number generator in PyTorch,https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Dropout.cu#L403
5. XPU Dropout random number generator in PyTorch, https://github.com/intel/torch-xpu-ops/blob/main/src/ATen/native/xpu/sycl/Dropout.cpp#L399
6. Dropout operation in Tensorflow, https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
7. Dropout random number generator algorithms in Tensorflow, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/random_ops_util.py#L39
8. Dropout operation in ONNX, https://onnx.ai/onnx/operators/onnx__Dropout.html

[1]: https://arxiv.org/abs/1207.0580
[2]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
[3]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp#L94
[4]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Dropout.cu#L403
[5]: https://github.com/intel/torch-xpu-ops/blob/main/src/ATen/native/xpu/sycl/Dropout.cpp#L399
[6]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
[7]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/random_ops_util.py#L39
[8]: https://onnx.ai/onnx/operators/onnx__Dropout.html