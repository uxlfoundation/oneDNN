# GatedMLP Driver

## Usage
``` sh
    ./benchdnn --gated_mlp [benchdnn-knobs] [gated_mlp-knobs] [gated_mlp-desc] ...
```

where *gated_mlp-knobs* are:

 - `--dt={f32 [default], ...}` -- source, weight, and destination data types.
            Interface supports broadcasting, when a single input is provided,
            e.g., `--dt=f32`, the value is applied for all tensors. Five
            individual values can be specified in SRC, W_GATE, W_UP, W_DOWN,
            DST order.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={abx [default], ...}` -- memory format of the source tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={abx [default], ...}` -- memory format of the weight tensors
            (applied to all three: gate, up, down).
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={abx [default], ...}` -- memory format of the destination tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--activation={swish [default], gelu_erf, gelu_tanh}` -- specifies the
            gated activation function applied after the gate matmul.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *gated_mlp-desc* is a problem descriptor. The canonical form is:
```
    MBxICxOC
```
Here `x` is the delimiter for the three dimensions: `MB` (batch size), `IC`
(input channels / model dimension), and `OC` (intermediate dimension). All
tensor shapes are derived from these three values:
- Source: `[MB, IC]`
- Gate weights: `[IC, OC]`
- Up weights: `[IC, OC]`
- Down weights: `[OC, IC]`
- Destination: `[MB, IC]`

When a 3D tag is specified (e.g. `--stag=abc --wtag=cab --dtag=abc`), the
driver creates "fake 3D" tensors with a unit dimension matching the layout used
by OpenVINO networks:
- Source: `[MB, 1, IC]`
- Gate/Up weights: `[1, IC, OC]`
- Down weights: `[1, OC, IC]`
- Destination: `[MB, 1, IC]`

## Quantization Attributes

The driver supports `--attr-scales` and `--attr-zero-points` for input and
weight dequantization. Supported argument names are `src` (for quantized
activations), `wei`, `wei_up`, and `wei_down` (corresponding to the gate, up,
and down projection weights).

Supported policies:
- `common` -- single scale/zero-point shared across the entire tensor.
- `per_dim_1` -- one value per output channel (dimension 1 of the weight tensor).
- `per_dim_01` -- grouped quantization along both dimensions (group size
            specified inline, e.g., `per_dim_01:f16:32x1`).

Example: run u4 quantized weights with per-channel f32 scales:
``` sh
    ./benchdnn --gated_mlp --dt=f32:u4:u4:u4:f32 \
               --attr-scales=wei:per_dim_1:f32+wei_up:per_dim_1:f32+wei_down:per_dim_1:f32 \
               64x128x256
```

Example: grouped quantization with groups along the K-dimension:
``` sh
    ./benchdnn --gated_mlp --dt=f32:u4:u4:u4:f32 \
               --attr-scales=wei:per_dim_01:f32:32x1+wei_up:per_dim_01:f32:32x1+wei_down:per_dim_01:f32:32x1 \
               --attr-zero-points=wei:per_dim_01:s4:32x1+wei_up:per_dim_01:s4:32x1+wei_down:per_dim_01:s4:32x1 \
               64x128x256
```

Example: INT8 SRC with per-tensor SRC scale and quantized weights:
``` sh
    ./benchdnn --gated_mlp --dt=s8:u4:u4:u4:f16 \
               --attr-scales=src:common:f16+wei:per_dim_01:f16:128x1+wei_up:per_dim_01:f16:128x1+wei_down:per_dim_01:f16:128x1 \
               64x256x4864
```

## Post-ops

The driver supports post-ops applied to the final (down) matmul output.
Supported kinds include `sum`, `eltwise`, and `binary`.

Example: binary add post-op:
``` sh
    ./benchdnn --gated_mlp --dt=f16:u4:u4:u4:f16 \
               --attr-scales=wei:per_dim_01:f16:128x1+wei_up:per_dim_01:f16:128x1+wei_down:per_dim_01:f16:128x1 \
               --attr-post-ops=binary_add:f16:2:ab \
               64x128x4864
```

## Essence of Testing

The GatedMLP operation computes
`DST = (activation(SRC • W_gate) ⊙ (SRC • W_up)) • W_down`,
where `•` denotes matrix multiplication and `⊙` denotes element-wise
(Hadamard) product. The reference executes each step independently using f32
matmul and element-wise primitives on the CPU. The driver compares the fused
primitive output against this stepwise reference.

## Examples

Run the default validation set of GatedMLP using `inputs/gated_mlp/shapes_basic`
file:
``` sh
    ./benchdnn --gated_mlp --batch=inputs/gated_mlp/shapes_basic
```

Run f16 GatedMLP with gelu_erf activation on a small shape:
``` sh
    ./benchdnn --gated_mlp --dt=f16 --activation=gelu_erf 64x128x256
```

Run GatedMLP performance benchmark on GPU with an LLM shape:
``` sh
    ./benchdnn --mode=f --gated_mlp --engine=gpu --dt=f16 \
               --activation=swish 1024x896x4864
```

More examples with different driver options can be found at
inputs/gated_mlp/test_\*.
