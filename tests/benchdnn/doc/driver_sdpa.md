# SDPA Driver

## Usage
``` sh
    ./benchdnn --sdpa [benchdnn-knobs] [sdpa-knobs] [sdpa-desc] ...
```

where *sdpa-knobs* are:

 - `--dt={f32 [default], ...}` -- source (Q, K, V) and destination data types.
            Interface supports broadcasting, when a single input is provided,
            e.g., `--dt=f32`, the value is applied for all tensors. To specify
            individual data types, use colon-separated format:
            `--dt=Q_DT:K_DT:V_DT:DST_DT`.
            Refer to [data types](knobs_dt.md) for details.
 - `--qtag={abx [default], ...}` -- memory format of the queries tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--ktag={abx [default], ...}` -- memory format of the keys tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--vtag={abx [default], ...}` -- memory format of the values tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={abx [default], ...}` -- memory format of the destination tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--mask_type={none [default], buffer, causal_top_left, causal_bottom_right}`
            -- specifies the attention mask type.
            `none` uses no mask, `buffer` provides an explicit mask tensor,
            `causal_top_left` and `causal_bottom_right` apply a causal mask
            aligned to the respective corner.
 - `--scale_type={none [default], mul, div}` -- specifies how the attention
            scores are scaled. `none` uses the default `1/sqrt(head_size)`,
            `mul` multiplies scores by the scale value, `div` divides by it.
 - `--kv_head_number={0 [default], INT}` -- number of KV heads for grouped
            query attention (GQA) or multi-query attention (MQA). `0` means
            standard multi-head attention where KV heads equal Q heads.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.
 - Any attributes options. Refer to [attributes](knobs_attr.md) for details.

and *sdpa-desc* is a problem descriptor. The canonical form is:
```
    Q_DIMS:K_DIMS:V_DIMS
```
Here `x` is the delimiter for dimensions within a tensor and `:` is the
delimiter for tensors, provided in the order queries (Q), keys (K), and
values (V). The destination dimensions are derived automatically.

For a 4D problem (batch x heads x seq_len x head_size):
```
    BxHxSxD:BxHxDxT:BxHxTxV
```
where `B` is batch, `H` is heads, `S` is query sequence length, `D` is head
size, `T` is key sequence length, and `V` is value dimension.

## Essence of Testing

The SDPA (Scaled Dot-Product Attention) operation computes:
```
    DST = softmax((Q * K^T) * scale [+ mask]) * V
```

The driver validates the fused SDPA primitive by comparing its output against
a reference implementation that executes each step independently in f32 on the
CPU: matrix multiplication (Q * K^T), scaling, optional masking, softmax, and
the final matrix multiplication with V.

## Examples

Run the default validation set of SDPA using `inputs/sdpa/shapes_basic` file:
``` sh
    ./benchdnn --sdpa --batch=inputs/sdpa/shapes_basic
```

Run f16 SDPA with a causal mask on a transformer-like shape:
``` sh
    ./benchdnn --sdpa --dt=f16 --mask_type=causal_top_left \
               1x12x128x64:1x12x64x128:1x12x128x64
```

Run SDPA with explicit scale (division mode) and mixed data types:
``` sh
    ./benchdnn --sdpa --dt=f16:f16:f16:f32 --scale_type=div \
               2x8x32x64:2x8x64x32:2x8x32x64
```

Run SDPA performance benchmark on GPU:
``` sh
    ./benchdnn --mode=f --sdpa --engine=gpu --dt=f16 \
               --mask_type=causal_bottom_right \
               1x32x2048x128:1x32x128x2048:1x32x2048x128
```

More examples with different driver options can be found at
inputs/sdpa/test_\*.
