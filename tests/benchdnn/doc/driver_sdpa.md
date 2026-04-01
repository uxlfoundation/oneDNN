# SDPA Driver

**Note:** This driver tests internal SDPA primitive functionality that is not
part of the public oneDNN API.

## Usage
``` sh
    ./benchdnn --sdpa [benchdnn-knobs] [sdpa-knobs] [sdpa-desc] ...
```

where *sdpa-knobs* are:

 - `--dir={FWD_I [default], FWD_D, BWD_D}` -- propagation direction.
            `FWD_I` is forward inference, `FWD_D` is forward training (needed
            for workspace output used by backward), `BWD_D` is backward data
            which computes gradients for Q, K, and V.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32:f32:f32:f32 [default], ...}` -- source (Q, K, V) and destination
            data types. Interface supports broadcasting, when a single input is
            provided, e.g., `--dt=f32`, the value is applied for all tensors.
            Refer to [data types](knobs_dt.md) for details.
 - `--qtag={abx [default], ...}` -- memory format of the queries tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--ktag={abx [default], ...}` -- memory format of the keys tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--vtag={abx [default], ...}` -- memory format of the values tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={abx [default], ...}` -- memory format of the destination tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--mask={none [default], buffer, buffer_1d, buffer_2d, causal_top_left, causal_bottom_right}`
            -- specifies the attention mask type.
            `none` uses no mask, `buffer` provides an explicit mask tensor
            of shape `[B, H, S_q, S_kv]` (added element-wise to attention
            scores before softmax), `buffer_1d` provides a broadcast mask of
            shape `[1, 1, 1, S_kv]` (same mask for all batches, heads, and
            queries), `buffer_2d` provides a broadcast mask of shape
            `[1, 1, S_q, S_kv]` (same mask for all batches and heads),
            `causal_top_left` and `causal_bottom_right` apply a causal mask
            aligned to the respective corner.
 - `--mdt={f32 [default], f16, bf16}` -- data type of the attention mask
            buffer. Only used when `--mask` is one of `buffer`,
            `buffer_1d`, or `buffer_2d`.
 - `--scale={library [default], mul, div}` -- specifies how the scale
            value is passed to the primitive. Attention scores are always
            scaled; the knob controls the API path used. `library` passes
            `1/sqrt(head_size)` as a multiplicative scale, `mul` does the same
            explicitly via the `invert_scale=false` API path, `div` passes
            `sqrt(head_size)` via the `invert_scale=true` API path. All three
            produce the same mathematical result `scores / sqrt(head_size)`.
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
A typical 4D tensor layout is `BxHxSxD` where `B` is batch, `H` is heads,
`S` is sequence length, and `D` is head size.

## Essence of Testing

The SDPA operation computes
`DST = softmax((Q * K^T) * scale [+ mask]) * V`.
The reference executes each step independently using f32 matmul and softmax
primitives on the CPU. The driver compares the fused primitive output against
this stepwise reference.

For backward (`--dir=BWD_D`), the driver creates a forward-training primitive
(to produce workspace) followed by a backward primitive. The backward reference
recomputes the forward intermediates (softmax probabilities), then derives
gradients via softmax backward and transposed matmuls:
`dS = softmax_bwd(S2, dO * V^T)`, `dQ = scale(dS) * K^T`, `dK = Q^T * scale(dS)`,
`dV = S2^T * dO`, with GQA reduction where applicable.

## Examples

Run the default validation set of SDPA using `inputs/sdpa/shapes_basic` file:
``` sh
    ./benchdnn --sdpa --batch=inputs/sdpa/shapes_basic
```

Run f16 SDPA with a causal mask on a transformer-like shape:
``` sh
    ./benchdnn --sdpa --dt=f16 --mask=causal_top_left \
               1x12x128x64:1x12x64x128:1x12x128x64
```

Run SDPA with explicit scale (division mode) and mixed data types:
``` sh
    ./benchdnn --sdpa --dt=f16:f16:f16:f32 --scale=div \
               2x8x32x64:2x8x64x32:2x8x32x64
```

Run SDPA performance benchmark on GPU:
``` sh
    ./benchdnn --mode=f --sdpa --engine=gpu --dt=f16 \
               --mask=causal_bottom_right \
               1x32x2048x128:1x32x128x2048:1x32x2048x128
```

Run SDPA backward (training) on GPU:
``` sh
    ./benchdnn --sdpa --engine=gpu --dir=BWD_D --dt=f16 \
               1x12x128x64:1x12x64x128:1x12x128x64
```

More examples with different driver options can be found at
inputs/sdpa/test_\*.
