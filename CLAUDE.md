# CLAUDE.md — GEMM → Matmul migration notes

Durable knowledge for triaging JIT GEMM-as-matmul issues. Read once per session. Update the moment a non-obvious gotcha is discovered. **Not** a status doc — see `STATE.md` for that.

Phase 5 is done: `gemm_desc_t` / `gemm_pd_t` / `gemm_arg` / `gpu_gemm_list` / `gemm::primitive_t` / `intel::matmul::gemm_t` / `create_gemm_pd` / `rekey_attr_for_gemm` are all deleted. JIT GEMM impls (`gen_t`, `xe_hp_systolic_t`, `ref_t`, `with_post_ops_t`, `conv_t`) inherit `jit_gemm_pd_t` and register in `gpu_matmul_list`. IP/RNN/SDPA callers construct `matmul_desc_t` directly. The only `common/gemm_*` survivor is `gemm_types.hpp`, kept solely for the BLAS-style enums (`transpose_t`, `offsetc_t`, `sum_ab_t`).

Public C-API (`dnnl_sgemm` / `dnnl_gemm_*`) and brgemm (`brgemm_desc_t`) are out of scope — untouched.

---

## Core doctrine: init in matmul-natural + one FULL transpose

This is the load-bearing decision for `init_GEMMProblem`. Read it before touching anything in `jit_gemm_pd.cpp` or `problem_utils.cpp`.

**The pattern:** Build the entire `GEMMProblem` in matmul-natural convention (A=SRC, B=WEIGHTS, C=DST). At the end of init, if `swap_ab_` is set, call `problem.transpose()` **exactly once** — the **full math transpose** (C^T = B^T A^T). No undos. No resets. No post-transpose fix-ups. No `transposeAuxMatrices` parameter (the prior "half-transpose" mode was wrong and has been deleted).

**Goal of the doctrine:** for the same matmul shape/attrs/storage, worktree must produce a **field-for-field identical** `GEMMProblem` to base. The kernel and the catalog see *only* the `GEMMProblem`; there is no second channel. So when base picks `swap_ab_=0` (its baked-in BLAS view) and worktree picks `swap_ab_=1` (does the matmul→BLAS conversion explicitly via `transpose()`), the two paths must converge on the same `GEMMProblem`. Any divergence is a translation bug — either in `init_GEMMProblem`, in `GEMMProblem::transpose()`, or in `transfer_post_ops`.

### Layout-tag convention (MUST FOLLOW)

`MatrixLayout::T` = **row-major** (stride-1 along the N/column axis).
`MatrixLayout::N` = **column-major** (stride-1 along the M/row axis).

Pre-init A/B/C from matmul md storage: row-major → T, col-major → N. The full `transpose()` then flips T↔N for all matrices, landing the BLAS col-major view the catalog expects.

```cpp
// CORRECT — all three follow the same matmul-natural convention:
problem.A.layout = tr_a_matmul ? MatrixLayout::N : MatrixLayout::T;  // matmul SRC
problem.B.layout = tr_b_matmul ? MatrixLayout::N : MatrixLayout::T;  // matmul WEIGHTS
problem.C.layout = (get_trans(dst_md_) == transpose::trans)
        ? MatrixLayout::N : MatrixLayout::T;                          // matmul DST
// binary in transfer_post_ops — gated on is_multi_col, NOT is_multi_row.
// See gotcha #21.
bool layout_trans = is_multi_col && !src_rmd.inner_dim.is_innermost();
atype.layout = layout_trans ? MatrixLayout::N : MatrixLayout::T;
```

### `transfer_post_ops` broadcast_mask bit convention (MUST FOLLOW)

`relative_md_t::broadcast_mask` is built loop-from-innermost: bit 0 = innermost (last) dim, bit 1 = second-to-last, etc. So for **2D matmul**:
- **bit 0 = matmul-N axis** (the trailing/col dim of the matmul md).
- **bit 1 = matmul-M axis** (the leading/row dim).

The matmul-natural pre-transpose assignments for `binaryRow` / `binaryCol`:

```cpp
// binaryRow = "varies along matmul-M (row axis)" = bit 1 NOT set.
// binaryCol = "varies along matmul-N (col axis)" = bit 0 NOT set.
bool is_multi_row = (src_rmd.broadcast_mask & 2) == 0;
bool is_multi_col = (src_rmd.broadcast_mask & 1) == 0;
```

Reading `& 1` for row / `& 2` for col produces **kernel-canonical** (already-swapped) values; combined with the single full `transpose()` this double-flips. See gotcha #21.

### Rules going forward

1. **Pre-transpose init is matmul-side-keyed.** Every problem field is set as if `swap_ab_=false`. AO/AScale/Ag/sumA/Tao/Ta_scale/Tag/aqGroup{M,K}/aoPtrDims/asPtrDims come from matmul-SRC (kA). BO/BScale/Bg/sumB/Tbo/Tb_scale/Tbg/bqGroup{N,K}/boPtrDims/bsPtrDims come from matmul-WEIGHTS (kB). CO/cqGroup{M,N}/Tco/Tc_scale come from matmul-DST. No `swap_ab_` ternaries in the pre-transpose block.
2. **One single `if (swap_ab_) problem.transpose();` at the end of `init_GEMMProblem`.** Full transpose, no parameter. No `std::swap` patches, no `problem.X.transpose()` undos, no MatrixLayout resets, no per-field mirror branches for skinny-N or any other case. **Two documented exemptions**:
   - (a) **Skinny-N WEIGHTS layout/alignment fixup** (review.md #5): runs post-transpose because `transpose()` doesn't run under `swap_ab_=false` and the pre-transpose block can't host the swap-aware condition. Look for the `DOCTRINE EXEMPTION (review.md #5)` marker.
   - (b) **CO.layout reset to `N` when CO is unused under swap_ab** (gotcha #23): base's swap_ab branch does NOT touch CO; worktree's full `transpose()` unconditionally flips `CO.layout`. When `cOffset == None && !sumA && !sumB` the field is part of the catalog hash but otherwise unused, so we restore the default after `transpose()`. Block lives immediately after `if (swap_ab_) problem.transpose();`.

   Do not add further exceptions without explicit reasoning.
3. **Any field that needs to swap under `swap_ab_` MUST be swapped by `GEMMProblem::transpose()`.** If you discover a missed field, fix `problem_utils.cpp` — don't patch the caller. The current `transpose()` already covers: A/B, AO/BO, A_scale/B_scale, Ag/Bg, types (Ta/Tb/Ta_ext/Tb_ext/Tao/Tbo/Ta_scale/Tb_scale/Tag/Tbg), aOffset/bOffset, ao/bo/asPtrDims, aqGroupM/bqGroupN, aqGroupK/bqGroupK, cqGroupM/cqGroupN, forceGroupSumsA/B, sumA/sumB, postOps, and per-binary[].
4. **Outside `init_GEMMProblem`, the source of truth is the stowed `GEMMProblem`, not `desc()`/`attr()`/pd shadow fields.**

### `GEMMProblem` is the SOLE post-init store

User doctrine, verbatim:
> "the idea is to do matmul desc/pd -> gemm problem conversion and stop accessing desc/pd after this point. gemm problem can apply transpose for some cases so it's the source of truth for all jit gemms most of the time! I don't want to make pd a/b/c getters to be swap ab aware."
>
> "no, it doesn't look right — attribute must not be used directly, all accesses must go via transformed gemmproblem"
>
> "to clarify - this has to be stored in gemmproblem! no intermediate fields!"

Practical rules:
- **No pd-side intermediates shadowing problem state.** Do NOT keep `a_group_m_`, `b_group_k_`, `src_group_m_`, `wei_group_k_`, or any rename thereof. Same for `{a,b,c}_scale_md_`, `{a,b,c}_zp_md_`, `{a,b}_gs_md_`, `cmask_{a,b,c}_`, `with_sum_`, `sum_at_begin_`, `bias_via_binary_`, `wei_decomp_`, `quant_enabled_`, `with_sround_`, `with_mx_scale_`, `lda_`, `ldb_`, etc. If `GEMMProblem` already has a slot — write there. If a piece of post-init state has no slot, **extend `GEMMProblem`** rather than adding a pd shadow.
- **Init writes straight to the problem.** `init_GEMMProblem` reads `attr().scales_.get(DNNL_ARG_SRC/WEIGHTS/DST)` and writes directly into `problem.aqGroupM`, `problem.bqGroupK`, etc. Quant mds populate `problem.A_scale`/`B_scale`/`AO`/`BO` directly. Post-op binary metadata writes to `problem.binary[]`. Then `if (swap_ab_) problem.transpose()` once at the end maps SRC↔WEIGHTS / `aq*↔bq*` / `Ag↔Bg` correctly.
- **No `attr()` access post-init.** Attr is read exclusively inside `init_GEMMProblem` to populate problem fields; from then on the problem carries the transformed quant state.
- **No swap-aware pd getters consumed by kernel code.** Pd swap-aware helpers (`m()/n()/k()/lda()/...`) exist only for init-time use while building the problem. Kernel code (`gen_t::execute`, `launch_nocopy`, `with_post_ops_t::execute`, `ref_t::execute`, `xe_hp_systolic_t::execute`) reads from the stowed `GEMMProblem`. For sizes/strides not on `GEMMProblem`, expose `problem_m()/n()/k()/lda()/...` helpers — the *explicit* post-init boundary, named to make it impossible to confuse with pd-side state.
- **`attr_` itself stays in matmul (user-facing) convention.** `apply_swap_ab` MUST NOT rekey scales/zp/precomputed_reductions via `swap_entries(kA, kB)` and MUST NOT run `swap_entry_mn_axes`. Framework binds buffers by `attr()` keys, so leaving `attr_` un-mutated keeps `DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS` resolving to non-null storage at exec.

### Diagnostic procedure — when a case fails

Always: `ONEDNN_DUMP_GEMM_PROBLEM=W` in worktree + same with `=B` in `../base`, diff. Any field-level divergence is a translation bug to identify and fix. Do NOT conclude "this is a catalog mismatch / new entry needed / accept the regression" without first eliminating field-level divergences.

When examining divergences, separate them into:
- **Real**: fields the kernel actually reads for this config (changes test outcome).
- **Cosmetic**: fields the kernel ignores for this config (only affects catalog hash; may or may not matter).

Cosmetic divergences sometimes resolve on their own once a real field is fixed.

---

## Design notes (still load-bearing)

### JIT GEMM is column-major; matmul md is row-major
JIT kernels are column-major. The matmul desc stays in row-major matmul convention (SRC = M×K, WEIGHTS = K×N, DST = M×N). `jit_gemm_pd_t::init()` decides internally whether to swap A/B (`swap_ab_`); the swap is applied via `problem.transpose()` once at the end of `init_GEMMProblem`. Quant entries (scales, zero_points) are stored in matmul attrs keyed by `DNNL_ARG_SRC/WEIGHTS`; the pd reads them and the `transpose()` rotates them into the kernel-canonical view.

Swap rules per impl:
- `jit_xe_hp_systolic_t` and `ref_t` ALWAYS swap (kernel cannot do row-major).
- `gen_t` swaps unless: N==1, C is row-major, weights-only compression off (`jit.hpp:66-77`).
- The swap is a hardware/codegen requirement, not a perf optimization.

### `transc()` does not exist on matmul — infer from dst_md strides
Matmul has no `transc` flag; the dst memory_desc carries strides that directly express layout. `gen_t`'s skinny-N un-swap heuristic and any other col-major-DST check goes through `get_trans(dst_md_)` (port of the BLAS-style trans inference from strides). Don't reintroduce a desc-flag-style `transc()`.

### sum_ab via `matmul_reduce_kind_t`
`matmul_reduce_kind_t` (internal-only enum in `src/common/c_types_map.hpp`) carries the sum-A-row / sum-B-col direction. Reduce shape comes from `reduce_desc` (`memory_desc_t`); reduce data type comes from `reduce_desc.data_type`. IP BWD_W picks `wei_tr() ? sum_b_col : sum_a_row`. No new field on `matmul_desc_t`.

### Exec-time arg routing
`exec_args_from_matmul(ctx, swap_ab)` in `src/gpu/intel/gemm/exec_types.hpp` builds an `exec_args_t` from a matmul `exec_ctx_t`. Matmul-keyed `DNNL_ARG_SRC/WEIGHTS/DST/BIAS` map to `args.a/b/c/bias`; quant/dropout args carry their `DNNL_ARG_ATTR_*` keys; `sum_ab` reads `DNNL_ARG_REDUCE`. `args.route_by_swap_ab(swap_ab)` does the A↔B flip. Called from each JIT impl's `execute(impl::exec_ctx_t)` entry.

---

## Gotchas — read before editing

### 2. JIT GEMM kernels are column-major
- `jit_xe_hp_systolic` and `ref` ALWAYS swap (kernel cannot do row-major).
- `gen_t` (the main JIT GEMM) swaps unless: N==1, C is row-major, weights-only compression off (`jit.hpp:66-77`).
- The swap is not a perf optimization — it's a hardware/codegen requirement.

### 3. Post-ops binary operands carry M/N indices
When the pd decides to swap, post-op operand descriptors must have their M/N axes swapped consistently — this runs inside `jit_gemm_pd_t::init_post_ops` after `swap_ab_` is decided. Silent breakage at runtime if missed.

### 4. sum_ab buffer is wired via `DNNL_ARG_REDUCE` (was `DNNL_ARG_DIFF_BIAS` pre-migration)
IP BWD_W needs a row/col sum of the gemm output to compute diff_bias. Post-migration the sum_ab output buffer comes from `CTX_OUT_STORAGE(DNNL_ARG_REDUCE)` via the matmul reduce slot.

### 10. `get_trans` short-circuit silently mis-classifies degenerate matrices
`jit_gemm_pd_t::get_trans` originally had `inner_n != 1 && strides[ndims-1] != 1 → TRANS`. For a `[m, 1]` matrix with strides `[1, m]` (single-column col-major), `inner_n == 1` short-circuits to NOTRANS even though the n-axis carries the larger stride. `get_ld` on the result then returns `strides[m_axis] = 1` instead of `m`. This bit RNN BWD_DW for M=1 (matmul DST `[1, N]` strides `[N, 1]` mn-transposed via `ld_kernel_view` → `[N, 1]` strides `[1, N]` → NOTRANS short-circuit → ldc=1 → corrupt output). Fixed by dropping the `inner_n != 1` gate: `strides[ndims-1] != 1` alone governs trans-ness.

### 11. Don't `mn-transpose` matmul mds to compute kernel-view ld/trans
The pre-fix `ld_kernel_view`/`trans_kernel_view` mn-transposed `matmul_a_md()/matmul_b_md()/dst_md_` under `swap_ab_` and then ran `get_ld`/`get_trans`; `compute_transa/b` additionally NEGATEd the result. For non-degenerate matrices the round-trip is identity, but for K=1 / N=1 / M=1 single-axis cases the mn-transposed md hits a different `get_ld`/`get_trans` branch and produces a different value. The fix: drop the mn-transpose entirely. `compute_lda = get_ld(matmul_a_md())`, `compute_transa = get_trans(matmul_a_md())`, no NEGATE. Doctrine: the matmul md is the source of truth; the post-init `GEMMProblem` carries any swap-induced rotation via `problem.transpose()`.

### 12. Matmul-natural skinny-N needs a `transb_/ldb_` flip in `gen_t::init`
Base's `jit.hpp:107` had `if (swap_ab_ && !transa_ && m==1) { transa_=true; lda_=k(); }`. Base's BLAS-m mapped to matmul-N, so the flip fired when matmul-N==1. In matmul-natural convention (`swap_ab_=false`, N==1, want_un_swap=true), the analogous flip is on the kernel-B side (kernel-B = matmul-WEIGHTS, K-axis): `transb_=true; ldb_=k();`. Mirror this branch immediately before the existing skinny-row/col pad, otherwise the pad-lda branch over-pads a 1-element K stride and corrupts output for integer N=1 cases (`s8:s8:f32 ... 2048x256:256x1` etc.).

### 14. `with_post_ops_t::pd_t::query` must route ALL `exec_arg_md` to the outer
Initial fix used an "exemption list" (BIAS, POST_OP_BASE+, DROPOUT_*) and forwarded everything else to the inner. Wrong: the inner pd is constructed from `desc_copy` with `dst_desc.data_type = intermediate_c_type` (f32) and `attributes_without_po` (post_ops_/scales_/zero_points_/dropout_ all stripped). So forwarding `DNNL_ARG_DST` to the inner returned a md with type **f32 instead of the user's s8** — the framework then allocated a 4× buffer and read the kernel's s8 output as f32, producing values like `1.56e-33`. Same hazard for scales/zp queries: the inner has them stripped.

Correct rule: `query(exec_arg_md, ...)` ALWAYS goes to the outer `gpu_matmul_pd_t::query`. The outer is the user-facing pd and owns the canonical view of every user-bound arg (SRC, WEIGHTS, DST, BIAS, scales, zps, post-op binaries, dropout). The inner exists only to run the scratchpad-backed gemm; its mds/attr are internal. Non-exec queries (scratchpad_md_id, nested-prim queries, etc.) still fall through to the inner. Affects every dst-type-conversion case (f16:f16:s8, bf16:bf16:s8, …), per_oc wei_scale, bias-mask-3/6, IP per_oc, and anything else that lands in `ocl:with_po:any`.

### 16. `gen_t::init` must force apply_swap_ab for K==1
The `(k()==1 && !trans_a()) || (m_==1 && trans_a())` pad branch sets `lda_ = rnd_up(lda_, 16)`. Intent: align leading dim for vectorized block reads — safe only when the matrix the kernel reads at that lda has 1 row (so lda is unused per row). Base routed kernel-A to matmul-WEIGHTS via `apply_swap_ab`; for K==1 matmul-WEIGHTS has gemm-M==1, pad benign. Worktree's matmul-natural un-swap path puts kernel-A on matmul-SRC (full M dimension); pad inflates lda from 1 to 16, kernel over-reads matmul-SRC by 16×, dst is garbage. Repro: `--matmul --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab 100x1:1x1`. Fix: `want_un_swap &= (k() > 1)` in `gen_t::init`, before `apply_swap_ab()`. For K==1 always swap, mirroring base.

### 17. Column-major DST IS a `want_un_swap` trigger
Under `swap_ab_=true` worktree converts matmul-natural → BLAS via `problem.transpose()`. That puts kernel-A = matmul-WEIGHTS, kernel-B = matmul-SRC. The kernel then computes `C_kern = (matmul-A · matmul-B)^T` col-major, which user reads as **row-major** matmul-DST. Correct for `dst:ab` but wrong for `dst:ba` (user wants col-major output, not row-major). For col-major DST, the kernel should compute `matmul-A · matmul-B` directly with kernel-A = matmul-SRC, kernel-B = matmul-WEIGHTS — matmul-natural orientation, `swap_ab_=false`.

Base's apparent contradiction (base set `swap_ab_=true` for col-major DST and still passed) is a different doctrine: base's `swap_ab_=true` path did `std::swap(a_quant, b_quant)` at the top of init_GEMMProblem plus a per-aux `.transpose()` block at the end — effectively inverting the matmul→BLAS conversion baked into base's gemm_desc accessors. Net effect: base's `swap_ab_=1 + col-major DST` produces kernel-A = matmul-SRC, same as worktree's `swap_ab_=0`. Worktree achieves the same end state more directly: just un-swap.

Fix: `want_un_swap |= (user_c_trans == transpose::trans)` in `gen_t::init` (`jit.hpp:79-80`).

Diagnostic note: always use `ONEDNN_DUMP_GEMM_PROBLEM` to verify GEMMProblem equality with base BEFORE concluding a case needs catalog work — the catalog is usually fine; the translation logic is wrong.

### 18. `aScale2D()` / `bScale2D()` are kernel-keyed; `a_scales_2d()` / `b_scales_2d()` are matmul-keyed
`gen_t::execute` was gating `a_scales = &GEMM_ARG_STORAGE(a_scales)` on `pd()->a_scales_2d()` (reads `attr().scales_.get(kA=SRC).ndims()`). Under `swap_ab_=true` with WEIGHTS-only 2D scales, `pd()->a_scales_2d()` is false (SRC has no scales) but `problem->aScale2D()` is true (after `problem.transpose()` the WEIGHTS scale rotated to slot A). `a_scales` stayed null while launch_nocopy dereferenced it → segfault on `f16:f4_e2m1:f16 wei:per_ocic:f16:128x1` (weight-decompression). Fix: gate by `problem.aScale2D()` / `problem.bScale2D()`. Mirror to any other matmul-vs-kernel-keyed slot guard you spot.

### 19. xe_hp_systolic packed `lda/ldb/ldc` index map: base used BLAS-swapped descs, worktree uses matmul-keyed mds
Base's `ld{a,b,c}_packed` read from `desc()->{a_desc,b_desc,c_desc}` which were the BLAS-swapped views (a_desc = SRC, b_desc = WEIGHTS) with reversed dim ordering vs matmul. Worktree's `xe_hp_systolic_t::pd_t::ld{a,b,c}_packed` migrated to read from `{weights,src,dst}_md_` directly but reused base's stride-index formula verbatim — wrong axis. Correct mapping (xe_hp_systolic always swaps):
- `lda_packed` ← `weights_md_` (K×N), kernel-M = matmul-N → `strides[ndims-1] = with_batch() ? 2 : 1`.
- `ldb_packed` ← `src_md_` (M×K), kernel-N = matmul-M → `strides[ndims-2] = with_batch() ? 1 : 0`.
- `ldc_packed` ← `dst_md_` (M×N), kernel-M = matmul-M → `strides[ndims-2] = with_batch() ? 1 : 0`.
Surfaces as `ldc=32` instead of `544` on s8:s8:f16 dst:AB32a32b → ~99% errors. When migrating any "packed leading dim" / "stride-by-axis" helper, write the index formula from the matmul-side dim layout (SRC = M,K · WEIGHTS = K,N · DST = M,N), not from base's swapped view.

### 21. `transfer_post_ops` broadcast_mask bit indexing — bit 0 is the INNERMOST dim, not matmul-M
`relative_md_t::broadcast_mask` (`src/gpu/intel/post_ops.hpp:240`) is built loop-from-innermost: `mask_bit` starts at 1 and shifts left as the loop walks from `ndims-1` down to `0`. So **bit 0 = innermost (trailing) dim**, bit 1 = second-to-last, etc. For 2D matmul `[M, N]` row-major: bit 0 = N, bit 1 = M.

Earlier code in `transfer_post_ops` had `is_multi_row = (mask & 1) == 0` — that's "varies along innermost = matmul-N", semantically the **kernel-canonical** (post-swap) "binaryCol". Combined with the full `problem.transpose()` doctrine, this double-flipped under `swap_ab_` and produced a `GEMMProblem` field-divergent from base for every post-op binary case (bias-via-binary, scales-as-binary, user binary post-ops with non-trivial broadcast).

Fix in `jit_gemm_pd.cpp:689-696`:
```cpp
bool is_multi_row = (src_rmd.broadcast_mask & 2) == 0;  // bit 1 = matmul-M
bool is_multi_col = (src_rmd.broadcast_mask & 1) == 0;  // bit 0 = matmul-N
```
Pre-transpose then holds matmul-natural `binaryRow/binaryCol`, and the single full `transpose()` at the swap_ab boundary lands at base's kernel-view values.

Mental model: the broadcast_mask formula was inherited from base where it was applied AFTER `apply_swap_ab` had already rotated the binary's md axes. In the worktree-natural flow we read the mask BEFORE any swap, so the bit-to-axis mapping is matmul-natural (bit 0 = innermost matmul axis = N for 2D row-major).

**Doctrine guard:** anywhere else that reads `broadcast_mask` (or any other innermost-indexed mask) and assigns to a field that `transpose()` is going to flip, the same care applies — the formula must produce the matmul-natural value, not the kernel-canonical one.

### 22. `A_scale.layout` / `B_scale.layout` pre-transpose rule
Base's pre-init sets `A_scale.layout = N` (always) and `B_scale.layout = T if !bScale2D else N`, where `bScale2D` is measured on **base's kernel-B side** (= matmul-SRC under `base.swap_ab=0`, matmul-WEIGHTS under `base.swap_ab=1` after the `std::swap(a_quant, b_quant)`). Both layouts are then per-field `.transpose()`-flipped only under base's swap_ab.

In worktree's matmul-natural pre-init, `A_scale` is the matmul-SRC scale slot and `B_scale` is the matmul-WEIGHTS scale slot — so `bScale2D()` there reads `wei_scale_2d`, NOT base's kernel-B-side meaning. Copying base's pre-rule literally (`B_scale = T if !bScale2D else N`) produces wrong post-state because (a) the full `problem.transpose()` includes `std::swap(A_scale, B_scale)` plus per-field transposes (base's per-field transpose has no swap), and (b) the 2D-ness measurement is matmul-side keyed, not kernel-side.

Solving for the matmul-natural pre-values that converge on base's post-state for both `worktree.swap_ab=1 ↔ base.swap_ab=0` and `worktree.swap_ab=0 ↔ base.swap_ab=1` orientations:
```cpp
problem.A_scale.layout = problem.aScale2D() ? MatrixLayout::T : MatrixLayout::N;
problem.B_scale.layout = MatrixLayout::T;
```
The `B_scale = T` is unconditional (the `if (!bScale2D) B_scale = T` from the previous pd is wrong here — keep it gone). `A_scale` is conditional on matmul-SRC 2D-ness.

Surfaced as 5 matmul failures pre-fix (`f4_e2m1` mx-scales, `bf16:u8` broadcast-batch, `f16:u4` weight-decomp, `f4_e2m1` e8m0 mx-scales) — every case with non-trivial `attr-scales`, regardless of dt or 4-bit-ness. The earlier `late_scale_path = !wei_decomp_ && !any_4bit` guard plus `tr_a_matmul` / `tr_b_matmul` conditions were ad-hoc patches for a subset; deleting them and applying the universal rule above fixes the rest by construction.

**Doctrine guard:** any other layout-tag field where base's pre-init rule looks "side-keyed but kernel-keyed under the hood" (i.e., the rule's `bScale2D`-style flag changes meaning under `std::swap(a_quant, b_quant)`) needs the same derive-from-post-state treatment, not a literal port.

### 23. `CO.layout` reset to `N` when CO is unused under swap_ab
`GEMMProblem::transpose()` unconditionally flips `CO.layout` (because real `c_offset` / `sum_ab` users need it swapped). Base's swap_ab branch in `init_GEMMProblem` doesn't touch CO at all (`../base/src/gpu/intel/gemm/jit/pd.cpp:807-815`). So when `cOffset == COffset::None && !sumA && !sumB` — i.e. CO is unused — base leaves `CO.layout = N` (default) while worktree's full transpose ends at `T` under `swap_ab_`. CO is part of the catalog hash, so the divergence flips kernel selection in branches the post-op binary path otherwise behaves identically.

Fix block lives immediately after `if (swap_ab_) problem.transpose();` in `init_GEMMProblem` (`jit_gemm_pd.cpp:1447`). Restore `CO.layout = N` only when the field has no real consumer. The `c_offset` / `sum_ab` path sets `CO.layout` explicitly pre-transpose, so under those branches the post-transpose value is the desired one and the reset doesn't fire.

**Doctrine guard:** if you find another field that `transpose()` flips but base's swap_ab branch never touches, the same pattern applies — restore to the default after transpose under the matching "field is unused" predicate.

### 24. `transfer_post_ops` binary `atype.layout` is gated on `is_multi_col`, not `is_multi_row`
The natural-looking matmul-natural pre-transpose layout rule is `atype.layout = is_multi_row && !innermost ? N : T` (mirroring A/B). That round-trips for "both vary" and "only N varies" cases, but for "only M varies" (e.g. prelu per_oc on a 2D-reshaped matmul) it produces `worktree.atype.layout = N` while base lands at `T`. Base reads bit 0 of `broadcast_mask` (kernel-row = matmul-N post-swap) for "row varies"; worktree reads bit 1 (matmul-M) for matmul-natural "row varies" — the two definitions diverge for axes that the catalog hash distinguishes.

Empirically the correct gate is the DUAL of base's: `is_multi_col && !src_rmd.inner_dim.is_innermost()` keeps the matmul-natural→kernel-canonical round-trip aligned for all four 1D/2D varying-axis cases (`only-M`, `only-N`, `both`, `neither`). The full `problem.transpose()` at the swap_ab boundary then lands on base's kernel-canonical value.

```cpp
bool layout_trans = is_multi_col && !src_rmd.inner_dim.is_innermost();
atype.layout = layout_trans ? MatrixLayout::N : MatrixLayout::T;
```

**Doctrine guard:** when porting a base rule that reads `broadcast_mask` bit 0 (kernel-row), the matmul-natural analogue is `mask & 2` (matmul-M = bit 1), but if the post-transpose target field is "row varies"-keyed in base's kernel view, you may need the DUAL — derive from the desired post-state, not by literal axis substitution.

---

## File map

| Concept | Path |
|---|---|
| matmul_desc_t | `src/common/opdesc.hpp:263` |
| matmul_pd_t | `src/common/matmul_pd.hpp:53` |
| BLAS-style enums (transpose_t, sum_ab_t, offsetc_t) | `src/common/gemm_types.hpp` |
| `jit_gemm_pd_t` (shared base for JIT GEMM impls) | `src/gpu/intel/gemm/jit/jit_gemm_pd.{hpp,cpp}` |
| `init_GEMMProblem` | `src/gpu/intel/gemm/jit/jit_gemm_pd.cpp` |
| `transfer_post_ops` (binary postop translation) | `src/gpu/intel/gemm/jit/jit_gemm_pd.cpp` |
| `GEMMProblem` definition | `src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp:166` |
| `GEMMProblem::transpose()` | `src/gpu/intel/gemm/jit/generator/pieces/problem_utils.cpp` |
| `gen_t::pd_t` swap-AB logic | `src/gpu/intel/gemm/jit.hpp:66-79` |
| `xe_hp_systolic_t` (always-swap impl) | `src/gpu/intel/gemm/jit_xe_hp_systolic.{hpp,cpp}` |
| Exec-args conversion (matmul ctx → JIT args) | `src/gpu/intel/gemm/exec_types.hpp` |
| GPU matmul impl list | `src/gpu/gpu_matmul_list.cpp` |
| GEMMProblem dump helper | `src/gpu/intel/gemm/jit/problem_dump.hpp` |
| Internal IP/matmul wrapper | `src/gpu/intel/ip/matmul.{hpp,cpp}` |
| Internal RNN/matmul calls | `src/gpu/intel/rnn/grid.cpp` |
| SDPA layout-query usage | `src/gpu/intel/sdpa/micro.cpp` |

---

## Workflow

### Use subagents aggressively
Large multi-file migration. **Default to launching subagents (Explore/general-purpose/Plan) for anything that crosses more than one or two files** — surveys, cross-file consistency checks, "find all callers of X". Run independent subagents in parallel (single message, multiple Agent calls). Reserve direct file reads/edits for the spot where you actually write the change. The main-context budget is precious — burning it on enumeration work is the easy way to lose continuity.

### Verification via random-sampled benchdnn
For quick correctness checks:
```
/nfs/site/disks/hal9000/echeresh/sandbox/scripts/benchdnnsample.sh matmul
/nfs/site/disks/hal9000/echeresh/sandbox/scripts/benchdnnsample.sh ip
/nfs/site/disks/hal9000/echeresh/sandbox/scripts/benchdnnsample.sh rnn
```
Generates a shape list via `benchdnnlist.py`, shuffles, takes first 100, writes to `test.in`, then runs `./build/tests/benchdnn/benchdnn -v5 --engine=gpu --mode-modifier=P --$driver --batch=test.in`. Run from worktree root.

### GEMMProblem dump for fast triage — USE THIS FIRST
`src/gpu/intel/gemm/jit/problem_dump.hpp` is a header-only helper. Set `ONEDNN_DUMP_GEMM_PROBLEM=<tag>` (any non-empty string) and rerun; every `init_GEMMProblem` call prints all GEMMProblem fields prefixed with `[GPDUMP:<tag>]`. To compare against `../base`, copy `problem_dump.hpp` over and add `maybe_dump_gemm_problem(problem);` at the end of `init_GEMMProblem` in `../base/src/gpu/intel/gemm/jit/pd.cpp` (same namespace). Then `diff` the two outputs — the first differing line is almost always the bug. Much faster than ad-hoc `fprintf` chasing.

### Targeted triage: tracing + ../base comparison
For pinpointing a specific regression (wrong output, segfault, dispatch mismatch), prefer **manual tracing + diffing against the `../base` worktree** over re-running broad benchdnn samples.
- `../base` is a sibling worktree from the pre-migration baseline. Same `./build/tests/benchdnn/benchdnn` layout. Run the failing case there to confirm "this passed before the migration."
- Add temporary `printf`/`gpu_warning` traces at the boundary you suspect (apply_swap_ab → init_GEMMProblem → execute → launch_nocopy). Diff worktree vs `../base` to see which value first diverges.
- Useful trace points for swap-related bugs: `swap_ab_` after apply_swap_ab; `problem.A.layout`/`problem.B.layout`/`problem.sumA`/`problem.sumB` after init_GEMMProblem; `m`/`n`/`k`/`lda`/`ldb`/`ldc` at launch_nocopy entry; post-op binary `src1_desc.dims`/`broadcast_mask` after canonicalize/swap.
- Rebuild both sides with the same `-O0` / verbose printfs to make traces line up. Strip the traces before committing.

### Keep CLAUDE.md alive
Update CLAUDE.md *the moment* you discover something non-obvious: a gotcha, an architectural surprise, a constraint that bit you. Don't wait for end-of-session. STATE.md tracks progress; CLAUDE.md is the durable knowledge base.

---

## Style

- Don't preserve `gemm_*` names in new code. New JIT-internal types use `jit_gemm_*` — "gemm" here means "the kernel family", not "the descriptor type".
- Comments only when explaining a non-obvious swap/transpose decision.
- `dnnl_sgemm` / `dnnl_gemm_*` (BLAS C-API) goes straight to CPU and never used the internal pd — leave it alone. Same for brgemm.
