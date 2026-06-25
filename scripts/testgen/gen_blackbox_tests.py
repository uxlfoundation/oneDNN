#!/usr/bin/env python3
"""
gen_blackbox_tests.py — Black-box augmentation generator for benchdnn matmul and conv.

Produces 250 matmul tests + 250 conv tests that cover input-space corners
unlikely to appear in production nightly logs:
  - GEMV/GEMM edge dimensions (M=1, N=1, K=1)
  - GPU tail sizes (127, 129, 255, 257, 511 — alignment boundary stress)
  - Layout corners (ba:ba, bca:cab, 3D permutations)
  - LLM patterns (attention Q·Kᵀ, FFN projections, int4 group quantization)
  - fpmath modes (tf32, bf16, f16 applied to f32 src/dst)
  - 1D/3D conv, small IC (1,2,3,4), asymmetric/even/large kernels
  - Depthwise with dilation and asymmetric kernels
  - NXC (axb) layout for conv
  - BWD_D/BWD_WB for 1D, 3D, asymmetric conv

Usage:
    python3 gen_blackbox_tests.py [--output-dir DIR]

Output files:
    <output-dir>/matmul_blackbox.txt
    <output-dir>/conv_blackbox.txt
"""
import argparse
import os

# ── Shape validation helpers ─────────────────────────────────────────────────

def _oh(ih, kh, ph, sh, dh):
    """Compute output size using floor division (oneDNN formula), or None if invalid."""
    eff_k = kh + (kh - 1) * dh
    num = ih + 2 * ph - eff_k
    if num < 0:
        return None
    oh = num // sh + 1
    return oh if oh >= 1 else None


# ── Conv shape builders ───────────────────────────────────────────────────────

def c2d(mb, ic, ih, oc, kh, kw=None, ph=0, pw=None, sh=1, sw=None, dh=0, dw=None,
        g=1, iw=None, dt=None, dir=None, stag=None, dtag=None, bia=None, postop=None,
        extra=None):
    if kw is None: kw = kh
    if pw is None: pw = ph
    if sw is None: sw = sh
    if dw is None: dw = dh
    if iw is None: iw = ih
    oh = _oh(ih, kh, ph, sh, dh)
    ow = _oh(iw, kw, pw, sw, dw)
    if oh is None or ow is None:
        return None
    if g > 1 and (ic % g != 0 or oc % g != 0):
        return None
    parts = []
    if dir:    parts.append(f"--dir={dir}")
    if dt:     parts.append(f"--dt={dt}")
    if stag:   parts.append(f"--stag={stag}")
    if dtag:   parts.append(f"--dtag={dtag}")
    if bia:    parts.append(f"--bia-dt={bia}")
    if extra:  parts.append(extra)
    if postop: parts.append(f"--attr-post-ops={postop}")
    prefix = f"g{g}" if g > 1 else ""
    shape = f"{prefix}mb{mb}ic{ic}"
    if ih == iw:
        shape += f"ih{ih}"
    else:
        shape += f"ih{ih}iw{iw}"
    shape += f"oc{oc}"
    shape += f"oh{oh}"
    if oh != ow:
        shape += f"ow{ow}"
    shape += f"kh{kh}"
    if kh != kw:
        shape += f"kw{kw}"
    if sh != 1 or sw != 1:
        shape += f"sh{sh}"
        if sh != sw:
            shape += f"sw{sw}"
    if ph != 0 or pw != 0:
        shape += f"ph{ph}"
        if ph != pw:
            shape += f"pw{pw}"
    if dh != 0 or dw != 0:
        shape += f"dh{dh}"
        if dh != dw:
            shape += f"dw{dw}"
    parts.append(shape)
    return " ".join(parts)


def c1d(mb, ic, iw, oc, kw, pw=0, sw=1, dw=0, g=1,
        dt=None, dir=None, stag=None, bia=None, postop=None, extra=None):
    ow = _oh(iw, kw, pw, sw, dw)
    if ow is None:
        return None
    if g > 1 and (ic % g != 0 or oc % g != 0):
        return None
    parts = []
    if dir:    parts.append(f"--dir={dir}")
    if dt:     parts.append(f"--dt={dt}")
    if stag:   parts.append(f"--stag={stag}")
    if bia:    parts.append(f"--bia-dt={bia}")
    if extra:  parts.append(extra)
    if postop: parts.append(f"--attr-post-ops={postop}")
    prefix = f"g{g}" if g > 1 else ""
    shape = f"{prefix}mb{mb}ic{ic}iw{iw}oc{oc}ow{ow}kw{kw}"
    if sw != 1: shape += f"sw{sw}"
    if pw != 0: shape += f"pw{pw}"
    if dw != 0: shape += f"dw{dw}"
    parts.append(shape)
    return " ".join(parts)


def c3d(mb, ic, id_, ih, iw, oc, kd, kh, kw,
        pd=0, ph=0, pw=0, sd=1, sh=1, sw=1, dd=0, dh=0, dw=0,
        g=1, dt=None, dir=None):
    od = _oh(id_, kd, pd, sd, dd)
    oh = _oh(ih, kh, ph, sh, dh)
    ow = _oh(iw, kw, pw, sw, dw)
    if od is None or oh is None or ow is None:
        return None
    if g > 1 and (ic % g != 0 or oc % g != 0):
        return None
    parts = []
    if dir: parts.append(f"--dir={dir}")
    if dt:  parts.append(f"--dt={dt}")
    prefix = f"g{g}" if g > 1 else ""
    shape = f"{prefix}mb{mb}ic{ic}id{id_}ih{ih}iw{iw}oc{oc}od{od}oh{oh}ow{ow}kd{kd}kh{kh}kw{kw}"
    if sd != 1 or sh != 1 or sw != 1: shape += f"sd{sd}sh{sh}sw{sw}"
    if pd != 0 or ph != 0 or pw != 0: shape += f"pd{pd}ph{ph}pw{pw}"
    if dd != 0 or dh != 0 or dw != 0: shape += f"dd{dd}dh{dh}dw{dw}"
    parts.append(shape)
    return " ".join(parts)


def add(lst, test):
    if test:
        lst.append(test)


# ── Matmul black-box tests ────────────────────────────────────────────────────

def build_matmul_tests():
    tests = []

    # GEMV: M=1 (LLM decode / inference)
    # Note: s8:s8:s32 GEMV uses scale=1.0 to avoid FP tolerance issues with large K
    for k, n, dt, layout in [
        (1024, 512,  "f16:f16:f16",    "--stag=ab --wtag=ba"),
        (4096, 4096, "f16:f16:f16",    "--stag=ab --wtag=ba"),
        (4096, 1024, "bf16:bf16:bf16", "--stag=ab --wtag=ba"),
        (2048, 8192, "f32:f32:f32",    "--stag=ab --wtag=ba"),
        (512, 512,   "s8:s8:s32",      "--attr-scales=src:common:1+wei:common:1"),
        (2048, 512,  "f16:s4:f16",     "--attr-fpmath=f16:true --skip-impl=ref"),
        (4096, 1024, "f16:u4:f16",     "--attr-fpmath=f16:true --skip-impl=ref"),
        (768,  768,  "f16:f16:f16",    "--stag=ab --wtag=ba"),
        (256,  128,  "f32:f32:f32",    "--stag=ab --wtag=ba"),
        (512,  256,  "bf16:bf16:bf16", "--stag=ab --wtag=ba"),
    ]:
        tests.append(f"--dt={dt} {layout} 1x{k}:{k}x{n}")

    # N=1 (output size 1 per batch row)
    for m, k, dt in [
        (1024, 512, "f16:f16:f16"), (2048, 512, "bf16:bf16:bf16"),
        (4096, 1024, "f32:f32:f32"), (512, 256, "s8:s8:s32"),
    ]:
        e = " --attr-scales=src:common:0.1+wei:common:0.1" if dt == "s8:s8:s32" else ""
        tests.append(f"--dt={dt}{e} {m}x{k}:{k}x1")

    # K=1 (outer product); M=N=K=1 degenerates
    for m, n, dt in [(128, 256, "f32:f32:f32"), (512, 512, "f16:f16:f16"),
                     (256, 512, "f32:f32:f32")]:
        tests.append(f"--dt={dt} {m}x1:1x{n}")
    for dt in ["f32:f32:f32", "f16:f16:f16", "bf16:bf16:bf16", "s8:s8:s32", "f64:f64:f64"]:
        e = " --attr-scales=src:common:0.1+wei:common:0.1" if dt == "s8:s8:s32" else ""
        tests.append(f"--dt={dt}{e} 1x1:1x1")

    # Non-power-of-2 / prime sizes (alignment stress)
    for m, k, n, dt in [
        (127, 251, 127, "f16:f16:f16"),  (97, 167, 83, "f32:f32:f32"),
        (113, 223, 113, "bf16:bf16:bf16"),(37, 73, 151, "f16:f16:f16"),
        (129, 257, 129, "f16:f16:f16"),  (255, 127, 255, "bf16:bf16:bf16"),
        (511, 511, 511, "f16:f16:f16"),  (513, 513, 513, "f32:f32:f32"),
        (33, 65, 33, "f16:f16:f16"),     (17, 33, 17, "f32:f32:f32"),
        (3, 7, 3, "f16:f16:f16"),        (7, 15, 7, "bf16:bf16:bf16"),
        (15, 31, 15, "f32:f32:f32"),     (31, 63, 31, "f16:f16:f16"),
        (48, 96, 48, "f32:f32:f32"),     (96, 192, 96, "f16:f16:f16"),
        (192, 384, 192, "bf16:bf16:bf16"),(384, 768, 384, "f16:f16:f16"),
        (768, 1536, 768, "f16:f16:f16"), (1152, 2304, 1152, "bf16:bf16:bf16"),
    ]:
        tests.append(f"--dt={dt} {m}x{k}:{k}x{n}")

    # Power-of-2 squares
    for n, dt in [(64, "f32:f32:f32"), (128, "f16:f16:f16"), (256, "f16:f16:f16"),
                  (512, "bf16:bf16:bf16"), (1024, "f16:f16:f16"), (2048, "bf16:bf16:bf16"),
                  (4096, "f32:f32:f32")]:
        tests.append(f"--dt={dt} {n}x{n}:{n}x{n}")

    # Large K (heavy reduction)
    for m, k, n, dt in [
        (32, 8192, 32, "f16:f16:f16"), (16, 16384, 16, "f32:f32:f32"),
        (1, 32768, 32, "f16:f16:f16"),
    ]:
        tests.append(f"--dt={dt} {m}x{k}:{k}x{n}")

    # 3D batch shapes
    for b, m, k, n, dt in [
        (1, 64, 256, 128, "f16:f16:f16"),   (1, 1, 512, 256, "bf16:bf16:bf16"),
        (128, 32, 256, 128, "f16:f16:f16"), (64, 64, 512, 256, "f32:f32:f32"),
        (32, 128, 512, 256, "bf16:bf16:bf16"),(8, 1, 256, 256, "f16:f16:f16"),
        (4, 8, 64, 32, "f16:f16:f16"),      (16, 32, 256, 128, "f32:f32:f32"),
        (4, 64, 512, 256, "f16:f16:f16"),   (2, 64, 256, 64, "f32:f32:f32"),
    ]:
        tests.append(f"--dt={dt} {b}x{m}x{k}:{b}x{k}x{n}")

    # 4D attention patterns
    for b1, b2, s, d, dt in [
        (2, 8, 64, 64, "f16:f16:f16"),    (4, 16, 32, 128, "bf16:bf16:bf16"),
        (1, 12, 128, 64, "f32:f32:f32"),  (2, 8, 1, 64, "f16:f16:f16"),
        (4, 16, 512, 64, "bf16:bf16:bf16"),(2, 32, 64, 128, "f16:f16:f16"),
    ]:
        tests.append(f"--dt={dt} {b1}x{b2}x{s}x{d}:{b1}x{b2}x{d}x{s}")
    tests.append("--dt=f16:f16:f16 --stag=abcd --wtag=abcd 2x8x64x64:2x8x64x64")
    tests.append("--dt=bf16:bf16:bf16 --stag=abdc --wtag=abcd 2x8x64x32:2x8x32x64")
    for b1, b2, m, k, n, dt in [
        (1, 4, 16, 64, 32, "f16:f16:f16"),  (2, 4, 32, 128, 64, "bf16:bf16:bf16"),
        (1, 16, 1, 64, 64, "f16:f16:f16"),  (4, 8, 1, 128, 64, "bf16:bf16:bf16"),
        (2, 16, 128, 64, 128, "f16:f16:f16"),
    ]:
        tests.append(f"--dt={dt} {b1}x{b2}x{m}x{k}:{b1}x{b2}x{k}x{n}")

    # Layout corners: ba:ba, ba:ab, 3D permutations
    for m, k, n, dt in [
        (256, 512, 256, "f16:f16:f16"), (512, 1024, 512, "bf16:bf16:bf16"),
        (128, 256, 128, "f32:f32:f32"), (64, 128, 64, "f32:f32:f32"),
        (192, 384, 192, "f32:f32:f32"), (1024, 1024, 1024, "f16:f16:f16"),
    ]:
        tests.append(f"--dt={dt} --stag=ba --wtag=ba {m}x{k}:{k}x{n}")
    for m, k, n, dt in [(512, 256, 512, "f16:f16:f16"), (256, 128, 256, "f32:f32:f32")]:
        tests.append(f"--dt={dt} --stag=ba --wtag=ab {m}x{k}:{k}x{n}")
    for stag, wtag in [("bac","bac"),("cba","abc"),("bca","abc"),("abc","bca"),("bca","bac"),("cab","bac")]:
        tests.append(f"--dt=f16:f16:f16 --stag={stag} --wtag={wtag} 4x32x64:4x64x32")

    # f32 baseline
    for m, k, n in [(128,256,128),(512,1024,512),(1024,1024,1024)]:
        tests.append(f"--dt=f32:f32:f32 {m}x{k}:{k}x{n}")
    tests.append("--dt=f32:f32:f32 --stag=ab --wtag=ba 512x1024:1024x512")
    # f64
    tests.append("--dt=f64:f64:f64 128x256:256x128")
    tests.append("--dt=f64:f64:f64 64x128:128x64")
    # s8:s8:s32
    for m, k, n in [(256,256,256),(512,512,512),(128,256,128),(1024,1024,1024)]:
        tests.append(f"--dt=s8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 {m}x{k}:{k}x{n}")
    tests.append("--dt=s8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 --stag=ab --wtag=ba 512x512:512x512")
    tests.append("--dt=s8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 4x128x256:4x256x128")
    # u8:s8:s32
    tests.append("--dt=u8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 --attr-zero-points=src:common:128 256x256:256x256")
    tests.append("--dt=u8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 --attr-zero-points=src:common:1 4x64x256:4x256x128")
    tests.append("--dt=u8:s8:u8 --attr-scales=src:per_tensor:f32:1x1+wei:per_tensor:f32:1x1+dst:per_tensor:f32:1x1 --attr-zero-points=src:common:128+dst:common:128 256x256:256x256")
    # f8 2D
    for dt, stag, wtag, m, k, n in [
        ("f8_e4m3:f8_e4m3:f16",  "ab","ba",256,256,256),
        ("f8_e5m2:f8_e5m2:bf16", "ab","ba",128,256,128),
        ("f8_e4m3:f8_e5m2:f32",  "ab","ba",512,512,512),
        ("bf16:f8_e5m2:bf16",    "ab","ba",256,512,256),
        ("f32:f8_e4m3:f32",      "ab","ba",512,1024,512),
        ("f8_e4m3:f8_e4m3:f32",  "ab","ab",256,512,256),
        ("f8_e5m2:f8_e5m2:f16",  "ab","ab",512,1024,512),
    ]:
        tests.append(f"--dt={dt} --stag={stag} --wtag={wtag} --skip-impl=ref {m}x{k}:{k}x{n}")

    # int4 group sizes (validate K divisibility)
    for gs, m, k, n in [(16,128,256,128),(32,64,256,64),(64,128,256,128),
                         (128,128,256,128),(256,1,256,512),(512,1,512,1024)]:
        tests.append(f"--dt=f16:s4:f16 --attr-scales=wei:per_ocic:f16:{gs}x1 --attr-zero-points=wei:per_ocic:s4:{gs}x1 --attr-fpmath=f16:true --skip-impl=ref {m}x{k}:{k}x{n}")
    tests.append("--dt=f16:u4:f16 --attr-scales=wei:per_ocic:f16:64x1 --attr-zero-points=wei:per_ocic:u4:64x1 --attr-fpmath=f16:true --skip-impl=ref 4x128x256:4x256x128")

    # Post-ops
    for postop, m, k, n, dt in [
        ("gelu_tanh", 256,512,256,"f16:f16:f16"),
        ("gelu_tanh", 4,64,256,"bf16:bf16:bf16"),
        ("swish:1",  256,512,256,"f16:f16:f16"),
        ("tanh",     128,256,128,"f32:f32:f32"),
        ("hardswish",256,512,256,"f32:f32:f32"),
        ("mish",     128,256,128,"f16:f16:f16"),
        ("sum+relu", 4,32,256,"f16:f16:f16"),
        ("sum+gelu_tanh",256,512,256,"bf16:bf16:bf16"),
        ("gelu_erf", 256,512,256,"f32:f32:f32"),
        ("elu:0.5",  128,256,128,"f32:f32:f32"),
        ("clip:0:1", 256,512,256,"f16:f16:f16"),
        ("relu:0.1", 128,256,128,"bf16:bf16:bf16"),
        ("logistic", 128,256,128,"f32:f32:f32"),
    ]:
        tests.append(f"--dt={dt} --attr-post-ops={postop} {m}x{k}:{k}x{n}")

    # Bias varieties
    for m, k, n, dt, mask, bia_dt in [
        (256,512,256,"f32:f32:f32",2,"f32"),   (512,1024,512,"f16:f16:f16",2,"f16"),
        (4,64,256,"f32:f32:f32",4,"f32"),      (4,128,64,"bf16:bf16:bf16",6,"bf16"),
    ]:
        tests.append(f"--dt={dt} --bia-dt={bia_dt} --bia_mask={mask} {m}x{k}:{k}x{n}")
    tests.append("--dt=u8:s8:f16 --bia-dt=f16 --attr-scales=src:common:0.1+wei:common:0.1 256x256:256x256")

    # fpmath modes
    for fpmath in ["tf32", "bf16", "f16"]:
        tests.append(f"--dt=f32:f32:f32 --attr-fpmath={fpmath} 256x512:512x256")
        tests.append(f"--dt=f32:f32:f32 --attr-fpmath={fpmath} 4x64x256:4x256x128")

    # Runtime dims
    tests.append("--dt=f16:f16:f16 --runtime_dims_masks=15:15 64x256:256x128")
    tests.append("--dt=bf16:bf16:bf16 --runtime_dims_masks=15:15 --dtag=abc 2x64x128:2x128x64")
    tests.append("--dt=f32:f32:f32 --runtime_dims_masks=3:3 256x512:512x256")
    tests.append("--dt=u8:s8:u8 --runtime_dims_masks=15:15 --attr-scales=src:common:0.1+wei:common:0.1 --attr-zero-points=src:common:128 128x256:256x128")

    # LLM FFN projections
    for tok, d, dt in [
        (1, 4096, "f16:f16:f16"),   (16, 4096, "bf16:bf16:bf16"),
        (64, 4096, "f16:f16:f16"),  (256, 4096, "bf16:bf16:bf16"),
        (1, 8192, "f16:f16:f16"),   (64, 2048, "f32:f32:f32"),
        (4, 4096, "f16:f16:f16"),   (128, 4096, "bf16:bf16:bf16"),
        (512, 2048, "f16:f16:f16"), (1024, 1024, "f32:f32:f32"),
    ]:
        tests.append(f"--dt={dt} --stag=ab --wtag=ba {tok}x{d}:{d}x{d}")

    # More GEMV: diverse decode shapes
    for k, n, dt in [
        (2048, 2048, "f16:f16:f16"), (4096, 11008, "bf16:bf16:bf16"),
        (4096, 14336, "f16:f16:f16"), (8192, 28672, "bf16:bf16:bf16"),
        (2048, 5504, "f16:f16:f16"),  (1024, 4096, "f32:f32:f32"),
    ]:
        tests.append(f"--dt={dt} --stag=ab --wtag=ba 1x{k}:{k}x{n}")

    # More batched matmul (MHA)
    for b, m, k, n, dt in [
        (4, 512, 64, 512, "f16:f16:f16"),  (8, 128, 64, 128, "f16:f16:f16"),
        (16, 64, 64, 64, "bf16:bf16:bf16"), (32, 32, 64, 32, "f32:f32:f32"),
        (4, 1, 64, 512, "f16:f16:f16"),     (8, 512, 64, 1, "bf16:bf16:bf16"),
        (2, 128, 128, 128, "f32:f32:f32"),  (4, 256, 256, 256, "f16:f16:f16"),
        (2, 512, 512, 512, "bf16:bf16:bf16"),
    ]:
        tests.append(f"--dt={dt} {b}x{m}x{k}:{b}x{k}x{n}")

    # More 4D shapes
    for b1, b2, m, k, n, dt in [
        (4, 8, 128, 64, 128, "f16:f16:f16"),   (2, 12, 512, 64, 512, "f16:f16:f16"),
        (1, 32, 64, 128, 64, "bf16:bf16:bf16"), (4, 4, 256, 64, 256, "f16:f16:f16"),
        (1, 8, 2048, 64, 2048, "f16:f16:f16"),  (2, 16, 256, 256, 256, "bf16:bf16:bf16"),
    ]:
        tests.append(f"--dt={dt} {b1}x{b2}x{m}x{k}:{b1}x{b2}x{k}x{n}")

    # More diverse post-ops
    for postop, dt in [
        ("exp",        "f32:f32:f32"), ("log",       "f32:f32:f32"),
        ("abs",        "f32:f32:f32"), ("square",    "f32:f32:f32"),
        ("sqrt",       "f32:f32:f32"), ("clip_v2:0:6","f32:f32:f32"),
        ("relu",       "f16:f16:f16"), ("swish:0.5", "f16:f16:f16"),
        ("linear:2:-1","f16:f16:f16"), ("round",     "f32:f32:f32"),
    ]:
        tests.append(f"--dt={dt} --attr-post-ops={postop} 256x512:512x256")

    # More ba:ba / transposed
    for m, k, n, dt in [
        (32, 64, 32, "f16:f16:f16"),    (48, 96, 48, "f32:f32:f32"),
        (320, 640, 320, "f16:f16:f16"), (640, 1280, 640, "bf16:bf16:bf16"),
    ]:
        tests.append(f"--dt={dt} --stag=ba --wtag=ba {m}x{k}:{k}x{n}")

    # u8:s8:s32 / per-tensor quantization
    for m, k, n in [(128,256,128),(256,512,256),(512,1024,512)]:
        tests.append(f"--dt=u8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 --attr-zero-points=src:common:128 {m}x{k}:{k}x{n}")

    # Per-oc weight scales (typical int8 inference)
    tests.append("--dt=u8:s8:f32 --attr-scales=src:common:0.1+wei:per_oc --attr-zero-points=src:common:1 256x512:512x256")
    tests.append("--dt=u8:s8:s32 --attr-scales=src:common:0.25+wei:per_oc --attr-zero-points=src:common:128 4x128x256:4x256x128")

    # Chained post-ops
    tests.append("--dt=f16:f16:f16 --attr-post-ops=sum+relu+clip:0:6 4x32x256:4x256x128")
    tests.append("--dt=f32:f32:f32 --attr-post-ops=gelu_tanh+mul:f32:per_dim_0 8x64x128:8x128x64")

    # bf16 accumulation of f32 inputs
    tests.append("--dt=f32:f32:f32 --attr-fpmath=bf16 --stag=ab --wtag=ba 1024x4096:4096x1024")
    tests.append("--dt=f32:f32:f32 --attr-fpmath=tf32 --stag=ab --wtag=ba 4096x4096:4096x4096")

    # Extra shapes to reach 250 — model dimensions from popular architectures
    _extra = [
        # BERT / Transformer encoder shapes
        "--dt=f16:f16:f16 384x768:768x768",
        "--dt=f16:f16:f16 384x768:768x3072",
        "--dt=f16:f16:f16 384x3072:3072x768",
        "--dt=bf16:bf16:bf16 512x1024:1024x1024",
        "--dt=bf16:bf16:bf16 512x1024:1024x4096",
        "--dt=f32:f32:f32 --attr-fpmath=bf16 128x768:768x768",
        # GPT-2 style
        "--dt=f16:f16:f16 1x768:768x2304",
        "--dt=f16:f16:f16 16x768:768x2304",
        "--dt=bf16:bf16:bf16 1x1024:1024x3072",
        # Mixed stag/wtag with 3D
        "--dt=f16:f16:f16 --stag=acb --wtag=abc 4x64x256:4x256x64",
        "--dt=bf16:bf16:bf16 --stag=abc --wtag=acb 8x32x128:8x128x32",
        "--dt=f32:f32:f32 --stag=acb --wtag=acb 2x128x256:2x256x128",
        # s8 with 3D
        "--dt=s8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 4x32x128:4x128x64",
        "--dt=u8:s8:s32 --attr-scales=src:common:0.1+wei:common:0.1 --attr-zero-points=src:common:128 8x64x256:8x256x128",
        # f16 int4 3D
        "--dt=f16:s4:f16 --attr-scales=wei:per_ocic:f16:32x1 --attr-fpmath=f16:true --skip-impl=ref 4x32x256:4x256x128",
        "--dt=f16:u4:f16 --attr-scales=wei:per_ocic:f16:64x1 --attr-fpmath=f16:true --skip-impl=ref 8x64x512:8x512x256",
        # More prime GPU tail sizes
        "--dt=f16:f16:f16 23x47:47x89",
        "--dt=bf16:bf16:bf16 41x83:83x167",
        "--dt=f32:f32:f32 101x199:199x397",
        "--dt=f16:f16:f16 53x107:107x211",
        "--dt=f16:f16:f16 79x157:157x313",
        # 2D identity-like (M=N)
        "--dt=f16:f16:f16 --stag=ab --wtag=ba 256x256:256x256",
        "--dt=bf16:bf16:bf16 --stag=ab --wtag=ba 512x512:512x512",
        "--dt=f32:f32:f32 --stag=ab --wtag=ba 1024x1024:1024x1024",
        # Ternary shapes (MxK, KxN where M≠N)
        "--dt=f16:f16:f16 128x4096:4096x256",
        "--dt=bf16:bf16:bf16 256x1024:1024x4096",
        "--dt=f32:f32:f32 64x2048:2048x512",
        # With bias add
        "--dt=f16:f16:f16 --bia-dt=f16 --bia_mask=2 --attr-post-ops=relu 256x512:512x256",
        "--dt=f32:f32:f32 --bia-dt=f32 --bia_mask=2 --attr-post-ops=gelu_tanh 512x1024:1024x512",
        # s8 per-channel quantization
        "--dt=s8:s8:f32 --attr-scales=src:common:0.1+wei:per_oc --attr-post-ops=relu 256x512:512x256",
        "--dt=u8:s8:u8 --attr-scales=src:common:0.5+wei:per_oc+dst:common:0.25 --attr-zero-points=src:common:128 256x256:256x256",
        # Fill to 250: diverse types and sizes
        "--dt=f16:f16:f16 320x640:640x320",   "--dt=bf16:bf16:bf16 160x320:320x160",
        "--dt=f32:f32:f32 80x160:160x80",     "--dt=f16:f16:f16 40x80:80x40",
        "--dt=bf16:bf16:bf16 20x40:40x20",    "--dt=f32:f32:f32 10x20:20x10",
        "--dt=f16:f16:f16 --stag=ba --wtag=ab 320x640:640x320",
        "--dt=bf16:bf16:bf16 --stag=ba --wtag=ab 160x320:320x160",
        "--dt=f32:f32:f32 16x1024:1024x512",  "--dt=f16:f16:f16 32x2048:2048x256",
        "--dt=bf16:bf16:bf16 64x4096:4096x128","--dt=f32:f32:f32 2x512:512x2",
        "--dt=f16:f16:f16 3x768:768x3",       "--dt=bf16:bf16:bf16 5x1024:1024x5",
        "--dt=f32:f32:f32 7x1024:1024x7",     "--dt=f16:f16:f16 11x512:512x11",
        "--dt=bf16:bf16:bf16 13x256:256x13",  "--dt=f32:f32:f32 --attr-fpmath=tf32 1x512:512x1024",
        "--dt=f16:f16:f16 --attr-post-ops=sum 4x64x256:4x256x512",
    ]
    tests.extend(_extra)

    return tests[:250]


# ── Conv black-box tests ──────────────────────────────────────────────────────

def build_conv_tests():
    tests = []

    # 1D — many kernel widths
    for kw, pw in [(1,0),(3,1),(5,2),(7,3),(9,4),(11,5),(13,6),(15,7),(17,8),(21,10)]:
        for ic, oc, iw in [(16,32,64),(32,64,128)]:
            add(tests, c1d(1, ic, iw, oc, kw, pw=pw))
    # 1D strides
    for sw in [2,3,4]:
        add(tests, c1d(4, 32, 128, 64, 3, pw=1, sw=sw))
        add(tests, c1d(4, 64, 64, 128, 5, pw=2, sw=sw))
    # 1D dilation
    for dw in [1,2,3]:
        pw = dw + 1
        add(tests, c1d(1, 16, 64, 32, 3, pw=pw, dw=dw))
        add(tests, c1d(2, 32, 128, 64, 3, pw=pw, dw=dw))
    # 1D depthwise
    for g, iw in [(32,64),(64,128),(16,56),(128,28)]:
        add(tests, c1d(1, g, iw, g, 3, pw=1, g=g))
        add(tests, c1d(4, g, iw, g, 5, pw=2, g=g))
    # 1D grouped
    add(tests, c1d(2, 32, 64, 64, 3, pw=1, g=4))
    # 1D int8
    add(tests, c1d(4, 32, 56, 64, 3, pw=1, sw=2, dt="u8:s8:u8",
                   extra="--attr-scales=src:common:0.1+wei:common:0.1"))
    # 1D BWD + NXC
    add(tests, c1d(4, 32, 56, 64, 3, pw=1, sw=2, dir="BWD_D"))
    add(tests, c1d(4, 32, 56, 64, 3, pw=1, sw=2, dir="BWD_WB"))
    add(tests, c1d(4, 32, 56, 64, 3, pw=1, sw=2, stag="axb"))

    # 3D — 3×3×3
    for (id_, ih) in [(8,8),(16,16),(4,8)]:
        add(tests, c3d(1, 16, id_, ih, ih, 32, 3, 3, 3, pd=1, ph=1, pw=1))
        add(tests, c3d(2, 32, id_, ih, ih, 64, 3, 3, 3, pd=1, ph=1, pw=1))
    # 3D 1×1×1 pointwise
    add(tests, c3d(4, 64, 8, 8, 8, 128, 1, 1, 1))
    add(tests, c3d(1, 128, 4, 7, 7, 256, 1, 1, 1))
    # 3D 1×3×3 temporal
    add(tests, c3d(2, 32, 8, 14, 14, 64, 1, 3, 3, ph=1, pw=1))
    # 3D strided
    add(tests, c3d(4, 64, 8, 8, 8, 128, 3, 3, 3, pd=1, ph=1, pw=1, sd=2, sh=2, sw=2))
    add(tests, c3d(2, 32, 16, 16, 16, 64, 3, 3, 3, pd=1, ph=1, pw=1, sd=2, sh=2, sw=2))
    # 3D depthwise
    add(tests, c3d(1, 32, 8, 8, 8, 32, 3, 3, 3, pd=1, ph=1, pw=1, g=32))
    add(tests, c3d(2, 16, 8, 14, 14, 16, 1, 3, 3, ph=1, pw=1, g=16))
    # 3D BWD
    add(tests, c3d(1, 16, 8, 8, 8, 32, 3, 3, 3, pd=1, ph=1, pw=1, dir="BWD_D"))
    add(tests, c3d(1, 16, 8, 8, 8, 32, 3, 3, 3, pd=1, ph=1, pw=1, dir="BWD_WB"))
    add(tests, c3d(4, 64, 8, 8, 8, 128, 3, 3, 3, pd=1, ph=1, pw=1, sd=2, sh=2, sw=2, dir="BWD_D"))
    # 3D dilation
    add(tests, c3d(1, 16, 16, 16, 16, 32, 3, 3, 3, pd=2, ph=2, pw=2, dd=1, dh=1, dw=1))
    # 3D int8 / f16
    add(tests, c3d(1, 16, 8, 8, 8, 32, 3, 3, 3, pd=1, ph=1, pw=1, dt="u8:s8:u8"))
    add(tests, c3d(2, 32, 8, 14, 14, 64, 3, 3, 3, pd=1, ph=1, pw=1, sh=2, sw=2, dt="f16:f16:f16"))
    # More 3D
    for (mb, ic, id_, ih, oc, kd, kh) in [
        (2,16,16,14,32,3,3),(4,32,8,8,64,3,3),(1,64,4,7,128,1,3),
        (2,32,8,8,32,3,3),(4,64,4,4,64,3,3),
    ]:
        add(tests, c3d(mb, ic, id_, ih, ih, oc, kd, kh, kh, pd=kd//2, ph=kh//2, pw=kh//2))

    # Small IC
    for ih, oc, kh, ph, sh in [
        (28,32,5,0,1),(32,16,3,1,1),(224,64,7,3,2),(64,8,3,1,1),(128,32,3,1,1)]:
        add(tests, c2d(1, 1, ih, oc, kh, ph=ph, sh=sh))
    for ih, ic, kh, ph in [(28,16,3,1),(16,32,5,2),(32,8,3,1)]:
        add(tests, c2d(4, ic, ih, 1, kh, ph=ph))
    add(tests, c2d(1, 1, 32, 1, 3, ph=1))
    for ih, oc, kh, ph in [(28,16,3,1),(32,32,5,2),(56,32,3,1)]:
        add(tests, c2d(1, 2, ih, oc, kh, ph=ph))
    # IC=3 (RGB)
    add(tests, c2d(1, 3, 224, 64, 7, ph=3, sh=2))
    add(tests, c2d(4, 3, 128, 32, 3, ph=1, sh=2))
    add(tests, c2d(4, 3, 32, 32, 3, ph=1))
    add(tests, c2d(4, 3, 224, 64, 7, ph=3, sh=2,
                   dt="u8:s8:u8", extra="--attr-scales=src:common:0.1+wei:common:0.1"))
    add(tests, c2d(4, 3, 32, 16, 3, ph=1, dir="BWD_D"))
    # IC=4 grouped
    add(tests, c2d(4, 4, 32, 4, 3, ph=1, g=4))

    # Asymmetric kernels
    for kh, kw, ph, pw in [
        (1,3,0,1),(3,1,1,0),(1,5,0,2),(5,1,2,0),(1,7,0,3),(7,1,3,0),
        (1,9,0,4),(9,1,4,0),(1,11,0,5),(3,7,1,3),(7,3,3,1),
        (3,5,1,2),(5,3,2,1),(2,3,0,1),(3,2,1,0),(2,4,0,1),(4,2,1,0),
    ]:
        add(tests, c2d(4, 32, 28, 64, kh, kw, ph, pw))
    # Asymmetric + stride + NXC
    add(tests, c2d(4, 32, 56, 64, 1, 3, 0, 1, sh=2, sw=2))
    add(tests, c2d(4, 32, 56, 64, 3, 1, 1, 0, sh=2, sw=2))
    add(tests, c2d(4, 32, 28, 64, 1, 3, 0, 1, stag="axb", dtag="axb"))
    add(tests, c2d(4, 32, 28, 64, 3, 1, 1, 0, stag="axb", dtag="axb"))
    # Asymmetric BWD
    add(tests, c2d(4, 32, 28, 64, 1, 3, 0, 1, dir="BWD_D"))
    add(tests, c2d(4, 32, 28, 64, 3, 1, 1, 0, dir="BWD_WB"))

    # Even kernel sizes
    for kh, kw in [(2,2),(4,4),(6,6),(2,4),(4,2),(2,3),(3,2)]:
        oh = _oh(28, kh, 0, 1, 0)
        ow = _oh(28, kw, 0, 1, 0)
        if oh and ow:
            tests.append(f"mb4ic32ih28iw28oc64oh{oh}ow{ow}kh{kh}kw{kw}")
    add(tests, c2d(4, 32, 56, 64, 2, 2, 0, 0, sh=2, sw=2))
    add(tests, c2d(4, 16, 56, 32, 4, 4, 0, 0, sh=2, sw=2))

    # Dilated 2D
    for kh, dh in [(3,1),(3,2),(3,3),(5,1),(5,2),(3,7)]:
        ph = kh + (kh-1)*dh - 1
        add(tests, c2d(4, 32, 28, 64, kh, ph=ph, dh=dh))
    add(tests, c2d(4, 32, 28, 64, 3, 3, ph=2, pw=1, dh=1, dw=0))
    add(tests, c2d(4, 32, 28, 32, 3, 3, ph=2, pw=2, dh=1, dw=1, g=32))  # dilated depthwise

    # Depthwise
    for g, ih, kh, sh, dh in [
        (32,28,3,1,0),(32,56,3,2,0),(64,56,3,1,0),(64,28,3,1,1),
        (16,112,5,2,0),(128,14,3,1,0),(256,7,3,1,0),(32,28,7,1,0),
        (64,28,5,1,0),(32,28,3,1,0),
    ]:
        ph = dh + 1 if dh > 0 else kh//2
        add(tests, c2d(4, g, ih, g, kh, ph=ph, sh=sh, dh=dh, g=g))
    add(tests, c2d(4, 32, 28, 32, 1, 3, ph=0, pw=1, g=32))
    add(tests, c2d(4, 64, 28, 64, 3, ph=1, sh=2, g=64, dt="u8:s8:u8",
                   extra="--attr-scales=src:common:0.1+wei:common:0.1"))

    # Large kernels
    for kh in [9,11,13,15]:
        add(tests, c2d(1, 16, 32, 32, kh, ph=kh//2))
        add(tests, c2d(2, 16, 32, 32, kh, ph=kh//2))

    # Spatial edge cases
    add(tests, c2d(4, 64, 1, 128, 1))
    add(tests, c2d(4, 256, 1, 512, 1))
    add(tests, c2d(1, 3, 512, 16, 3, ph=1))
    for ih in [27,13,7]:
        add(tests, c2d(4, 32, ih, 64, 3, ph=1))
    # Stride > kernel
    add(tests, c2d(4, 32, 112, 64, 3, sh=4, sw=4))
    add(tests, c2d(4, 32, 56, 64, 1, sh=4, sw=4))
    # Asymmetric stride
    add(tests, c2d(4, 32, 56, 64, 3, 3, ph=1, pw=1, sh=2, sw=1))
    add(tests, c2d(4, 32, 28, 64, 3, 3, ph=1, pw=1, sh=1, sw=2))

    # Data types
    for dt, ih in [("f32:f32:f32",28),("bf16:bf16:bf16",28),("f16:f16:f16",28),
                   ("f8_e5m2:f8_e5m2:f8_e5m2",14),("f16:f32:f16",28)]:
        add(tests, c2d(4, 32, ih, 64, 3, ph=1, dt=dt))
    add(tests, c2d(4, 32, 28, 64, 3, ph=1, dt="u8:s8:f32", bia="f32",
                   extra="--attr-scales=src:common:0.1+wei:common:0.1"))

    # NXC (axb) layout
    for ic, oc, ih, kh in [(32,64,28,3),(64,128,14,3),(16,32,56,3),(128,256,7,3)]:
        add(tests, c2d(4, ic, ih, oc, kh, ph=kh//2, stag="axb", dtag="axb"))
        add(tests, c2d(4, ic, ih, oc, kh, ph=kh//2, sh=2, stag="axb", dtag="axb"))

    # BWD coverage
    for dir_ in ["BWD_D","BWD_WB","BWD_W"]:
        add(tests, c2d(4, 32, 28, 64, 3, ph=1, dir=dir_))
        add(tests, c2d(4, 32, 56, 64, 3, ph=1, sh=2, dir=dir_))

    # Post-ops
    for po in ["gelu_tanh","swish:1","mish","hardswish","sum+relu","relu","gelu_erf"]:
        add(tests, c2d(4, 32, 28, 64, 3, ph=1, postop=po))

    # Grouped conv
    for g, ic, oc, kh in [(2,32,64,3),(4,64,128,3),(8,64,64,3),(16,64,64,3)]:
        add(tests, c2d(4, ic, 28, oc, kh, ph=kh//2, g=g))

    # More quantized variants
    for dt in ["u8:s8:s32","s8:s8:s8","u8:s8:u8"]:
        add(tests, c2d(4, 32, 28, 64, 3, ph=1, dt=dt,
                       extra="--attr-scales=src:common:0.1+wei:common:0.1"))
        add(tests, c2d(4, 32, 28, 64, 1, dt=dt,
                       extra="--attr-scales=src:common:0.1+wei:common:0.1"))

    # More BWD with diverse kernels and sizes
    for (ic, oc, ih, kh, ph, sh, dir_) in [
        (16, 32, 28, 3, 1, 1, "BWD_D"),  (32, 64, 14, 3, 1, 1, "BWD_D"),
        (64, 128, 7, 3, 1, 1, "BWD_WB"), (16, 32, 28, 5, 2, 1, "BWD_D"),
        (32, 64, 14, 1, 0, 1, "BWD_WB"), (64, 128, 28, 3, 1, 2, "BWD_D"),
    ]:
        add(tests, c2d(4, ic, ih, oc, kh, ph=ph, sh=sh, dir=dir_))

    # More NXC with various dtypes
    for dt in ["f32:f32:f32","f16:f16:f16","bf16:bf16:bf16"]:
        add(tests, c2d(4, 32, 28, 64, 3, ph=1, stag="axb", dtag="axb", dt=dt))
    add(tests, c2d(4, 32, 28, 64, 1, stag="axb", dtag="axb"))

    # Depthwise with bias
    for g, kh, ph in [(32,3,1),(64,5,2),(32,7,3)]:
        add(tests, c2d(4, g, 28, g, kh, ph=ph, g=g, bia="f32"))

    # Larger batch sizes
    for mb in [16, 32, 64, 128]:
        add(tests, c2d(mb, 32, 28, 64, 3, ph=1))

    # Spatial combinations for 3D conv (more)
    for (id_, kd) in [(16,3),(32,3),(8,1),(4,3),(2,3),(16,1)]:
        add(tests, c3d(2, 32, id_, 7, 7, 64, kd, 3, 3, pd=kd//2, ph=1, pw=1))

    # 1D with more kernel sizes
    for kw in [25, 31, 33]:
        add(tests, c1d(1, 16, 64, 32, kw, pw=kw//2))

    # Conv with f8 dtype
    add(tests, c2d(4, 32, 28, 64, 3, ph=1, dt="f8_e4m3:f8_e4m3:f32"))
    add(tests, c2d(4, 32, 14, 64, 1, dt="f8_e5m2:f8_e5m2:f16"))

    # Checkerboard: odd input sizes with even kernel
    for (ic, oc, ih, kh) in [(32,64,27,4),(32,64,13,2),(64,128,7,4)]:
        oh = _oh(ih, kh, 0, 1, 0)
        if oh:
            tests.append(f"mb4ic{ic}ih{ih}oc{oc}oh{oh}kh{kh}")

    # Pointwise on large spatial (cheap but important)
    for ih in [112, 224, 56, 14]:
        add(tests, c2d(4, 64, ih, 128, 1))
        add(tests, c2d(4, 128, ih, 256, 1))

    # Fill to 250: explicit valid conv shapes not covered above
    _conv_fill = [
        # Depthwise with dilation on various group sizes
        c2d(4, 32, 56, 32, 3, ph=2, dh=1, g=32),  # depthwise dh=1 on 56
        c2d(4, 64, 56, 64, 3, ph=2, dh=1, g=64),
        c2d(4, 128, 56, 128, 3, ph=1, g=128),
        c2d(4, 32, 28, 32, 5, ph=2, g=32),
        # Grouped 2D with various g
        c2d(4, 32, 28, 32, 3, ph=1, g=32),    # g=ic=oc (depthwise)
        c2d(4, 16, 28, 16, 5, ph=2, g=16),
        c2d(4, 64, 14, 64, 3, ph=1, g=64),
        c2d(4, 48, 14, 96, 1, g=16),           # grouped 1x1
        c2d(4, 64, 14, 128, 3, ph=1, g=4),
        c2d(4, 128, 7, 128, 3, ph=1, g=8),
        # 1D with NXC
        c1d(4, 32, 56, 64, 3, pw=1, stag="axb"),
        c1d(1, 16, 56, 32, 5, pw=2, stag="axb"),
        # 3D mixed shapes
        c3d(1, 32, 4, 14, 14, 64, 3, 3, 3, pd=1, ph=1, pw=1, dt="f16:f16:f16"),
        c3d(2, 16, 8, 7, 7, 32, 3, 3, 3, pd=1, ph=1, pw=1, dir="BWD_D"),
        # Asymmetric 3D kernels
        c3d(2, 16, 8, 14, 14, 32, 1, 1, 3, pw=1),
        c3d(2, 16, 8, 14, 14, 32, 1, 3, 1, ph=1),
        # Per-oc scales conv
        c2d(4, 32, 28, 64, 3, ph=1, dt="u8:s8:f32",
            extra="--attr-scales=src:common:0.1+wei:per_oc"),
        c2d(4, 32, 56, 64, 3, ph=1, sh=2, dt="u8:s8:u8",
            extra="--attr-scales=src:common:0.1+wei:per_oc"),
        # f16 + bia
        c2d(4, 32, 28, 64, 3, ph=1, dt="f16:f16:f16", bia="f16"),
        c2d(4, 64, 14, 128, 1, dt="bf16:bf16:bf16", bia="f32"),
        # Extra IC=1/2/3
        c2d(4, 1, 56, 32, 3, ph=1),
        c2d(4, 2, 56, 32, 3, ph=1),
        c2d(4, 3, 56, 32, 3, ph=1),
    ]
    for t in _conv_fill:
        if t:
            tests.append(t)

    return tests[:250]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    matmul_tests = build_matmul_tests()
    conv_tests   = build_conv_tests()

    matmul_path = os.path.join(args.output_dir, "matmul_blackbox.txt")
    conv_path   = os.path.join(args.output_dir, "conv_blackbox.txt")

    with open(matmul_path, "w") as f:
        for t in matmul_tests:
            f.write(f"--reset {t}\n")
    with open(conv_path, "w") as f:
        for t in conv_tests:
            f.write(f"--reset {t}\n")

    print(f"matmul: {len(matmul_tests)} tests → {matmul_path}")
    print(f"conv:   {len(conv_tests)} tests → {conv_path}")


if __name__ == "__main__":
    main()
