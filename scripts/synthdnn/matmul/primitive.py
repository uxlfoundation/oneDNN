################################################################################
# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import itertools
import fnmatch


class Dims:
    def __init__(self, b, m, n, k):
        # b is a list due to variable size
        self.b = b
        self.m = m
        self.n = n
        self.k = k

    def __str__(self):
        a_dims = self.b + [self.m, self.k]
        b_dims = self.b + [self.k, self.n]
        a_str = "x".join([str(x) for x in a_dims])
        b_str = "x".join([str(x) for x in b_dims])
        return f"{a_str}:{b_str}"

    def __eq__(self, other):
        return (self.b, self.m, self.n, self.k) == (
            other.b,
            other.m,
            other.n,
            other.k,
        )

    def __hash__(self):
        return hash((self.b, self.m, self.n, self.k))


class Layouts:
    class Layout:
        def __init__(self, layout):
            self.A, self.B, self.C = layout.split(":")

        def benchdnn_str(self):
            return f"--stag={self.A} --wtag={self.B} --dtag={self.C}"

    def __init__(self, layouts, ndims):
        if layouts == "all":
            self.values = self.supported(ndims)
        else:
            self.values = [self.Layout(x) for x in layouts.split(",")]

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported(ndims):
        if ndims < 2 or ndims > 6:
            raise RuntimeError(f"No support for ndims={ndims}")
        dims_base = "abcdef"
        gemm_kn = dims_base[ndims - 1]
        gemm_mk = dims_base[ndims - 2]
        perms = [
            "".join(p)
            for p in itertools.permutations(dims_base[:ndims])
            if p[-1] == gemm_kn or p[-1] == gemm_mk
        ]
        perms.insert(0, "any")
        return [
            Layouts.Layout(f"{a}:{b}:{c}")
            for a, b, c in itertools.product(perms, perms, perms)
            if c == "any" or c[-1] == gemm_kn
        ]


class Types:
    class Type:
        def __init__(self, type_str):
            s = type_str.split("(")
            self.A, self.B, self.C = s[0].split(":")
            self.A, self.B, self.C = self.index_match(self.A, self.B, self.C)
            if len(s) < 2:
                self.mode = None
            else:
                self.mode = s[1].strip(")")

        @staticmethod
        def index_match(A, B, C):
            dts = [A, B, C]
            for i, dt in enumerate(dts):
                if fnmatch.fnmatch(dt, r"%[0-9]"):
                    idx = int(dt[1:])
                    dts[i] = dts[idx]
            return dts

        def __str__(self):
            mode_str = ""
            if self.mode:
                mode_str = f"({self.mode})"
            return f"{self.A}:{self.B}:{self.C}{mode_str}"

        def benchdnn_str(self):
            mode_str = ""
            if not self.mode is None:
                mode_str = f"--attr-fpmath={self.mode}"
            return f"--dt={self.A}:{self.B}:{self.C} {mode_str}"

        def matches(self, A, B, C):
            dts = [self.A, self.B, self.C]
            patterns = [A, B, C]

            # XXX: Careful here - we want indexing operators to apply to self
            # (not A/B/C), so that we aren't matching any wildcards/globs.
            res = True
            for i in range(3):
                # Resolve index operators
                if fnmatch.fnmatch(patterns[i], r"%[0-9]"):
                    idx = int(patterns[i][1:])
                    res = res and dts[i] == dts[idx]
                else:
                    # Resolve globbing matches
                    res = res and fnmatch.fnmatch(dts[i], patterns[i])
            return res

        def __eq__(self, other):
            return (self.A, self.B, self.C, self.mode) == (
                other.A,
                other.B,
                other.C,
                other.mode,
            )

        def __hash__(self):
            return hash(self.__str__())

    @staticmethod
    def expand(A, B, C, cases=None):
        if cases is None:
            cases = Types.supported()

        return set(case for case in cases if case.matches(A, B, C))

    def __init__(self, types: str):
        self.values = set()
        for x in types.split(","):
            if x.count(":") == 0:
                configs = [f"{x}:*:*", f"*:{x}:*", f"*:*:{x}"]
            else:
                configs = [x]

            for config in configs:
                A, B, C = config.split(":")
                self.values = self.values.union(Types.expand(A, B, C))

    def __str__(self):
        return ",".join([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported():
        support_matrix = [
            [["f64"], ["f64"], ["f64"]],
            [["f32"], ["f32"], ["f32"]],
            [["f32"], ["u8", "s8"], ["f32", "f16", "bf16"]],
            [
                ["f16", "bf16"],
                ["%0", "u8", "s8", "u4", "s4"],
                ["f32", "%0", "u8", "s8"],
            ],
            [["u8", "s8"], ["u8", "s8", "u4", "s4"], ["f32", "bf16", "f16", "s32", "u8", "s8"]],
            [
                ["f8_e5m2", "f8_e4m3"],
                ["f8_e5m2", "f8_e4m3"],
                ["f32", "bf16", "f16", "f8_e5m2", "f8_e4m3"],
            ],
            [
                ["f16", "bf16"],
                ["f8_e5m2", "f8_e4m3"],
                ["f32", "bf16", "f16", "f8_e5m2", "f8_e4m3"],
            ],
        ]

        def is_int_type(t):
            return t in ["u4", "s4", "u8", "s8", "s32"]

        def get_accumulator(wei):
            if is_int_type(wei):
                return "s32"
            if wei == "f64":
                return "f64"
            return "f32"

        def get_fpmath_modes(src, wei, dst):
            src, wei, dst = Types.Type.index_match(src, wei, dst)
            if get_accumulator(wei) == "f32":
                ret = [""]
                if "f32" in [src, wei]:
                    ret.append("(tf32)")
                if "f32" in [src, wei] and not "f16" in [src, wei]:
                    ret.append("(bf16)")
                if "f32" in [src, wei] and not "bf16" in [src, wei]:
                    ret.append("(f16)")
                return ret
            if (
                get_accumulator(wei) == "s32"
                and not is_int_type(dst)
                and not is_int_type(src)
            ):
                ret = []
                if "f32" in [src, wei]:
                    ret.append("(strict:true)")
                    ret.append("(tf32:true)")
                if "f16" not in [src, wei]:
                    ret.append("(bf16:true)")
                if "bf16" not in [src, wei]:
                    ret.append("(f16:true)")
                return ret
            return [""]

        out = []
        for c in support_matrix:
            for src, wei, dst in itertools.product(c[0], c[1], c[2]):
                for math in get_fpmath_modes(src, wei, dst):
                    out.append(Types.Type(f"{src}:{wei}:{dst}{math}"))
        return out


# Kind represents problem parameters that do not make sense to consider
# in aggregate for optimization purposes as these features require significant
# changes within generated implementations or the implementation dispatching.
class Kind:
    def __init__(self, layout, type):
        self.layout = layout
        self.type = type

    def benchdnn_str(self):
        return f"{self.layout.benchdnn_str()} {self.type.benchdnn_str()}"


class Primitive:
    def __init__(self, kind, dims, quantization):
        self.kind: Kind = kind
        self.dims = dims
        self.quantization = quantization

    def benchdnn_str(self, optional_args = ""):
        return f"{optional_args}{self.kind.benchdnn_str()} {self.quantization} {self.dims}"
