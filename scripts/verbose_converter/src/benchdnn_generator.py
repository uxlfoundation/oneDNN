################################################################################
# Copyright 2020-2024 Intel Corporation
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

from collections import defaultdict
from typing import Any, Dict, List, Optional

def everyone_is(list, value="None"):
    if [value == "None"]:
        value = list[0]
    return [e for e in list if e != value] == []


primitives_with_algs = (
    "binary",
    "convolution",
    "deconvolution",
    "eltwise",
    "lrn",
    "pooling",
    "reduction",
    "resampling",
    "rnn",
)


def alg_remove_primitive(alg):
    for p in primitives_with_algs:
        if alg.find(p) != -1:
            alg = alg[(alg.find(p) + len(p) + 1) :]
    return alg


def convert_driver(prop_kind):
    driver = {
        "batch_normalization": "bnorm",
        "binary": "binary",
        "brgemm": "brgemm",
        "concat": "concat",
        "convolution": "conv",
        "deconvolution": "deconv",
        "eltwise": "eltwise",
        "group_normalization": "gnorm",
        "inner_product": "ip",
        "layer_normalization": "lnorm",
        "lrn": "lrn",
        "matmul": "matmul",
        "pooling": "pool",
        "prelu": "prelu",
        "reduction": "reduction",
        "reorder": "reorder",
        "resampling": "resampling",
        "rnn": "rnn",
        "shuffle": "shuffle",
        "softmax": "softmax",
        "sum": "sum",
    }.get(prop_kind)
    return driver


def convert_engine(engine):
    return f"--engine={engine}"


def convert_dir(entry):
    # get base direction
    dir = {
        "forward_training": "FWD_D",
        "forward_inference": "FWD_I",
        "backward_data": "BWD_D",
        "backward_weights": "BWD_W",
        "backward": "BWD_DW",
    }.get(entry["prop_kind"])

    if not dir:
        return ""

    found_bias = [
        e for e in entry["mds"] if "bia" == e["arg"] and e["data_type"] != "undef"
    ]
    dir = "FWD_B" if "FWD" in dir and found_bias else dir
    dir = "BWD_WB" if dir == "BWD_W" and found_bias else dir
    if entry["prim_kind"] == "rnn":
        return f"--prop={dir}"
    else:
        return f"--dir={dir}"


def convert_aux(entry):
    if entry.get("aux") != None:
        alg = entry["aux"]["alg"] if entry["aux"].get("alg") != None else ""
        pk = entry["prim_kind"]
        if pk == "convolution":
            str = ""
            alg = alg_remove_primitive(alg)
            algs = {"winograd": "WINO", "direct": "direct"}
            alg = algs.get(alg)
            if alg != None:
                str = f"--alg={alg}"
            return str
        if pk == "eltwise":
            alpha = entry["aux"]["alpha"]
            beta = entry["aux"]["beta"]
            alg += f" --alpha={alpha} --beta={beta}"
            return f"--alg={alg}"
        elif pk == "concat":
            axis = entry["aux"]["axis"]
            return f"--axis={axis}"
        elif pk in [
            "batch_normalization",
            "layer_normalization",
            "group_normalization",
        ]:
            flags = entry["aux"]["flags"]
            return f"--flags={flags}"
        elif pk == "lrn":
            str = ""
            alg = alg_remove_primitive(alg)
            algs = {"across_channels": "ACROSS", "within_channel": "WITHIN"}
            alg = algs.get(alg)
            if alg != None:
                str = f"--alg={alg}"
            return str
        elif pk == "reduction":
            p = entry["aux"]["p"]
            eps = entry["aux"]["eps"]
            alg += f" --p={p} --eps={eps}"
            return f"--alg={alg}"
        elif pk == "rnn":
            str = ""
            algs = {
                "vanilla_rnn": "VANILLA_RNN",
                "vanilla_lstm": "VANILLA_LSTM",
                "vanilla_gru": "VANILLA_GRU",
                "vanilla_augru": "VANILLA_AUGRU",
                "lbr_gru": "LBR_GRU",
                "lbr_augru": "LBR_AUGRU",
            }
            alg = algs.get(alg)
            if alg != None:
                str += f"--alg={alg}"
            ir_dir = entry["aux"]["direction"]
            dirs = {
                "unidirectional_left2right": "left2right",
                "unidirectional_right2left": "right2left",
                "bidirectional_sum": "sum",
                "bidirectional_concat": "concat",
            }
            dir = dirs.get(ir_dir)
            if dir is not None:
                str += f" --direction={dir}"
            ir_act = entry["aux"]["activation"]
            acts = {
                "eltwise_relu": "RELU",
                "eltwise_logistic": "LOGISTIC",
                "eltwise_tanh": "TANH",
            }
            act = acts.get(ir_act)
            if act is not None:
                str += f" --activation={act}"
            flags = entry["aux"]["flags"]
            if flags is not None:
                str += f" --flags={flags}"
            return str
        elif pk == "shuffle":
            axis = entry["aux"]["axis"]
            group = entry["aux"]["group"]
            return f"--axis={axis} --group={group}"
        elif pk == "softmax":
            axis = entry["aux"]["axis"]
            return f"--alg={alg} --axis={axis}"
        elif pk == "pooling":
            return f"--alg={alg}"
        elif pk == "matmul":
            runtime_dims_masks = (
                entry["aux"]["runtime_dims_masks"]
                if entry["aux"].get("runtime_dims_masks") != None
                else ""
            )
            return f"--runtime_dims_masks={runtime_dims_masks}"
        elif pk == "reorder":
            runtime_dim_mask = (
                entry["aux"]["runtime-dim-mask"]
                if entry["aux"].get("runtime-dim-mask") != None
                else ""
            )
            return f"--runtime-dim-mask={runtime_dim_mask}"
        elif pk == "brgemm":
            bs = entry["aux"]["bs"] if entry["aux"].get("bs") != None else ""
            beta = entry["aux"]["beta"] if entry["aux"].get("beta") != None else ""
            return f"--bs={bs} --beta={beta}"
        else:
            alg = alg_remove_primitive(alg)
            if alg != "":
                return f"--alg={alg}"
    return ""


def convert_bias_mask(mds):
    bia_mds = [md for md in mds if md["arg"] == "bia"]
    if len(bia_mds) != 0:
        bia_md = bia_mds[0]
        flags = bia_md["flags"]["value"].split("_")
        if len(flags) > 1:
            mask = flags[1][4:]
            return f"--bia_mask={mask}"
    return ""


def convert_dts(mds, prim_kind):
    def convert_dts_common(mds):
        dts = [md["data_type"] for md in mds if md["data_type"] != "undef"]
        dt = dts[0]
        return f"--dt={dt}"

    def convert_dts_cfg_rnn(mds):
        cfg = "--cfg="
        args = ["src_iter", "src_iter_c", "src_layer", "dst_iter", "dst_layer", "bias"]
        mds_strip = [md for md in mds if md["arg"] in args]
        # ws is not part of cfg
        mds_strip = [md for md in mds_strip if "ws" not in md["arg"]]
        # bias is not part of cfg
        mds_strip = [md for md in mds_strip if "bia" not in md["arg"]]
        common_dt = everyone_is([md["data_type"] for md in mds_strip])
        if common_dt and mds_strip[0]["data_type"] in ["f32", "f16"]:
            cfg += mds_strip[0]["data_type"]
        elif common_dt and mds_strip[0]["data_type"] == "bf16":
            cfg += mds_strip[0]["data_type"]
            # bias is part of cfg for bf16
            bias_md = [md for md in mds if md["arg"] == "bias"][0]
            bias_dt = bias_md["data_type"]
            if bias_dt != mds_strip[0]["data_type"]:
                cfg += bias_dt
        else:
            for arg in args:
                for md in mds_strip:
                    if md["arg"] == arg:
                        # src iter is skipped if it is f32
                        if arg == "src_iter_c" and md["data_type"] == "f16":
                            continue
                        cfg += md["data_type"]
        return cfg

    def convert_dts_all(mds):
        dts = ""
        md_args = ""
        for md in mds:
            md_arg = md["arg"][0]
            if md_args.find(md_arg) == -1:
                md_dt = md["data_type"]
                dts += f" --{md_arg}dt={md_dt}"
                md_args += md_arg
        return dts

def maybe_make_any_tag(md: ir.MemoryDescriptor):
    return "any" if "a" in md.properties else md.tag


    def convert_dts_multiple_src(mds):
        src_dts = ""
        dts = ""
        first_src = True
        for md in mds:
            md_dt = md["data_type"]
            md_arg = md["arg"]
            if md_arg == "src":
                if not first_src:
                    src_dts += f":{md_dt}"
                else:
                    src_dts += f" --{md_arg[0]}dt={md_dt}"
                    first_src = False
            else:
                if md_dt != "undef":
                    dts += f" --{md_arg[0]}dt={md_dt}"
        return src_dts + dts

    def convert_dts_with_bias(mds):
        dt = convert_dts_multiple(mds)
        mds_bias = [md for md in mds if "bia" in md["arg"]]
        if len(mds_bias) != 0:
            md_bias = mds_bias[0]
            bias_dt = md_bias["data_type"]
            dt += " " + f"--bia_dt={bias_dt}"
        return dt

    def convert_dts_with_ss(mds):
        dt = convert_dts_multiple(mds)
        mds_scale = [md for md in mds if "scale" in md["arg"]]
        mds_shift = [md for md in mds if "shift" in md["arg"]]

        if len(mds_scale) != 0:
            md_scale = mds_scale[0]
            scale_dt = md_scale["data_type"]
            dt += " " + f"--ss_dt={scale_dt}"
        elif len(mds_shift) != 0:
            md_shift = mds_shift[0]
            shift_dt = md_shift["data_type"]
            dt += " " + f"--ss_dt={shift_dt}"

        return dt

    convert_dts = {
        "batch_normalization": convert_dts_common,
        "binary": convert_dts_multiple_src,
        "brgemm": convert_dts_multiple,
        "concat": convert_dts_all,
        "convolution": convert_dts_multiple,
        "deconvolution": convert_dts_multiple,
        "eltwise": convert_dts_common,
        "inner_product": convert_dts_multiple,
        "group_normalization": convert_dts_multiple,
        "layer_normalization": convert_dts_with_ss,
        "lrn": convert_dts_common,
        "matmul": convert_dts_with_bias,
        "pooling": convert_dts_multiple,
        "prelu": convert_dts_prelu,
        "reduction": convert_dts_all,
        "reorder": convert_dts_all,
        "resampling": convert_dts_all,
        "rnn": convert_dts_cfg_rnn,
        "shuffle": convert_dts_common,
        "softmax": convert_dts_all,
        "sum": convert_dts_multiple_src,
    }

    @property
    def dts(self):
        for md in self.entry.mds:
            if md.data_type == "undef":
                continue
            return f"--dt={md.data_type}"
        return ""

    @property
    def tags(self):
        for md in self.entry.mds:
            if not md.tag:
                continue
            return f"--tag={md.tag}"  # XXX: Don't use maybe_make_any_tag
        return ""

    @property
    def flags(self):
        return ""

    def _get_nondefault_args(self, values, defaults):
        parts: List[str] = []
        pairs = list(zip(values, defaults))
        seen_nondefault = False
        for value, default in reversed(pairs):
            if value != default:
                seen_nondefault = True
            if seen_nondefault:
                parts.append(str(value))
        return list(reversed(parts))

    def _convert_dw_post_op(self, po: ir.DepthwisePostOp):
        return f"dw:{po.ksp}:{po.dst_dt}"

    def _convert_sum_post_op(self, po: ir.SumPostOp):
        values = po.scale, po.zp, po.dt
        args = self._get_nondefault_args(values, defaults=(1.0, 0, ""))
        return ":".join(["sum"] + args)

    def _convert_prelu_post_op(self, po: ir.PreLUPostOp):
        if po.mask != 0:
            return f"prelu:{self.policy(po.mask)}"
        return "prelu"

    def _convert_eltwise_post_op(self, po: ir.EltwisePostOp):
        values = po.alpha, po.beta, po.scale
        args = self._get_nondefault_args(values, defaults=(0.0, 0.0, 1.0))
        return ":".join([po.alg] + args)

    def _convert_binary_post_op(self, po: ir.BinaryPostOp):
        return f"{po.alg}:{po.dt}:{po.mask}:{po.tag}"

    @property
    def post_ops(self):
        post_ops = self.entry.exts.post_ops
        if post_ops is None:
            return ""
        results = []
        for post_op in post_ops:
            if post_op.alg == "dw":
                results.append(self._convert_dw_post_op(post_op))
            elif post_op.alg == "sum":
                results.append(self._convert_sum_post_op(post_op))
            elif post_op.alg == "prelu":
                results.append(self._convert_prelu_post_op(post_op))
            elif post_op.alg.startswith("binary"):
                results.append(self._convert_binary_post_op(post_op))
            elif post_op.alg.startswith("eltwise"):
                results.append(self._convert_eltwise_post_op(post_op))
        return "--attr-post-ops=" + "+".join(results)

    def _get_quantization(
        self,
        params: Optional[Dict[str, ir.QuantizationParam]],
        def_value: float,
        def_type: str,
    ):
        if params is None:
            return ""
        results = []
        for arg, param in params.items():
            policy = self.policy(param.mask)
            result = f"{arg}:{policy}"
            if policy == "common":
                result += f":{def_value}"
            dt = param.data_type
            groups = param.groups
            if dt != def_type or groups:
                result += f":{dt}"
            if groups:
                result += f":{groups}"
            results.append(result)
        return "+".join(results)

    @property
    def scales(self):
        params = self._get_quantization(self.entry.exts.scales, 0.5, "f32")
        return f"--attr-scales={params}"

    @property
    def zero_points(self):
        params = self._get_quantization(self.entry.exts.zero_points, 1, "s32")
        return f"--attr-zero-points={params}"

    @property
    def rounding_mode(self):
        rounding_modes = self.entry.exts.rounding_mode
        if rounding_modes is None:
            return ""
        results = []
        for arg, mode in rounding_modes.items():
            results.append(f"{arg}:{mode!s}")
        return "--attr-rounding-mode=" + "+".join(results)

    scratchpad_mode = attribute_flag("scratchpad")
    fpmath_mode = attribute_flag("fpmath")
    acc_mode = attribute_flag("acc")

    @property
    def dropout(self):
        dropout = self.entry.exts.dropout
        if dropout is None:
            return ""
        # Use default p=0.5 and seed=12345 since those values are user data and
        # can't be obtained properly.
        result = "0.5:12345"
        if dropout.tag:
            result += f":{dropout.tag}"
        return f"--attr-dropout={result}"

    deterministic = attribute_flag("deterministic")

    @property
    def attrs(self):
        attrs = (
            self.post_ops,
            self.scales,
            self.zero_points,
            self.scratchpad_mode,
            self.fpmath_mode,
            self.acc_mode,
            self.rounding_mode,
            self.dropout,
            self.deterministic,
        )
        return " ".join(attr for attr in attrs if attr)

    @property
    def aux(self):
        alg = self._get_alg()
        if alg is not None:
            return f"--alg={alg}"
        return ""

    @property
    def shapes(self):
        return self.entry.shapes


class AlgorithmMixin:
    entry: ir.Entry

    def _get_alg(self):
        alg = self.entry.aux.get("alg")
        if alg is None:
            return None
        return alg.split(self.entry.prim_kind, 1)[1][1:]


class MultiSourceMixin:
    entry: ir.Entry

    @property
    def dts(self):
        src_dts: List[str] = []
        other_dts: Dict[str, str] = {}
        for md in self.entry.mds:
            dt = md.data_type
            if md.arg == "src":
                src_dts.append(dt)
            elif dt != "undef":
                other_dts[md.arg[0]] = dt
        sdt_flags = "--sdt=" + ":".join(src_dts)
        other_dt_flags = " ".join(f"--{k}dt={v}" for k, v in other_dts.items())
        return f"{sdt_flags} {other_dt_flags}".strip()

    @property
    def tags(self):
        src_tags: List[str] = []
        other_tags: Dict[str, str] = {}
        for md in self.entry.mds:
            if md.arg == "src":
                src_tags.append(maybe_make_any_tag(md))
            elif md.tag:
                other_tags[md.arg[0]] = maybe_make_any_tag(md)
        stag_flags = "--stag=" + ":".join(src_tags)
        other_tag_flags = " ".join(
            f"--{k}tag={v}" for k, v in other_tags.items()
        )
        return f"{stag_flags} {other_tag_flags}".strip()


class CommonDataTypeMixin:
    entry: ir.Entry

    @property
    def dts(self):
        dts: Dict[str, str] = {}
        for md in self.entry.mds:
            c = md.arg[0]
            if c in dts:
                continue
            dts[c] = md.data_type
        return " ".join(f"--{k}dt={v}" for k, v in dts.items())


class TagTripletMixin:
    entry: ir.Entry

    @property
    def tags(self):
        md_map = {md.arg: md for md in self.entry.mds}
        has_fused_dw = "src_fused" in md_map
        # Fused dw defines dst tag by src_fused argument
        dst_name = "src_fused" if has_fused_dw else "dst"
        tags = []
        if "src" in md_map:
            md = md_map["src"]
            tag = maybe_make_any_tag(md)
            tags.append(f"--stag={tag}")
        if "wei" in md_map:
            md = md_map["wei"]
            tag = maybe_make_any_tag(md)
            # pass wtag any for cases with compensation
            if str(md.flags.value) != "f0":
                tag = "any"
            tags.append(f"--wtag={tag}")
        if dst_name in md_map:
            md = md_map[dst_name]
            tag = maybe_make_any_tag(md)
            tags.append(f"--dtag={tag}")
        return " ".join(tags)


class StridesMixin(TagTripletMixin):
    @property
    def tags(self):
        tags = []
        strides = []

        def add_strides_or_tag(arg, md):
            tag = maybe_make_any_tag(md)
            if arg == "wei" and str(md.flags.value) != "f0":
                tag = "any"
            if tag != "any" and tag.lower() == tag and md.strides:
                strides.append(md.strides)
            else:
                tags.append(f"--{arg[0]}tag={tag}")
                strides.append("")

        md_map = {md.arg: md for md in self.entry.mds}
        args = "src", "wei", "dst"
        for arg in args:
            if arg not in md_map:
                continue
            md = md_map[arg]
            add_strides_or_tag(arg, md)
        stride_flag = "--strides=" + ":".join(strides)
        return " ".join(tags + [stride_flag])


class MultiDataTypeMixin:
    entry: ir.Entry

    @property
    def dts(self):
        dt_map = {md.arg: md.data_type for md in self.entry.mds}
        # Fused dw defines dst_dt by src_fused argument
        has_fused_dw = "src_fused" in dt_map
        dst_name = "src_fused" if has_fused_dw else "dst"
        dts = [
            dt_map.get("src", ""),
            dt_map.get("wei", ""),
            dt_map.get(dst_name, ""),
        ]
        return "--dt=" + ":".join(dt for dt in dts if dt)


class NormalizationMixin:
    entry: ir.Entry

    @property
    def aux(self):
        flags = self.entry.aux.get("flags")
        if flags is not None:
            return f"--flags={flags}"
        return ""


class BatchNormalizationConverter(NormalizationMixin, Converter):
    driver: str = "bnorm"


class BinaryConverter(AlgorithmMixin, MultiSourceMixin, Converter):
    driver: str = "binary"

    @property
    def shapes(self):
        return self.entry.shapes.split(" ", 1)[0]


class BRGEMMConverter(MultiDataTypeMixin, Converter):
    driver: str = "brgemm"

    @property
    def aux(self):
        bs = self.entry.aux.get("bs", "")
        beta = self.entry.aux.get("beta", "")
        return f"--bs={bs} --beta={beta}"


class ConcatConverter(CommonDataTypeMixin, MultiSourceMixin, Converter):
    driver: str = "concat"

    @property
    def aux(self):
        axis = self.entry.aux.get("axis")
        if axis is None:
            return ""
        return f"--axis={axis}"


class ConvolutionConverter(
    AlgorithmMixin,
    TagTripletMixin,
    MultiDataTypeMixin,
    Converter,
):
    driver: str = "conv"

    @property
    def aux(self):
        alg = self._get_alg()
        if alg is not None:
            return f"--alg={alg}"
        return ""


class DeconvolutionConverter(ConvolutionConverter):
    driver: str = "deconv"


class EltwiseConverter(Converter):
    driver: str = "eltwise"

    @property
    def aux(self):
        alpha = self.entry.aux.get("alpha")
        beta = self.entry.aux.get("beta")
        flags = [f"--alg={self._get_alg()}"]
        if alpha is not None:
            flags.append(f"--alpha={alpha}")
        if beta is not None:
            flags.append(f"--beta={beta}")
        return " ".join(flags)


class GroupNormalizationConverter(
    MultiDataTypeMixin,
    BatchNormalizationConverter,
):
    driver: str = "gnorm"

    # --tag=SRC_TAG[:WEI_TAG][:DST_TAG]
    @property
    def tags(self):
        tag_map = {md.arg: maybe_make_any_tag(md) for md in self.entry.mds}
        args = "src", "wei", "dst"
        tags = [tag_map[arg] for arg in args if arg in tag_map]
        return "--tag=" + ":".join(tags)


class InnerProductConverter(TagTripletMixin, MultiDataTypeMixin, Converter):
    driver: str = "ip"


class LayerNormalizationConverter(GroupNormalizationConverter):
    driver: str = "lnorm"

    @property
    def dts(self):
        dts = super().dts
        shift_flag = None
        for md in self.entry.mds:
            if "scale" in md.arg:
                return f"{dts} --ss_dt={md.data_type}".strip()
            if "shift" in md.arg and shift_flag is None:
                shift_flag = f"--ss_dt={md.data_type}"
        if shift_flag is not None:
            return f"{dts} {shift_flag}".strip()
        return dts

    def convert_tags_multiple_src(mds):
        src_tags = ""
        tags = ""
        first_src = False
        for md in mds:
            md_tag = md["tag"]
            md_arg = md["arg"]
            if md_arg == "src":
                if first_src:
                    if "a" in md["properties"]:
                        src_tags += f":any"
                    else:
                        src_tags += f":{md_tag}"
                else:
                    if "a" in md["properties"]:
                        src_tags += f" --{md_arg[0]}tag=any"
                    else:
                        src_tags += f" --{md_arg[0]}tag={md_tag}"
                    first_src = True
            else:
                if md_tag != "":
                    if "a" in md["properties"]:
                        tags += f" --{md_arg[0]}tag=any"
                    else:
                        tags += f" --{md_arg[0]}tag={md_tag}"
        return src_tags + tags

    def convert_tags_prelu(mds):
        # FIXME: fix benchdnn input template
        data_md = [md for md in mds if "data" in md["arg"]][0]
        weights_md = [md for md in mds if "wei" in md["arg"]][0]

        data_tag = data_md["tag"]
        if "a" in data_md["properties"]:
            data_tag = "any"
        weights_tag = weights_md["tag"]
        if "a" in weights_md["properties"]:
            weights_tag = "any"

        return f" --stag={data_tag}:{weights_tag}"

    def convert_tags_rnn(mds):
        tags = "--tag="
        with_proj = ""
        with_peep = ""
        skip_colon = True

        # Tags for backward are driven by diff tensors, query them instead of
        # forward tensors. Latter will always have `any` format.
        has_diff_tensors = False
        for md in mds:
            if md["arg"].find("diff") != -1:
                has_diff_tensors = True

        for md in mds:
            md_arg = md["arg"]
            md_tag = md["tag"]
            if has_diff_tensors == True:
                if md_arg in ["diff_src_layer", "diff_wei_layer", "diff_dst_layer"]:
                    if not skip_colon:
                        tags += f":"
                    if "a" in md["properties"]:
                        tags += f"any"
                    else:
                        tags += f"{md_tag}"
                    skip_colon = False
            else:
                if md_arg in ["src_layer", "wei_layer", "dst_layer"]:
                    if not skip_colon:
                        tags += f":"
                    if "a" in md["properties"]:
                        tags += f"any"
                    else:
                        tags += f"{md_tag}"
                    skip_colon = False

            if md_arg == "wei_proj" and md_tag != "undef":
                with_proj = " --with-projection=true"
            if md_arg == "wei_peephole" and md_tag != "undef":
                with_peep = " --with-peephole=true"

        return tags + with_proj + with_peep

    def convert_tags_lnorm(mds):
        tag = convert_tags_multiple(mds)
        stat_md = ""
        for md in mds:
            if md["arg"] == "stats":
                stat_tag = md["tag"]

        return f"{tag} --stat_tag={stat_tag}"

    cvt_tags = {
        "batch_normalization": convert_tags_common,
        "binary": convert_tags_multiple_src,
        "concat": convert_tags_multiple_src,
        "convolution": convert_tags_all,
        "deconvolution": convert_tags_all,
        "eltwise": convert_tags_common,
        "inner_product": convert_tags_all,
        "group_normalization": convert_tags_multiple,
        "layer_normalization": convert_tags_lnorm,
        "lrn": convert_tags_common,
        "matmul": convert_tags_and_strides,
        "pooling": convert_tags_common,
        "prelu": convert_tags_prelu,
        "reduction": convert_tags_all,
        "reorder": convert_tags_and_strides,
        "resampling": convert_tags_common,
        "rnn": convert_tags_rnn,
        "shuffle": convert_tags_common,
        "softmax": convert_tags_all,
        "sum": convert_tags_multiple_src,
    }


class LRNConverter(AlgorithmMixin, Converter):
    driver: str = "lrn"

    @property
    def aux(self):
        alg = self._get_alg()
        algs = {"across_channels": "ACROSS", "within_channel": "WITHIN"}
        if alg not in algs:
            return ""
        return f"--alg={algs[alg]}"


class MatmulConverter(StridesMixin, MultiDataTypeMixin, Converter):
    driver: str = "matmul"

    @staticmethod
    def _get_policies():
        return "common", "per_oc", "per_ocic"

    @staticmethod
    def _get_policy_map():
        return 0, 1, 1, 2, 1, 3, 2, 3, 1, 3, 3, 3, 2

    @property
    def bias_mask(self):
        for md in self.entry.mds:
            if md.arg != "bia":
                continue
            if "_" in md.flags.value:
                mask = md.flags.value.split("_")[1][4:]
                return f"--bia_mask={mask}"
        return ""

    @property
    def dts(self):
        dts = super().dts
        for md in self.entry.mds:
            if md.arg != "bia":
                continue
            return f"{dts} --bia_dt={md.data_type}".strip()
        return dts

    @property
    def aux(self):
        rt_dim_masks = self.entry.aux.get("runtime_dims_masks", "")
        return f"--runtime_dims_masks={rt_dim_masks}"


class PoolingConverter(MultiDataTypeMixin, Converter):
    driver: str = "pool"

    @property
    def aux(self):
        return f"--alg={self._get_alg()}"


class PreLUConverter(Converter):
    driver: str = "prelu"

    @property
    def dts(self):
        data_dt, wei_dt = "", ""
        for md in self.entry.mds:
            if "data" in md.arg and not data_dt:
                data_dt = md.data_type
            if "wei" in md.arg and not wei_dt:
                wei_dt = md.data_type
            if data_dt and wei_dt:
                break
        return f"--sdt={data_dt}:{wei_dt}"

    @property
    def tags(self):
        data_tag, wei_tag = "", ""
        for md in self.entry.mds:
            if "data" in md.arg and not data_tag:
                data_tag = maybe_make_any_tag(md)
            if "wei" in md.arg and not wei_tag:
                wei_tag = maybe_make_any_tag(md)
            if data_tag and wei_tag:
                break
        return f"--stag={data_tag}:{wei_tag}"


class ReductionConverter(
    AlgorithmMixin,
    TagTripletMixin,
    CommonDataTypeMixin,
    Converter,
):
    driver: str = "reduction"

    @property
    def aux(self):
        p = self.entry.aux.get("p")
        eps = self.entry.aux.get("eps")
        args = [f"--alg={self._get_alg()}"]
        if p is not None:
            args.append(f"--p={p}")
        if eps is not None:
            args.append(f"--eps={eps}")
        return " ".join(args)


class ReorderConverter(StridesMixin, CommonDataTypeMixin, Converter):
    driver: str = "reorder"

    def _convert_flag(self, prefix, md: ir.MemoryDescriptor):
        flags = []
        fields = md.flags
        if fields.s8_comp_mask is not None:
            flags.append(f"s8s8_comp:{fields.s8_comp_mask}")
        if fields.zp_comp_mask is not None:
            flags.append(f"zp_comp:{fields.zp_comp_mask}")
        if flags:
            return f"--{prefix}flag=" + "+".join(flags)
        return ""

    start_idx += len(type) + 1
    end_symbol = ";"
    if type == "post_ops":
        start_idx += 1
        end_symbol = "'"
    end_idx = attrs.find(end_symbol, start_idx)
    if type == "post_ops":
        start_idx -= 1
        end_idx += 1
    return attrs[start_idx:end_idx]


def convert_scale_policy(value, prim_kind):
    if prim_kind == "reorder":
        masks = {0: "common", 1: "per_dim_0", 2: "per_dim_1", 3: "per_dim_01"}
    elif prim_kind == "matmul":
        masks = {
            0: "common",
            1: "per_oc",
            2: "per_oc",
            3: "per_ocic",
            4: "per_oc",
            6: "per_ocic",
            8: "per_oc",
            12: "per_ocic",
        }
    else:
        masks = {0: "common", 1: "per_oc", 2: "per_oc", 3: "per_oc"}

    mask = masks.get(int(value))
    if mask:
        return mask
    # this is a workaround for tensors with mask more than 4
    return "per_tensor"


def convert_zp_policy(value, prim_kind):
    if prim_kind == "matmul":
        masks = {
            0: "common",
            2: "per_oc",
            3: "per_ocic",
            4: "per_oc",
            6: "per_ocic",
            12: "per_ocic",
        }
    else:
        masks = {0: "common", 2: "per_dim_1"}
    mask = masks.get(int(value))
    if mask:
        return mask
    # this is a workaround for tensors with mask more than 4
    return "per_tensor"


def convert_post_ops(post_ops, prim_kind):
    def convert_binary_post_op(post_op):
        po = post_op["alg"] + ":" + post_op["dt"] + ":" + post_op["mask"]
        if post_op["tag"] != None:
            po += ":" + post_op["tag"]
        return po

    def convert_dw_post_op(post_op):
        po = post_op["alg"] + ":" + post_op["ksp"] + ":" + post_op["dst_dt"]
        return po

    def convert_eltwise_post_op(post_op):
        benchdnn_p_op = post_op["alg"]
        alpha = post_op["alpha"]
        beta = post_op["beta"]
        scale = post_op["scale"]
        if alpha != "1.0":
            benchdnn_p_op += ":" + alpha
            if beta != "0.0":
                benchdnn_p_op += ":" + beta
                if alpha != "1.0":
                    benchdnn_p_op += ":" + scale
        return benchdnn_p_op

    def convert_sum_post_op(post_op):
        benchdnn_p_op = post_op["alg"]
        if post_op["scale"] != 1.0:
            benchdnn_p_op += ":" + post_op["scale"]
            if post_op["zp"] != 0:
                benchdnn_p_op += ":" + post_op["zp"]
                if post_op["dt"] != "":
                    benchdnn_p_op += ":" + post_op["dt"]
        return benchdnn_p_op

    def convert_prelu_post_op(post_op):
        benchdnn_p_op = post_op["alg"]
        if post_op["mask"] != 0:
            policy = convert_scale_policy(post_op["mask"], prim_kind)
            benchdnn_p_op += ":" + policy
        return benchdnn_p_op

    convert = {
        "binary": convert_binary_post_op,
        "dw": convert_dw_post_op,
        "eltwise": convert_eltwise_post_op,
        "sum": convert_sum_post_op,
        "prelu": convert_prelu_post_op,
    }

    @property
    def dts(self):
        args = ["src_iter", "src_iter_c", "src_layer", "dst_iter", "dst_layer"]
        cfg_dts: str
        common_dt = True
        shared_dt = None
        bias_dt = None
        md_map: Dict[Optional[str], ir.MemoryDescriptor] = {}
        for md in self.entry.mds:
            md_map[md.arg] = md
            if md.arg == "bias":
                bias_dt = md.data_type
            elif md.arg in args:
                if shared_dt is None:
                    shared_dt = md.data_type
                elif md.data_type != shared_dt:
                    common_dt = False
        if common_dt and shared_dt in ["f32", "f16"]:
            cfg_dts = shared_dt
        elif common_dt and shared_dt == "bf16":
            cfg_dts = shared_dt
            # bias is part of cfg for bf16
            if bias_dt is not None and bias_dt != shared_dt:
                cfg_dts += bias_dt
        else:
            cfg_dts = ""
            for arg in args:
                if arg not in md_map:
                    continue
                md = md_map[arg]
                # src iter is skipped if it is f16
                if arg == "src_iter_c" and md.data_type == "f16":
                    continue
                cfg_dts += md.data_type
        return f"--cfg={cfg_dts}"

    @property
    def tags(self):
        # Tags for backward are driven by diff tensors, query them instead of
        # forward tensors. Latter will always have `any` format.
        has_diff_tensors = False
        for md in self.entry.mds:
            if "diff" in md.arg:
                has_diff_tensors = True
                break
    return benchdnn_postops


def convert_quantization(q_param, prim_kind, def_value, def_type):
    res = []
    for arg in q_param.keys():
        p = q_param[arg]
        policy = convert_scale_policy(p["mask"], prim_kind)
        benchdnn_p = arg + ":" + policy
        if policy == "common":
            benchdnn_p += ":" + def_value
        dt = p["data_type"]
        groups = p["groups"]
        if dt != def_type or groups != "":
            benchdnn_p += ":" + dt
        if groups != "":
            benchdnn_p += ":" + groups
        res.append(benchdnn_p)
    return "+".join(res)


def convert_scales(scales, prim_kind):
    return convert_quantization(
        q_param=scales, prim_kind=prim_kind, def_value="0.5", def_type="f32"
    )


def convert_zero_points(zero_points, prim_kind):
    return convert_quantization(
        q_param=zero_points, prim_kind=prim_kind, def_value="1", def_type="s32"
    )

def convert_rounding_mode(rounding_modes, prim_kind):
    res = []
    for arg in rounding_modes.keys():
        res.append(arg + ":" + rounding_modes[arg])
    return "+".join(res)

def convert_scratchpad_mode(scratchpad_mode, prim_kind):
    return scratchpad_mode


def convert_fpmath_mode(fpmath_mode, prim_kind):
    return fpmath_mode


def convert_acc_mode(acc_mode, prim_kind):
    return acc_mode


def convert_dropout(dropout, prim_kind):
    ret = dropout["p"]
    if dropout["seed"] != None:
        ret += ":" + dropout["seed"]
        if dropout["tag"] != None:
            ret += ":" + dropout["tag"]
    return ret


def convert_deterministic(deterministic, prim_kind):
    return deterministic


def convert_attrs(exts, prim_kind):
    converters = {
        "attr-post-ops": convert_post_ops,
        "attr-scales": convert_scales,
        "attr-zero-points": convert_zero_points,
        "attr-scratchpad": convert_scratchpad_mode,
        "attr-fpmath": convert_fpmath_mode,
        "attr-acc": convert_acc_mode,
        "attr-rounding-mode": convert_rounding_mode,
        "attr-dropout": convert_dropout,
        "attr-deterministic": convert_deterministic,
    }

        layer_names = ["src_layer", "wei_layer", "dst_layer"]
        if has_diff_tensors:
            layer_names = [f"diff_{name}" for name in layer_names]
        tags = []
        other_flags = []
        for md in self.entry.mds:
            arg = md.arg
            tag = maybe_make_any_tag(md)
            if arg in layer_names:
                tags.append(tag)
            elif md.tag == "undef":
                continue
            elif arg == "wei_proj":
                other_flags.append("--with-projection=true")
            elif arg == "wei_peephole":
                other_flags.append("--with-peephole=true")
        tag_flag = "--tag=" + ":".join(tags)
        return " ".join([tag_flag] + other_flags)


class ShuffleConverter(Converter):
    driver: str = "shuffle"

    @property
    def aux(self):
        axis = self.entry.aux.get("axis")
        group = self.entry.aux.get("group")
        args = []
        if axis is not None:
            args.append(f"--axis={axis}")
        if group is not None:
            args.append(f"--group={group}")
        return " ".join(args)


class SoftmaxConverter(TagTripletMixin, CommonDataTypeMixin, Converter):
    driver: str = "softmax"

    @property
    def aux(self):
        axis = self.entry.aux.get("axis")
        flags = f"--alg={self._get_alg()}"
        if axis is not None:
            flags += f" --axis={axis}"
        return flags


class SumConverter(MultiSourceMixin, Converter):
    driver: str = "sum"


class ZeroPadConverter(Converter):
    driver: str = "zeropad"

    @property
    def dts(self):
        return f"--dt={self.entry.mds[0].data_type}"

    @property
    def tags(self):
        return f"--tag={maybe_make_any_tag(self.entry.mds[0])}"


def get_converter(primitive: str) -> ConverterMeta:
    converters: Dict[str, ConverterMeta] = {
        "batch_normalization": BatchNormalizationConverter,
        "binary": BinaryConverter,
        "brgemm": BRGEMMConverter,
        "concat": ConcatConverter,
        "convolution": ConvolutionConverter,
        "deconvolution": DeconvolutionConverter,
        "eltwise": EltwiseConverter,
        "group_normalization": GroupNormalizationConverter,
        "inner_product": InnerProductConverter,
        "layer_normalization": LayerNormalizationConverter,
        "lrn": LRNConverter,
        "matmul": MatmulConverter,
        "pooling": PoolingConverter,
        "prelu": PreLUConverter,
        "reduction": ReductionConverter,
        "reorder": ReorderConverter,
        "resampling": ResamplingConverter,
        "rnn": RNNConverter,
        "shuffle": ShuffleConverter,
        "softmax": SoftmaxConverter,
        "sum": SumConverter,
        "zero_pad": ZeroPadConverter,
    }
    return converters[primitive]


class InputGenerator:
    """
    Generates an input for benchdnn from internal representation.
    """

    def __init__(self, _: Any = None):  # Maintain old interface
        pass

    def _generate_case(self, entry: ir.Entry):
        Converter = get_converter(entry.prim_kind)
        converter = Converter(entry)
        args = [
            "--reset",
            "--allow-enum-tags-only=0",
            converter.engine,
            converter.dir,
            converter.aux,
            converter.bias_mask,
            converter.dts,
            converter.tags,
            converter.flags,
            converter.attrs,
            converter.shapes,
        ]
        return converter.driver, " ".join(arg for arg in args if arg)

    def generate(self, input, split_by_driver=False):
        data: Dict[str, List[str]] = defaultdict(list)
        for value in input.values():
            driver, args = self._generate_case(value)
            if not split_by_driver:
                driver, args = "all", f"--{driver} {args}"
            data[driver].append(args)
        return {k: "\n".join(v) for k, v in data.items()}
