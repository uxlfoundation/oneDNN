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

from matmul.primitive import Primitive as matmul

import report.plotter as plotter
import report.metrics as metrics


class Sample:
    def __init__(self, sample_line):
        _, name, prb, flops, bandwidth = sample_line.strip().split(",")
        self.name = name
        if "--matmul" in sample_line:
            self.primitive = matmul.from_repro(prb)
        else:
            raise RuntimeError(f"Unimplemented primitive: {sample_line}")
        self.flops = float(flops)
        self.bandwidth = float(bandwidth)

    def kind(self):
        if self.name != "":
            return self.name + ": " + self.primitive.kind.benchdnn_str()
        else:
            return self.primitive.kind.benchdnn_str()

    def type(self):
        return self.primitive.kind.type.benchdnn_str()


class ReportList:
    def __init__(self, scatter_plots):
        self.reports = []
        for s in scatter_plots:
            args = s.split(",")
            x_idx = args[0]
            y_idx = None
            metric = metrics.metric_from_str(args[1])
            scale = metrics.SampleRelative()
            if metric is None:
                y_idx = args[1]
                metric = metrics.metric_from_str(args[2])
                if len(args) > 3:
                    scale = metrics.scale_from_str(args[3])
            else:
                if len(args) > 2:
                    scale = metrics.scale_from_str(args[2])
            self.reports.append(plotter.Scatter(x_idx, y_idx, metric, scale))
            self.has_plot = True

        if self.has_plot:
            plotter.initialize()

    def add_line(self, line):
        if line.startswith("sample"):
            sample = Sample(line)
            for r in self.reports:
                r.add(sample)

    def finalize(self):
        for r in self.reports:
            r.update()
        if self.has_plot:
            plotter.show()
