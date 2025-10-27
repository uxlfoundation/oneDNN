#! /bin/python3
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

import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib import colors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import math

import report.metrics as metrics


# Helpers for workaround for https: //github.com/matplotlib/matplotlib/issues/209
def log_tick_formatter(val, pos=None):
    return "{:.0f}".format(2**val)


def rescale(data):
    return math.log2(data)


def add_plot(proj=None):
    fig = plt.gcf()
    subplots = fig.get_axes()
    cols = len(subplots) + 1
    gs = gridspec.GridSpec(1, cols)

    # Update the existing subplot positions
    for i, ax in enumerate(subplots):
        ax.set_position(gs[i].get_position(fig))
        ax.set_subplotspec(gs[i])

    # Add the new subplot
    return fig.add_subplot(gs[cols - 1], projection=proj)


class Scatter:
    class Data:
        def __init__(self, perf_metric):
            self.xs = []
            self.ys = []
            self.metrics: metrics.MetricData = perf_metric

        def add(self, x, y, sample):
            self.xs.append(x)
            if y is not None:
                self.ys.append(y)
            self.metrics.add(sample)

    def __init__(self, x_label, y_label, metricValue, scaling):
        self.ax = add_plot(None if y_label is None else "3d")
        self.scaling = scaling
        self.metric_value = metricValue
        self.title = f"{self.scaling.title} {self.metric_value.title}"
        self.x_label = x_label
        self.y_label = y_label
        self.metric_label = metricValue.title
        self.data: dict[String, self.Data] = {}

        super().__init__()

    def update(self):
        self.ax.cla()

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter)
        )

        if self.y_label is None:
            # 2D Plot
            self.ax.set_ylabel(self.metric_label)
            for key, value in self.data.items():
                self.ax.scatter(value.xs, value.metrics.get(), s=1, label=key)
        else:
            # 3D Plot
            self.ax.set_ylabel(self.y_label)
            self.ax.set_zlabel(self.metric_label)

            self.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(log_tick_formatter)
            )

            for key, value in self.data.items():
                self.ax.scatter(
                    value.xs, value.ys, value.metrics.get(), s=1, label=key
                )

        self.ax.legend()

    def add(self, sample):
        x = rescale(sample.primitive[self.x_label])
        y = None
        if self.y_label is not None:
            y = rescale(sample.primitive[self.y_label])
        dt = sample.primitive["dt"]
        entry = sample.primitive["kind"]
        if sample.name is not None:
            entry = sample.name + ": " + entry
        if not entry in self.data:
            self.data[entry] = self.Data(
                metrics.MetricData(self.scaling, self.metric_value)
            )
        self.data[entry].add(x, y, sample)


def initialize():
    plt.style.use('Solarize_Light2')

def show():
    plt.show()
