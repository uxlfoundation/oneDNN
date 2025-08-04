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


class Scaling:
    title = ""

    def __init__(self):
        self.max_value = 0

class SampleRelative(Scaling):
    title = "Sample Relative"

    def update(self, value):
        if self.max_value < value:
            self.max_value = value
        return

class Absolute(Scaling):
    title = "Absolute"

    def update(self, value):
        return

def scale_from_str(name):
    normalized = name.lower()
    if normalized == "absolute":
        return Absolute()
    if normalized == "relative":
        return SampleRelative()
    return None


class Metric:
    title = ""

    def get(self, sample):
        return 0


class Bandwidth(Metric):
    title = "Bandwidth"

    def get(self, sample):
        return sample.bandwidth


class Flops(Metric):
    title = "GFLOPS"

    def get(self, sample):
        return sample.flops


def metric_from_str(name):
    normalized = name.lower()
    if normalized == "bandwidth":
        return Bandwidth()
    if normalized == "flops":
        return Flops()
    return None


class MetricData:
    def __init__(self, scaling, value):
        self.scaling: Scaling = scaling
        self.value: Metric = value
        self.base_data: list[float] = []

    @property
    def title(self):
        return f"{self.scaling.title} {self.value.title}"

    def add(self, sample):
        val = self.value.get(sample)
        self.scaling.update(val)
        self.base_data.append(val)

    def get(self):
        if self.scaling.max_value == 0 or self.scaling.max_value == 1:
            return self.base_data
        return [x / self.scaling.max_value for x in self.base_data]
