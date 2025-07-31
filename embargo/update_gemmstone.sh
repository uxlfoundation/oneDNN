#!/bin/bash

set -e

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm $root/../third_party/gemmstone -rf
git clone https://github.com/intel-innersource/libraries.performance.math.gemmstone.git $root/../third_party/gemmstone
rm -rf $root/../third_party/gemmstone/.git
