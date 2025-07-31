#!/bin/bash

set -e

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm $root/../third_party/ngen -rf
git clone https://github.com/intel-innersource/libraries.gpu.ngen.git $root/../third_party/ngen
rm -rf $root/../third_party/ngen/{.git,.gitignore,docs,examples,scripts,README.md}
git checkout $root/third_party/ngen/COPYRIGHT
