#!/bin/bash

set -e

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm $root/../third_party/ngen -rf
git clone https://github.com/intel-innersource/libraries.gpu.ngen.git $root/../third_party/ngen
find $root/../third_party/ngen -name "*pp" -type f -exec $root/../third_party/ngen/scripts/recopyright.sh {} \;
rm -rf $root/../third_party/ngen/{.git,.gitignore,docs,examples,scripts,tests,README.md}
git checkout $root/../third_party/ngen/COPYRIGHT
