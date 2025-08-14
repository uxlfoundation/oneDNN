#!/bin/bash

if [ -z "$1" ]; then
    echo "Script for nGEN downstreaming."
    echo "Usage: $0 <nGEN git repository address>"
    exit 0
fi

set -e
rm -rf third_party/ngen
git clone $@ third_party/ngen
./third_party/ngen/scripts/public.sh
git checkout third_party/ngen/COPYRIGHT
rm -rf third_party/ngen/{.git,scripts}
echo "Update completed."
