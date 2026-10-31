#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build once on the hosted build runner, then share with all test partitions.
set -euo pipefail

root=$(pwd)
version=11.1.3
checksum=5a8289c7940dc77d3873c81a6c29ba3afe1446fc9d36e77fa1658cdbecb84b22
work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/onednn-qemu.XXXXXX")
cd "$work"
curl -fL --retry 3 "https://download.qemu.org/qemu-${version}.tar.xz" -o qemu.tar.xz
printf '%s  qemu.tar.xz\n' "$checksum" | sha256sum --check --strict
tar -xf qemu.tar.xz
cd "qemu-${version}"
./configure --target-list=riscv64-linux-user --static --disable-pie \
    --disable-system --disable-tools --disable-docs --disable-guest-agent \
    --disable-plugins --disable-werror
ninja -C build -j "$(nproc)" qemu-riscv64
mkdir -p "$root/qemu-bin"
install -m 755 build/qemu-riscv64 "$root/qemu-bin/qemu-riscv64"
file "$root/qemu-bin/qemu-riscv64" | grep 'statically linked'
"$root/qemu-bin/qemu-riscv64" --version
(cd "$root/qemu-bin" && sha256sum qemu-riscv64 > SHA256SUMS)
