#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Register only RISC-V in the ephemeral GitHub-hosted test runner.
set -euo pipefail

(cd qemu-bin && sha256sum --check SHA256SUMS)
sudo install -m 755 qemu-bin/qemu-riscv64 /usr/local/bin/qemu-riscv64
/usr/local/bin/qemu-riscv64 --version
if [[ ! -e /proc/sys/fs/binfmt_misc/register ]]; then
    sudo mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
fi
if [[ -e /proc/sys/fs/binfmt_misc/qemu-riscv64 ]]; then
    printf '%s\n' -1 | sudo tee /proc/sys/fs/binfmt_misc/qemu-riscv64
fi
printf '%s\n' ':qemu-riscv64:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/local/bin/qemu-riscv64:OCF' \
    | sudo tee /proc/sys/fs/binfmt_misc/register
cat /proc/sys/fs/binfmt_misc/qemu-riscv64
