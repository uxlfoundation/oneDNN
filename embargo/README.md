oneDNN Development Branch with Support for Embargoed GPU Platforms
===========================================

# How To Merge with the Main Branch

> NOTE: The script needs a clean tree (right after git clone) as all unstaged changes are lost.

```bash
git clone https://github.com/intel-innersource/libraries.performance.math.onednn.git onednn-prv-gpu
cd onednn-prv-gpu
./embargo/merge_main.sh

# Check git status to see conflicts to resolve
# Use git add after conflicts are resolved
# Use git commit/rebase to squash the update into a single commit and open a PR
```


# How to Downstream nGEN and gemmstone

```bash
./embargo/update_ngen.sh
./embargo/update_gemmstone.sh
```
