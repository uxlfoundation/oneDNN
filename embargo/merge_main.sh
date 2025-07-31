#!/bin/bash

set -e

if [[ -n "$(git status --porcelain --ignored)" ]]; then
    echo "There are uncommitted changes (unstaged or untracked). The script needs a clean tree."
    exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workdir=$(mktemp -d -t embargo-XXX)
githash=$(cat $root/main.hash)

echo "=== Get the diff between old hash main and current branch"
cd $root/..
git fetch
git diff "$githash" -- . ':!third_party/ngen' > $workdir/prv.diff
# TODO: Use this command to skip gemmstone once it is moved to third_party.
#git diff "$githash" -- . ':!third_party/ngen' ':!src/gpu/intel/jit/gemm/generator' ':!src/gpu/intel/jit/gemm/include' ':!src/gpu/intel/jit/gemm/selector' > $workdir/prv.diff

echo "=== Copy the file tree from the new main"
git restore --source=origin/main .
git checkout -- third_party/ngen embargo
git add * .github
git commit -am 'wip'

echo "=== Applying patches one by one"
python $root/genapply.py $workdir $workdir/prv.diff
bash $workdir/apply.sh

echo "=== Add untracked files - these were new files in prv-gpu"
git ls-files --others --exclude-standard | xargs git add

echo "=== Update synced main hash"
git rev-parse origin/main > embargo/main.hash
git add embargo/main.hash

cd $workdir

cat merge.log | grep error > error.log
echo "=== Found errors during merge, check $workdir/error.log"
cat error.log
