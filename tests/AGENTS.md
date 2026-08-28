# AGENTS.md — tests

See root `/AGENTS.md` for build/test/style/commit conventions. This file
covers the test tools themselves.

## benchdnn

The main correctness and performance tool, run per-primitive:

```sh
./benchdnn --DRIVER [COMMON-OPTIONS] [DRIVER-OPTIONS] PROBLEM-DESCRIPTION
# example:
./benchdnn --conv --dir=FWD_B --dt=f32 mb2_ic3oc16_ih7oh7kh3sh1dh0ph1
./benchdnn --mode=P --conv mb2_ic64oc64_ih56oh56kh3sh1dh0ph1   # perf mode
```

`DRIVER` is one per primitive (`conv`, `matmul`, `bnorm`, `graph`, ...) — see
`tests/benchdnn/README.md`'s driver list for the full set and its own
`tests/benchdnn/doc/driver_*.md` for that driver's options.

## gtests

Plain googletest binaries. Run one directly with `--gtest_filter=<pattern>`.

Fixing a bug: add a regression test under `gtests/regression/`, named
`test_regression_<description>.cpp` — that's the convention the existing
tests there follow.
