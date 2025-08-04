# Synthdnn

Synthdnn is a suite of scripts for collecting and analyzing oneDNN performance
across a randomly generated data. The general architecture is intended to follow
a data pipeline composed of synthetic problem generation, data collection, and
data analysis. The `synthdnn.py` script provides a command line interface to
these tools.

Basic syntax:
```
python3 synthdnn.py <collect|primitive> [controls]
```
Currently, `primitive` can only be `matmul`.

### Collect
Collect data from an existing problem set/batch file.
```
python3 synthdnn.py collect [collect controls] -b <batch_file> <benchdnn_executable>
```
Collect controls:
- `--batch-file|-b <filename>`: Path to a batch file containing the problem set.
- `benchdnn_executable`: Path to a benchdnn executable.
- `--engine <cpu|gpu>`: (default `cpu`) Select the engine to use during execution.
- `--impl|--skip-impl <impl-name>`: Select an implementation to use (or skip) for benchdnn execution
- `--collect <corr|perf>`: (default `corr`) Select what type of data to collect.
### Matmul
Generate a problem set for the matmul primitive.
```
python3 synthdnn.py matmul [matmul controls]
```
Matmul controls:
- `--batch-file|-b <filename>`: Direct the problem set into a batch file which can be run via benchdnn using the `--batch` argument. If not supplied, the test set will be written to stdout.
- `--types|-t <dt>[:dt:dt][(fpmath_mode)][,dt...]`: (default `*`) A comma-separated list of data types to be used in the test set (each entry will add configurations). Details below:
  - `dt:dt:dt`: Selects supported type configurations matching `src:wei:dst`
  - `dt`: Selects any supported type configuration that uses `dt` for any buffer
  - Data types use globbing-style wildcard matching, `[su]4` will match both `s4` and `u4` for example
  - The special data type `%N` matches the same data type as the Nth supplied data type (using zero-indexing), after wildcards are expanded. For example `*f16:*:%0` will match `f16:*:f16` and `bf16:*:bf16` configurations, but not `f16:*:bf16`.
- `--layouts|-l <stag:wtag:dtag>`: (default `all`) Selects the format tags to be used for src, weights, and dst tensors. The default selects all possible plain formats.
- `--iter-mode|-m <zip|product>`: (default `zip`) Changes the order of problem iteration in the test set, which may affect the problems generated.
- `--region|-r <pt_min>:<pt_max>:<pt_align>`: Each point has the syntax `([b,]m,n,k)` - selects the region of problem space to sample tests in. Each dimension varies from the `min` size to the `max` size with spacing equal to `align`.
- `--samples|-s <num>`: (default 1000) Selects the number of sample points, which determines the test set's total size.
### Matmul
Generate a report from performance collected data
```
python3 synthdnn.py report [report controls] [logs ...]
```
Report controls:
- `--scatter x_idx,[y_idx],metric,[scale]`: Create a 2d or 3d scatter plot of the given metric (bandwidth/flops) and given scale (relative/absolute)

## Examples

```
# Generate an LLM 1st token matmul problem set (each dimension varies from 64-16384 elements, in multiples of 16) with layers that use f16 or bf16, 100 total samples
python3 synthdnn.py matmul -r "(64,64,64):(16384,16384,16384):(16,16,16)" -s 100 -t *f16

# Generate a matmul problem set with int4 weights, and then collect GPU performance and correctness data on it
python3 synthdnn.py matmul -t *:[su]4:* -b matmul.batch
python3 synthdnn.py collect --collect corr --engine gpu -b matmul.batch build/tests/benchdnn/benchdnn
python3 synthdnn.py collect --collect perf --engine gpu -b matmul.batch build/tests/benchdnn/benchdnn

# Generate scatter plots from performance data collected with synthdnn collect
python3 synthdnn.py report --scatter m,flops --scatter n,flops --scatter k,flops base.log tune.log
python3 synthdnn.py report --scatter m,bandwidth,absolute --scatter n,bandwidth,absolute --scatter k,bandwidth,absolute base.log tune.log

```
