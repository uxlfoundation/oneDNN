# Summary information

## Introduction
Benchdnn is designed to operate with batch files, and while some of them can be
small, some of them might be really large. While benchdnn provides tools to get
information from a single test case, it might be challenging to acquire the
interested data from the whole batch file, e.g., when running the benchmark
manually, the screen buffer may not fit the whole output and the data is simply
lost if not redirected to the file.

This is where the summary options come in handy to provide the batch level
statistics.

## Usage
```
    --summary=[no-]SETTING1[+[no-]SETTING2...]
```

The `--summary` knob is a global state of benchdnn and provides the summary
statistics at the end of the run. Different options are separated with `+`
delimiter. To negate the effect of the particular summary option, use the "no-"
prefix in front of the option value.

If the same setting is specified multiple times, only the latter value is
considered.

## Failed cases summary

### Usage
```
    --summary=[no-]failures
```

This knob provides a list of failures up to ten entries starting from the
beginning of the run to help to identify problematic cases during the run
without necessity to process the whole output manually. Enabled by default.

## Implementations summary

### Usage
```
    --summary=[no-]impl
    --summary=[no-]impl-csv
```

This knob provides a list of implementation names and the number of hits used
for problems. A table view is enabled by default, and CSV-style output is
disabled by default.
