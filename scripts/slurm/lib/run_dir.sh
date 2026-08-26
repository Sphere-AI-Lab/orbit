#!/bin/bash

# prepare_run_dir <requested-path> <default-path>
#
# Create an empty per-run directory and print its canonical absolute path.
# Slurm's --export uses commas as separators, so commas cannot be represented
# safely by submit.sh's current export format.
prepare_run_dir() {
    local requested=$1 default_path=$2 run_dir canonical_run_dir
    run_dir=${requested:-$default_path}

    if [[ -e "$run_dir" && ! -d "$run_dir" ]]; then
        echo "FATAL: RUN_DIR exists and is not a directory: $run_dir" >&2
        return 73  # EX_CANTCREAT
    fi
    if ! mkdir -p -- "$run_dir"; then
        echo "FATAL: could not create RUN_DIR: $run_dir" >&2
        return 73  # EX_CANTCREAT
    fi
    if ! canonical_run_dir=$(cd -- "$run_dir" && pwd -P); then
        echo "FATAL: could not resolve RUN_DIR: $run_dir" >&2
        return 73  # EX_CANTCREAT
    fi
    if [[ "$canonical_run_dir" == *,* ]]; then
        echo "FATAL: RUN_DIR must not contain ',' (Slurm --export separator): $canonical_run_dir" >&2
        return 78  # EX_CONFIG
    fi
    if [[ -n "$(find "$canonical_run_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "FATAL: RUN_DIR already contains run artifacts: $canonical_run_dir" >&2
        return 73  # EX_CANTCREAT
    fi

    printf '%s\n' "$canonical_run_dir"
}
