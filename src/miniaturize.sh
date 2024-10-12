#!/usr/bin/bash
source ~/.profile_minigraphs

# Assign inputs
GRAPH_NAME=$1
FRACTION=$2
N_CHANGES=$3
N_STEPS=$4
N_SUBSTEPS=$5

# Intermediate paths
NET_DIR="$DATA_DIR/networks/$GRAPH_NAME"

# Final paths
METRICS_FILE="$NET_DIR/metrics.json"
PARAMS_FILE="$NET_DIR/parameters/params_$FRACTION.json"
OUTPUT_DIR="$NET_DIR/miniatures"

# Verify output directory exists
mkdir -p "$OUTPUT_DIR"

mpiexec -n 6 python -u miniaturize.py "$METRICS_FILE" "$PARAMS_FILE" "$FRACTION" --output_dir="$OUTPUT_DIR" --n_changes="$N_CHANGES" --n_steps="$N_STEPS" --n_substeps="$N_SUBSTEPS"