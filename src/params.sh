#!/usr/bin/bash
source ~/.profile_minigraphs

# Assign inputs
GRAPH_NAME=$1
FRACTION=$2
N_CHANGES=$3
N_SAMPLES=$4
N_ITERATIONS=$5

# Directory containing the original network
NET_DIR="$DATA_DIR/networks/$GRAPH_NAME"

# Final paths
METRICS_FILE="$NET_DIR/metrics.json"
OUTPUT_DIR="$NET_DIR/parameters"

# Verify output directory exists
mkdir -p "$OUTPUT_DIR"

# Calculate parameters
python -u params.py "$METRICS_FILE" "$FRACTION" --output_dir="$OUTPUT_DIR" --n_changes="$N_CHANGES" --n_samples="$N_SAMPLES" --n_iterations="$N_ITERATIONS"