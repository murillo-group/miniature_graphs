#!/bin/bash
#SBATCH --output=nips-params-10_changes/output_%A_%a.out
#SBATCH --error=nips-params-10_changes/error_%A_%a.err
#SBATCH --array=1-9
#SBATCH --time=15:00:00  # Time limit
#SBATCH --mem=7G 

GRAPH_NAME=$1

# Source profile
source ~/.profile_minigraphs

# Calculate miniaturization size
VALUE=$(echo "scale=3; $SLURM_ARRAY_TASK_ID * 0.1" | bc)

# Run your Python script with the selected input file
python -u "/mnt/home/martjor/repos/dev_pt/src/params.py" "$GRAPH_NAME" "$VALUE" 10 10 100
