#!/bin/bash
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --array=1-9
#SBATCH --time=02:00:00  # Time limit
#SBATCH --mem=4G 

GRAPH_NAME=$1

# Source profile
source ~/.profile_minigraphs

# Calculate miniaturization size
VALUE=$(echo "scale=3; $SLURM_ARRAY_TASK_ID * 0.1" | bc)

# Run your Python script with the selected input file
python -u "/mnt/home/martjor/repos/dev_pt/src/params.py" "$GRAPH_NAME" "$VALUE" 100 10
