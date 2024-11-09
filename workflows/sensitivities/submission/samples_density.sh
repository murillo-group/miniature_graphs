#!/bin/bash
#SBATCH --job-name=sDensity
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --output=out_%x/output_%A.out

source profile
OUTPUT_DIR="$DATA_DIR/samples/$SLURM_JOB_ID"

python -u sample.py $OUTPUT_DIR "params_density.yaml"