#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH --time=12:00:00

# Source profile
module load Conda/3

# Activate conda module
conda activate paper_miniaturize

echo "Timing generators..."
python prof_generators.py "$1"

echo "Done."


