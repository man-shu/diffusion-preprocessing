#!/bin/bash
#
#SBATCH --job-name=full
#SBATCH -c100
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out

srun python runners/run_tracto_drago.py


