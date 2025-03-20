#!/bin/bash
#
#SBATCH --job-name=full
#SBATCH -c100
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out

srun python /data/parietal/store3/work/haggarwa/diffusion/diffusion_preprocessing/runners/run_tracto_drago.py


