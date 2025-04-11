#!/bin/bash
#
#SBATCH --job-name=wand_preproc
#SBATCH -c100
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out

srun python /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/runners/run_preproc_drago_WAND.py


