#!/bin/bash
#
#SBATCH --job-name=wand_concat_preproc
#SBATCH -c100
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out

srun singularity exec --env-file singularity_env.txt --bind ./data:/home/input diffusion_pipelines.sif /opt/miniconda3/bin/diffusion_pipelines -< config-concat.cfg
