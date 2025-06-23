#!/bin/bash
#
#SBATCH --job-name=wand_concat_preproc
#SBATCH -c20
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out
#SBATCH --mem=0

module load singularity

srun singularity exec --env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt --bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif /opt/miniconda3/bin/diffusion_pipelines -< /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/config-concat.cfg
