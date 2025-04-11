#!/bin/bash
#
#SBATCH --job-name=bedpostx_gpu
#SBATCH --partition=gpu,parietal
#SBATCH --error error_%A_%a.out
#SBATCH --gres=gpu:1

srun python /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/runners/run_tracto_drago.py
