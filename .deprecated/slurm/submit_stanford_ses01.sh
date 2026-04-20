#!/bin/bash
#SBATCH --job-name=prep_ses01_stanford
#SBATCH --output=log_slurm/jobid_%A_%a.out 
#SBATCH --error=log_slurm/jobid_%A_%a.err
#SBATCH --partition=normal,parietal
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --array=0-95

dirs=(/data/parietal/store3/work/haggarwa/diffusion/data/stanford-bids/sub-*)
echo ${dirs[${SLURM_ARRAY_TASK_ID}]:69}

srun singularity exec \
--env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
--bind /data/parietal/store3/work/haggarwa/diffusion/data:/home/input \
/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
/opt/miniconda3/bin/diffusion_pipelines \
/home/input/stanford-bids \
/home/input/stanford-bids/derivatives \
--work-dir /home/input/cache \
--output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
--cifti-output 91k \
--nprocs 1 \
--omp-nthreads 10 \
--participant-label ${dirs[${SLURM_ARRAY_TASK_ID}]:69} \
--session-label ses-01 \
--subject-anatomical-reference sessionwise \
--recon \
--preproc \
--no-msm \
--debug