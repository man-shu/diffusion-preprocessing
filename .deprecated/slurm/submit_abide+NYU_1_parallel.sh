#!/bin/bash
#SBATCH --job-name=diffusion_pipelines
#SBATCH --output=log_slurm/jobid_%A_%a.out 
#SBATCH --error=log_slurm/jobid_%A_%a.err
#SBATCH --partition=normal,parietal
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=90:00:00
#SBATCH --array=0-78%100

dirs=(/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/abide/NYU_1/sub-*)
echo ${dirs[${SLURM_ARRAY_TASK_ID}]:91}

module load singularity

srun singularity exec \
--env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
--bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input \
/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
/opt/miniconda3/bin/diffusion_pipelines \
/home/input/abide/NYU_1 \
/home/input/abide/NYU_1/derivatives \
--work-dir /home/input/cache \
--output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
--cifti-output 91k \
--nprocs 1 \
--omp-nthreads 10 \
--participant-label ${dirs[${SLURM_ARRAY_TASK_ID}]:91} \
--bids-filter-file /home/input/bids_filter_abide_nyu2.json \
--preproc \
--debug