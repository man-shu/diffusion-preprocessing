#!/bin/bash
#SBATCH --job-name=diffusion_pipelines
#SBATCH --output=log_slurm/jobid_%A_%a.out 
#SBATCH --error=log_slurm/jobid_%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=1-151%100

dirs=(/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/WAND-concat/sub-*)
echo ${dirs[${SLURM_ARRAY_TASK_ID}]:91}

module load singularity

srun singularity exec \ 
--env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
--bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input \
/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
/opt/miniconda3/bin/diffusion_pipelines \
/home/input/WAND-concat \
/home/input/WAND-concat/derivatives \
--work-dir /home/input/cache \
--fs-subjects-dir /home/input/WAND-concat/derivatives/freesurfer \
--output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
--cifti-output 91k \
--nprocs 1 \
--omp-nthreads 8 \
--participant-label ${dirs[${SLURM_ARRAY_TASK_ID}]:91} \
--acquisition AxCaliberConcat \
--no-msm \
--fs-no-resume \
--no-submm-recon \
--preproc