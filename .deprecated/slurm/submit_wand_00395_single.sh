#!/bin/bash
#SBATCH --job-name=sub-00395
#SBATCH --output=log_slurm/jobid_%A.out 
#SBATCH --error=log_slurm/jobid_%A.err
#SBATCH --partition=normal,parietal
#SBATCH -c20

module load singularity

srun singularity exec \
--env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
--bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input \
/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
/opt/miniconda3/bin/diffusion_pipelines \
/home/input/WAND-concat \
/home/input/WAND-concat/derivatives \
--work-dir /home/input/cache \
--output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
--cifti-output 91k \
--nprocs 1 \
--omp-nthreads 20 \
--participant-label sub-00395 \
--acquisition AxCaliberConcat \
--no-msm \
--fs-no-resume \
--no-submm-recon \
--recon \
--preproc \
--debug