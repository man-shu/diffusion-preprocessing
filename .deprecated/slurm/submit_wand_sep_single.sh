#!/bin/bash
#SBATCH --job-name=diffusion_pipelines
#SBATCH --output=log_slurm/jobid_%A.out 
#SBATCH --error=log_slurm/jobid_%A.err
#SBATCH --partition=normal,parietal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=10

module load singularity

srun singularity exec \
--env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
--bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input \
/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
/opt/miniconda3/bin/diffusion_pipelines \
/home/input/WAND-sep \
/home/input/WAND-sep/derivatives \
--work-dir /home/input/cache \
--fs-subjects-dir /home/input/WAND-sep/derivatives/freesurfer \
--output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
--cifti-output 91k \
--nprocs 1 \
--omp-nthreads 20 \
--participant-label sub-01187 \
--acquisition AxCaliber1 \
--no-msm \
--fs-no-resume \
--no-submm-recon \
--preproc