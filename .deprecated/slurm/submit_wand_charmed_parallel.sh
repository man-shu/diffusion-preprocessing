#!/bin/bash
#SBATCH --job-name=diffusion_pipelines
#SBATCH --output=log_slurm/jobid_%A_%a.out 
#SBATCH --error=log_slurm/jobid_%A_%a.err
#SBATCH --partition=normal,parietal
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --array=1-151%100

dirs=(/data/parietal/store4/data/WAND/sub-*)
echo ${dirs[${SLURM_ARRAY_TASK_ID}]:36}

module load singularity

srun singularity exec \
--env-file /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/singularity_env.txt \
--bind /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data:/home/input \
--bind /data/parietal/store4/data:/home/input \
/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/diffusion-preprocessing_main_singularity.sif \
/opt/miniconda3/bin/diffusion_pipelines \
/home/input/WAND \
/home/input/WAND-concat/derivatives \
--work-dir /home/input/cache \
--output-spaces fsLR:den-32k MNI152NLin6Asym T1w fsaverage5 \
--cifti-output 91k \
--nprocs 1 \
--omp-nthreads 10 \
--participant-label ${dirs[${SLURM_ARRAY_TASK_ID}]:36} \
--acquisition CHARMED_dir-AP \
--preproc \
--preproc-t1 /home/input/WAND-concat/derivatives/smriprep/sub-${dirs[${SLURM_ARRAY_TASK_ID}]:36}/ses-02/anat/sub-${dirs[${SLURM_ARRAY_TASK_ID}]:36}_ses-02_desc-preproc_T1w.nii.gz \
--preproc-t1-mask /home/input/WAND-concat/derivatives/smriprep/sub-${dirs[${SLURM_ARRAY_TASK_ID}]:36}/ses-02/anat/sub-${dirs[${SLURM_ARRAY_TASK_ID}]:36}_ses-02_desc-brain_mask.nii.gz \
--fs-native-to-t1w-xfm /home/input/WAND-concat/derivatives/smriprep/sub-${dirs[${SLURM_ARRAY_TASK_ID}]:36}/ses-02/anat/sub-${dirs[${SLURM_ARRAY_TASK_ID}]:36}_ses-02_from-fsnative_to-T1w_mode-image_xfm.txt \
--debug