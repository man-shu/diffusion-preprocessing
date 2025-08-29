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
--preproc \
--preproc-t1 /home/input/WAND-concat/derivatives/smriprep/sub-00395/ses-02/anat/sub-00395_ses-02_desc-preproc_T1w.nii.gz \
--preproc-t1-mask /home/input/WAND-concat/derivatives/smriprep/sub-00395/ses-02/anat/sub-00395_ses-02_desc-brain_mask.nii.gz \
--fs-native-to-t1w-xfm /home/input/WAND-concat/derivatives/smriprep/sub-00395/ses-02/anat/sub-00395_ses-02_from-fsnative_to-T1w_mode-image_xfm.txt \
--debug