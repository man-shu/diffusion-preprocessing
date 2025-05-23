# %%
# downsample existing data
from nilearn.image import resample_img
from glob import glob
import os
import numpy as np

# Define the input and output file paths
root_path = "/Users/himanshu/Desktop/diffusion/data/WAND-bids/"

# get all nifti files
nifti_files = glob(f"{root_path}**/*.nii*", recursive=True)

# Define the output directory
output_dir = "/Users/himanshu/Desktop/diffusion/data/WAND-downsampled/"
os.makedirs(output_dir, exist_ok=True)

# Define the target resolution as 5mm isotropic
target_resolution = (5, 5, 5)
# %%
# Downsample all nifti files
for nii in nifti_files:
    try:
        img = resample_img(nii, target_affine=np.diag(target_resolution))
    except Exception as e:
        print(f"Error: {e}")
        continue
    _output_dir = os.path.dirname(nii.replace(root_path, output_dir))
    os.makedirs(_output_dir, exist_ok=True)
    output_path = os.path.join(_output_dir, os.path.basename(nii))
    img.to_filename(output_path)

# %%
# also downsample the templates
template_dir = "/Users/himanshu/Desktop/diffusion/mni_icbm152_nlin_sym_09a"
template_files = glob(f"{template_dir}/*.nii*")

for nii in template_files:
    img = resample_img(nii, target_affine=np.diag(target_resolution))
    output_path = os.path.join(output_dir, os.path.basename(nii))
    img.to_filename(output_path)

# %%
# downsample the rois
roi_dir = "/Users/himanshu/Desktop/diffusion/rois"
roi_files = glob(f"{roi_dir}/*.nii*")
for nii in roi_files:
    img = resample_img(nii, target_affine=np.diag(target_resolution))
    output_path = os.path.join(output_dir, os.path.basename(nii))
    img.to_filename(output_path)

# %%
