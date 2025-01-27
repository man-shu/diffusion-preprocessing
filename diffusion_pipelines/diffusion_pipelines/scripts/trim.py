# %%
# trim the downsampled existing data
from nilearn.image import index_img
from glob import glob
import os
import numpy as np

# Define the input and output file paths
root_path = "/Users/himanshu/Desktop/diffusion/downsampled_data/sub-7014/dwi"

# get the dwi nifti file
nifti_file = glob(f"{root_path}/*.nii*")[0]

# Define the output directory
output_dir = "/Users/himanshu/Desktop/diffusion/downsampled_trimmed"
os.makedirs(output_dir, exist_ok=True)

# %%
# Trim all nifti files
img = index_img(nifti_file, slice(0, 26))

# %%
# Save the trimmed nifti file
output_path = os.path.join(output_dir, os.path.basename(nifti_file))

img.to_filename(output_path)

# %%
# also trim the bval and bvec files
bval_file = glob(f"{root_path}/*.bval")[0]
bvec_file = glob(f"{root_path}/*.bvec")[0]

# %%
# Read the bval and bvec files
bval = np.loadtxt(bval_file)
bvec = np.loadtxt(bvec_file)

# %%
# Trim the bval and bvec files
bval_trimmed = bval[:26]
bvec_trimmed = bvec[:, :26]

# %%
# Save the trimmed bval and bvec files
output_path_bval = os.path.join(output_dir, os.path.basename(bval_file))
output_path_bvec = os.path.join(output_dir, os.path.basename(bvec_file))

# save the trimmed bval as row vector and as integers (not floats)
np.savetxt(output_path_bval, bval_trimmed, fmt="%d", newline=" ")

# save the trimmed bvec as column vector and as decimals (not integers)
np.savetxt(output_path_bvec, bvec_trimmed, fmt="%.6f")
# %%
