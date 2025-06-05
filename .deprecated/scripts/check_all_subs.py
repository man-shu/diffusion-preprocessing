from nilearn.image import concat_imgs
from glob import glob
from pathlib import Path
import os
import numpy as np
from joblib import Parallel, delayed
from nibabel import load as nib_load
from drop_unavailable import get_unavailable_subjects


def check_file_lengths(sub_dir, protocols):
    for protocol in protocols:
        for extension in ["nii.gz", "bval", "bvec"]:
            if extension == "nii.gz":
                dwi_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.{extension}")
                )
                dwi_files.sort()
                # print(dwi_files)
                dwi_img = nib_load(dwi_files[0])
                dwi_shape = dwi_img.shape
                # print(dwi_shape)
            elif extension == "bval":
                # find all bval files
                bval_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.{extension}")
                )
                bval_files.sort()
                # print(bval_files)
                # concatenate all bval files
                for bval_file in bval_files:
                    bval = np.loadtxt(bval_file)
                    bval_shape = bval.shape
                    # print(f"bval shape: {bval_shape}")
            elif extension == "bvec":
                # find all bvec files
                bvec_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.bvec")
                )
                bvec_files.sort()
                # print(bvec_files)
                for bvec_file in bvec_files:
                    bvec = np.loadtxt(bvec_file)
                    bvec_shape = bvec.shape
                    # print(f"bvec shape: {bvec_shape}")

    if dwi_shape[3] != bval_shape[0] or dwi_shape[3] != bvec_shape[1]:
        print(
            f"Mismatch in file lengths for {sub_dir.name}:"
            f"dwi: {dwi_shape[3]}, bval: {bval_shape[0]}, bvec: {bvec_shape[1]}"
        )


if __name__ == "__main__":

    protocols = ["AxCaliberConcat"]
    root_directory = Path(
        "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/WAND-concat"
    )
    unavail_root_directory = Path(
        "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/.deprecated/scripts"
    )
    sub_dirs = list(root_directory.glob("sub-*"))
    print(f"Found {len(sub_dirs)} subjects in {root_directory}")
    sub_dirs.sort()
    unavailable_subjects = get_unavailable_subjects(unavail_root_directory)
    for sub_dir in sub_dirs:
        if sub_dir.name in unavailable_subjects:
            print(f"Skipping unavailable subject: {sub_dir.name}")
            continue
        else:
            print(f"Checking {sub_dir.name}")
        check_file_lengths(sub_dir, protocols)
