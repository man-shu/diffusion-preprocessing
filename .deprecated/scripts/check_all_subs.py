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
                if sub_dir.name == "sub-01945":
                    dwi_files = list(
                        sub_dir.glob(
                            f"ses-02/dwi/*{protocol}*run-01_dwi.{extension}"
                        )
                    )
                    dwi_files.sort()
                    dwi_files2 = list(
                        sub_dir.glob(
                            f"ses-02/dwi/*{protocol}*mag_dwi.{extension}"
                        )
                    )
                    dwi_files2.sort()
                    dwi_files = dwi_files + dwi_files2[:2]
                if sub_dir.name == "sub-11220":
                    continue
                dwi_img = nib_load(dwi_files[0])
                print(dwi_files)
                print(dwi_img.shape)
            elif extension == "bval":
                # find all bval files
                bval_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.{extension}")
                )
                bval_files.sort()
                print(bval_files)
                # concatenate all bval files
                for bval_file in bval_files:
                    bval = np.loadtxt(bval_file)
                    print(f"bval shape: {bval.shape}")
            elif extension == "bvec":
                # find all bvec files
                bvec_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.bvec")
                )
                bvec_files.sort()
                print(bvec_files)
                for bvec_file in bvec_files:
                    bvec = np.loadtxt(bvec_file)
                    print(f"bvec shape: {bvec.shape}")


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
