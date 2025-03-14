from nilearn.image import concat_imgs
from glob import glob
from pathlib import Path
import os
import numpy as np

root_directory = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/WAND/sub-00395"
)

for protocol in ["AxCaliber", "CHARMED"]:
    # find all dwi.nii.gz files
    dwi_files = list(root_directory.glob(f"ses-02/dwi/*{protocol}*dwi.nii.gz"))
    dwi_files.sort()
    if protocol == "AxCaliber":
        dwi_files = dwi_files[:-1]

    # concatenate all dwi files
    concatenated_dwi = concat_imgs(dwi_files)

    # save the concatenated dwi file
    os.makedirs(root_directory / "dwi", exist_ok=True)

    concatenated_dwi.to_filename(
        root_directory
        / "dwi"
        / f"sub-00395_ses-02_acq-{protocol}_dwi_concat.nii.gz"
    )

    # find all bval files
    bval_files = list(root_directory.glob(f"ses-02/dwi/*{protocol}*dwi.bval"))
    bval_files.sort()
    if protocol == "AxCaliber":
        bval_files = bval_files[:-1]
    output_path_bval = (
        root_directory
        / "dwi"
        / f"sub-00395_ses-02_acq-{protocol}_dwi_concat.bval"
    )
    concatenated_bval = []

    # concatenate all bval files
    for bval_file in bval_files:
        bval = np.loadtxt(bval_file)
        concatenated_bval = np.concatenate((concatenated_bval, bval))

    # save the concatenated bval as row vector and as integers (not floats)
    np.savetxt(output_path_bval, concatenated_bval, fmt="%d", newline=" ")

    # find all bvec files
    bvec_files = list(root_directory.glob(f"ses-02/dwi/*{protocol}*dwi.bvec"))
    bvec_files.sort()
    if protocol == "AxCaliber":
        bvec_files = bvec_files[:-1]

    output_path_bvec = (
        root_directory
        / "dwi"
        / f"sub-00395_ses-02_acq-{protocol}_dwi_concat.bvec"
    )
    concatenated_bvec = []

    # concatenate all bvec files
    for bvec_file in bvec_files:
        bvec = np.loadtxt(bvec_file)
        concatenated_bvec.append(bvec)

    concatenated_bvec = np.concatenate(concatenated_bvec, axis=1)

    # save the concatenated bvec as column vector and as decimals (not integers)
    np.savetxt(output_path_bvec, concatenated_bvec, fmt="%.6f")
