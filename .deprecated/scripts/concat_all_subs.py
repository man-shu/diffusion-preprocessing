from nilearn.image import concat_imgs
from glob import glob
from pathlib import Path
import os
import numpy as np

protocols = ["AxCaliber"]
root_directory = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND-bids"
)
out_dir = Path("/data/parietal/store3/work/haggarwa/diffusion/data/WAND-bids")
sub_dirs = list(root_directory.glob("sub-*"))
sub_dirs.sort()
dry = False

for sub_dir in sub_dirs:
    for protocol in protocols:
        for extension in ["nii.gz", "bval", "bvec"]:
            output_path_dir = out_dir / sub_dir.name / "ses-02" / "dwi"
            os.makedirs(output_path_dir, exist_ok=True)
            if extension == "nii.gz":
                dwi_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.{extension}")
                )
                dwi_files.sort()
                if protocol == "AxCaliber":
                    dwi_files = dwi_files[:-1]
                concatenated_dwi = concat_imgs(dwi_files)
                output_path_dwi = output_path_dir / (
                    f"{sub_dir.name}_ses-02_acq-{protocol}"
                    f"Concat_dwi.{extension}"
                )
                if not dry:
                    concatenated_dwi.to_filename(output_path_dwi)
                print(f"Saved {output_path_dwi}")
            elif extension == "bval":
                # find all bval files
                bval_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.{extension}")
                )
                bval_files.sort()
                if protocol == "AxCaliber":
                    bval_files = bval_files[:-1]
                output_path_bval = output_path_dir / (
                    f"{sub_dir.name}_ses-02_acq-{protocol}"
                    f"Concat_dwi.{extension}"
                )
                concatenated_bval = []
                # concatenate all bval files
                for bval_file in bval_files:
                    bval = np.loadtxt(bval_file)
                    concatenated_bval = np.concatenate(
                        (concatenated_bval, bval)
                    )
                if not dry:
                    # save the concatenated bval as row vector and as integers
                    # (not floats)
                    np.savetxt(
                        output_path_bval,
                        concatenated_bval,
                        fmt="%d",
                        newline=" ",
                    )
                print(f"Saved {output_path_bval}")
            elif extension == "bvec":
                # find all bvec files
                bvec_files = list(
                    sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.bvec")
                )
                bvec_files.sort()
                if protocol == "AxCaliber":
                    bvec_files = bvec_files[:-1]
                output_path_bvec = output_path_dir / (
                    f"{sub_dir.name}_ses-02_acq-{protocol}"
                    f"Concat_dwi.{extension}"
                )
                concatenated_bvec = []
                # concatenate all bvec files
                for bvec_file in bvec_files:
                    bvec = np.loadtxt(bvec_file)
                    concatenated_bvec.append(bvec)

                concatenated_bvec = np.concatenate(concatenated_bvec, axis=1)
                if not dry:
                    # save the concatenated bvec as column vector and as
                    # decimals (not integers)
                    np.savetxt(output_path_bvec, concatenated_bvec, fmt="%.6f")
                print(f"Saved {output_path_bvec}")
