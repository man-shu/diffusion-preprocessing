from glob import glob
from pathlib import Path
import os
from joblib import Parallel, delayed

protocols = ["T1w"]
root_directory = Path(
    "/data/parietal/store3/work/ggomezji/datasets/camcan/cc700/BIDS_20190411/anat"
)
out_dir = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/camcan"
)
sub_dirs = list(root_directory.glob("sub-*"))
sub_dirs.sort()
dry = True


def copy_files(out_dir, sub_dir):
    for protocol in protocols:
        for extension in ["nii.gz", "json"]:
            output_path_dir = out_dir / sub_dir.name / "anat"
            os.makedirs(output_path_dir, exist_ok=True)
            anat_files = list(sub_dir.glob(f"anat/*{protocol}.{extension}"))
            anat_files.sort()
            print(anat_files)
            for anat_file in anat_files:
                output_path_anat = output_path_dir / (
                    f"{sub_dir.name}_{protocol}.{extension}"
                )
                if not dry:
                    # copy the file to the output directory
                    os.system(f"cp {anat_file} {output_path_anat}")
                print(f"Copied {anat_file} to {output_path_anat}")


# Run the parallel processing and collect results
results = Parallel(n_jobs=20)(
    delayed(copy_files)(out_dir, sub_dir) for sub_dir in sub_dirs
)
