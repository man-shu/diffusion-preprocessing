from nilearn.image import concat_imgs
from glob import glob
from pathlib import Path
import os
import numpy as np
from joblib import Parallel, delayed
from nibabel import load as nib_load
from drop_unavailable import get_unavailable_subjects


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
available_subjects = []
for sub_dir in sub_dirs:
    if sub_dir.name in unavailable_subjects:
        print(f"Skipping unavailable subject: {sub_dir.name}")
        continue
    else:
        print(f"Checking {sub_dir.name}")
    available_subjects.append(sub_dir.name)

# write the available subject to a file
available_subjects_file = unavail_root_directory / "available_subjects.txt"
with open(available_subjects_file, "w") as f:
    f.writelines("\n".join(available_subjects))
print(f"Available subjects: {available_subjects}")
