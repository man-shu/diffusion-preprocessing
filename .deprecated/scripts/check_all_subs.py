from nilearn.image import concat_imgs
from glob import glob
from pathlib import Path
import os
import numpy as np
from joblib import Parallel, delayed
from nibabel import load as nib_load

protocols = ["AxCaliberConcat"]
root_directory = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND"
)
out_dir = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND-concat"
)
sub_dirs = list(root_directory.glob("sub-*"))
sub_dirs.sort()
dry = True


def safe_check_file_lengths(out_dir, sub_dir):
    """Wrapper function that catches exceptions and identifies the failing subject"""
    try:
        check_file_lengths(out_dir, sub_dir)
        return {"subject": sub_dir.name, "status": "success"}
    except Exception as e:
        import traceback

        return {
            "subject": sub_dir.name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def check_file_lengths(out_dir, sub_dir):
    for protocol in protocols:
        for extension in ["nii.gz", "bval", "bvec"]:
            output_path_dir = out_dir / sub_dir.name / "ses-02" / "dwi"
            os.makedirs(output_path_dir, exist_ok=True)
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


# Run the parallel processing and collect results
results = Parallel(n_jobs=20)(
    delayed(safe_check_file_lengths)(out_dir, sub_dir) for sub_dir in sub_dirs
)

# Check results for any failures
failures = [res for res in results if res.get("status") == "failed"]
if failures:
    print("\n===== FAILED SUBJECTS =====")
    for failure in failures:
        print(f"\nSubject {failure['subject']} failed with error:")
        print(f"  {failure['error']}")
        print("\nTraceback:")
        print(failure["traceback"])
else:
    print("\nAll subjects processed successfully!")
