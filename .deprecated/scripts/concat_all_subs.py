from nilearn.image import concat_imgs
from glob import glob
from pathlib import Path
import os
import numpy as np
from joblib import Parallel, delayed

protocols = ["AxCaliber"]
root_directory = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND"
)
out_dir = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND-concat"
)
sub_dirs = list(root_directory.glob("sub-*"))
sub_dirs.sort()
dry = True


def safe_concat_files(out_dir, sub_dir):
    """Wrapper function that catches exceptions and identifies the failing subject"""
    try:
        concat_files(out_dir, sub_dir)
        return {"subject": sub_dir.name, "status": "success"}
    except Exception as e:
        import traceback

        return {
            "subject": sub_dir.name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def concat_files(out_dir, sub_dir):
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
                if protocol == "AxCaliber":
                    dwi_files = dwi_files[:-1]
                print(dwi_files)
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


# Run the parallel processing and collect results
results = Parallel(n_jobs=20)(
    delayed(safe_concat_files)(out_dir, sub_dir) for sub_dir in sub_dirs
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
