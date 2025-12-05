from glob import glob
from pathlib import Path
import os
import shutil
from joblib import Parallel, delayed

protocols = ["AxCaliberConcat"]
root_directory = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/WAND-concat/derivatives/preprocess/"
)
out_dir = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/WAND-concat/derivatives/diffusion_preprocess/"
)
sub_dirs = list(root_directory.glob("sub-*"))
sub_dirs.sort()
dry = False


def safe_copy_files(out_dir, sub_dir):
    """Wrapper function that catches exceptions and identifies the failing subject"""
    try:
        copy_files(out_dir, sub_dir)
        return {"subject": sub_dir.name, "status": "success"}
    except Exception as e:
        import traceback

        return {
            "subject": sub_dir.name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def copy_files(out_dir, sub_dir):
    for protocol in protocols:
        for extension in ["nii.gz", "bvec"]:
            output_path_dir = out_dir / sub_dir.name / "ses-02" / "dwi"
            os.makedirs(output_path_dir, exist_ok=True)

            # Find the single file for this extension
            dwi_files = list(sub_dir.glob(f"*{protocol}*dwi.{extension}"))
            dwi_files.sort()

            for dwi_file in dwi_files:
                print(f"Found {extension} file: {dwi_file}")
                if not dry:
                    shutil.copy2(dwi_file, str(output_path_dir) + "/")
                print(f"Copied {dwi_file} -> {str(output_path_dir) + '/'}")


# Run the parallel processing and collect results
results = Parallel(n_jobs=20)(
    delayed(safe_copy_files)(out_dir, sub_dir) for sub_dir in sub_dirs
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
