from glob import glob
from pathlib import Path
import os
import shutil
from joblib import Parallel, delayed

protocols = ["CHARMED_dir-AP"]
root_directory = Path("/data/parietal/store4/data/WAND")
out_dir = Path(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND-concat"
)
sub_dirs = list(root_directory.glob("sub-*"))
sub_dirs.sort()
dry = True


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
        for extension in ["nii.gz", "bval", "bvec"]:
            output_path_dir = out_dir / sub_dir.name / "ses-02" / "dwi"
            os.makedirs(output_path_dir, exist_ok=True)

            # Find the single file for this extension
            dwi_files = list(
                sub_dir.glob(f"ses-02/dwi/*{protocol}*dwi.{extension}")
            )
            dwi_files.sort()

            if dwi_files:
                # Take the first (and should be only) file
                dwi_file = dwi_files[0]
                print(f"Found {extension} file: {dwi_file}")

                output_path = output_path_dir / (
                    f"{sub_dir.name}_ses-02_acq-{protocol}_dwi.{extension}"
                )
                if not dry:
                    shutil.copy2(dwi_file, output_path)
                print(f"Copied {dwi_file} -> {output_path}")
            else:
                print(
                    f"Warning: No {extension} file found for protocol {protocol}"
                )


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
