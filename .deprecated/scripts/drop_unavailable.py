import pandas as pd
import os
from pathlib import Path
from glob import glob


def get_unavailable_subjects(root_directory=None):
    if root_directory is None:
        unavailable_file = Path("unavailable.txt")
    else:
        unavailable_file = root_directory / "unavailable.txt"
    if not unavailable_file.exists():
        print(f"{unavailable_file} does not exist.")
        return []

    with open(unavailable_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def delete_unavailable_subjects(
    root_directory, unavailable_subjects, dry=True
):
    for subject in unavailable_subjects:
        subject_dir = root_directory / subject
        if subject_dir.exists():
            print(f"Deleting {subject_dir}")
            if not dry:
                os.system(f"rm -rf {subject_dir}")
        else:
            print(f"{subject_dir} does not exist, skipping deletion.")


def update_participants_tsv(root_directory, unavailable_subjects, dry=True):
    participants_tsv = root_directory / "participants.tsv"
    if not participants_tsv.exists():
        print(f"{participants_tsv} does not exist.")
        return

    df = pd.read_csv(participants_tsv, sep="\t")
    print(f"Original participants.tsv: {df.shape[0]} subjects")
    df = df[~df["participant_id"].isin(unavailable_subjects)]
    print(f"Filtered participants.tsv: {df.shape[0]} subjects")
    if not dry:
        df.to_csv(participants_tsv, sep="\t", index=False)
    print(f"Updated {participants_tsv} to remove unavailable subjects.")


if __name__ == "__main__":
    root_directory = Path(
        "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/data/WAND-concat"
    )
    unavail_root_directory = Path(
        "/data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/.deprecated/scripts"
    )
    dry = False  # Set to False to actually delete files and update TSV
    unavailable_subjects = get_unavailable_subjects(unavail_root_directory)
    delete_unavailable_subjects(root_directory, unavailable_subjects, dry=dry)
    update_participants_tsv(root_directory, unavailable_subjects, dry=dry)
