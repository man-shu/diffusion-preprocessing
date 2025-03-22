from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import BIDSDataGrabber, SelectFiles
from configparser import ConfigParser
from pathlib import Path


def bidsdata_node(config, name="bidsdata"):
    # String template with {}-based strings
    templates = {
        "T1": "sub-{subject_id}/*/anat/sub-{subject_id}*_T1w.nii.gz",
        "dwi": (
            "sub-{subject_id}/*/dwi/sub-{subject_id}*_acq-{acquisition}"
            "*_dwi.nii.gz"
        ),
        "bval": (
            "sub-{subject_id}/*/dwi/sub-{subject_id}*_acq-{acquisition}"
            "*_dwi.bval"
        ),
        "bvec": (
            "sub-{subject_id}/*/dwi/sub-{subject_id}*_acq-{acquisition}"
            "*_dwi.bvec"
        ),
    }

    # Create SelectFiles node
    sf = Node(
        SelectFiles(
            templates,
            base_directory=config["DATASET"]["directory"],
            sort_filelist=True,
        ),
        name="selectfiles",
    )

    sf.inputs.acquisition = config["DATASET"]["acquisition"]
    layout = BIDSLayout(Path(config["DATASET"]["directory"]))
    # set subjects as iterables
    # if subject is not specified, all subjects will be processed
    if config["DATASET"]["subject"] == "all":
        sf.iterables = [("subject_id", layout.get_subjects())]
    # otherwise, only the specified subjects will be processed
    elif isinstance(config["DATASET"]["subject"], list):
        for subject in config["DATASET"]["subject"]:
            if subject not in layout.get_subjects():
                raise ValueError(f"Subject {subject} not found in dataset")
        sf.iterables = [("subject_id", config["DATASET"]["subject"])]
    return sf
