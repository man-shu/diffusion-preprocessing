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
    # set subjects as iterables
    # if subject is not specified, all subjects will be processed
    if (
        "subject" not in config["DATASET"]
        or config["DATASET"]["subject"] == "all"
    ):
        layout = BIDSLayout(Path(config["DATASET"]["directory"]))
        sf.iterables = [("subject_id", layout.get_subjects())]
    # otherwise pick the one specified in the config file
    else:
        sf.inputs.subject_id = config["DATASET"]["subject"]
    return sf
