from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import BIDSDataGrabber
from configparser import ConfigParser
from pathlib import Path


def bidsdata_node(config, name="bidsdata"):
    bidsdata = Node(BIDSDataGrabber(), name=name)
    bidsdata.inputs.base_dir = Path(config["DATASET"]["directory"])
    # set subjects as iterables
    # if subject is not specified, all subjects will be processed
    if (
        config["DATASET"]["subject"] == "all"
        or "subject" not in config["DATASET"]
    ):
        layout = BIDSLayout(Path(config["DATASET"]["directory"]))
        bidsdata.inputs.iterables = ("subject", layout.get_subjects())
    # otherwise pick the one specified in the config file
    else:
        bidsdata.inputs.subject = config["DATASET"]["subject"]
    bidsdata.inputs.output_query = {
        "dwis": dict(suffix="dwi", extension="nii.gz"),
        "bvals": dict(suffix="dwi", extension="bval"),
        "bvecs": dict(suffix="dwi", extension="bvec"),
        "T1": dict(suffix="T1w", extension="nii.gz"),
    }
    return bidsdata
