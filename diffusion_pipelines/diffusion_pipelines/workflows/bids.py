from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import BIDSDataGrabber
from configparser import ConfigParser
from pathlib import Path


def _set_inputs(config, wf):
    wf.inputs.input_data.directory = Path(config["DATASET"]["directory"])
    if config["DATASET"]["subject"] == "all" or None:
        layout = BIDSLayout(Path(config["DATASET"]["directory"]))
        wf.inputs.input_data.iterables = ("subject", layout.get_subjects())
    else:
        wf.inputs.input_data.iterables = (
            "subject",
            config["DATASET"]["subject"],
        )
    return wf


def _bidsdata_wf(name="bidsdata_wf"):
    input_data = Node(
        IdentityInterface(
            fields=["directory", "subject"],
        ),
        name="input_data",
    )
    bidsdata = Node(BIDSDataGrabber(), name="bidsdata")
    bidsdata.inputs.output_query = {
        "dwis": dict(suffix="dwi", extension="nii.gz"),
        "bvals": dict(suffix="dwi", extension="bval"),
        "bvecs": dict(suffix="dwi", extension="bvec"),
        "T1": dict(suffix="T1w", extension="nii.gz"),
    }
    workflow = Workflow(name=name)
    workflow.connect(input_data, "directory", bidsdata, "base_dir")
    workflow.connect(input_data, "subject", bidsdata, "subject")
    return workflow


def init_bidsdata_wf(config):
    wf = _bidsdata_wf()
    wf = _set_inputs(config, wf)
    return wf
