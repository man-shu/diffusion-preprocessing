from bids.layout import BIDSLayout
from nipype.pipeline import Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import BIDSDataGrabber


def _get_config(config_file):
    config = ConfigParser()
    config.read(config_file)
    return config


def _set_inputs(config, wf):
    wf.inputs.base_dir = Path(config["DATASET"]["directory"])
    layout = BIDSLayout(Path(config["DATASET"]["directory"]))
    if config["DATASET"]["subject"] == "all" or None:
        wf.inputs.input_data.iterables = ("subject", layout.get_subjects())
    else:
        wf.inputs.input_data.iterables = (
            "subject",
            config["DATASET"]["subject"],
        )
    return wf


def _bidsdata_wf(name="bidsdata", output_dir="."):
    input_data = Node(
        IdentityInterface(
            fields=["directory", "subject"],
        ),
        name="input_data",
    )
    bidsdata = Node(BIDSDataGrabber(), name="bids-grabber")
    bidsdata.inputs.output_query = {
        "dwis": dict(suffix="dwi", extension="nii.gz"),
        "bvals": dict(suffix="dwi", extension="bval"),
        "bvecs": dict(suffix="dwi", extension="bvec"),
        "T1": dict(suffix="T1w", extension="nii.gz"),
    }
    workflow = Workflow(name="data_wf", base_dir=output_dir)
    workflow.connect(input_data, "directory", bidsdata, "base_dir")
    workflow.connect(input_data, "subject", bidsdata, "subject")
    return workflow


def init_bidsdata_wf(config_file, output_dir="."):
    wf = _bidsdata_wf(output_dir=output_dir)
    config = _get_config(config_file)
    wf = _set_inputs(config, wf)
    return wf
