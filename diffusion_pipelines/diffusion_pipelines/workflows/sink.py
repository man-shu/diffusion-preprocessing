from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataSink
from configparser import ConfigParser


def _get_config(config_file):
    config = ConfigParser()
    config.read(config_file)
    return config


def _set_inputs(config, wf):
    wf.inputs.base_directory = Path(config["OUTPUT"]["directory"])
    if config["DATASET"]["subject"] == "all" or None:
        layout = BIDSLayout(Path(config["DATASET"]["directory"]))
        wf.inputs.input_data.iterables = ("subject", layout.get_subjects())
    else:
        wf.inputs.input_data.iterables = (
            "subject",
            config["OUTPUT"]["subject"],
        )
    return wf


def _sink_wf(name="sink_wf"):
    input_data = Node(
        IdentityInterface(
            fields=["directory", "subject"],
        ),
        name="input_data",
    )
    sink = Node(DataSink(), name="sink")
    workflow = Workflow(name=name)
    workflow.connect(input_data, "directory", sink, "base_directory")
    workflow.connect(input_data, "subject", sink, "container")
    return workflow


def init_sink_wf(config_file):
    wf = _sink_wf()
    config = _get_config(config_file)
    wf = _set_inputs(config, wf)
    return wf
