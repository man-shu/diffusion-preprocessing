from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataSink
from configparser import ConfigParser
from pathlib import Path


def sink_node(config, name="sink_wf"):
    sink = Node(DataSink(), name="sink")
    sink.inputs.base_directory = config["OUTPUT"]["directory"]
    # set subjects as iterables
    # if subject is not specified, all subjects will be processed
    if (
        config["DATASET"]["subject"] == "all"
        or "subject" not in config["DATASET"]
    ):
        layout = BIDSLayout(Path(config["DATASET"]["directory"]))
        sink.inputs.iterables = ("subject", layout.get_subjects())
    # otherwise pick the one specified in the config file
    else:
        sink.inputs.subject = config["DATASET"]["subject"]
    return sink
