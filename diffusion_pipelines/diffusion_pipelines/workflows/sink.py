from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataSink
from configparser import ConfigParser
from pathlib import Path


def sink_node(config, name="sink_wf"):
    sink = Node(DataSink(), name="sink")
    sink.inputs.base_directory = config["OUTPUT"]["directory"]
    return sink
