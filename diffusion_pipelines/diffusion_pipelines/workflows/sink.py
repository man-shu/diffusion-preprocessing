from bids.layout import BIDSLayout
from nipype import IdentityInterface, Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataSink
from configparser import ConfigParser
from pathlib import Path


def init_sink_wf(config, name="sink_wf"):

    inputnode = Node(
        IdentityInterface(fields=["bids_entities"]), name="sinkinputnode"
    )

    ### build the full file name
    def build_substitutions(bids_entities):

        def _build_bids(bids_entities):
            replacements = {
                "subject": "sub-",
                "session": "_ses-",
                "acquisition": "_acq-",
                "direction": "_dir-",
                "part": "_part-",
            }
            bids_name = ""
            for key, value in bids_entities.items():
                if key in replacements:
                    bids_name += f"{replacements[key]}{value}"
            return bids_name

        bids_name = _build_bids(bids_entities)
        substitutions = [
            ("_subject_id_", "sub-"),
            (
                "mean_bzero_brain_mask_trans",
                f"{bids_name}_space-individualT1_desc-mask_dwi",
            ),
            (
                "vol0000_flirt_merged_trans",
                f"{bids_name}_space-individualT1_dwi",
            ),
            ("vol0000_flirt_merged", f"{bids_name}_desc-eddycorrected_dwi"),
        ]
        return substitutions

    BuildSubstitutions = Function(
        input_names=["bids_entities"],
        output_names=["substitutions"],
        function=build_substitutions,
    )
    build_substitutions = Node(BuildSubstitutions, name="build_substitutions")

    ### DataSink node
    sink = Node(DataSink(), name="sink")
    sink.inputs.base_directory = config["OUTPUT"]["derivatives"]

    # Create the workflow
    sink_wf = Workflow(name=name)
    sink_wf.connect(
        [
            (
                inputnode,
                build_substitutions,
                [("bids_entities", "bids_entities")],
            ),
            (build_substitutions, sink, [("substitutions", "substitutions")]),
        ]
    )
    return sink_wf
