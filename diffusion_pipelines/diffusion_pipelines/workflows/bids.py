from bids.layout import BIDSLayout, parse_file_entities
from nipype import Node, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import SelectFiles
from niworkflows.interfaces.bids import BIDSFreeSurferDir
import os
from pathlib import Path


def init_bidsdata_wf(config, name="bidsdata_wf"):

    ### SelectFiles node
    # String template with {}-based strings
    templates = {
        "preprocessed_t1": (
            str(
                Path(
                    (
                        "derivatives/smriprep/sub-{subject_id}"
                        "/ses-{session_id}/anat/sub-{subject_id}"
                        "_ses-{session_id}*_desc-preproc_T1w.nii.gz"
                    )
                )
            )
            if config.preproc_t1 is None and config.recon is True
            else str(config.preproc_t1)
        ),
        "preprocessed_t1_mask": (
            str(
                Path(
                    (
                        "derivatives/smriprep/sub-{subject_id}"
                        "/ses-{session_id}/anat/sub-{subject_id}"
                        "_ses-{session_id}*_desc-brain_mask.nii.gz"
                    )
                )
            )
            if config.preproc_t1_mask is None and config.recon is True
            else str(config.preproc_t1_mask)
        ),
        "fsnative2t1w_xfm": (
            str(
                Path(
                    (
                        "derivatives/smriprep/sub-{subject_id}"
                        "/ses-{session_id}/anat/sub-{subject_id}_ses-"
                        "{session_id}*_from-fsnative_to-T1w_mode-image_xfm.txt"
                    )
                )
            )
            if config.fs_native_to_t1w_xfm is None and config.recon is True
            else str(config.fs_native_to_t1w_xfm)
        ),
        "plot_recon_surface_on_t1": (
            str(
                Path(
                    (
                        "derivatives/smriprep/sub-{subject_id}/figures"
                        "/sub-{subject_id}_ses-{session_id}*"
                        "_desc-reconall_T1w.svg"
                    )
                )
            )
        ),
        "plot_recon_segmentations_on_t1": (
            str(
                Path(
                    (
                        "derivatives/smriprep/sub-{subject_id}/figures"
                        "/sub-{subject_id}_ses-{session_id}*_dseg.svg"
                    )
                )
            )
        ),
        "dwi": (
            str(
                Path(
                    (
                        "sub-{subject_id}/ses-{session_id}/dwi/"
                        "sub-{subject_id}_ses-{session_id}*"
                        "_acq-{acquisition}*_dwi.nii.gz"
                    )
                )
            )
        ),
        "bval": (
            str(
                Path(
                    (
                        "sub-{subject_id}/ses-{session_id}/dwi/"
                        "sub-{subject_id}_ses-{session_id}*"
                        "_acq-{acquisition}*_dwi.bval"
                    )
                )
            )
        ),
        "bvec": (
            str(
                Path(
                    (
                        "sub-{subject_id}/ses-{session_id}/dwi/"
                        "sub-{subject_id}_ses-{session_id}*"
                        "_acq-{acquisition}*_dwi.bvec"
                    )
                )
            )
        ),
    }

    # Create SelectFiles node
    sf = Node(
        SelectFiles(
            templates, base_directory=str(config.bids_dir), sort_filelist=True
        ),
        name="selectfiles",
    )

    sf.inputs.acquisition = config.acquisition
    layout = BIDSLayout(str(config.bids_dir))
    # set subjects as iterables
    # if subject is not specified, all subjects will be processed
    if config.participant_label == ["all"]:
        sf.iterables = [("subject_id", layout.get_subjects())]
    # otherwise, only the specified subjects will be processed
    elif isinstance(config.participant_label, list):
        for subject in config.participant_label:
            if subject not in layout.get_subjects():
                raise ValueError(f"Subject {subject} not found in dataset")
        sf.iterables = [("subject_id", config.participant_label)]

    if config.session_label is not None:
        sf.iterables.append(("session_id", config.session_label))
    # all sessions
    else:
        sf.iterables.append(("session_id", layout.get_sessions()))

    ### Node to decode entities
    def decode_entities(file_name):
        from bids.layout import parse_file_entities

        print(f"Decoding entities from {file_name}")
        return parse_file_entities(file_name)

    DecodeEntities = Function(
        input_names=["file_name"],
        output_names=["bids_entities"],
        function=decode_entities,
    )

    decode_entities = Node(DecodeEntities, name="decode_entities")

    ### Node to only pick one file
    def pick_files(file_name_list):
        from bids.layout import parse_file_entities

        if isinstance(file_name_list, str):
            return file_name_list

        # Filter files based on space
        for file_name in file_name_list:
            entities = parse_file_entities(file_name)
            if entities["space"] is None:
                return file_name

    PickFiles = Function(
        input_names=["file_name_list"],
        output_names=["file_name"],
        function=pick_files,
    )
    pick_t1 = Node(PickFiles, name="pick_t1")
    pick_mask = pick_t1.clone("pick_mask")
    pick_xfm = pick_t1.clone("pick_xfm")

    output = Node(
        IdentityInterface(
            fields=[
                "preprocessed_t1",
                "preprocessed_t1_mask",
                "fsnative2t1w_xfm",
                "dwi",
                "bval",
                "bvec",
                "plot_recon_surface_on_t1",
                "plot_recon_segmentations_on_t1",
            ]
        ),
        name="output",
    )

    bidsdata_wf = Workflow(name=name)
    bidsdata_wf.connect(
        [
            (sf, decode_entities, [("dwi", "file_name")]),
            (sf, pick_t1, [("preprocessed_t1", "file_name_list")]),
            (sf, pick_mask, [("preprocessed_t1_mask", "file_name_list")]),
            (sf, pick_xfm, [("fsnative2t1w_xfm", "file_name_list")]),
            (sf, output, [("dwi", "dwi")]),
            (sf, output, [("bval", "bval")]),
            (sf, output, [("bvec", "bvec")]),
            (
                sf,
                output,
                [("plot_recon_surface_on_t1", "plot_recon_surface_on_t1")],
            ),
            (
                sf,
                output,
                [
                    (
                        "plot_recon_segmentations_on_t1",
                        "plot_recon_segmentations_on_t1",
                    )
                ],
            ),
            (pick_t1, output, [("file_name", "preprocessed_t1")]),
            (pick_mask, output, [("file_name", "preprocessed_t1_mask")]),
            (pick_xfm, output, [("file_name", "fsnative2t1w_xfm")]),
        ]
    )

    return bidsdata_wf
