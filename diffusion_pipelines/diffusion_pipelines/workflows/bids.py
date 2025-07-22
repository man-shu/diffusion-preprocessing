from bids.layout import BIDSLayout, parse_file_entities
from nipype import Node, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import SelectFiles


def init_bidsdata_wf(config, name="bidsdata_wf"):

    ### SelectFiles node
    # String template with {}-based strings
    templates = {
        "preprocessed_t1": (
            (
                "derivatives/smriprep/sub-{subject_id}/*/anat/sub-{subject_id}"
                "_ses-??_desc-preproc_T1w.nii.gz"
            )
            if config.preproc_t1 is None
            else str(config.preproc_t1)
        ),
        "preprocessed_t1_mask": (
            (
                "derivatives/smriprep/sub-{subject_id}/*/anat/sub-{subject_id}"
                "_ses-??_desc-brain_mask.nii.gz"
            )
            if config.preproc_t1_mask is None
            else str(config.preproc_t1_mask)
        ),
        "plot_recon_surface_on_t1": (
            "derivatives/smriprep/sub-{subject_id}/figures"
            "/sub-{subject_id}*_desc-reconall_T1w.svg"
        ),
        "plot_recon_segmentations_on_t1": (
            "derivatives/smriprep/sub-{subject_id}/figures"
            "/sub-{subject_id}*_dseg.svg"
        ),
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

    bidsdata_wf = Workflow(name=name)
    bidsdata_wf.connect(sf, "dwi", decode_entities, "file_name")

    return bidsdata_wf
