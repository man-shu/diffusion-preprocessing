from bids.layout import BIDSLayout, parse_file_entities, Query
from bids.utils import listify
from nipype import Node, Workflow, IdentityInterface
from nipype.interfaces.utility import Function
from nipype.interfaces.io import SelectFiles
from niworkflows.interfaces.bids import BIDSFreeSurferDir
import os
from pathlib import Path

DEFAULT_BIDS_QUERIES = {
    "dwi": {
        "datatype": "dwi",
        "extension": [".nii", ".nii.gz", ".bval", ".bvec"],
    },
    "t1w": {
        "datatype": "anat",
        "suffix": "T1w",
        "desc": "preproc",
        "space": None,
    },
    "brain_mask": {
        "datatype": "anat",
        "suffix": "mask",
        "desc": "brain",
        "space": None,
    },
    "fsnative2t1w_xfm": {
        "datatype": "anat",
        "suffix": "xfm",
        "to": "T1w",
        "extension": [".txt"],
    },
    "figures": {
        "extension": [".svg"],
    },
}


def collect_data(
    bids_dir,
    participant_label,
    session_id=None,
    bids_validate=True,
    bids_filters=None,
):
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    queries = copy.deepcopy(DEFAULT_BIDS_QUERIES)

    session_id = session_id or Query.OPTIONAL
    layout_get_kwargs = {
        "return_type": "file",
        "extension": [".nii", ".nii.gz"],
        "subject": participant_label,
        "session": session_id,
    }

    reserved_entities = [
        ("subject", participant_label),
        ("session", session_id),
    ]

    bids_filters = bids_filters or {}
    for acq, entities in bids_filters.items():
        # BIDS filters will not be able to override subject / session entities
        for entity, param in reserved_entities:
            if param == Query.OPTIONAL:
                continue
            if entity in entities and listify(param) != listify(
                entities[entity]
            ):
                raise ValueError(
                    f'Conflicting entities for "{entity}" found:'
                    f" {entities[entity]} // {param}"
                )

        queries[acq].update(entities)

    for acq, entities in queries.items():
        for entity in list(layout_get_kwargs.keys()):
            if acq == "figures" and entity == "extension":
                continue
            if entity in entities:
                queries[acq][entity] = listify(
                    layout_get_kwargs[entity]
                ) + listify(entities[entity])
                queries[acq][entity] = set(queries[acq][entity])
                queries[acq][entity] = list(queries[acq][entity])
            else:
                queries[acq][entity] = layout_get_kwargs[entity]

    subj_data = {
        dtype: sorted(layout.get(**query)) for dtype, query in queries.items()
    }

    # Filter out unwanted files
    # DWI: only raw files (no derivatives)
    # T1w, brain_mask, fsnative2t1w_xfm: only derivatives
    for dtype, files in subj_data.items():
        selected = []
        for f in files:
            if dtype == "dwi":
                if "derivative" not in f:
                    selected.append(f)
            else:
                if "derivative" in f:
                    selected.append(f)
        subj_data[dtype] = selected

        if (
            dtype != "dwi"
            and len(subj_data[dtype]) == 0
            and not config.recon
            and config.preproc
        ):
            raise FileNotFoundError(
                f"No {dtype} files found for participant {participant_label}."
                "If you are running diffusion preprocessing without "
                "reconstruction, please ensure that the necessary files "
                "are available. Otherwise, use the --recon flag to enable "
                "reconstruction and generate the required files."
            )
    return subj_data, layout


def init_bidsdata_wf(config, name="bidsdata_wf"):

    bids_filters = (
        json.loads(config.bids_filter_file.read_text())
        if config.bids_filter_file
        else None
    )

    layout = BIDSLayout(str(config.bids_dir), validate=False, derivatives=True)

    subject_data, layout = collect_data(
        layout,
        config.participant_label,
        session_id=config.session_label,
        bids_filters=bids_filters,
        queries=DEFAULT_BIDS_QUERIES,
        bids_validate=False,
    )

    # Create SelectFiles node
    sf = Node(
        SelectFiles(
            templates, base_directory=str(config.bids_dir), sort_filelist=True
        ),
        name="selectfiles",
    )

    sf.inputs.acquisition = config.acquisition
    sf.inputs.phase_encoding_direction = config.phase_encoding_direction
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
