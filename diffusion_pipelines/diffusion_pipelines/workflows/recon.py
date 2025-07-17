#!/bin/env python
from pathlib import Path
from smriprep.workflows.base import init_smriprep_wf
from bids.layout import BIDSLayout
from bids.layout.index import BIDSLayoutIndexer
from niworkflows.utils.spaces import Reference, SpatialReferences


def init_recon_wf(output_dir=".", config=None):
    spaces = SpatialReferences(spaces=["MNI152NLin2009aSym", "fsaverage5"])
    spaces.checkpoint()
    wf = init_smriprep_wf(
        output_dir=config["OUTPUT"]["derivatives"],
        work_dir=config["OUTPUT"]["cache"],
        subject_list=config["DATASET"]["subject"],
        layout=BIDSLayout(
            root=Path(config["DATASET"]["directory"]),
            validate=True,
        ),
        # other parameters
        sloppy=True,
        debug=False,
        derivatives=[],
        freesurfer=True,
        fs_subjects_dir=Path(config["OUTPUT"]["derivatives"], "freesurfer"),
        hires=False,
        fs_no_resume=False,
        longitudinal=False,
        low_mem=False,
        msm_sulc=False,
        omp_nthreads=20,
        run_uuid="123",
        skull_strip_mode="auto",
        skull_strip_fixed_seed=True,
        skull_strip_template=Reference.from_string("OASIS30ANTs")[0],
        spaces=spaces,
        bids_filters=None,
        cifti_output="91k",
    )
    return wf
