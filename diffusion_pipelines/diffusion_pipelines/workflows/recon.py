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

    # Build main workflow
    retval["workflow"] = init_smriprep_wf(
        sloppy=opts.sloppy,
        debug=False,
        derivatives=derivatives,
        freesurfer=opts.run_reconall,
        fs_subjects_dir=opts.fs_subjects_dir,
        hires=opts.hires,
        fs_no_resume=opts.fs_no_resume,
        layout=layout,
        longitudinal=opts.longitudinal,
        low_mem=opts.low_mem,
        msm_sulc=opts.msm_sulc,
        omp_nthreads=omp_nthreads,
        output_dir=str(output_dir),
        run_uuid=run_uuid,
        skull_strip_fixed_seed=opts.skull_strip_fixed_seed,
        skull_strip_mode=opts.skull_strip_mode,
        skull_strip_template=opts.skull_strip_template[0],
        spaces=output_spaces,
        subject_session_list=subject_session_list,
        work_dir=str(work_dir),
        bids_filters=bids_filters,
        cifti_output=opts.cifti_output,
    )

    return wf
