#!/bin/env python
from pathlib import Path
from smriprep.workflows.base import init_smriprep_wf
from bids.layout import BIDSLayout


def init_recon_wf(config):
    output_spaces = config.output_spaces.checkpoint()

    # Initialize BIDSLayout
    layout = BIDSLayout(root=config.bids_dir, validate=True)

    wf = init_smriprep_wf(
        output_dir=str(config.output_dir),
        work_dir=str(config.work_dir),
        subject_list=config.participant_label,
        layout=layout,
        # other parameters
        sloppy=config.sloppy,
        debug=False,
        derivatives=[],
        freesurfer=config.run_reconall,
        fs_subjects_dir=Path(config.output_dir, "freesurfer"),
        hires=config.hires,
        fs_no_resume=config.fs_no_resume,
        longitudinal=config.longitudinal,
        low_mem=config.low_mem,
        msm_sulc=config.msm_sulc,
        omp_nthreads=config.omp_nthreads,
        run_uuid=config.run_uuid,
        skull_strip_mode=config.skull_strip_mode,
        skull_strip_fixed_seed=config.skull_strip_fixed_seed,
        skull_strip_template=config.skull_strip_template[0],
        spaces=output_spaces,
        bids_filters=None,
        cifti_output=config.cifti_output,
    )
    return wf
