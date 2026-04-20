#!/bin/env python
from pathlib import Path
from smriprep.workflows.base import init_smriprep_wf
from bids.layout import BIDSLayout
from niworkflows.utils.bids import collect_participants


def _subject_session_list(config):
    # First check that bids_dir looks like a BIDS folder
    bids_dir = config.bids_dir.resolve()
    layout = BIDSLayout(str(bids_dir), validate=False)
    subject_list = collect_participants(
        layout, participant_label=config.participant_label
    )
    session_list = config.session_label or []

    subject_session_list = []
    for subject in subject_list:
        sessions = (
            layout.get_sessions(
                scope="raw",
                subject=subject,
                session=session_list or Query.OPTIONAL,
                suffix=[
                    "T1w",
                    "T2w",
                ],
            )
            or None
        )

        if config.subject_anatomical_reference == "sessionwise":
            if not sessions:
                raise RuntimeError(
                    '--subject-anatomical-reference "sessionwise" was,'
                    " requested, but no sessions "
                    f"found for subject {subject}."
                )
            for session in sessions:
                subject_session_list.append((subject, session))
        else:
            # This will use all sessions either found by layout or passed in
            # via --session-id
            subject_session_list.append((subject, sessions))

    return subject_session_list


def init_recon_wf(output_dir, config):
    config.output_spaces.checkpoint()

    # Initialize BIDSLayout
    layout = BIDSLayout(root=config.bids_dir, validate=True)

    wf = init_smriprep_wf(
        output_dir=str(config.output_dir),
        work_dir=str(config.work_dir),
        layout=layout,
        # other parameters
        sloppy=config.sloppy,
        debug=False,
        derivatives=[],
        freesurfer=True,
        fs_subjects_dir=config.fs_subjects_dir,
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
        spaces=config.output_spaces,
        bids_filters=None,
        cifti_output=config.cifti_output,
        subject_session_list=_subject_session_list(config),
    )
    return wf
