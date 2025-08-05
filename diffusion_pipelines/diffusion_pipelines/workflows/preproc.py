import inspect
from nipype import IdentityInterface, Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.interfaces.freesurfer as fs
from niflow.nipype1.workflows.dmri.fsl.epi import create_eddy_correct_pipeline
from nipype.interfaces import utility
from nipype.interfaces.utility.wrappers import Function
from .report import init_report_wf
from .bids import init_bidsdata_wf
from .sink import init_sink_wf
from pathlib import Path
from niworkflows.anat.coregistration import init_bbreg_wf
from niworkflows.interfaces.bids import BIDSFreeSurferDir
import os


def _set_inputs_outputs(config, preproc_wf):
    # bids dataset
    bidsdata_wf = init_bidsdata_wf(config=config)
    # outputs
    sink_wf = init_sink_wf(config=config)
    # get freesurfer directory
    fsdir = Node(
        BIDSFreeSurferDir(
            derivatives=config.output_dir,
            freesurfer_home=os.getenv("FREESURFER_HOME"),
            spaces=config.output_spaces.get_fs_spaces(),
        ),
        name="fsdir_preproc",
    )
    # create the full workflow
    preproc_wf.connect(
        [
            (
                bidsdata_wf,
                preproc_wf.get_node("input_subject"),
                [
                    ("selectfiles.preprocessed_t1", "preprocessed_t1"),
                    (
                        "selectfiles.preprocessed_t1_mask",
                        "preprocessed_t1_mask",
                    ),
                    (
                        "selectfiles.fsnative2t1w_xfm",
                        "fsnative2t1w_xfm",
                    ),
                    ("selectfiles.dwi", "dwi"),
                    ("selectfiles.bval", "bval"),
                    ("selectfiles.bvec", "bvec"),
                    ("decode_entities.bids_entities", "bids_entities"),
                    (
                        "selectfiles.plot_recon_surface_on_t1",
                        "plot_recon_surface_on_t1",
                    ),
                    (
                        "selectfiles.plot_recon_segmentations_on_t1",
                        "plot_recon_segmentations_on_t1",
                    ),
                ],
            ),
            (
                fsdir,
                preproc_wf.get_node("input_subject"),
                [("subjects_dir", "fs_subjects_dir")],
            ),
            (
                bidsdata_wf,
                sink_wf,
                [
                    (
                        "decode_entities.bids_entities",
                        "sinkinputnode.bids_entities",
                    )
                ],
            ),
            (
                preproc_wf.get_node("output"),
                sink_wf.get_node("sink"),
                [
                    (
                        "dwi_rigid_registered",
                        "diffusion_preprocess.@registered_dwi",
                    ),
                    ("eddy_corrected", "diffusion_preprocess.@eddy_corrected"),
                    ("mask", "diffusion_preprocess.@mask"),
                    (
                        "registered_mean_bzero",
                        "diffusion_preprocess.@registered_mean_bzero",
                    ),
                    ("bvec_rotated", "diffusion_preprocess.@bvec_rotated"),
                ],
            ),
            (
                preproc_wf.get_node("report"),
                sink_wf.get_node("sink"),
                [
                    (
                        "report_outputnode.out_file",
                        "diffusion_preprocess.@report",
                    )
                ],
            ),
        ]
    )
    return preproc_wf


def _preprocess_wf(
    config, name="diffusion_preprocess", bet_frac=0.34, output_dir="."
):

    def _get_mean_bzero(
        dwi_file, bval, prefix, bval_threshold=config.b0_threshold
    ):
        """Mean of the b=0 volumes of the input dwi file."""
        import os
        from nilearn.image import index_img, mean_img

        import numpy as np

        bvals = np.loadtxt(bval)
        # get the index of the b=0 volumes
        bzero_index = np.where(bvals <= bval_threshold)[0]
        # get the mean image of the b=0 volumes
        mean_bzero_img = mean_img(index_img(dwi_file, bzero_index))
        # save the mean image
        out_file = os.path.join(os.getcwd(), f"{prefix}_mean_bzero.nii.gz")
        mean_bzero_img.to_filename(out_file)

        return out_file

    # define a function to get the mean of b=0 of the input dwi file
    MeanBZero = Function(
        input_names=["dwi_file", "bval", "prefix"],
        output_names=["out"],
        function=_get_mean_bzero,
    )
    # this node is used to get the mean of b=0 of the input dwi file
    get_intial_mean_bzero = Node(MeanBZero, name="get_intial_mean_bzero")
    get_intial_mean_bzero.inputs.prefix = "initial"

    # this node is used to get the mean of b=0 of the eddy-corrected dwi file
    get_eddy_mean_bzero = get_intial_mean_bzero.clone("get_eddy_mean_bzero")
    get_eddy_mean_bzero.inputs.prefix = "eddy"

    # this node is used to get the mean of b=0 of the eddy-corrected and registered dwi file
    get_registered_mean_bzero = get_intial_mean_bzero.clone(
        "get_registered_mean_bzero"
    )
    get_registered_mean_bzero.inputs.prefix = "registered"

    def get_subject_id(bids_entities):
        """Get the subject id from the BIDS entities."""
        return f"sub-{bids_entities['subject']}"

    GetSubjectID = Function(
        input_names=["bids_entities"],
        output_names=["subject_id"],
        function=get_subject_id,
    )
    get_subject_id_node = Node(GetSubjectID, name="get_subject_id")

    def rotate_gradients_lta(lta_file, gradient_file):
        import os
        import os.path
        import numpy as np
        from scipy.linalg import polar

        # Parse the LTA file to extract the transformation matrix
        with open(lta_file, "r") as f:
            lines = f.readlines()

        # Find the matrix section (after "1 4 4" line)
        matrix_start = None
        for i, line in enumerate(lines):
            if line.strip() == "1 4 4":
                matrix_start = i + 1
                break

        if matrix_start is None:
            raise ValueError(
                "Could not find transformation matrix in LTA file"
            )

        # Read the 4x4 transformation matrix
        matrix_lines = lines[matrix_start : matrix_start + 4]
        affine = np.array(
            [list(map(float, line.strip().split())) for line in matrix_lines]
        )

        # Extract rotation component using polar decomposition
        u, p = polar(affine[:3, :3], side="right")

        # Load and rotate gradients
        gradients = np.loadtxt(gradient_file)
        new_gradients = np.linalg.solve(u, gradients).T

        # Save rotated gradients
        name, ext = os.path.splitext(os.path.basename(gradient_file))
        output_name = os.path.join(os.getcwd(), f"{name}_rot{ext}")
        np.savetxt(output_name, new_gradients)

        return output_name

    RotateGradientsLTA = Function(
        input_names=["lta_file", "gradient_file"],
        output_names=["rotated_gradients"],
        function=rotate_gradients_lta,
    )

    input_subject = Node(
        IdentityInterface(
            fields=[
                "preprocessed_t1",
                "preprocessed_t1_mask",
                "fsnative2t1w_xfm",
                "fs_subjects_dir",
                "dwi",
                "bval",
                "bvec",
                "bids_entities",
                "plot_recon_surface_on_t1",
                "plot_recon_segmentations_on_t1",
            ],
        ),
        name="input_subject",
    )
    input_template = Node(
        IdentityInterface(
            fields=["T2", "mask"],
        ),
        name="input_template",
    )

    output = Node(
        IdentityInterface(
            fields=[
                "dwi_rigid_registered",
                "bval",
                "bvec_rotated",
                "mask",
                "rigid_dwi_2_t1",
                "eddy_corrected",
                "dwi_initial",
                "registered_mean_bzero",
                "dwi_masked",
                "bet_mask",
                "t1_initial",
                "t1_masked",
                "bids_entities",
                "plot_recon_surface_on_t1",
                "plot_recon_segmentations_on_t1",
            ]
        ),
        name="output",
    )

    fslroi = Node(interface=fsl.ExtractROI(), name="fslroi")
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    strip_mean_bzero = Node(interface=fsl.ApplyMask(), name="strip_mean_bzero")

    strip_dwi = Node(interface=fsl.ApplyMask(), name="strip_dwi")

    strip_t1 = Node(interface=fsl.ApplyMask(), name="strip_t1")

    bet = Node(interface=fsl.BET(), name="bet")
    bet.inputs.mask = True
    bet.inputs.frac = bet_frac

    eddycorrect = create_eddy_correct_pipeline("eddycorrect")
    eddycorrect.inputs.inputnode.ref_num = 0

    rotate_gradients = Node(
        interface=RotateGradientsLTA, name="rotate_gradients"
    )

    apply_registration = Node(
        interface=fs.ApplyVolTransform(), name="apply_registration"
    )

    apply_registration_mask = Node(
        interface=fs.ApplyVolTransform(), name="apply_registration_mask"
    )

    bbreg_wf = init_bbreg_wf(
        name="bbreg_wf",
        omp_nthreads=config.omp_nthreads,
        use_bbr=True,
        epi2t1w_dof=12,
    )

    report = init_report_wf(
        name="report", calling_wf_name=name, output_dir=output_dir
    )

    workflow = Workflow(name=name, base_dir=output_dir)
    workflow.connect(
        [
            # get mean of b=0 volumes of the input dwi file
            (input_subject, get_intial_mean_bzero, [("dwi", "dwi_file")]),
            (input_subject, get_intial_mean_bzero, [("bval", "bval")]),
            # get mask from the mean b=0 volumes
            (get_intial_mean_bzero, bet, [("out", "in_file")]),
            # apply mask to mean b=0 output
            (get_intial_mean_bzero, strip_mean_bzero, [("out", "in_file")]),
            (bet, strip_mean_bzero, [("mask_file", "mask_file")]),
            # apply the mask to the dwi
            (input_subject, strip_dwi, [("dwi", "in_file")]),
            (bet, strip_dwi, [("mask_file", "mask_file")]),
            # apply mask to the preprocessed subject T1
            (input_subject, strip_t1, [("preprocessed_t1", "in_file")]),
            (input_subject, strip_t1, [("preprocessed_t1_mask", "mask_file")]),
            # edddy correct the skull-stripped dwi
            (strip_dwi, eddycorrect, [("out_file", "inputnode.in_file")]),
            # compute the mean of the b=0 eddycorrected volumes
            (
                eddycorrect,
                get_eddy_mean_bzero,
                [("outputnode.eddy_corrected", "dwi_file")],
            ),
            (input_subject, get_eddy_mean_bzero, [("bval", "bval")]),
            # register the skull-stripped dwi to the skull-stripped subject T1
            (
                get_eddy_mean_bzero,
                bbreg_wf,
                [("out", "inputnode.in_file")],
            ),
            (
                input_subject,
                bbreg_wf,
                [("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm")],
            ),
            (
                input_subject,
                get_subject_id_node,
                [("bids_entities", "bids_entities")],
            ),
            (
                get_subject_id_node,
                bbreg_wf,
                [("subject_id", "inputnode.subject_id")],
            ),
            (
                input_subject,
                bbreg_wf,
                [("fs_subjects_dir", "inputnode.subjects_dir")],
            ),
            # some matrix format conversions
            # rotate the gradients using the LTA file directly
            (input_subject, rotate_gradients, [("bvec", "gradient_file")]),
            (
                bbreg_wf.get_node("bbregister"),
                rotate_gradients,
                [("out_lta_file", "lta_file")],
            ),
            # apply the registration to the skull-stripped and eddy-corrected
            # dwi using FreeSurfer's ApplyVolTransform
            (
                bbreg_wf.get_node("bbregister"),
                apply_registration,
                [("out_lta_file", "lta_file")],
            ),
            (
                eddycorrect,
                apply_registration,
                [("outputnode.eddy_corrected", "source_file")],
            ),
            (
                strip_t1,
                apply_registration,
                [("out_file", "target_file")],
            ),
            # get a mean b=0 image of the registered dwi
            (
                apply_registration,
                get_registered_mean_bzero,
                [("transformed_file", "dwi_file")],
            ),
            (input_subject, get_registered_mean_bzero, [("bval", "bval")]),
            # also apply the registration to the mask
            (
                bbreg_wf.get_node("bbregister"),
                apply_registration_mask,
                [("out_lta_file", "lta_file")],
            ),
            (bet, apply_registration_mask, [("mask_file", "source_file")]),
            (
                strip_t1,
                apply_registration_mask,
                [("out_file", "target_file")],
            ),
            # collect all the outputs in the output node
            # get subject id
            (input_subject, output, [("bids_entities", "bids_entities")]),
            # get the plots from smriprep
            (
                input_subject,
                output,
                [("plot_recon_surface_on_t1", "plot_recon_surface_on_t1")],
            ),
            (
                input_subject,
                output,
                [
                    (
                        "plot_recon_segmentations_on_t1",
                        "plot_recon_segmentations_on_t1",
                    )
                ],
            ),
            (strip_dwi, output, [("out_file", "dwi_masked")]),
            (
                bbreg_wf.get_node("bbregister"),
                output,
                [("out_lta_file", "rigid_dwi_2_t1")],
            ),
            (input_subject, output, [("preprocessed_t1", "t1_initial")]),
            (strip_t1, output, [("out_file", "t1_masked")]),
            (
                apply_registration,
                output,
                [("transformed_file", "dwi_rigid_registered")],
            ),
            (
                rotate_gradients,
                output,
                [("rotated_gradients", "bvec_rotated")],
            ),
            (input_subject, output, [("bval", "bval")]),
            (bet, output, [("mask_file", "bet_mask")]),
            (apply_registration_mask, output, [("transformed_file", "mask")]),
            (
                eddycorrect,
                output,
                [("outputnode.eddy_corrected", "eddy_corrected")],
            ),
            (input_subject, output, [("dwi", "dwi_initial")]),
            (
                get_registered_mean_bzero,
                output,
                [("out", "registered_mean_bzero")],
            ),
            # connect the report workflow
            (
                output,
                report,
                [
                    ("dwi_initial", "report_inputnode.dwi_initial"),
                    ("dwi_masked", "report_inputnode.dwi_masked"),
                    ("bval", "report_inputnode.bval"),
                    (
                        "t1_initial",
                        "report_inputnode.t1_initial",
                    ),
                    (
                        "t1_masked",
                        "report_inputnode.t1_masked",
                    ),
                    ("eddy_corrected", "report_inputnode.eddy_corrected"),
                    ("mask", "report_inputnode.mask"),
                    ("bet_mask", "report_inputnode.bet_mask"),
                    (
                        "dwi_rigid_registered",
                        "report_inputnode.dwi_rigid_registered",
                    ),
                    ("bids_entities", "report_inputnode.bids_entities"),
                    (
                        "plot_recon_surface_on_t1",
                        "report_inputnode.plot_recon_surface_on_t1",
                    ),
                    (
                        "plot_recon_segmentations_on_t1",
                        "report_inputnode.plot_recon_segmentations_on_t1",
                    ),
                ],
            ),
        ]
    )

    return workflow


def init_preprocess_wf(output_dir=".", config=None):
    wf = _preprocess_wf(output_dir=output_dir, config=config)
    wf = _set_inputs_outputs(config, wf)
    return wf
