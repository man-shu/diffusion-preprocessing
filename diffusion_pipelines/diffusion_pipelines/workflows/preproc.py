import inspect
from nipype import IdentityInterface, Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from niflow.nipype1.workflows.dmri.fsl.epi import create_eddy_correct_pipeline
from nipype.interfaces import utility
from nipype.interfaces.utility.wrappers import Function
from .report import init_report_wf
from .bids import init_bidsdata_wf
from .sink import init_sink_wf
from pathlib import Path


def _set_inputs_outputs(config, preproc_wf):
    # inputs from the config file
    preproc_wf.inputs.input_template.T2 = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["t2"]
    )
    preproc_wf.inputs.input_template.mask = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["mask"]
    )
    # bids dataset
    bidsdata_wf = init_bidsdata_wf(config=config)
    # outputs
    sink_wf = init_sink_wf(config=config)
    # create the full workflow
    preproc_wf.connect(
        [
            (
                bidsdata_wf,
                preproc_wf.get_node("input_subject"),
                [
                    ("selectfiles.T1", "preprocessed_T1"),
                    ("selectfiles.dwi", "dwi"),
                    ("selectfiles.bval", "bval"),
                    ("selectfiles.bvec", "bvec"),
                    ("decode_entities.bids_entities", "bids_entities"),
                ],
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
                        "preprocess.@registered_dwi",
                    ),
                    ("eddy_corrected", "preprocess.@eddy_corrected"),
                    ("mask", "preprocess.@mask"),
                ],
            ),
            (
                preproc_wf.get_node("report"),
                sink_wf.get_node("sink"),
                [("report_outputnode.out_file", "preprocess.@report")],
            ),
        ]
    )
    return preproc_wf


def _preprocess_wf(name="preprocess", bet_frac=0.34, output_dir="."):

    def _get_mean_bzero(dwi_file, bval):
        """Mean of the b=0 volumes of the input dwi file."""
        import os
        from nilearn.image import index_img, mean_img

        import numpy as np

        bvals = np.loadtxt(bval)
        # get the index of the b=0 volumes
        bzero_index = np.where(bvals == 0)[0]
        # get the mean image of the b=0 volumes
        mean_bzero_img = mean_img(index_img(dwi_file, bzero_index))
        # save the mean image
        out_file = os.path.join(os.getcwd(), "mean_bzero.nii.gz")
        mean_bzero_img.to_filename(out_file)

        return out_file

    # define a function to get the mean of b=0 of the input dwi file
    MeanBZero = Function(
        input_names=["dwi_file", "bval"],
        output_names=["out"],
        function=_get_mean_bzero,
    )
    # this node is used to get the mean of b=0 of the input dwi file
    get_intial_mean_bzero = Node(MeanBZero, name="get_intial_mean_bzero")

    def convert_affine_itk_2_ras(input_affine):
        import subprocess
        import os, os.path

        output_file = os.path.join(
            os.getcwd(), f"{os.path.basename(input_affine)}.ras"
        )
        subprocess.check_output(
            f"c3d_affine_tool "
            f"-itk {input_affine} "
            f"-o {output_file} -info-full ",
            shell=True,
        ).decode("utf8")
        return output_file

    ConvertAffine2RAS = Function(
        input_names=["input_affine"],
        output_names=["affine_ras"],
        function=convert_affine_itk_2_ras,
    )

    def rotate_gradients_(input_affine, gradient_file):
        import os
        import os.path
        import numpy as np
        from scipy.linalg import polar

        affine = np.loadtxt(input_affine)
        u, p = polar(affine[:3, :3], side="right")
        gradients = np.loadtxt(gradient_file)
        new_gradients = np.linalg.solve(u, gradients).T
        name, ext = os.path.splitext(os.path.basename(gradient_file))
        output_name = os.path.join(os.getcwd(), f"{name}_rot{ext}")
        np.savetxt(output_name, new_gradients)

        return output_name

    RotateGradientsAffine = Function(
        input_names=["input_affine", "gradient_file"],
        output_names=["rotated_gradients"],
        function=rotate_gradients_,
    )

    input_subject = Node(
        IdentityInterface(
            fields=["dwi", "bval", "bvec", "bids_entities"],
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
                "dwi_masked",
                "bet_mask",
                "t1_initial",
                "t1_masked",
                "bids_entities",
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

    rigid_registration = Node(
        interface=ants.RegistrationSynQuick(), name="affine_reg"
    )
    rigid_registration.inputs.num_threads = 8
    rigid_registration.inputs.transform_type = "a"

    conv_affine = Node(
        interface=ConvertAffine2RAS, name="convert_affine_itk_2_ras"
    )

    rotate_gradients = Node(
        interface=RotateGradientsAffine, name="rotate_gradients"
    )

    transforms_to_list = Node(
        interface=utility.Merge(1), name="transforms_to_list"
    )

    apply_registration = Node(
        interface=ants.ApplyTransforms(), name="apply_registration"
    )
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 3
    apply_registration.inputs.interpolation = "NearestNeighbor"

    apply_registration_mask = Node(
        interface=ants.ApplyTransforms(), name="apply_registration_mask"
    )
    apply_registration_mask.inputs.dimension = 3
    apply_registration_mask.inputs.input_image_type = 3
    apply_registration_mask.inputs.interpolation = "NearestNeighbor"

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
            # register the skull-stripped dwi to the skull-stripped subject T1
            (
                strip_mean_bzero,
                rigid_registration,
                [("out_file", "moving_image")],
            ),
            (
                strip_t1,
                rigid_registration,
                [("out_file", "fixed_image")],
            ),
            # some matrix format conversions
            (rigid_registration, transforms_to_list, [("out_matrix", "in1")]),
            (
                rigid_registration,
                conv_affine,
                [("out_matrix", "input_affine")],
            ),
            # rotate the gradients
            (input_subject, rotate_gradients, [("bvec", "gradient_file")]),
            (conv_affine, rotate_gradients, [("affine_ras", "input_affine")]),
            # apply the registration to the skull-stripped and eddy-corrected
            # dwi
            (transforms_to_list, apply_registration, [("out", "transforms")]),
            (
                eddycorrect,
                apply_registration,
                [("outputnode.eddy_corrected", "input_image")],
            ),
            (
                strip_t1,
                apply_registration,
                [("out_file", "reference_image")],
            ),
            (
                transforms_to_list,
                apply_registration_mask,
                [("out", "transforms")],
            ),
            # also apply the registration to the mask
            (bet, apply_registration_mask, [("mask_file", "input_image")]),
            (
                strip_t1,
                apply_registration_mask,
                [("out_file", "reference_image")],
            ),
            # collect all the outputs in the output node
            # get subject id
            (input_subject, output, [("bids_entities", "bids_entities")]),
            (strip_dwi, output, [("out_file", "dwi_masked")]),
            (conv_affine, output, [("affine_ras", "rigid_dwi_2_t1")]),
            (input_subject, output, [("preprocessed_t1", "t1_initial")]),
            (strip_t1, output, [("out_file", "t1_masked")]),
            (
                apply_registration,
                output,
                [("output_image", "dwi_rigid_registered")],
            ),
            (
                rotate_gradients,
                output,
                [("rotated_gradients", "bvec_rotated")],
            ),
            (input_subject, output, [("bval", "bval")]),
            (bet, output, [("mask_file", "bet_mask")]),
            (apply_registration_mask, output, [("output_image", "mask")]),
            (
                eddycorrect,
                output,
                [("outputnode.eddy_corrected", "eddy_corrected")],
            ),
            (input_subject, output, [("dwi", "dwi_initial")]),
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
                ],
            ),
        ]
    )

    return workflow


def init_preprocess_wf(output_dir=".", config=None):
    wf = _preprocess_wf(output_dir=output_dir)
    wf = _set_inputs_outputs(config, wf)
    return wf
