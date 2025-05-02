#!/bin/env python
import os
import sys
import subprocess
import configparser
from nipype import IdentityInterface, Node, Workflow, Merge, MapNode
from nipype.interfaces.utility.wrappers import Function
from nipype.interfaces import utility
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from nipype.interfaces.freesurfer import ReconAll, MRIsConvert, MRIConvert
from .bids import init_bidsdata_wf
from .sink import init_sink_wf
from pathlib import Path
from smriprep.workflows.base import init_smriprep_wf
from bids.layout import BIDSLayout
from bids.layout.index import BIDSLayoutIndexer
from niworkflows.utils.spaces import Reference, SpatialReferences


def _set_inputs_outputs(config, recon_wf):
    # inputs from the config file
    recon_wf.inputs.input_template.T1 = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["t1"]
    )
    recon_wf.inputs.input_subject.subjects_dir = config["OUTPUT"]["cache"]
    # bids dataset
    bidsdata_wf = init_bidsdata_wf(config=config)

    def extract_subject_id(bids_entities):
        return f"sub-{bids_entities['subject']}"

    # set node for extracting the subject id
    extract_subject_id = Node(
        Function(
            input_names=["bids_entities"],
            output_names=["subject_id"],
            function=extract_subject_id,
        ),
        name="extract_subject_id",
    )

    # outputs
    sink_wf = init_sink_wf(config=config)
    # create the full workflow
    recon_wf.connect(
        [
            (
                bidsdata_wf,
                recon_wf.get_node("input_subject"),
                [("selectfiles.T1", "T1")],
            ),
            (
                bidsdata_wf,
                extract_subject_id,
                [("decode_entities.bids_entities", "bids_entities")],
            ),
            (
                extract_subject_id,
                recon_wf.get_node("input_subject"),
                [("subject_id", "subject_id")],
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
                recon_wf.get_node("output"),
                sink_wf.get_node("sink"),
                [
                    (
                        "shrunk_surface",
                        "recon.@shrunk_surface",
                    ),
                    (
                        "mri_convert_reference_image",
                        "recon.@mri_convert_reference_image",
                    ),
                    (
                        "reg_nl_forward_transforms",
                        "recon.@reg_nl_forward_transforms",
                    ),
                ],
            ),
        ]
    )
    return recon_wf


def freesurfer_get_ras_conversion_matrix(subjects_dir, subject_id):
    from os.path import join
    from os import getcwd
    import subprocess

    f = join(subjects_dir, subject_id, "mri", "brain.finalsurfs.mgz")
    res = subprocess.check_output("mri_info %s" % f, shell=True)
    res = res.decode("utf8")
    lines = res.splitlines()
    translations = dict()
    for c, coord in (("c_r", "x"), ("c_a", "y"), ("c_s", "z")):
        tr = [l for l in lines if c in l][0].split("=")[4]
        translations[coord] = float(tr)

    output = (
        f'1 0 0 {translations["x"]}\n'
        f'0 1 0 {translations["y"]}\n'
        f'0 0 1 {translations["z"]}\n'
        f"0 0 0 1\n"
    )

    output_file = join(getcwd(), "ras_c.mat")
    with open(output_file, "w") as f:
        f.write(output)

    return output_file


def freesurfer_gii_2_native(
    freesurfer_gii_surface, ras_conversion_matrix, warps
):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    if isinstance(warps, str):
        warps = [warps]

    if "lh" in freesurfer_gii_surface:
        structure_name = "CORTEX_LEFT"
    elif "rh" in freesurfer_gii_surface:
        structure_name = "CORTEX_RIGHT"

    if "inflated" in freesurfer_gii_surface:
        surface_type = "INFLATED"
    elif "sphere" in freesurfer_gii_surface:
        surface_type = "SPHERICAL"
    else:
        surface_type = "ANATOMICAL"

    if "pial" in freesurfer_gii_surface:
        secondary_type = "PIAL"
    if "white" in freesurfer_gii_surface:
        secondary_type = "GRAY_WHITE"

    output_file = join(getcwd(), basename(freesurfer_gii_surface))
    output_file = output_file.replace(".gii", ".surf.gii")
    subprocess.check_call(
        f"cp {freesurfer_gii_surface} {output_file}", shell=True
    )

    subprocess.check_call(
        f"wb_command -set-structure {output_file} {structure_name} "
        f"-surface-type {surface_type} -surface-secondary-type {secondary_type}",
        shell=True,
    )

    subprocess.check_call(
        f"wb_command -surface-apply-affine {freesurfer_gii_surface} {ras_conversion_matrix} {output_file}",
        shell=True,
    )

    # for warp in warps:
    #    subprocess.check_call(
    #        f'wb_command -surface-apply-warpfield {output_file} {warp} {output_file}',
    #        shell=True
    #    )

    return output_file


def surface_signed_distance_image(surface, image):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    output_file = str(join(getcwd(), basename(surface)))
    output_file = output_file.replace(".surf.gii", "signed_dist.nii.gz")

    subprocess.check_call(
        f"wb_command -create-signed-distance-volume {surface} {image} {output_file}",
        shell=True,
    )

    return output_file


def shrink_surface_fun(surface, image, distance):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    output_file = str(join(getcwd(), basename(surface)))
    output_file = output_file.replace(".surf.gii", "_shrunk.surf.gii")

    subprocess.check_call(
        f"shrink_surface -surface {surface} -reference {image} "
        f"-mm {distance} -out {output_file}",
        shell=True,
    )

    if "lh" in output_file:
        structure_name = "CORTEX_LEFT"
    elif "rh" in output_file:
        structure_name = "CORTEX_RIGHT"

    if "inflated" in output_file:
        surface_type = "INFLATED"
    elif "sphere" in output_file:
        surface_type = "SPHERICAL"
    else:
        surface_type = "ANATOMICAL"

    if "pial" in output_file:
        secondary_type = "PIAL"
    if "white" in output_file:
        secondary_type = "GRAY_WHITE"

    subprocess.check_call(
        f"wb_command -set-structure {output_file} {structure_name} "
        f"-surface-type {surface_type} -surface-secondary-type {secondary_type}",
        shell=True,
    )

    return output_file


def bvec_flip(bvecs_in, flip):
    from os.path import join, basename
    from os import getcwd

    import numpy as np

    print(bvecs_in)
    bvecs = np.loadtxt(bvecs_in).T * flip

    output_file = str(join(getcwd(), basename(bvecs_in)))
    np.savetxt(output_file, bvecs)

    return output_file


def _recon_wf(name="recon", output_dir="."):

    input_subject = Node(
        IdentityInterface(
            fields=["T1", "subject_id", "subjects_dir"],
        ),
        name="input_subject",
    )

    input_template = Node(
        IdentityInterface(
            fields=["T1"],
        ),
        name="input_template",
    )

    output = Node(
        IdentityInterface(
            fields=[
                "mri_convert_reference_image",
                "reg_nl_forward_transforms",
                "reg_nl_forward_invert_flags",
                "shrunk_surface",
            ]
        ),
        name="output",
    )

    recon_all = Node(interface=ReconAll(), name="recon_all")
    recon_all.inputs.directive = "all"
    recon_all.inputs.openmp = 20
    recon_all.inputs.mprage = True
    recon_all.inputs.parallel = True
    recon_all.interface.num_threads = 20
    recon_all.inputs.flags = "-no-isrunning"

    ras_conversion_matrix = Node(
        interface=Function(
            input_names=["subjects_dir", "subject_id"],
            output_names=["output_mat"],
            function=freesurfer_get_ras_conversion_matrix,
        ),
        name="ras_conversion_matrix",
    )

    mris_convert = MapNode(
        interface=MRIsConvert(), name="mris_convert", iterfield=["in_file"]
    )
    mris_convert.inputs.out_datatype = "gii"
    # mris_convert.inputs.subjects_dir = subjects_dir

    mri_convert = Node(interface=MRIConvert(), name="mri_convert")
    mri_convert.inputs.out_type = "nii"
    # mri_convert.inputs.subjects_dir = subjects_dir

    freesurfer_surf_2_native = MapNode(
        interface=Function(
            input_names=[
                "freesurfer_gii_surface",
                "ras_conversion_matrix",
                "warps",
            ],
            output_names=["out_surf"],
            function=freesurfer_gii_2_native,
        ),
        name="freesurfer_surf_2_native",
        iterfield=["freesurfer_gii_surface"],
    )

    affine_initializer = Node(
        interface=ants.AffineInitializer(), name="affine_initializer"
    )
    affine_initializer.inputs.num_threads = 20
    affine_initializer.interface.num_threads = 20

    registration_affine = Node(interface=ants.Registration(), name="reg_aff")
    registration_affine.inputs.num_threads = 16
    registration_affine.inputs.metric = ["MI"] * 2
    registration_affine.inputs.metric_weight = [1] * 2
    registration_affine.inputs.radius_or_number_of_bins = [32] * 2
    registration_affine.inputs.sampling_strategy = ["Random", "Random"]
    registration_affine.inputs.sampling_percentage = [0.05, 0.05]
    registration_affine.inputs.convergence_threshold = [1.0e-6] * 2
    registration_affine.inputs.convergence_window_size = [10] * 2
    registration_affine.inputs.transforms = ["Rigid", "Affine"]
    registration_affine.inputs.output_transform_prefix = "output_"
    registration_affine.inputs.transform_parameters = [(0.1,), (0.1,)]
    registration_affine.inputs.number_of_iterations = [
        [1000, 500, 250, 0],
        [1000, 500, 250, 0],
    ]
    registration_affine.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2
    registration_affine.inputs.sigma_units = ["vox"] * 2
    registration_affine.inputs.shrink_factors = [[8, 4, 2, 1]] * 2
    registration_affine.inputs.use_histogram_matching = [
        True,
        True,
    ]  # This is the default
    registration_affine.inputs.output_warped_image = (
        "output_warped_image.nii.gz"
    )

    registration_nl = Node(interface=ants.Registration(), name="reg_nl")
    registration_nl.inputs.num_threads = 16
    registration_nl.inputs.metric = ["MI"]
    registration_nl.inputs.metric_weight = [1]
    registration_nl.inputs.radius_or_number_of_bins = [32]
    registration_nl.inputs.sampling_strategy = [None]
    registration_nl.inputs.sampling_percentage = [None]
    registration_nl.inputs.convergence_threshold = [1.0e-6]
    registration_nl.inputs.convergence_window_size = [10]
    registration_nl.inputs.transforms = ["SyN"]
    registration_nl.inputs.output_transform_prefix = "output_"
    registration_nl.inputs.transform_parameters = [(0.1, 3.0, 0.0)]
    registration_nl.inputs.number_of_iterations = [[1000, 700, 400, 100]]
    registration_nl.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    registration_nl.inputs.sigma_units = ["vox"]
    registration_nl.inputs.shrink_factors = [[8, 4, 2, 1]]
    registration_nl.inputs.use_histogram_matching = [
        True
    ]  # This is the default
    registration_nl.inputs.output_warped_image = "output_warped_image.nii.gz"

    select_nl_transform = Node(
        interface=utility.Select(), name="select_nl_transform"
    )
    select_nl_transform.inputs.index = [1]

    registration = Node(interface=ants.Registration(), name="reg")
    registration.inputs.num_threads = 16
    registration.inputs.metric = ["MI", "MI", "MI"]
    registration.inputs.metric_weight = [1] * 3
    registration.inputs.radius_or_number_of_bins = [32] * 3
    registration.inputs.sampling_strategy = ["Random", "Random", None]
    registration.inputs.sampling_percentage = [0.05, 0.05, None]
    registration.inputs.convergence_threshold = [1.0e-6] * 3
    registration.inputs.convergence_window_size = [10] * 3
    registration.inputs.transforms = ["Rigid", "Affine", "SyN"]
    registration.inputs.output_transform_prefix = "output_"
    registration.inputs.transform_parameters = [
        (0.1,),
        (0.1,),
        (0.1, 3.0, 0.0),
    ]
    registration.inputs.number_of_iterations = [
        [1000, 500, 250, 0],
        [1000, 500, 250, 0],
        [1000, 700, 400, 100],
    ]
    registration.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2 + [[3, 2, 1, 0]]
    registration.inputs.sigma_units = ["vox"] * 3
    registration.inputs.shrink_factors = [[8, 4, 2, 1]] * 2 + [[8, 4, 2, 1]]
    registration.inputs.use_estimate_learning_rate_once = [True, True, True]
    registration.inputs.use_histogram_matching = [
        True,
        True,
        True,
    ]  # This is the default
    registration.inputs.output_warped_image = "output_warped_image.nii.gz"

    shrink_surface_node = MapNode(
        interface=Function(
            input_names=["surface", "image", "distance"],
            output_names=["out_file"],
            function=shrink_surface_fun,
        ),
        name="surface_shrink_surface",
        iterfield=["surface"],
    )
    shrink_surface_node.inputs.distance = 3
    workflow = Workflow(name=name, base_dir=output_dir)

    workflow.connect(
        [
            (input_subject, recon_all, [("T1", "T1_files")]),
            (input_subject, recon_all, [("subject_id", "subject_id")]),
            (input_subject, recon_all, [("subjects_dir", "subjects_dir")]),
            (
                recon_all,
                ras_conversion_matrix,
                [
                    ("subjects_dir", "subjects_dir"),
                    ("subject_id", "subject_id"),
                ],
            ),
            (recon_all, mris_convert, [("subjects_dir", "subjects_dir")]),
            (recon_all, mris_convert, [("white", "in_file")]),
            (recon_all, mri_convert, [("subjects_dir", "subjects_dir")]),
            (recon_all, mri_convert, [("brain", "in_file")]),
            (
                mris_convert,
                freesurfer_surf_2_native,
                [("converted", "freesurfer_gii_surface")],
            ),
            (mri_convert, affine_initializer, [("out_file", "moving_image")]),
            (
                input_template,
                affine_initializer,
                [("T1", "fixed_image")],
            ),
            (mri_convert, registration_affine, [("out_file", "moving_image")]),
            (
                input_template,
                registration_affine,
                [
                    ("T1", "fixed_image"),
                ],
            ),
            (
                affine_initializer,
                registration_affine,
                [("out_file", "initial_moving_transform")],
            ),
            (mri_convert, registration_nl, [("out_file", "moving_image")]),
            (
                input_template,
                registration_nl,
                [
                    ("T1", "fixed_image"),
                ],
            ),
            (
                registration_affine,
                registration_nl,
                [
                    ("forward_transforms", "initial_moving_transform"),
                    (
                        "forward_invert_flags",
                        "invert_initial_moving_transform",
                    ),
                ],
            ),
            (
                ras_conversion_matrix,
                freesurfer_surf_2_native,
                [("output_mat", "ras_conversion_matrix")],
            ),
            (
                registration_nl,
                select_nl_transform,
                [("forward_transforms", "inlist")],
            ),
            (
                select_nl_transform,
                freesurfer_surf_2_native,
                [("out", "warps")],
            ),
            (
                freesurfer_surf_2_native,
                shrink_surface_node,
                [("out_surf", "surface")],
            ),
            (
                mri_convert,
                shrink_surface_node,
                [("out_file", "image")],
            ),
            (
                mri_convert,
                output,
                [("out_file", "mri_convert_reference_image")],
            ),
            (
                registration_nl,
                output,
                [
                    ("forward_transforms", "reg_nl_forward_transforms"),
                    ("forward_invert_flags", "reg_nl_forward_invert_flags"),
                ],
            ),
            (shrink_surface_node, output, [("out_file", "shrunk_surface")]),
        ]
    )

    return workflow


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
        sloppy=False,
        debug=False,
        derivatives=[],
        freesurfer=True,
        fs_subjects_dir=Path(config["OUTPUT"]["derivatives"], "freesurfer"),
        hires=False,
        fs_no_resume=False,
        longitudinal=False,
        low_mem=False,
        msm_sulc=False,
        omp_nthreads=config["NIPYPE"]["n_jobs"],
        run_uuid="123",
        skull_strip_mode="auto",
        skull_strip_fixed_seed=True,
        skull_strip_template=Reference.from_string("OASIS30ANTs")[0],
        spaces=spaces,
        bids_filters=None,
        cifti_output=False,
    )
    return wf
