#!/bin/env python
import inspect
import os

from nipype import IdentityInterface, Node, Workflow
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from niflow.nipype1.workflows.dmri.fsl.epi import create_eddy_correct_pipeline
from nipype.interfaces import utility
from nipype.interfaces.utility.wrappers import Function


def convert_affine_itk_2_ras(input_affine):
    import subprocess
    import os, os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(input_affine)}.ras'
    )
    subprocess.check_output(
        f'c3d_affine_tool '
        f'-itk {input_affine} '
        f'-o {output_file} -info-full ',
        shell=True
    ).decode('utf8')
    return output_file


ConvertAffine2RAS = Function(
    input_names=['input_affine'], output_names=['affine_ras'],
    function=convert_affine_itk_2_ras
  )


def rotate_gradients_(input_affine, gradient_file):
  import os
  import os.path
  import numpy as np
  from scipy.linalg import polar

  affine = np.loadtxt(input_affine)
  u, p = polar(affine[:3, :3], side='right') 
  gradients = np.loadtxt(gradient_file)
  new_gradients = np.linalg.solve(u, gradients.T).T
  name, ext = os.path.splitext(os.path.basename(gradient_file))
  output_name = os.path.join(
      os.getcwd(),
      f'{name}_rot{ext}'
  )
  np.savetxt(output_name, new_gradients)
          
  return output_name

RotateGradientsAffine = Function(
  input_names=['input_affine', 'gradient_file'],
  output_names=['rotated_gradients'],
  function=rotate_gradients_
)


def create_diffusion_prep_pipeline(name='dMRI_preprocessing', bet_frac=0.34):
  input_subject = Node(
    IdentityInterface(
      fields=['dwi', 'bval', 'bvec'],
    ),
    name='input_subject'
  )

  input_template = Node(
    IdentityInterface(
      fields=['T1', 'T2'],
    ),
    name='input_template'
  )

  output = Node(
    IdentityInterface(
      fields=[
        'dwi_rigid_registered', 'bval', 'bvec_rotated', 'mask', 'rigid_dwi_2_template'
      ]
    ),
    name='output'
  )

  fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
  fslroi.inputs.t_min = 0
  fslroi.inputs.t_size = 1

  bet = Node(interface=fsl.BET(), name='bet')
  bet.inputs.mask = True
  bet.inputs.frac = bet_frac

  eddycorrect = create_eddy_correct_pipeline('eddycorrect')
  eddycorrect.inputs.inputnode.ref_num = 0

  rigid_registration = Node(
      interface=ants.RegistrationSynQuick(),
      name='affine_reg'
  )
  rigid_registration.inputs.num_threads = 8
  rigid_registration.inputs.transform_type = 'a'

  conv_affine = Node(
      interface=ConvertAffine2RAS,
      name='convert_affine_itk_2_ras'
  )

  rotate_gradients = Node(
      interface=RotateGradientsAffine,
      name='rotate_gradients'
  )

  transforms_to_list = Node(
      interface=utility.Merge(1),
      name='transforms_to_list'
  )

  apply_registration = Node(
      interface=ants.ApplyTransforms(),
      name='apply_registration'
  )
  apply_registration.inputs.dimension = 3
  apply_registration.inputs.input_image_type = 3
  apply_registration.inputs.interpolation = 'NearestNeighbor'

  apply_registration_mask = Node(
      interface=ants.ApplyTransforms(),
      name='apply_registration_mask'
  )
  apply_registration_mask.inputs.dimension = 3
  apply_registration_mask.inputs.input_image_type = 3
  apply_registration_mask.inputs.interpolation = 'NearestNeighbor'

  workflow = Workflow(
      name=name,
  )
  workflow.connect([
    (input_subject, fslroi, [('dwi', 'in_file')]),
    (fslroi, bet, [('roi_file', 'in_file')]),
    (input_subject, eddycorrect, [('dwi', 'inputnode.in_file')]),
    (fslroi, rigid_registration, [('roi_file', 'moving_image')]),
    (input_template, rigid_registration, [('T2', 'fixed_image')]),
    (rigid_registration, transforms_to_list, [('out_matrix', 'in1')]),
    (rigid_registration, conv_affine, [('out_matrix', 'input_affine')]),
    (input_subject, rotate_gradients, [('bvec', 'gradient_file')]),
    (conv_affine, rotate_gradients, [('affine_ras', 'input_affine')]),
    (transforms_to_list, apply_registration, [('out', 'transforms')]),
    (eddycorrect, apply_registration, [('outputnode.eddy_corrected', 'input_image')]),
    (input_template, apply_registration, [('T2', 'reference_image')]),

    (transforms_to_list, apply_registration_mask, [('out', 'transforms')]),
    (bet, apply_registration_mask, [('mask_file', 'input_image')]),
    (input_template, apply_registration_mask, [('T2', 'reference_image')]),

    (conv_affine, output, [('affine_ras', 'rigid_dwi_2_template')]),
    (apply_registration, output, [('output_image', 'dwi_rigid_registered')]),
    (rotate_gradients, output, [('rotated_gradients', 'bvec_rotated')]),
    (input_subject, output, [('bval', 'bval')]),
    (apply_registration_mask, output, [('output_image', 'mask')]),
  ])

  return workflow
