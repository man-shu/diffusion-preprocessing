#!/bin/env python
import os
import sys
import subprocess
import configparser

from nipype import DataGrabber, DataSink, IdentityInterface, Node, Workflow, MapNode, JoinNode, Merge
from niflow.nipype1.workflows.dmri.fsl import bedpostx_parallel 
from nipype.interfaces.fsl.utils import CopyGeom
from nipype.interfaces import utility
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from nipype.interfaces.ants.base import ANTSCommand
ANTSCommand.set_default_num_threads(16)
from nipype.interfaces.freesurfer import ReconAll, MRIsConvert, MRIConvert
from nipype.interfaces.utility import Function
from datetime import datetime
#from diffusion_pipelines import diffusion_preprocessing

from nipype import config, logging
config.update_config({'logging': {'log_directory': os.path.join(os.getcwd(), 'logs'),
                                  'workflow_level': 'DEBUG',
                                  'interface_level': 'DEBUG',
                                  'log_to_file': True,
                                  },
                      'execution': {'stop_on_first_crash': False,
                                    'keep_inputs': True},
                    })
config.enable_debug_mode()
logging.update_logging(config)
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




def create_diffusion_prep_pipeline(name='dMRI_preprocessing', bet_frac=0.34):
  ConvertAffine2RAS = Function(
        input_names=['input_affine'], output_names=['affine_ras'],
        function=convert_affine_itk_2_ras
      )

  RotateGradientsAffine = Function(
      input_names=['input_affine', 'gradient_file'],
      output_names=['rotated_gradients'],
      function=rotate_gradients_
    )
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
        'dwi_rigid_registered', 'bval', 'bvec_rotated', 'mask', 'rigid_dwi_2_template',
	'dwi_subject_space', 'mask_subject_space', 'bvec_subject_space', 'transform_subject_2_template',
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


    (eddycorrect, output, [('outputnode.eddy_corrected', 'dwi_subject_space')]),
    (input_subject, output, [('bvec', 'bvec_subject_space')]),
    (bet, output, [('mask_file', 'mask_subject_space')]),
    (transforms_to_list, output, [('out', 'transform_subject_2_template')]),

    (conv_affine, output, [('affine_ras', 'rigid_dwi_2_template')]),
    (apply_registration, output, [('output_image', 'dwi_rigid_registered')]),
    (rotate_gradients, output, [('rotated_gradients', 'bvec_rotated')]),
    (input_subject, output, [('bval', 'bval')]),
    (apply_registration_mask, output, [('output_image', 'mask')]),
  ])
  #workflow.write_graph(graph2use='colored', dotfilename='/oak/stanford/groups/menon/projects/cdla/2019_dwi_mathfun/scripts/2019_dwi_pipeline_mathfun/dmri_preprocessing_graph_orig.dot')
  return workflow

def freesurfer_get_ras_conversion_matrix(subjects_dir, subject_id):
    from os.path import join
    from os import getcwd
    import subprocess

    f = join(subjects_dir, subject_id, 'mri', 'brain.finalsurfs.mgz')
    res = subprocess.check_output('mri_info %s'%f, shell=True)
    res = res.decode('utf8')
    lines = res.splitlines()
    translations = dict()
    for c, coord in (('c_r', 'x'), ('c_a', 'y'), ('c_s', 'z')):
        tr = [l for l in lines if c in l][0].split('=')[4]
        translations[coord] = float(tr)

    output = (
        f'1 0 0 {translations["x"]}\n'
        f'0 1 0 {translations["y"]}\n'
        f'0 0 1 {translations["z"]}\n'
        f'0 0 0 1\n'
    )

    output_file = join(getcwd(), 'ras_c.mat')
    with open(output_file, 'w') as f:
        f.write(output)

    return output_file


def freesurfer_gii_2_native(freesurfer_gii_surface, ras_conversion_matrix, warps):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    if isinstance(warps, str):
        warps = [warps]

    if 'lh' in freesurfer_gii_surface:
        structure_name = 'CORTEX_LEFT'
    elif 'rh' in freesurfer_gii_surface:
        structure_name = 'CORTEX_RIGHT'

    if 'inflated' in freesurfer_gii_surface:
        surface_type = 'INFLATED'
    elif 'sphere' in freesurfer_gii_surface:
        surface_type = 'SPHERICAL'
    else:
        surface_type = 'ANATOMICAL'

    if 'pial' in freesurfer_gii_surface:
        secondary_type = 'PIAL'
    if 'white' in freesurfer_gii_surface:
        secondary_type = 'GRAY_WHITE'
   
    output_file = join(getcwd(), basename(freesurfer_gii_surface))
    output_file = output_file.replace('.gii', '.surf.gii')
    subprocess.check_call(f'cp {freesurfer_gii_surface} {output_file}', shell=True)

    subprocess.check_call(
        f'wb_command -set-structure {output_file} {structure_name} '
        f'-surface-type {surface_type} -surface-secondary-type {secondary_type}',
        shell=True
    )

    subprocess.check_call(
        f'wb_command -surface-apply-affine {freesurfer_gii_surface} {ras_conversion_matrix} {output_file}',
        shell=True
    )

    #for warp in warps:
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
    output_file = output_file.replace('.surf.gii', 'signed_dist.nii.gz')

    subprocess.check_call(
        f'wb_command -create-signed-distance-volume {surface} {image} {output_file}',
        shell=True
    )

    return output_file


def shrink_surface_fun(surface, image, distance):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    output_file = str(join(getcwd(), basename(surface)))
    output_file = output_file.replace('.surf.gii', '_shrunk.surf.gii')

    subprocess.check_call(
            f'shrink_surface -surface {surface} -reference {image} '
            f'-mm {distance} -out {output_file}',
            shell=True
    )


    if 'lh' in output_file:
        structure_name = 'CORTEX_LEFT'
    elif 'rh' in output_file:
        structure_name = 'CORTEX_RIGHT'

    if 'inflated' in output_file:
        surface_type = 'INFLATED'
    elif 'sphere' in output_file:
        surface_type = 'SPHERICAL'
    else:
        surface_type = 'ANATOMICAL'

    if 'pial' in output_file:
        secondary_type = 'PIAL'
    if 'white' in output_file:
        secondary_type = 'GRAY_WHITE'
 
    subprocess.check_call(
        f'wb_command -set-structure {output_file} {structure_name} '
        f'-surface-type {surface_type} -surface-secondary-type {secondary_type}',
        shell=True
    )
    
    return output_file

    
def bvec_flip(bvecs_in, flip):
    from os.path import join, basename
    from os import getcwd

    import numpy as np

    bvecs = np.loadtxt(bvecs_in).T * flip

    output_file = str(join(getcwd(), basename(bvecs_in)))
    np.savetxt(output_file, bvecs)
    
    return output_file


if __name__ == '__main__':
    print('start')
    dmri_preprocess_workflow = create_diffusion_prep_pipeline(
        'dmri_preprocess')
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    #PATH = '/oak/stanford/groups/menon/projects/cdla/2019_dwi_mathfun/results/' 
    #PATH = '/tmp/cdla/dwi_mathfun_gpu_group/'#+sys.argv[1].split('.')[0]
    subjects=config['DEFAULT']['id_list']
    if len(subjects)==4:
        subject_list=[subjects]
    else:
        subject_list = config['DEFAULT']['id_list'].split(' ')
        subject_list = [str(x) for x in subject_list]
    #PATH = '/tmp/cdla/dwi_mathfun_gpu_group_050420/'
   # PATH = '/scratch/users/cdla/dwi_mathfun_gpu_group_%s'%(subject_list[0])#%s/'%(datetime.now().strftime("%Y%m%d_%H%M%S"))
    PATH='/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun'

    visits = [config['DEFAULT']['visits']]
    subjects_dir = config['DEFAULT']['subjects_dir']
    sessions = [config['DEFAULT']['sessions']]
    use_cpu = config['DEFAULT']['use_cpu']
    print(subject_list)
    print(visits)
    print(sessions)
    infosource = Node(IdentityInterface(fields=['subject_id', 'visit','session']),
                      name='subjects')
    infosource.iterables = [('subject_id', subject_list), ('visit', visits), ('session', sessions)]
    #infosource.iterables = [('visit', visits), ('session', sessions)]
    #infosource.inputs.subject_id=list(subject_list)
    #infosource.inputs.visit=['1']
    #infosource.inputs.session=['1']
    def compose_id(subject_id,visit):
        if type(subject_id) == list:
            subject_id=subject_id[0]
        if type(visit) == list:
            visit=visit[0]
        composite_id=subject_id + '_' + visit

        return composite_id

    subject_id_visit = Node(
        interface=Function(
            input_names=['subject_id', 'visit'], output_names=['composite_id'],
            function=compose_id#
            #lambda subject_id, visit: '{}_{}'.format(str(subject_id), str(visit))
        ),
        name='subject_id_visit'
    )


    data_source = Node(DataGrabber(infields=['subject_id', 'visit','session'],
                                   outfields=['dwi', 'bval', 'bvec', 'T1']),
                       name='data_grabber')
    data_source.inputs.sort_filelist = True
    data_source.inputs.base_directory = config['DEFAULT']['base_directory']
    data_source.inputs.template = ''
    data_source.inputs.field_template = {
        'T1': '%s/visit%s/session%s/anat/T1w.nii',
        'dwi': '%s/visit%s/session%s/dwi/dwi_raw.nii.gz',
        'bval': '%s/visit%s/session%s/dwi/dti_raw.bvals',
        'bvec': '%s/visit%s/session%s/dwi/dti_raw.bvecs'
    }
    data_source.inputs.template_args = {
        template: [['subject_id', 'visit','session']]
        for template in data_source.inputs.field_template.keys()
    }

    flip_bvectors_node = Node(
        interface=Function(
            input_names=['bvecs_in', 'flip'], output_names=['bvecs_out'],
            function=bvec_flip
        ),
        name='flip_bvecs',
    )
    flip_bvectors_node.inputs.flip = (-1, 1, 1)

    template_source = Node(DataGrabber(infields=[], outfields=['T1', 'T1_brain', 'T1_mask', 'T2', 'T2_brain', 'T2_mask']),
                           name='mni_template')
    template_source.inputs.sort_filelist = True
    template_source.inputs.base_directory = config['TEMPLATE']['directory']
    template_source.inputs.template = ''
    template_source.inputs.field_template = {
        'T1': config['TEMPLATE']['T1'],
        'T1_brain': config['TEMPLATE']['T1_brain'],
        'T1_mask': config['TEMPLATE']['T1_mask'],
        'T2': config['TEMPLATE']['T2'],
        'T2_brain': config['TEMPLATE']['T2_brain'],
    }
    template_source.inputs.template_args = {
        template: []
        for template in template_source.inputs.field_template.keys()
    }

    roi_source = Node(DataGrabber(infields=[]),
                           name='rois')
    roi_source.inputs.sort_filelist = True
    roi_source.inputs.base_directory = config['ROIS']['directory']
    roi_source.inputs.template = '*.nii.gz'

    recon_all = Node(interface=ReconAll(), name='recon_all')
    recon_all.inputs.directive = 'all'
    recon_all.inputs.subjects_dir = subjects_dir
    recon_all.inputs.openmp = 16 
    recon_all.inputs.mprage = True
    recon_all.inputs.parallel = True
    recon_all.interface.num_threads =16
    recon_all.n_procs = 16
    recon_all.plugin_args={
	'sbatch_args':'--time=48:00:00 -c 16 --mem=16G --oversubscribe --exclude=node[22-32] ',
	'overwrite':True
    }

    ras_conversion_matrix = Node(
            interface=Function(
                input_names=['subjects_dir', 'subject_id'],
                output_names=['output_mat'],
                function=freesurfer_get_ras_conversion_matrix
            ),
            name='ras_conversion_matrix'
    )
    
    mris_convert = MapNode(interface=MRIsConvert(), name='mris_convert', iterfield=['in_file'])
    mris_convert.inputs.out_datatype = 'gii'
    mris_convert.inputs.subjects_dir = subjects_dir

    mri_convert = Node(interface=MRIConvert(), name='mri_convert')
    mri_convert.inputs.out_type = 'nii'
    mri_convert.inputs.subjects_dir = subjects_dir
    fslcpgeom_mask = Node(
            interface=CopyGeom(),name='fsl_cpgeom_mask')

    fslcpgeom_roi = MapNode(interface=CopyGeom(),name='fslcpgeom_roi',iterfield=['dest_file'])

    freesurfer_surf_2_native = MapNode(
        interface=Function(
            input_names=['freesurfer_gii_surface', 'ras_conversion_matrix', 'warps'], output_names=['out_surf'],
            function=freesurfer_gii_2_native
        ),
        name='freesurfer_surf_2_native',
        iterfield=['freesurfer_gii_surface']
    )

    #bedpostx = Node(interface=fsl.BEDPOSTX5(), name='bedpostx', iterfield=['dwi'])
    bedpostx = bedpostx_parallel(
	params=dict(
		fudge=1,
		burn_in=1000,
		n_jumps=1250,
		sample_every=25,
		n_fibres=3,
	)
    )
    bedpostx.get_node('xfibres').plugin_args={
	'sbatch_args':'--time=72:00:00 -c 4 -n 1 --mem=16G --oversubscribe --exclude=node[22-32]',
	'max_jobs': 4,
	'overwrite':True
    }

    #bedpostx.inputs.n_fibres = 3
    #bedpostx.inputs.fudge = 1
    #bedpostx.inputs.burn_in = 1000
    #bedpostx.inputs.n_jumps=1250
    #bedpostx.inputs.sample_every=25
    #bedpostx.inputs.use_gpu = False
    #bedpostx.interface.num_threads=16
    #bedpostx.n_procs=16


    #bedpostx.plugin_args={'sbatch_args':'--time=4:00:00 -c 16 --mem=128G --account=menon --partition=nih_s10 --gres=gpu:1','overwrite':True}
    #bedpostx.plugin_args={'sbatch_args':'--time=8:00:00 -c 4 --mem=16G --partition=gpu --gpus=1','overwrite':True}
    #bedpostx.plugin_args={'sbatch_args':'--time=72:00:00 -c 4 -n 1 --mem=16G --oversubscribe --comment="7014"','overwrite':True}
    join_seeds = Node(
        interface=Merge(2),
        name='join_seeds',
    )

    pbx2 = Node(
        interface=fsl.ProbTrackX2(),
        name='probtrackx2',
    )
    pbx2.inputs.n_samples = 5000
    pbx2.inputs.n_steps = 2000
    pbx2.inputs.step_length = 0.5
    pbx2.inputs.omatrix1 = True
    pbx2.inputs.distthresh1 = 5
    pbx2.inputs.args = " --ompl --fibthresh=0.01 --verbose=1 "
    pbx2.plugin_args={'sbatch_args':'--time=48:00:00 -c 4 --mem=16G','overwrite':True}
    pbx2.interface.num_threads=16
    pbx2.n_procs=16

    fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    affine_initializer = Node(interface=ants.AffineInitializer(), name='affine_initializer')
    affine_initializer.inputs.num_threads =16 
    
    affine_initializer.interface.num_threads = 16
    affine_initializer.n_procs=16
    
    registration_affine = Node(interface=ants.Registration(), name='reg_aff')
    registration_affine.inputs.num_threads = 16
    registration_affine.n_procs=4
    registration_affine.inputs.metric = ['MI'] * 2
    registration_affine.inputs.metric_weight = [1] * 2
    registration_affine.inputs.radius_or_number_of_bins = [32] * 2
    registration_affine.inputs.sampling_strategy = ['Random', 'Random']
    registration_affine.inputs.sampling_percentage = [0.05, 0.05]
    registration_affine.inputs.convergence_threshold = [1.e-6] * 2
    registration_affine.inputs.convergence_window_size = [10] * 2
    registration_affine.inputs.transforms = ['Rigid', 'Affine']
    registration_affine.inputs.output_transform_prefix = "output_"
    registration_affine.inputs.transform_parameters = [(0.1,), (0.1,)]
    registration_affine.inputs.number_of_iterations = [[1000, 500, 250, 0], [1000, 500, 250, 0]]
    registration_affine.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2
    registration_affine.inputs.sigma_units = ['vox'] * 2
    registration_affine.inputs.shrink_factors = [[8, 4, 2, 1]] * 2
    registration_affine.inputs.use_estimate_learning_rate_once = [True, True]
    registration_affine.inputs.use_histogram_matching = [True, True] # This is the default
    registration_affine.inputs.output_warped_image = 'output_warped_image.nii.gz'

    registration_nl = Node(interface=ants.Registration(), name='reg_nl')
    registration_nl.inputs.num_threads = 16
    registration_nl.interface.num_threads=16
    registration_nl.n_procs=4
    registration_nl.inputs.metric = ['MI']
    registration_nl.inputs.metric_weight = [1]
    registration_nl.inputs.radius_or_number_of_bins = [32]
    registration_nl.inputs.sampling_strategy = [None]
    registration_nl.inputs.sampling_percentage = [None]
    registration_nl.inputs.convergence_threshold = [1.e-6]
    registration_nl.inputs.convergence_window_size = [10]
    registration_nl.inputs.transforms = ['SyN']
    registration_nl.inputs.output_transform_prefix = "output_"
    registration_nl.inputs.transform_parameters = [(0.1, 3.0, 0.0)]
    registration_nl.inputs.number_of_iterations = [[1000, 700, 400, 100]]
    registration_nl.inputs.smoothing_sigmas = [[3, 2, 1, 0]] 
    registration_nl.inputs.sigma_units = ['vox']
    registration_nl.inputs.shrink_factors = [[8, 4, 2, 1]]
    registration_nl.inputs.use_estimate_learning_rate_once = [True]
    registration_nl.inputs.use_histogram_matching = [True] # This is the default
    registration_nl.inputs.output_warped_image = 'output_warped_image.nii.gz'

    select_nl_transform = Node(interface=utility.Select(), name='select_nl_transform')
    select_nl_transform.inputs.index = [1]

    registration = Node(interface=ants.Registration(), name='reg')
    registration.inputs.num_threads = 16
    registration.interface.num_threads = 16
    registration.n_procs=16
    registration.inputs.metric = ['MI', 'MI', 'MI']
    registration.inputs.metric_weight = [1] * 3
    registration.inputs.radius_or_number_of_bins = [32] * 3
    registration.inputs.sampling_strategy = ['Random', 'Random', None]
    registration.inputs.sampling_percentage = [0.05, 0.05, None]
    registration.inputs.convergence_threshold = [1.e-6] * 3
    registration.inputs.convergence_window_size = [10] * 3
    registration.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    registration.inputs.output_transform_prefix = "output_"
    registration.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    registration.inputs.number_of_iterations = [[1000, 500, 250, 0], [1000, 500, 250, 0], [1000, 700, 400, 100]]
    registration.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2 + [[3, 2, 1, 0]] 
    registration.inputs.sigma_units = ['vox'] * 3
    registration.inputs.shrink_factors = [[8, 4, 2, 1]] * 2 + [[8, 4, 2, 1]]
    registration.inputs.use_estimate_learning_rate_once = [True, True, True]
    registration.inputs.use_histogram_matching = [True, True, True] # This is the default
    registration.inputs.output_warped_image = 'output_warped_image.nii.gz'


    apply_registration = MapNode(interface=ants.ApplyTransforms(),
                              name='apply_registration', iterfield=['input_image'])
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 3
    apply_registration.inputs.interpolation = 'NearestNeighbor'


    shrink_surface_node = MapNode(
            interface=Function(
                input_names=['surface', 'image', 'distance'],
                output_names=['out_file'],
                function=shrink_surface_fun
            ),
            name='surface_shrink_surface',
            iterfield=['surface']
    )
    shrink_surface_node.inputs.distance = 3

    workflow = Workflow('diffusion_workflow_new_mgt', base_dir=PATH)
    workflow.connect([
        (infosource, data_source, [('subject_id', 'subject_id'),
                                   ('visit', 'visit'),
                   ('session','session')]),
        (data_source, flip_bvectors_node, [('bvec', 'bvecs_in')]),

        (infosource, subject_id_visit, [
            ('subject_id', 'subject_id'),
            ('visit', 'visit')
        ]),
        (data_source, recon_all, [('T1', 'T1_files')]),
        (subject_id_visit, recon_all, [('composite_id', 'subject_id')]),
        (recon_all, ras_conversion_matrix, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id')
        ]),
        (recon_all, mris_convert, [('white', 'in_file')]),
        (recon_all, mri_convert, [('brain', 'in_file')]),
        (mris_convert, freesurfer_surf_2_native, [('converted', 'freesurfer_gii_surface')]),
        
        (mri_convert, affine_initializer, [('out_file', 'moving_image')]),
        (template_source, affine_initializer, [('T1_brain', 'fixed_image')]),

        (mri_convert, registration_affine, [('out_file', 'moving_image')]),
        (template_source, registration_affine, [
            ('T1_brain', 'fixed_image'),
        ]),
        (affine_initializer, registration_affine, [('out_file', 'initial_moving_transform')]),

        (mri_convert, registration_nl, [('out_file', 'moving_image')]),
        (template_source, registration_nl, [
            ('T1_brain', 'fixed_image'),
        ]),
        (registration_affine, registration_nl, [
            ('forward_transforms', 'initial_moving_transform'),
            ('forward_invert_flags', 'invert_initial_moving_transform')
        ]),

        (ras_conversion_matrix, freesurfer_surf_2_native, [('output_mat', 'ras_conversion_matrix')]),
        (registration_nl, select_nl_transform, [('forward_transforms', 'inlist')]),
        (select_nl_transform, freesurfer_surf_2_native, [('out', 'warps')]),

        (registration_nl, apply_registration, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags'),
        ]),
        (roi_source, apply_registration, [('outfiles', 'input_image')]),
        (mri_convert, apply_registration, [('out_file', 'reference_image')]),

        (data_source, dmri_preprocess_workflow,
         [
          ('dwi', 'input_subject.dwi'), 
          ('bval', 'input_subject.bval'),
         ]
        ),
        (flip_bvectors_node, dmri_preprocess_workflow, [('bvecs_out', 'input_subject.bvec')]),

        (mri_convert, dmri_preprocess_workflow, [
          ('out_file', 'input_template.T1'),
          ('out_file', 'input_template.T2')
        ]),
        (
            dmri_preprocess_workflow,
            bedpostx,
            #[('output.bval', 'bvals'),
            # ('output.bvec_rotated', 'bvecs'),
            # ('output.dwi_rigid_registered', 'dwi'),
            # ('output.mask', 'mask')],
            [('output.bval', 'inputnode.bvals'),
             ('output.bvec_subject_space', 'inputnode.bvecs'),
             ('output.dwi_subject_space', 'inputnode.dwi'),
             ('output.mask_subject_space', 'inputnode.mask')],

        ),
        (
            freesurfer_surf_2_native,   
            shrink_surface_node,
            [('out_surf', 'surface')],
        ),
        (
            mri_convert,   
            shrink_surface_node,
            [('out_file', 'image')],
        ),
        # (
        #      bedpostx, pbx2,
        #      [
        #       ('merged_thsamples', 'thsamples'),
        #       ('merged_fsamples', 'fsamples'),
        #       ('merged_phsamples', 'phsamples'),
        #      ]
        # ),

        (
            dmri_preprocess_workflow, fslcpgeom_mask,
            [
             ('output.mask', 'dest_file'),
            ]
        ),
        (
            mri_convert,fslcpgeom_mask,
            [
            ('out_file','in_file')
            ]
        ),
#         (
#             fslcpgeom_mask, pbx2,
#             [
#              ('out_file', 'mask')
#             ]
#         ),
        (
            shrink_surface_node, join_seeds,
            [
                ('out_file', 'in1')
            ]
        ),
        
        (
            apply_registration, fslcpgeom_roi,
            [
                ('output_image', 'dest_file')
            ]
        ),
        (
            mri_convert,fslcpgeom_roi,
            [
            ('out_file','in_file')
            ]
        ),
        (
            fslcpgeom_roi,join_seeds,
            [
                ('out_file','in2')
            ]
        ),
#         (
#             join_seeds, pbx2,
#             [
#              ('out', 'seed'),
#             ]
#         ),
    ])

    slurm_logs = (
	 f' -e {os.path.join(os.getcwd(), "slurm_out")}/slurm_%40j.out ' +
	 f'-o {os.path.join(os.getcwd(), "slurm_out")}/slurm_%40j.out '
    )

#    workflow.write_graph(format='pdf', simple_form=False)
    if False and (config['DEFAULT'].get('server', '').lower() == 'margaret'):
        workflow.run(plugin='SLURM',
                     plugin_args={
                         'dont_resubmit_completed_jobs':
                 True,
                         'sbatch_args':
                 '--oversubscribe ' +
                 '-N 1 -n 1 ' +
                 '--time 5-0 ' +
                 f'-e {os.path.join(os.getcwd(), "slurm_out")}/slurm_%40j.out ' +
                 f'-o {os.path.join(os.getcwd(), "slurm_out")}/slurm_%40j.out '
                 #'--exclude=node[25-32] '
                     })
    else:
        #workflow.run(plugin='SLURM',plugin_args={'dont_resubmit_completed_jobs': True,'max_jobs':128,'sbatch_args':'-p menon'})
        #workflow.run(plugin='Linear', plugin_args={'n_procs': 20, 'memory_gb' :32})
        #workflow.write_graph(graph2use='colored', dotfilename='/oak/stanford/groups/menon/projects/cdla/2019_dwi_mathfun/scripts/2019_dwi_pipeline_mathfun/graph_orig.dot')
        #workflow.run(plugin='MultiProc', plugin_args={'n_procs':16, 'memory_gb' :64})
        #workflow.run(plugin='SLURMGraph',plugin_args={'dont_resubmit_completed_jobs': True,'sbatch_args':' -p menon -c 4 --mem=16G -t 4:00:00'})
        workflow.run(plugin='SLURM',plugin_args={
        		'dont_resubmit_completed_jobs': True,'sbatch_args':'--mem=16G -t 6:00:00 --oversubscribe -n 2 --exclude=node[22-32] -c 2','max_jobs':4
	},)
