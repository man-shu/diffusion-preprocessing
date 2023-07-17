#!/bin/env python

import nipype.interfaces.mrtrix3 as mrt
import nipype.pipeline.engine as pe
from nipype.interfaces import fsl
from nipype.interfaces.fsl import BET, BEDPOSTX, BEDPOSTX5
import os
from nipype.interfaces import utility
from nipype import DataGrabber, DataSink, IdentityInterface, MapNode, JoinNode
from typing import List
import numpy 
from nilearn import plotting
import nibabel as nib
from nipype.interfaces.utility.wrappers import Function
import nipype.interfaces.ants as ants
from nipype.interfaces.ants.base import ANTSCommand
import configparser
import sys
import nipype.interfaces.fsl.utils as fslu
from nipype import Merge
from nipype.interfaces.freesurfer import ReconAll, MRIsConvert, MRIConvert
from nipype import config, logging, Workflow


from niflow.nipype1.workflows.dmri.fsl.dti import create_bedpostx_pipeline
import diffusion_pipelines.diffusion_preprocessing as dp



config.update_config({'logging': {'log_directory': os.path.join(os.getcwd(), 'logs'),
                                  'workflow_level': 'DEBUG',
                                  'interface_level': 'DEBUG',
                                  'log_to_file': True
                                  },
                      'execution': {'stop_on_first_crash': True},
                    })



# Define the paths to the input and output files
 
out_dir = '/data/parietal/store/work/zmohamed/mathfun/output'
PATH = '/data/parietal/store/work/zmohamed/mathfun/'
seed_num = 50000



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

    
def bvec_flip(bvecs_in, flip):
    from os.path import join, basename
    from os import getcwd

    import numpy as np

    print(bvecs_in)
    bvecs = np.loadtxt(bvecs_in).T * flip

    output_file = str(join(getcwd(), basename(bvecs_in)))
    np.savetxt(output_file, bvecs)
    
    return output_file


#dmri_preprocess_workflow = dp.create_diffusion_prep_pipeline(
#        'dmri_preprocess')
config = configparser.ConfigParser()
config.read(sys.argv[1])




subject_list = config['DEFAULT']['id_list'].split(" ")
visits = list(config['DEFAULT']['visits'])
subjects_dir = config['DEFAULT']['subjects_dir']
sessions = list(config['DEFAULT']['sessions'])

infosource = pe.Node(IdentityInterface(fields=['subject_id', 'visit','session']),
                      name='subjects')
infosource.iterables = [('subject_id', subject_list), ('visit', visits), ('session', sessions)]

subject_id_visit = pe.Node(
        interface=Function(
            input_names=['subject_id', 'visit'], output_names=['composite_id'],
            function=lambda subject_id, visit: '{}_{}'.format(subject_id, visit)
        ),
        name='subject_id_visit'
)  

data_source = pe.Node(DataGrabber(infields=[],
                                   outfields=['dwi', 'bval', 'bvec', 'mask', 'roi','template', 'T1', 'T1_brain', 'parc']),
                       name='input_node')

data_source.inputs.sort_filelist = True
data_source.inputs.base_directory = config['DEFAULT']['base_directory']
data_source.inputs.template = ''


data_source.inputs.field_template = {
        'T1': '%s/visit%s/session%s/anat/T1w.nii',
        'dwi': '%s/visit%s/session%s/dwi/dwi_raw.nii.gz',
        'bval': '%s/visit%s/session%s/dwi/dti_raw.bvals',
        'bvec': '%s/visit%s/session%s/dwi/dti_raw.bvecs',
        
    }
data_source.inputs.template_args = {
        template: [['subject_id', 'visit','session']]
        for template in data_source.inputs.field_template.keys()
    }


roi_source = pe.Node(DataGrabber(infields=[]),
                           name='rois')
roi_source.inputs.sort_filelist = True
roi_source.inputs.base_directory = config['ROIS']['directory']
roi_source.inputs.template = 'combined_BN*bin*.nii.gz'


template_source = pe.Node(DataGrabber(infields=[], outfields=['T1', 'T1_brain', 'T1_mask', 'T2', 'T2_brain', 'T2_mask']),
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


dwiextract = pe.Node(
    interface=mrt.DWIExtract(
        bzero=True,
        out_file='b0.nii.gz'
    ),
    name='dwiextract'
)

reduce_dimension = pe.Node(
    interface=mrt.MRMath(
        operation='mean',
        axis = 3 ,
        out_file='b0_mean.nii.gz'
    ),
    name='reduce_dimension'
)


flip_bvectors_node = pe.Node(
    interface=Function(
        input_names=['bvecs_in', 'flip'], output_names=['bvecs_out'],
        function=bvec_flip
    ),
    name='flip_bvecs',
)
flip_bvectors_node.inputs.flip = (-1, 1, 1)


bet = pe.Node(BET(frac=0.2, mask=True), name='bet')

recon_all = pe.Node(interface=ReconAll(), name='recon_all')
recon_all.inputs.directive = 'all'
recon_all.inputs.subjects_dir = subjects_dir
recon_all.inputs.openmp = 20
recon_all.inputs.mprage = True
recon_all.inputs.parallel = True
recon_all.interface.num_threads = 20
recon_all.inputs.flags = "-no-isrunning"



ras_conversion_matrix = pe.Node(
        interface=Function(
            input_names=['subjects_dir', 'subject_id'],
            output_names=['output_mat'],
            function=freesurfer_get_ras_conversion_matrix
        ),
        name='ras_conversion_matrix'
)


affine_initializer = pe.Node(interface=ants.AffineInitializer(), name='affine_initializer')
affine_initializer.inputs.num_threads = 20
affine_initializer.interface.num_threads = 20

registration_affine = pe.Node(interface=ants.Registration(), name='reg_aff')
registration_affine.inputs.num_threads = 16
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

registration_nl = pe.Node(interface=ants.Registration(), name='reg_nl')
registration_nl.inputs.num_threads = 16
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

select_nl_transform = pe.Node(interface=utility.Select(), name='select_nl_transform')
select_nl_transform.inputs.index = [1]

registration = pe.Node(interface=ants.Registration(), name='reg')
registration.inputs.num_threads = 16
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

mris_convert = MapNode(interface=MRIsConvert(), name='mris_convert', iterfield=['in_file'])
mris_convert.inputs.out_datatype = 'gii'
mris_convert.inputs.subjects_dir = subjects_dir

mri_convert = pe.Node(interface=MRIConvert(), name='mri_convert')
mri_convert.inputs.out_type = 'nii'
mri_convert.inputs.subjects_dir = subjects_dir

freesurfer_surf_2_native = MapNode(
    interface=Function(
        input_names=['freesurfer_gii_surface', 'ras_conversion_matrix', 'warps'], output_names=['out_surf'],
        function=freesurfer_gii_2_native
    ),
    name='freesurfer_surf_2_native',
    iterfield=['freesurfer_gii_surface']
)


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

join_seeds = pe.Node( interface=Merge(2), name='join_seeds')

#bedp = pe.Node(BEDPOSTX5(n_fibres=5), name = 'bedpostx')
params = dict(n_fibres = 2, fudge = 1, burn_in = 1000, n_jumps = 1250, sample_every = 25)
bedp = create_bedpostx_pipeline('nipype_bedpostx', params=params)

pbx2 = pe.Node(
        interface=fsl.ProbTrackX2(),
        name='probtrackx2',
    )
pbx2.inputs.n_samples = 5000
pbx2.inputs.n_steps = 2000
pbx2.inputs.step_length = 0.5
pbx2.inputs.omatrix1 = True
pbx2.inputs.distthresh1 = 5
pbx2.inputs.args = " --ompl --fibthresh=0.01 "



data_sink = pe.Node(DataSink(), name="datasink")
data_sink.inputs.base_directory = out_dir


'''
merge_mean_S0samples = pe.Node(Merge(dimension='t'), name='merge_mean_S0samples')
merge_mean_thsamples = pe.Node(Merge(dimension='t'), name='merge_mean_thsamples')
merge_mean_phsamples = pe.Node(Merge(dimension='t'), name='merge_mean_phsamples')
merge_mean_fsamples = pe.Node(Merge(dimension='t'), name='merge_mean_fsamples')
merge_mean_dsamples = pe.Node(Merge(dimension='t'), name='merge_mean_dsamples')
'''

# Create a Nipype workflow

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

    (flip_bvectors_node, bedp, [('bvecs_out', 'inputnode.bvecs')]),

    (template_source,bedp,[('T1_mask', 'inputnode.mask')],),
    (data_source,bedp,[('dwi', 'inputnode.dwi'), ('bval', 'inputnode.bvals'),]),
    (freesurfer_surf_2_native,shrink_surface_node,[('out_surf', 'surface')],),
    (mri_convert,shrink_surface_node,[('out_file', 'image')],),
    (bedp, pbx2,
         [
          ('outputnode.thsamples', 'thsamples'),
          ('outputnode.fsamples', 'fsamples'),
          ('outputnode.phsamples', 'phsamples'),
         ]
     ),

     (bedp, data_sink,
         [
          ('outputnode.dyads_disp', 'dyads_dispersion'),
          ('outputnode.dyads', 'mean_S0samples'),
          ('outputnode.mean_thsamples', 'mean_dsamples'),
            
         ]
     ),

    (template_source, pbx2,[('T1_mask', 'mask'),]),
    (shrink_surface_node, join_seeds,[('out_file', 'in1')]
    ),
    (
        apply_registration, join_seeds,
        [
            ('output_image', 'in2')
        ]
    ),
    (
        join_seeds, pbx2,
        [
            ('out', 'seed'),
        ]
    ),
])

workflow.run(plugin='MultiProc', plugin_args={'n_procs':20, 'memory_gb': 8,'dont_resubmit_completed_jobs':True})



'''

tractography_wf = pe.Workflow(name='tractography_wf',  base_dir=PATH)

tractography_wf.connect(infosource, 'subject_id', data_source, 'subject_id')
tractography_wf.connect(infosource, 'visit', data_source, 'visit')
tractography_wf.connect(infosource, 'session', data_source, 'session')

tractography_wf.connect(infosource, 'subject_id', subject_id_visit, 'subject_id')
tractography_wf.connect(infosource, 'visit', subject_id_visit, 'visit')

tractography_wf.connect(data_source, 'dwi', dwiextract, 'in_file')
tractography_wf.connect(data_source, 'bval', dwiextract, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwiextract, 'in_bvec')


tractography_wf.connect(dwiextract, 'out_file', reduce_dimension, 'in_file')
tractography_wf.connect(reduce_dimension, 'out_file', bet, 'in_file')

tractography_wf.connect(bet, 'out_file', bedp, 'dwi')
#tractography_wf.connect(data_source, 'dwi', bedp, 'dwi')

tractography_wf.connect(data_source, 'bvec', bedp, 'bvecs')
tractography_wf.connect(data_source, 'bval', bedp, 'bvals')
tractography_wf.connect(template_source, 'T1_mask', bedp, 'mask')

tractography_wf.connect(bedp, 'mean_S0samples', merge_mean_S0samples, 'in_files')
tractography_wf.connect(bedp, 'mean_thsamples', merge_mean_thsamples, 'in_files')
tractography_wf.connect(bedp, 'mean_phsamples', merge_mean_phsamples, 'in_files')
tractography_wf.connect(bedp, 'mean_fsamples', merge_mean_fsamples, 'in_files')
tractography_wf.connect(bedp, 'mean_dsamples', merge_mean_dsamples, 'in_files')








# Run the workflow
tractography_wf.run(plugin='MultiProc', plugin_args={'n_procs':20, 'memory_gb': 8,'dont_resubmit_completed_jobs':True})
#tractography_wf.run(plugin='SLURMGraph',plugin_args={'dont_resubmit_completed_jobs': True})







'''
