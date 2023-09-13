#!/bin/env python

import nipype.interfaces.mrtrix3 as mrt
import nipype.pipeline.engine as pe
from nipype.interfaces import fsl
from nipype import DataGrabber, DataSink, IdentityInterface, MapNode, JoinNode
import numpy
from nipype.interfaces.utility.wrappers import Function
import nipype.interfaces.ants as ants
import configparser
import sys
from nipype import Merge
from nipype.interfaces.freesurfer import ReconAll, MRIConvert
from nipype import logging, Workflow
import argparse

from niflow.nipype1.workflows.dmri.fsl.dti import bedpostx_parallel
import diffusion_pipelines.diffusion_preprocessing as dp


def bvec_flip(bvecs_in, flip) -> str:
    from os.path import join, basename
    from os import getcwd
    import numpy as np

    print(bvecs_in)
    bvecs = np.loadtxt(bvecs_in).T * flip

    output_file = str(join(getcwd(), basename(bvecs_in)))
    np.savetxt(output_file, bvecs)
    return output_file


def create_workflow(id_values):
    dmri_preprocess_workflow = dp.create_diffusion_prep_pipeline(
        'dmri_preprocess')

    # Define the path to the input and output files

    PATH = '/data/parietal/store/work/zmohamed/mathfun/'

    subject_list = config['DEFAULT']['id_list'].split(" ")
    visits = list(config['DEFAULT']['visits'])
    subjects_dir = config['DEFAULT']['subjects_dir']
    sessions = list(config['DEFAULT']['sessions'])

    ''' 
    Identity mapping of subjects is based on 3 criteria. 
    The subject_id, the visit, and session number.
    '''

    infosource = pe.Node(IdentityInterface
                         (fields=['subject_id', 'visit', 'session']),
                         name='subjects'
                         )

    infosource.iterables = [('subject_id', [subject_list[id_values]]),
                            ('visit', visits), ('session', sessions)]

    # Concacetantion of subject id and the visit number.

    subject_id_visit = pe.Node(interface=Function(
            input_names=['subject_id', 'visit'], output_names=['composite_id'],
            function=lambda subject_id, visit: '{}_{}'.format(subject_id, visit)
            ),
        name='subject_id_visit'
    )

    data_source = pe.Node(DataGrabber(
                           infields=[],
                           outfields=['dwi', 'bval', 'bvec', 'mask',
                                      'roi', 'template', 'T1', 'T1_brain', 'parc']),
                          name='input_node'
                          )

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
        template: [['subject_id', 'visit', 'session']]
        for template in data_source.inputs.field_template.keys()
    }

    '''
    recon-all generates surfaces and parcellations of structural data
    from anatomical images of a subject.
    '''

    recon_all = pe.Node(interface=ReconAll(), name='recon_all')
    recon_all.inputs.directive = 'all'
    recon_all.inputs.subjects_dir = subjects_dir
    recon_all.inputs.openmp = 20
    recon_all.inputs.mprage = True
    recon_all.inputs.parallel = True
    recon_all.interface.num_threads = 20
    recon_all.inputs.flags = "-no-isrunning"

    flip_bvectors_node = pe.Node(
        interface=Function(
            input_names=['bvecs_in', 'flip'], output_names=['bvecs_out'],
            function=bvec_flip
        ),
        name='flip_bvecs',
    )
    flip_bvectors_node.inputs.flip = (-1, 1, 1)

    mrconvert_nifti_to_mif = pe.Node(interface=mrt.MRConvert(
                                     out_file='dwi.mif'),
                                     name='mrconvert'
                                     )

    '''
    Extract diffusion-weighted volumes, b=0 volumes, or certain shells from a
    DWI dataset
    '''

    dwiextract = pe.Node(interface=mrt.DWIExtract(
                         bzero=True, out_file='b0.mif'),
                         name='dwiextract'
                         )
    
    '''
    Eliminate the 4th dimension of the b0 image 
    by computing the mean based on 3 axes
    '''

    reduce_dimension = pe.Node(interface=mrt.MRMath(
                                operation='mean', axis=3, out_file='b0_mean.mif'),
                               name='reduce_dimension'
                               )

    mrconvert_mif_to_nifti_b0 = pe.Node(interface=mrt.MRConvert(
                                         out_file='b0.nii.gz'),
                                        name='mrconvert_mif_to_nifti_b0'
                                        )

    template_source = pe.Node(DataGrabber(infields=[],
                                          outfields=['T1', 'T1_brain', 'T1_mask',
                                                     'T2', 'T2_brain', 'T2_mask']),
                              name='mni_template'
                              )
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

    roi_source = pe.Node(DataGrabber(infields=[]),
                         name='rois'
                         )
    roi_source.inputs.sort_filelist = True
    roi_source.inputs.base_directory = config['ROIS']['directory']
    #roi_source.inputs.template = '*bin.nii.gz'
    roi_source.inputs.template = 'combined_BN*bin*.nii.gz'

    mri_convert = pe.Node(interface=MRIConvert(), name='mri_convert')
    mri_convert.inputs.out_type = 'nii'
    mri_convert.inputs.subjects_dir = subjects_dir

    affine_initializer = pe.Node(interface=ants.AffineInitializer(
                                  num_threads=20),
                                 name='affine_initializer'
                                 )
        
    '''    
    Construct rigid and affine transforms 
    '''

    registration_affine = pe.Node(interface=ants.Registration(),
                                  name='registration_affine')

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
    registration_affine.inputs.use_histogram_matching = [True, True]  # This is the default
    registration_affine.inputs.output_warped_image = 'output_warped_image.nii.gz'

    '''    
    Construct “SyN”: Symmetric normalization:
    Affine + deformable non linear transformation
    '''

    registration_nl = pe.Node(interface=ants.Registration(), name='registration_nl')
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
    registration_nl.inputs.use_histogram_matching = [True]  # This is the default
    registration_nl.inputs.output_warped_image = 'output_warped_image.nii.gz'

    '''
    Apply_registration, applied to an input image, transforms it 
    according to a reference image and a set of transforms
    '''

    apply_registration = pe.Node(interface=ants.ApplyTransforms(),
                                 name='apply_registration'
                                 )
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 0
    apply_registration.inputs.interpolation = 'NearestNeighbor'

    '''
    Bedpostx runs Markov Chain Monte Carlo sampling to build up 
    distributions on diffusion parameters at each voxel.
    It creates all the files necessary for running probabilistic tractography.
    '''

    params = dict(n_fibres=2, fudge=1, burn_in=1000,
                  n_jumps=1250, sample_every=25
                  )
    bedp = bedpostx_parallel('nipype_bedpostx', params=params)

    '''    
    Use FSL  probtrackx2 for probabalistic tractography on bedpostx results
    '''  

    pbx2 = pe.Node(
        interface=fsl.ProbTrackX2(),
        name='probtrackx2',
    )
    pbx2.inputs.n_samples = 5000
    pbx2.inputs.n_steps = 2000
    pbx2.inputs.step_length = 0.5
    pbx2.inputs.network = True
    pbx2.inputs.omatrix1 = True
    pbx2.inputs.out_dir = '/data/parietal/store/work/zmohamed/mathfun/tractography_wf/_session_1_subject_id_' \
        + subject_list[id_values] + '_visit_1/probtrackx2'
    pbx2.inputs.distthresh1 = 5
    pbx2.inputs.args = " --ompl --fibthresh=0.01 "


    # Create a Nipype workflow

    tractography_wf = pe.Workflow(name='tractography_wf',  base_dir=PATH)

    tractography_wf.connect(infosource, 'subject_id', data_source, 'subject_id')
    tractography_wf.connect(infosource, 'visit', data_source, 'visit')
    tractography_wf.connect(infosource, 'session', data_source, 'session')

    tractography_wf.connect(infosource, 'subject_id', subject_id_visit, 'subject_id')
    tractography_wf.connect(infosource, 'visit', subject_id_visit, 'visit')

    tractography_wf.connect(data_source, 'dwi', mrconvert_nifti_to_mif, 'in_file')
    tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', dwiextract, 'in_file')

    tractography_wf.connect(data_source, 'bval', dwiextract, 'in_bval')
    tractography_wf.connect(data_source, 'bvec', dwiextract, 'in_bvec')
    tractography_wf.connect(data_source, 'bvec', flip_bvectors_node, 'bvecs_in')

    tractography_wf.connect(dwiextract, 'out_file', reduce_dimension, 'in_file')
    tractography_wf.connect(reduce_dimension, 'out_file', mrconvert_mif_to_nifti_b0, 'in_file')

    tractography_wf.connect(template_source, 'T1_brain', affine_initializer, 'moving_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', affine_initializer, 'fixed_image')

    tractography_wf.connect(template_source, 'T1_brain', registration_affine, 'moving_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_affine, 'fixed_image')
    tractography_wf.connect(affine_initializer, 'out_file', registration_affine, 'initial_moving_transform')

    tractography_wf.connect(template_source, 'T1_brain', registration_nl, 'moving_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_nl, 'fixed_image')

    tractography_wf.connect(registration_affine, 'forward_transforms', registration_nl, 'initial_moving_transform')
    tractography_wf.connect(registration_affine, 'forward_invert_flags', registration_nl, 'invert_initial_moving_transform')

    tractography_wf.connect(registration_nl, 'forward_transforms', apply_registration, 'transforms')
    tractography_wf.connect(registration_nl, 'forward_invert_flags', apply_registration, 'invert_transform_flags')

    tractography_wf.connect(template_source, 'T1_mask', apply_registration, 'input_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', apply_registration, 'reference_image')

    tractography_wf.connect(template_source, 'T1', dmri_preprocess_workflow, 'input_template.T1')
    tractography_wf.connect(template_source, 'T2', dmri_preprocess_workflow, 'input_template.T2')

    '''
    The following connections can replace line 316, 317 in order 
    to include recon-all if necessary.

    tractography_wf.connect(data_source, 'T1', recon_all, 'T1_files')
    tractography_wf.connect(subject_id_visit, 'composite_id', recon_all, 'subject_id')
    tractography_wf.connect(recon_all, 'brain', mri_convert, 'in_file')
    tractography_wf.connect(mri_convert, 'out_file', dmri_preprocess_workflow, 'input_template.T1')
    tractography_wf.connect(mri_convert, 'out_file', dmri_preprocess_workflow, 'input_template.T2')
    '''

    tractography_wf.connect(flip_bvectors_node, 'bvecs_out', dmri_preprocess_workflow, 'input_subject.bvec')
    tractography_wf.connect(data_source, 'dwi', dmri_preprocess_workflow, 'input_subject.dwi')
    tractography_wf.connect(data_source, 'bval', dmri_preprocess_workflow, 'input_subject.bval')

    tractography_wf.connect(dmri_preprocess_workflow, 'output.bval', bedp, 'inputnode.bvals')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.bvec_rotated', bedp, 'inputnode.bvecs')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.dwi_rigid_registered', bedp, 'inputnode.dwi')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.mask', bedp, 'inputnode.mask')

    '''
    The following connections can replace line 334 -337, and line 356 in order 
    to eliminate the dependence on dmri_preprocess_workflow

    tractography_wf.connect(data_source, 'bval', bedp, 'inputnode.bvals')
    tractography_wf.connect(data_source, 'bvec', bedp, 'inputnode.bvecs')
    tractography_wf.connect(data_source, 'dwi', bedp, 'inputnode.dwi')
    tractography_wf.connect(apply_registration, 'output_image', bedp, 'inputnode.mask')
    tractography_wf.connect(apply_registration, 'output_image', pbx2, 'mask')

    '''

    tractography_wf.connect(bedp, 'outputnode.merged_thsamples', pbx2, 'thsamples')
    tractography_wf.connect(bedp, 'outputnode.merged_fsamples', pbx2, 'fsamples')
    tractography_wf.connect(bedp, 'outputnode.merged_phsamples', pbx2, 'phsamples')

    tractography_wf.connect(roi_source, 'outfiles', pbx2, 'seed')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.mask', pbx2, 'mask')


    # Run the workflow
    tractography_wf.run(plugin='MultiProc', plugin_args={'n_procs': 220, 'memory_gb': 320,
                                                         'dont_resubmit_completed_jobs': True})

config = configparser.ConfigParser()

parser = argparse.ArgumentParser()
parser.add_argument("--id_to_process", type=int)
parser.add_argument('--config', type=str)

args, _ = parser.parse_known_args()
config.read(args.config)
print("Args" + str(args.id_to_process))

# Call the run_workflow function with each id value from the config file
create_workflow(args.id_to_process)
