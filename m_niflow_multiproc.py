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
from nipype import logging, Workflow
import argparse
import submitit
import functools

from niflow.nipype1.workflows.dmri.fsl.dti import bedpostx_parallel
import diffusion_pipelines.diffusion_preprocessing as dp


#warnings.filterwarnings("ignore")

def create_workflow(id_values):



 
    '''config.update_config({'logging': {'log_directory': os.path.join(os.getcwd(), 'logs'),
                                    'workflow_level': 'DEBUG',
                                    'interface_level': 'DEBUG',
                                    'log_to_file': True
                                    },
                        'execution': {'stop_on_first_crash': True},
                    })'''




    # Define the paths to the input and output files

    out_dir = '/data/parietal/store/work/zmohamed/mathfun/output'
    PATH = '/data/parietal/store/work/zmohamed/mathfun/'


    subject_list = config['DEFAULT']['id_list'].split(" ")

    visits = list(config['DEFAULT']['visits'])
    subjects_dir = config['DEFAULT']['subjects_dir']
    sessions = list(config['DEFAULT']['sessions'])

    print("Key_val" + subject_list[id_values])
   

    infosource = pe.Node(IdentityInterface(fields=['subject_id', 'visit','session']),
                        name='subjects')
    #infosource.inputs.subject_id = subject_list[id_values] 
    #infosource.inputs.subject_id = id_values

    infosource.iterables = [('subject_id', [subject_list[id_values]] ), ('visit', visits), ('session', sessions)]
    


    data_source = pe.Node(DataGrabber(infields=[],
                                    outfields=['dwi', 'bval', 'bvec', 'mask', 'roi','template', 'T1', 'T1_brain', 'parc']),
                        name='input_node')

    data_source.inputs.sort_filelist = True
    data_source.inputs.base_directory = config['DEFAULT']['base_directory']
    #data_source.inputs.base_directory = '/data/parietal/store/work/zmohamed/mathfun/raw_data_visit1'
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

    mrconvert_nifti_to_mif = pe.Node(
    interface=mrt.MRConvert( 
        out_file='dwi.mif'
        
    ),
    name='mrconvert'
    )


    dwiextract = pe.Node(
    interface=mrt.DWIExtract(
        bzero=True,
        out_file='b0.mif'
    ),
    name='dwiextract'
    )


    reduce_dimension = pe.Node(
    interface=mrt.MRMath(
        operation='mean',
        axis = 3 ,
        out_file='b0_mean.mif'
    ),
    name='reduce_dimension'
    )


    mrconvert_mif_to_nifti_b0 = pe.Node(
    interface=mrt.MRConvert( 
        out_file='b0.nii.gz'
        
    ),
    name='mrconvert_mif_to_nifti_b0'
    )


    template_source = pe.Node(DataGrabber(infields=[], outfields=['T1', 'T1_brain', 'T1_mask', 'T2', 'T2_brain', 'T2_mask']),
                            name='mni_template')
    template_source.inputs.sort_filelist = True
    template_source.inputs.base_directory = config['TEMPLATE']['directory']
    #template_source.inputs.base_directory  = '/data/parietal/store/work/zmohamed/mathfun/hcp_templates'
    template_source.inputs.template = ''
    
    
    template_source.inputs.field_template = {
        'T1': config['TEMPLATE']['T1'],
        'T1_brain': config['TEMPLATE']['T1_brain'],
        'T1_mask': config['TEMPLATE']['T1_mask'],
        'T2': config['TEMPLATE']['T2'],
        'T2_brain': config['TEMPLATE']['T2_brain'],
    }


    '''template_source.inputs.field_template = {
        'T1':  'MNI152_T1_1mm.nii.gz',
        'T1_brain': 'MNI152_T1_1mm_brain.nii.gz',
        'T1_mask': 'MNI152_T1_1mm_brain_mask.nii.gz',
        'T2': 'MNI152_T2_1mm.nii.gz',
        'T2_brain': 'MNI152_T2_1mm_brain.nii.gz',
    }'''


    template_source.inputs.template_args = {
        template: []
        for template in template_source.inputs.field_template.keys()
    }


    affine_initializer = pe.Node(
    interface=ants.AffineInitializer(
        num_threads = 20

    ),                          
    name='affine_initializer')



    registration_affine = pe.Node(
    interface=ants.Registration(


    ), name='registration_affine'
    )
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
    registration_nl.inputs.use_histogram_matching = [True] # This is the default
    registration_nl.inputs.output_warped_image = 'output_warped_image.nii.gz'



    apply_registration = pe.Node(interface=ants.ApplyTransforms(),
                                name='apply_registration')
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 0
    apply_registration.inputs.interpolation = 'NearestNeighbor'


    #bedp = pe.Node(BEDPOSTX5(n_fibres=5), name = 'bedpostx')
    params = dict(n_fibres = 2, fudge = 1, burn_in = 1000, n_jumps = 1250, sample_every = 25)
    bedp = bedpostx_parallel('nipype_bedpostx', params=params)


    roi_source = pe.Node(DataGrabber(infields=[]),
                            name='rois')
    roi_source.inputs.sort_filelist = True
    roi_source.inputs.base_directory = config['ROIS']['directory']
    #roi_source.inputs.base_directory  = '/data/parietal/store/work/zmohamed/mathfun/dwi_rois/1mm'
    roi_source.inputs.template = 'combined_BN*bin*.nii.gz'



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



    # Create a Nipype workflow
    tractography_wf = pe.Workflow(name='tractography_wf',  base_dir=PATH)

    tractography_wf.connect(infosource, 'subject_id', data_source, 'subject_id')
    tractography_wf.connect(infosource, 'visit', data_source, 'visit')
    tractography_wf.connect(infosource, 'session', data_source, 'session')


    tractography_wf.connect(data_source, 'dwi', mrconvert_nifti_to_mif, 'in_file')
    tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', dwiextract, 'in_file')


    tractography_wf.connect(data_source, 'bval', dwiextract, 'in_bval')
    tractography_wf.connect(data_source, 'bvec', dwiextract, 'in_bvec')


    tractography_wf.connect(dwiextract, 'out_file', reduce_dimension, 'in_file')
    tractography_wf.connect(reduce_dimension, 'out_file', mrconvert_mif_to_nifti_b0, 'in_file')

    tractography_wf.connect(template_source, 'T1_brain', affine_initializer, 'moving_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', affine_initializer, 'fixed_image')

    tractography_wf.connect(template_source, 'T1_brain', registration_affine, 'moving_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_affine, 'fixed_image')
    tractography_wf.connect(affine_initializer, 'out_file', registration_affine , 'initial_moving_transform')

    tractography_wf.connect(template_source, 'T1_brain', registration_nl, 'moving_image')
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_nl, 'fixed_image')

    tractography_wf.connect(registration_affine, 'forward_transforms', registration_nl, 'initial_moving_transform')
    tractography_wf.connect(registration_affine, 'forward_invert_flags', registration_nl, 'invert_initial_moving_transform')

    tractography_wf.connect(registration_nl, 'forward_transforms', apply_registration, 'transforms')
    tractography_wf.connect(registration_nl, 'forward_invert_flags', apply_registration, 'invert_transform_flags')

    tractography_wf.connect(template_source, 'T1_mask', apply_registration, 'input_image')   # create the parcellations 
    tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', apply_registration, 'reference_image')


    tractography_wf.connect(data_source, 'bval', bedp, 'inputnode.bvals')
    tractography_wf.connect(data_source, 'bvec', bedp, 'inputnode.bvecs')
    tractography_wf.connect(data_source, 'dwi', bedp, 'inputnode.dwi')

    tractography_wf.connect(apply_registration, 'output_image', bedp, 'inputnode.mask')


    tractography_wf.connect(bedp, 'outputnode.merged_thsamples', pbx2, 'thsamples')
    tractography_wf.connect(bedp, 'outputnode.merged_fsamples', pbx2, 'fsamples')
    tractography_wf.connect(bedp, 'outputnode.merged_phsamples', pbx2, 'phsamples')

    tractography_wf.connect(roi_source, 'outfiles', pbx2, 'seed')
    #tractography_wf.connect(template_source, 'T1_mask', pbx2, 'mask')
    tractography_wf.connect(apply_registration, 'output_image', pbx2, 'mask')

    '''
    #BEDPOSTX 5 Connections

    tractography_wf.connect(data_source, 'bval', bedp, 'bvals')
    tractography_wf.connect(data_source, 'bvec', bedp, 'bvecs')
    tractography_wf.connect(data_source, 'dwi', bedp, 'dwi')
    tractography_wf.connect(apply_registration, 'output_image', bedp, 'mask')

    '''
    # Run the workflow
    tractography_wf.run(plugin='MultiProc', plugin_args={'n_procs':20, 'memory_gb': 8, "timeout": 3600, 'dont_resubmit_completed_jobs':True})





'''
id_values = ['7014','7035']
#visits = [1]
#sessions =[1]


log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder=log_folder)
# the following line tells the scheduler to only run\

#jobs = executor.map_array(create_workflow, id_values)  # just a list of jobs

fns = [functools.partial(create_workflow, a) for a in id_values]
executor.submit_array(fns)
'''



config = configparser.ConfigParser()


parser = argparse.ArgumentParser()
parser.add_argument("--id_to_process", type=int)
parser.add_argument('--config', type = str)


args, _ = parser.parse_known_args()
config.read(args.config)

print("Args" + str(args.id_to_process))


# Call the run_workflow function with each value from the config
create_workflow(args.id_to_process)




