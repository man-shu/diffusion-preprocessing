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


    
def bvec_flip(bvecs_in, flip):
    from os.path import join, basename
    from os import getcwd

    import numpy as np

    print(bvecs_in)
    bvecs = np.loadtxt(bvecs_in).T * flip

    output_file = str(join(getcwd(), basename(bvecs_in)))
    np.savetxt(output_file, bvecs)
    
    return output_file

config.update_config({'logging': {'log_directory': os.path.join(os.getcwd(), 'logs'),
                                  'workflow_level': 'DEBUG',
                                  'interface_level': 'DEBUG',
                                  'log_to_file': True
                                  },
                      'execution': {'stop_on_first_crash': True},
                    })


config = configparser.ConfigParser()
config.read(sys.argv[1])


# Define the paths to the input and output files
 
out_dir = '/data/parietal/store/work/zmohamed/mathfun/output'
PATH = '/data/parietal/store/work/zmohamed/mathfun/'
seed_num = 50000



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

flip_bvectors_node = pe.Node(
    interface=Function(
        input_names=['bvecs_in', 'flip'], output_names=['bvecs_out'],
        function=bvec_flip
    ),
    name='flip_bvecs',
)
flip_bvectors_node.inputs.flip = (-1, 1, 1)


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



bet = pe.Node(BET(frac=0.2, mask=True), name='bet')


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



# Create a Nipype workflow

workflow = Workflow('diffusion_workflow_new_mgt', base_dir=PATH)
workflow.connect([
    (infosource, data_source, [('subject_id', 'subject_id'),
                                ('visit', 'visit'),
                ('session','session')]),

    (infosource, subject_id_visit, [
        ('subject_id', 'subject_id'),
        ('visit', 'visit')
    ]),

    (data_source, flip_bvectors_node, [('bvec', 'bvecs_in')]),
 
    #(roi_source, apply_registration, [('outfiles', 'input_image')]),

    # input to bedp

    (flip_bvectors_node, bedp, [('bvecs_out', 'inputnode.bvecs')]),

    (template_source,bedp,[('T1_mask', 'inputnode.mask')]),
    
    (data_source,bedp,[('dwi', 'inputnode.dwi'), ('bval', 'inputnode.bvals')]),

    # bedp -- > pbx2
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

    (roi_source, pbx2, [('outfiles', 'seed')]),


    (template_source, pbx2,[('T1_mask', 'mask'),]),

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
