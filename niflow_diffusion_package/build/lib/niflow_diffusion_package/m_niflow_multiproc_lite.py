#!/bin/env python

import nipype.pipeline.engine as pe
from nipype.interfaces import fsl
from nipype import DataGrabber, IdentityInterface
import numpy
from nipype.interfaces.utility.wrappers import Function
import configparser
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


    flip_bvectors_node = pe.Node(
        interface=Function(
            input_names=['bvecs_in', 'flip'], output_names=['bvecs_out'],
            function=bvec_flip
        ),
        name='flip_bvecs',
    )
    flip_bvectors_node.inputs.flip = (-1, 1, 1)

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
    tractography_wf.connect(data_source, 'bvec', flip_bvectors_node, 'bvecs_in')

    tractography_wf.connect(template_source, 'T1', dmri_preprocess_workflow, 'input_template.T1')
    tractography_wf.connect(template_source, 'T2', dmri_preprocess_workflow, 'input_template.T2')

    tractography_wf.connect(flip_bvectors_node, 'bvecs_out', dmri_preprocess_workflow, 'input_subject.bvec')
    tractography_wf.connect(data_source, 'dwi', dmri_preprocess_workflow, 'input_subject.dwi')
    tractography_wf.connect(data_source, 'bval', dmri_preprocess_workflow, 'input_subject.bval')

    tractography_wf.connect(dmri_preprocess_workflow, 'output.bval', bedp, 'inputnode.bvals')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.bvec_rotated', bedp, 'inputnode.bvecs')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.dwi_rigid_registered', bedp, 'inputnode.dwi')
    tractography_wf.connect(dmri_preprocess_workflow, 'output.mask', bedp, 'inputnode.mask')

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
