#!/bin/env python

import nipype.interfaces.mrtrix3 as mrt
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import os
import nipype.interfaces.utility as util
from nipype import DataGrabber, DataSink
from nipype.interfaces.utility import Merge
from typing import List
import numpy 
from nilearn import plotting
import nibabel as nib
from nipype.interfaces.utility.wrappers import Function
import nipype.interfaces.ants as ants
from nipype.interfaces.ants.base import ANTSCommand


# Define the paths to the input and output files
 
out_dir = '/data/parietal/store/work/zmohamed/mathfun/output'

# Create a Nipype workflow

tractography_wf = pe.Workflow(name='tractography_wf')

data_source = pe.Node(DataGrabber(infields=[],
                                   outfields=['dwi', 'bval', 'bvec', 'mask', 'template']),
                       name='input_node')

data_source.inputs.sort_filelist = True
data_source.inputs.base_directory = '/data/parietal/store/work/zmohamed/mathfun/'
data_source.inputs.template = ''
data_source.inputs.field_template = {
        'mask': '/data/parietal/store/work/zmohamed/mathfun/hcp_templates/MNI152_T1_0.8mm_brain_mask.nii.gz' ,
        'dwi': '/data/parietal/store/work/zmohamed/mathfun/tp2/7014/visit2/session1/dwi/dwi_raw.nii.gz',
        'bval': '/data/parietal/store/work/zmohamed/mathfun/tp2/7014/visit2/session1/dwi/dti_raw.bvals',
        'bvec': '/data/parietal/store/work/zmohamed/mathfun/tp2/7014/visit2/session1/dwi/dti_raw.bvecs',
        'template': '/data/parietal/store/work/zmohamed/mathfun/hcp_templates/MNI152_T2_1mm.nii.gz',
    
    }



mrconvert_nifti_to_mif = pe.Node(
    interface=mrt.MRConvert( 
        out_file='dwi.mif'
        
    ),
    name='mrconvert'
)


#tractography_wf.connect ( data_source,('bvec','bval'), mrconvert, 'grad_fsl' )
tractography_wf.connect(data_source, 'dwi', mrconvert_nifti_to_mif, 'in_file')




dwiextract = pe.Node(
    interface=mrt.DWIExtract(
        bzero=True,
        out_file='b0.mif'
    ),
    name='dwiextract'
)

tractography_wf.connect(data_source, 'bval', dwiextract, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwiextract, 'in_bvec')
tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', dwiextract, 'in_file')

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



dwi2response = pe.Node(
    interface=mrt.ResponseSD(
        algorithm='tournier',
     ),
    name='dwi2response'
)
 
tractography_wf.connect(dwiextract, 'out_file', reduce_dimension, 'in_file')
tractography_wf.connect(reduce_dimension, 'out_file', mrconvert_mif_to_nifti_b0, 'in_file')
tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', dwi2response, 'in_file')
tractography_wf.connect(data_source, 'bval', dwi2response, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwi2response, 'in_bvec')



# Estimate the fiber orientation distribution (FOD) using constrained spherical deconvolution (CSD)
dwi2fod = pe.Node(
    interface=mrt.ConstrainedSphericalDeconvolution(
        algorithm='csd',
    
    ),
    name='dwi2fod'
)
#tractography_wf.connect(data_source, ('bvec','bval'), dwi2fod, 'grad_fsl')
 
tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', dwi2fod, 'in_file')
tractography_wf.connect(dwi2response, 'wm_file', dwi2fod, 'wm_txt')
tractography_wf.connect(data_source, 'bval', dwi2fod, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwi2fod, 'in_bvec')


tckgen = pe.Node(
    interface=mrt.Tractography(
        algorithm='iFOD2',
        select=10000,
        out_file='prob_tractography.tck',
        n_trials = 10000,
    ),
    name='tckgen'
)
tractography_wf.connect(dwi2fod, 'wm_odf', tckgen, 'in_file')
tractography_wf.connect(data_source, 'bval', tckgen, 'in_bval')
tractography_wf.connect(data_source, 'bvec', tckgen, 'in_bvec')
tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', tckgen, 'seed_image')



data_sink = pe.Node(DataSink(), name="datasink")
data_sink.inputs.base_directory = out_dir

""""
# Define the output node
output_node = pe.Node(
    interface=util.IdentityInterface(
        fields=['prob_tractography']
    ),
    name='output_node'
)
#tractography_wf.connect(tckgen, 'out_file', output_node, 'prob_tractography')
"""


tckmap = pe.Node(
    interface=mrt.ComputeTDI(
        out_file='b0.nii.gz',
     ),
    name='tckmap'
)




tckconn = pe.Node(
    interface=mrt.BuildConnectome(
        out_file='connectome.csv'
        
     ),
    name='tckconn'
)


tractography_wf.connect(tckgen, 'out_file', tckmap, 'in_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', tckmap, 'reference') #should be b0 image (or use transformation to get higher res)
#tractography_wf.connect(tckgen, 'out_file', tckconn, 'in_file')
#tractography_wf.connect(mrconvert_mif_to_nifti, 'out_file', tckconn, 'in_parc')  # this needs to be changed to b0 image as input to tckconn


tractography_wf.connect(tckgen, 'out_file', data_sink, 'prob_tractography')
tractography_wf.connect(tckmap, 'out_file', data_sink, 'tdi_out')
#tractography_wf.connect(tckconn, 'out_file', data_sink, 'conn_out')


def plotTDI (TDI_file, background):
    import nibabel as nib
    from nilearn import plotting 
    import os, os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(TDI_file)}.png'
    )
    stat_img = nib.load(TDI_file)
    
    plotting.plot_stat_map(stat_img,bg_img = background , display_mode='mosaic', cut_coords= 3, title='Statistical map', output_file= output_file)
    return output_file   

PlotTDI = Function(
    input_names=['TDI_file','background'], output_names=['output_image'],
    function=plotTDI
  )

plot_tdi =  pe.Node(interface = PlotTDI,
          name = 'plot_tdi')

tractography_wf.connect(tckmap, 'out_file', plot_tdi, 'TDI_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', plot_tdi, 'background')
tractography_wf.connect(plot_tdi, 'output_image', data_sink, 'tdi_image')




# Run the workflow
tractography_wf.run(plugin='MultiProc', plugin_args={'dont_resubmit_completed_jobs':True})


# plot with the b0 image 
# add Affine then non linear registration from MNI space to the subject's b0 image.
# warp parcellation (from nilearn / or ROI) using ANTS and the output of the registration.
# Conclusion : warp the parcellation from the MNI space to the subject space
# plot the parcellation on the subject space .




