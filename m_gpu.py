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
from nipype.interfaces import fsl


# Define the paths to the input and output files
 
out_dir = '/data/parietal/store/work/zmohamed/mathfun/output'
PATH = '/data/parietal/store/work/zmohamed/mathfun/'

# Create a Nipype workflow

tractography_wf = pe.Workflow(name='tractography_wf',  base_dir=PATH)

data_source = pe.Node(DataGrabber(infields=[],
                                   outfields=['dwi', 'bval', 'bvec', 'mask', 'roi','template', 'T1', 'T1_brain']),
                       name='input_node')

data_source.inputs.sort_filelist = True
data_source.inputs.base_directory = '/data/parietal/store/work/zmohamed/mathfun/'
data_source.inputs.template = ''
data_source.inputs.field_template = {
        'mask': '/data/parietal/store/work/zmohamed/mathfun/hcp_templates/MNI152_T1_1mm_brain_mask.nii.gz' ,
        'roi':'/data/parietal/store/work/zmohamed/mathfun/dwi_rois/1mm/L_PPC_6mm_-44_-42_52_1mmiso_roi.nii.gz',
        'dwi': '/data/parietal/store/work/zmohamed/mathfun/tp2/7014/visit2/session1/dwi/dwi_raw.nii.gz',
        'bval': '/data/parietal/store/work/zmohamed/mathfun/tp2/7014/visit2/session1/dwi/dti_raw.bvals',
        'bvec': '/data/parietal/store/work/zmohamed/mathfun/tp2/7014/visit2/session1/dwi/dti_raw.bvecs',
        'T1': '/data/parietal/store/work/zmohamed/mathfun/hcp_templates/MNI152_T1_1mm.nii.gz',
        'T1_brain': '/data/parietal/store/work/zmohamed/mathfun/hcp_templates/MNI152_T1_1mm_brain.nii.gz',
    
    }

    # Where can i find the MNI ? 



mrconvert_nifti_to_mif = pe.Node(
    interface=mrt.MRConvert( 
        out_file='dwi.mif'
        
    ),
    name='mrconvert'
)


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



#apply_registration = pe.MapNode(interface=ants.ApplyTransforms(),
#                              name='apply_registration', iterfield=['input_image'])

apply_registration = pe.Node(interface=ants.ApplyTransforms(),
                              name='apply_registration')
apply_registration.inputs.dimension = 3
apply_registration.inputs.input_image_type = 0
apply_registration.inputs.interpolation = 'NearestNeighbor'



dwi2response = pe.Node(
    interface=mrt.ResponseSD(
        algorithm='tournier',
     ),
    name='dwi2response'
)
 
tractography_wf.connect(dwiextract, 'out_file', reduce_dimension, 'in_file')
tractography_wf.connect(reduce_dimension, 'out_file', mrconvert_mif_to_nifti_b0, 'in_file')


# MNI to T1 

tractography_wf.connect(data_source, 'T1_brain', affine_initializer, 'moving_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', affine_initializer, 'fixed_image')

tractography_wf.connect(data_source, 'T1_brain', registration_affine, 'moving_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_affine, 'fixed_image')
tractography_wf.connect(affine_initializer, 'out_file', registration_affine , 'initial_moving_transform')


tractography_wf.connect(data_source, 'T1_brain', registration_nl, 'moving_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_nl, 'fixed_image')

tractography_wf.connect(registration_affine, 'forward_transforms', registration_nl, 'initial_moving_transform')
tractography_wf.connect(registration_affine, 'forward_invert_flags', registration_nl, 'invert_initial_moving_transform')

tractography_wf.connect(registration_nl, 'forward_transforms', apply_registration, 'transforms')
tractography_wf.connect(registration_nl, 'forward_invert_flags', apply_registration, 'invert_transform_flags')

tractography_wf.connect(data_source, 'roi', apply_registration, 'input_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', apply_registration, 'reference_image')




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


tdimap = pe.Node(
    interface=mrt.ComputeTDI(
        out_file='tdi.nii.gz',
     ),
    name='tdimap'
)


# Convert images from floating point to integer values

def convertfloat2int(float_image):
    import nibabel as nib
    import os, os.path
    import numpy as np
 
    # Load the floating-point image
    #my_float_img = nib.load(float_image)
    my_float_img = nib.load(float_image)

    data = my_float_img.get_fdata()
    data = data.astype('int16')

    # Convert to integer type
    output_int_img = nib.Nifti1Image(data, my_float_img.affine)
    
    filename = f"image_{output_int_img}.nii.gz"
    output_path = os.path.join(os.getcwd(), filename)

    nib.save(output_int_img, os.path.join(os.getcwd(),'output_image.nii.gz')) 

    #my_int_img = nib.load('/data/parietal/store/work/zmohamed/mathfun/output_image.nii.gz')
 
    return os.path.join(os.getcwd(),'output_image.nii.gz')
    

Convert_float2int = Function(
    input_names=['float_image'], output_names=['output_int_img'],
    function=convertfloat2int
  )

convert_float2int =  pe.Node(interface = Convert_float2int,
          name = 'convert_float2int')

tractography_wf.connect(apply_registration, 'output_image', convert_float2int, 'float_image')  
#tractography_wf.connect(data_source, 'roi', convert_float2int, 'float_image')  



tckconn = pe.Node(
    interface=mrt.BuildConnectome(
        out_file='connectome.csv'
        
     ),
    name='tckconn'
)


tractography_wf.connect(tckgen, 'out_file', tdimap, 'in_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', tdimap, 'reference') #should be b0 image (or use transformation to get higher res)


tractography_wf.connect(tckgen, 'out_file', tckconn, 'in_file')
tractography_wf.connect(convert_float2int, 'output_int_img', tckconn, 'in_parc')  

#tractography_wf.connect(data_source, 'roi', tckconn, 'in_parc')  # this needs to be changed to b0 image as input to tckconn


tractography_wf.connect(tckgen, 'out_file', data_sink, 'prob_tractography')
tractography_wf.connect(tdimap, 'out_file', data_sink, 'tdi_out')

tractography_wf.connect(tckconn, 'out_file', data_sink, 'conn_out')


# Quality check of B0 and TDI 

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

tractography_wf.connect(tdimap, 'out_file', plot_tdi, 'TDI_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', plot_tdi, 'background')
#tractography_wf.connect(apply_registration, 'output_image', plot_tdi, 'background')
tractography_wf.connect(plot_tdi, 'output_image', data_sink, 'tdi_image')


# Quality check of ROI and TDI 


def plotROI (ROI_file, background):
    import nibabel as nib
    from nilearn import plotting 
    import os, os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(ROI_file)}.png'
    )
    roi_img = nib.load(ROI_file)
    
    plotting.plot_roi(roi_img,bg_img = background , display_mode='mosaic', cut_coords= 3, title='ROI map', output_file= output_file)
    return output_file   

PlotROI = Function(
    input_names=['ROI_file','background'], output_names=['output_image'],
    function=plotROI
  )

plot_roi =  pe.Node(interface = PlotROI,
          name = 'plot_roi')



tractography_wf.connect(apply_registration, 'output_image', plot_roi, 'ROI_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', plot_roi, 'background')
tractography_wf.connect(plot_roi, 'output_image', data_sink, 'roi_image')

#tractography_wf.connect(apply_registration, 'output_image', data_sink, 'temp')
#tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', data_sink, 'temp1')




# Run the workflow
tractography_wf.run(plugin='MultiProc', plugin_args={'dont_resubmit_completed_jobs':True})


# plot with the b0 image 
# add Affine then non linear registration from MNI space to the subject's b0 image.
# warp parcellation (from nilearn / or ROI) using ANTS and the output of the registration.
# Conclusion : warp the parcellation from the MNI space to the subject space
# plot the parcellation on the subject space .




