from diffusion_pipelines.diffusion_preprocessing import (
    create_diffusion_prep_pipeline,
)
import os
import time

# Create output directory with timestamp in YYYYMMDD_HHMMSS format
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Define the root directory
# on local machine
root = "/Users/himanshu/Desktop/"
# on drago
# root = "/storage/store3/work/haggarwa/"

output_dir = os.path.join(
    f"{root}diffusion/internal_pipeline_output_{timestamp}"
)

# Create the diffusion preprocessing pipeline
internal_dmri_pipeline = create_diffusion_prep_pipeline(output_dir=output_dir)

# Provide inputs to the diffusion preprocessing pipeline
# All subject files
internal_dmri_pipeline.inputs.input_subject.dwi = f"{root}diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.nii.gz"
internal_dmri_pipeline.inputs.input_subject.bval = f"{root}diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.bval"
internal_dmri_pipeline.inputs.input_subject.bvec = f"{root}diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.bvec"

# All template files
internal_dmri_pipeline.inputs.input_template.T1 = f"{root}diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
internal_dmri_pipeline.inputs.input_template.T2 = f"{root}diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii"
internal_dmri_pipeline.inputs.input_template.mask = f"{root}diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"

# create a visual representation of the pipeline
internal_dmri_pipeline.write_graph(
    graph2use="flat",
    dotfilename=os.path.join(output_dir, "graph.dot"),
    format="svg",
)

# Run the diffusion preprocessing pipeline
internal_dmri_pipeline.run()
