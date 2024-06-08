from diffusion_pipelines.diffusion_preprocessing import (
    create_diffusion_prep_pipeline,
)
import os
from time import time

# Create output directory with timestamp in YYYYMMDD_HHMMSS format
timestamp = time()
output_dir = os.path.join(
    f"/storage/store3/work/haggarwa/diffusion/internal_pipeline_output_{timestamp:.0f}"
)

# Create the diffusion preprocessing pipeline
internal_dmri_pipeline = create_diffusion_prep_pipeline(output_dir=output_dir)

# Connect the input node to the diffusion preprocessing pipeline
internal_dmri_pipeline.inputs.input_subject.dwi = "/storage/store3/work/haggarwa/diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.nii.gz"
internal_dmri_pipeline.inputs.input_subject.bval = "/storage/store3/work/haggarwa/diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.bval"
internal_dmri_pipeline.inputs.input_subject.bvec = "/storage/store3/work/haggarwa/diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.bvec"
internal_dmri_pipeline.inputs.input_template.T1 = "/storage/store3/work/haggarwa/diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
internal_dmri_pipeline.inputs.input_template.T2 = "/storage/store3/work/haggarwa/diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii"

# create a visual representation of the pipeline
internal_dmri_pipeline.write_graph(
    graph2use="flat",
    dotfilename=os.path.join(output_dir, "graph.dot"),
    format="svg",
)

# Run the diffusion preprocessing pipeline
internal_dmri_pipeline.run()
