from diffusion_pipelines.workflows.recon import (
    init_recon_wf,
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

output_dir = os.path.join(f"{root}diffusion/recon_output_{timestamp}")

# Create the surface recon pipeline
recon = init_recon_wf(output_dir=output_dir)

# Provide inputs to the surface recon pipeline
# All subject files
recon.inputs.input_subject.T1 = (
    f"{root}diffusion/bids_data/sub-7014/anat/sub-7014_T1w.nii"
)
recon.inputs.input_subject.dwi = (
    f"{root}diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.nii.gz"
)
recon.inputs.input_subject.bval = (
    f"{root}diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.bval"
)
recon.inputs.input_subject.bvec = (
    f"{root}diffusion/bids_data/sub-7014/dwi/sub-7014_dwi.bvec"
)
recon.inputs.input_subject.subject_id = "sub-7014"
recon.inputs.input_subject.subjects_dir = f"{root}diffusion/bids_data/sub-7014"


# All template files
recon.inputs.input_template.T1 = f"{root}diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
recon.inputs.input_template.T2 = f"{root}diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii"
recon.inputs.input_template.mask = f"{root}diffusion/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"

# create a visual representation of the pipeline
recon.write_graph(
    graph2use="flat",
    dotfilename=os.path.join(output_dir, "graph.dot"),
    format="svg",
)

# Run the surface recon pipeline
recon.run()
