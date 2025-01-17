from diffusion_pipelines.workflows import init_tracto_wf
import os
import time

# Create output directory with timestamp in YYYYMMDD_HHMMSS format
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Define the root directory
# on local machine
root = "/Users/himanshu/Desktop/"
# on drago
# root = "/storage/store3/work/haggarwa/"

output_dir = os.path.join(f"{root}diffusion/preprocess_output_{timestamp}")

# Create the diffusion preprocess wf
preprocess = init_preprocess_wf(
    output_dir=output_dir, config_file="config.cfg"
)
