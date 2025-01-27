from diffusion_pipelines.workflows import init_tracto_wf
import os
import time
from nipype import config

config.enable_debug_mode()
# Create output directory with timestamp in YYYYMMDD_HHMMSS format
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Define the root directory
# on local machine
# root = "/Users/himanshu/Desktop/diffusion"
# on drago
root = "/storage/store3/work/haggarwa/diffusion"

output_dir = os.path.join(
    root, "result", f"tracto_downsampled_output_{timestamp}"
)
config_path = os.path.join(
    root, "diffusion-preprocessing", "configs", "config_drago_downsampled.cfg"
)

# Create the diffusion preprocess wf
tracto = init_tracto_wf(output_dir=output_dir, config_file=config_path)

# create a visual representation of the pipeline
tracto.write_graph(
    graph2use="flat",
    dotfilename=os.path.join(output_dir, "graph.dot"),
    format="svg",
)

# Run the diffusion preprocessing pipeline
tracto.run(plugin="MultiProc", plugin_args={"n_procs": 60})
