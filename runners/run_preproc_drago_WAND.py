from diffusion_pipelines.workflows.preprocess import (
    init_preprocess_wf,
)
import os
import time

# Create output directory with timestamp in YYYYMMDD_HHMMSS format
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Define the root directory
# on local machine
root = "/Users/himanshu/Desktop/"
# on drago
root = "/data/parietal/store3/work/haggarwa/diffusion"


output_dir = os.path.join(root, "result", f"preproc_output_{timestamp}")
config_path = os.path.join(
    root, "diffusion-preprocessing", "configs", "config_drago.cfg"
)

# Create the diffusion preprocess wf
preproc = init_preprocess_wf(output_dir=output_dir, config_file=config_path)

# create a visual representation of the pipeline
preproc.write_graph(
    graph2use="flat",
    dotfilename=os.path.join(output_dir, "graph.dot"),
    format="svg",
)

# Run the diffusion preprocessing pipeline
preproc.run(plugin="MultiProc", plugin_args={"n_procs": 100})
