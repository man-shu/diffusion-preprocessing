# Niflow Diffusion Package

This is a probabilistic tractography package primarily based on BEDPOSTX and ProbTrackX2. 

There are 2 packages that need to be preinstalled to run the niflow package succesfully : 
1. FSL - A comprehensive library of analysis tools for FMRI, MRI and diffusion brain imaging data. https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
2. Diffusion Pipelines - A custom built package contatining the necessary pre processing regitration that can be applied to the raw inputs.

The mode of execution used is Multiproc applied on slurm based parallel clustering . Hence, the package includes: 
1. submit.sh: Configures the parameters for cluster computation
2. mg_test.cfg: Configures the path directories of inputs needed for the main script. Note that the id-list has to have exactly one space charachter between each ID.
3. dti_raw_bvec: A raw bvec file that is used for the niflow_test script, to validate the bvec_flip function.

The main script is m_niflow_multiproc. It contains all the required libraries that can be used interchangeably to customize the diffusion pipline according to the preference of the user. The principal tractography algorithms are constant: bedpostx and probtrackx2.

m_niflow_multiproc_lite is dervied from the main script , and it contains the minimum amount of modules and ndes that are rquired ot run the pipeline. 