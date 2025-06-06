# How to run

- Clone this repository and navigate to the directory

  ```bash
  git clone git@github.com:man-shu/diffusion-preprocessing.git
  cd diffusion-preprocessing
  ```

- Create a config file, for example:

  - `config.cfg`

    ```python
    [DATASET]
    directory = /home/input/WAND-downsampled
    acquisition = AxCaliber1
    # Select a subset of subjects by separating them with commas
    # Select all of them by setting the value to all or deleting the line
    subject = 00395, 01187

    [TEMPLATE]
    directory = /home/input/mni_icbm152_nlin_sym_09a-downsampled
    T1 = mni_icbm152_t1_tal_nlin_sym_09a.nii
    T2 = mni_icbm152_t2_tal_nlin_sym_09a.nii
    mask = mni_icbm152_t1_tal_nlin_sym_09a_mask.nii

    [ROIS]
    directory = /home/input/rois-downsampled

    [OUTPUT]
    cache = /home/input/cache
    derivatives = /home/input/WAND-downsampled/derivatives

    # The pipelines to run
    [PIPELINE]
    # You can choose to run either of the preprocessing and reconstruction 
    # pipeline or both
    preprocessing = True
    reconstruction = False
    # If tractography is set to True, the pipeline will run the both 
    # preprocessing and reconstruction steps anyway
    tractography = False

    # Set number of threads to use for the pipeline
    [MULTIPROCESSING]
    n_jobs = 30
    ```

    - Note that `/home/input/` is the path inside the docker container and
     all the paths in the config file are relative to this path.
    - So here we will mount a directory from our host machine
     to `/home/input/` when running the container.
    - This directory on our host machine contains everything:
      - our dataset (indicated via `[DATASET]`)
      - the template (via `[TEMPLATE]`)
      - the ROIs (via `[ROIS]`)
      - and the output directory (via `[OUTPUT]`).
    - The intermediate outputs will be saved in the `cache` directory
    (under `[OUTPUT]`) and the final outputs will be saved in the `derivatives`
    directory (also under `[OUTPUT]`).

## Using Docker

- Pull the docker image

  ```bash
  docker pull haggarwa/diffusion_pipelines:latest
  ```

- **Optionally**, you can also build the docker image

  - If you're using a machine with x86_64 architecture (check with `uname -m`):

    ```bash
    docker image build --tag haggarwa/diffusion_pipelines .
    ```

  - If you're using a machine with ARM architecture (for example, Apple M1):

    ```bash
    docker image build --platform linux/x86_64 --tag haggarwa/diffusion_pipelines .
    ```

- Run the container

  ```bash
  docker container run --rm --interactive \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source=./data,target=/home/input \
  haggarwa/diffusion_pipelines:latest -< config.cfg 
  ```

  - If you're using a machine with ARM architecture (for example, Apple M1):

    ```bash
    docker container run --rm --interactive \
    --platform linux/x86_64 \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source=./data,target=/home/input \
    haggarwa/diffusion_pipelines:latest -< config.cfg 
    ```

## Using Singularity

- Build the singularity image

  ```bash
  singularity build diffusion_pipelines.sif docker://haggarwa/diffusion_pipelines:latest
  ```

**Note** that this build will work even if you are a non-root user. Only building singularity images from `.def` files requires root privileges.

- Run the singularity image

  ```bash
  singularity exec --env-file singularity_env.txt \
  --bind ./data:/home/input diffusion_pipelines.sif \
  /opt/miniconda3/bin/diffusion_pipelines -< config.cfg
  ```

- Alternatively, you can run the singularity image in an interactive shell

  ```bash
  singularity shell --env-file singularity_env.txt \
  --bind ./data:/home/input diffusion_pipelines.sif
  ```
