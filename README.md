# How to run

- Clone this repository and navigate to the directory

```bash
git clone git@github.com:man-shu/diffusion-preprocessing.git
cd diffusion-preprocessing
```

- Build the docker image

  - If you're using a machine with x86_64 architecture (check with `uname -m`):

    ```bash
    docker image build \
    --tag dmriprep-tracto \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" .
    ```

  - If you're using a machine with ARM architecture (for example, Apple M1):

    ```bash
    docker image build \
    --platform linux/x86_64 \
    --tag dmriprep-tracto \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" .
    ```

- Create a config file, for example:

  - `config.cfg`

    ```python
    [DATASET]
    directory = /home/input/data/WAND-bids
    acquisition = AxCaliberConcat
    # Select a subset of subjects by separating them with commas
    # Select all of them by setting the value to all or deleting the line
    subject = 00395, 01187

    [TEMPLATE]
    directory = /home/input/data/mni_icbm152_nlin_sym_09a
    T1 = mni_icbm152_t1_tal_nlin_sym_09a.nii
    T2 = mni_icbm152_t2_tal_nlin_sym_09a.nii
    mask = mni_icbm152_t1_tal_nlin_sym_09a_mask.nii

    [ROIS]
    directory = /home/input/data/rois

    [OUTPUT]
    cache = /home/input/dmriprep-tracto_cache/
    derivatives = /home/input/data/WAND-bids/derivatives/

    # The pipelines to run
    [PIPELINE]
    # You can choose to run either of the preprocessing and 
    # reconstruction pipeline or both
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
    - That directory on our host machine contains everything:
      - our dataset (indicated via `[DATASET]`)
      - the template (via `[TEMPLATE]`)
      - the ROIs (via `[ROIS]`)
      - and the output directory (via `[OUTPUT]`).
    - The intermediate outputs will be saved in the `cache` directory
    (under `[OUTPUT]`) and the final outputs will be saved in the `derivatives`
     directory (also under `[OUTPUT]`).

- Run the container

    ```bash
    docker container run --rm --interactive \
    --mount type=bind,source=/data/parietal/store3/work/haggarwa/diffusion,target=/home/input \
    dmriprep-tracto:latest -< /data/parietal/store3/work/haggarwa/diffusion/diffusion-preprocessing/configs/config_dockerdrago_WAND.cfg 
    ```
