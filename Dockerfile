# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Get the user and group IDs from build arguments
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=diffusion_pipelines

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    bc \
    python3 \
    python3-pip \
    unzip \
    libgomp1 \
    cmake \
    graphviz \
    tcsh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create user with specified UID/GID
RUN groupadd -f -g $GROUP_ID $USER_NAME && \
    useradd -u $USER_ID -g $GROUP_ID -m $USER_NAME

# Set HOME explicitly
ENV HOME="/home/$USER_NAME"

# Create directories and set permissions
RUN mkdir -p $HOME/ANTS \
    $HOME/freesurfer \
    $HOME/miniconda3 \
    $HOME/fsl \
    $HOME/niflow \
    $HOME/Convert3D && \
    chown -R $USER_ID:$GROUP_ID $HOME

# Switch to the created user
USER $USER_NAME

# Set HOME explicitly
ENV HOME="/home/$USER_NAME"
WORKDIR $HOME

# Install ANTS
RUN cd $HOME/ANTS && \
    wget https://github.com/ANTsX/ANTs/releases/download/v2.4.4/ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    unzip ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    rm ants-2.4.4-ubuntu-22.04-X64-gcc.zip

# Set up environment variables for ANTs
ENV ANTSPATH="$HOME/ANTS/ants-2.4.4/bin"
ENV PATH="$ANTSPATH:$PATH"

# Install FreeSurfer
COPY --chown=$USER_ID:$GROUP_ID docker/files/freesurfer7.3.2-exclude.txt $HOME/freesurfer/freesurfer7.3.2-exclude.txt
RUN cd $HOME/freesurfer
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz \
    | tar zxv --no-same-owner -C $HOME/freesurfer --exclude-from=$HOME/freesurfer/freesurfer7.3.2-exclude.txt
RUN rm $HOME/freesurfer/freesurfer7.3.2-exclude.txt

# Simulate SetUpFreeSurfer.sh
ENV OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="$HOME/freesurfer/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

# Install conda
RUN cd $HOME/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda3/miniconda.sh && \
    bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3 && \
    rm $HOME/miniconda3/miniconda.sh

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="$HOME/miniconda3/bin:$PATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

# Install selected FSL conda packages
COPY --chown=$USER_ID:$GROUP_ID docker/files/fsl_deps.txt $HOME/fsl/fsl_deps.txt
RUN conda install --yes --file $HOME/fsl/fsl_deps.txt -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ -c conda-forge

# Set up environment variables for FSL
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLDIR="$HOME/miniconda3" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# Install niflow
RUN cd $HOME/niflow && \
    git clone https://github.com/niflows/nipype1-workflows.git && \
    cd nipype1-workflows/package && \
    pip install .

# Download and install Convert3D
RUN cd $HOME/Convert3D && \
    wget https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz && \
    tar -xzf c3d-1.0.0-Linux-x86_64.tar.gz && \
    rm c3d-1.0.0-Linux-x86_64.tar.gz
ENV PATH="$HOME/Convert3D/c3d-1.0.0-Linux-x86_64/bin:$PATH"

# Install diffusion-pipelines
COPY --chown=$USER_ID:$GROUP_ID diffusion_pipelines $HOME/diffusion_pipelines
RUN cd $HOME/diffusion_pipelines && \
    pip install -e .

# copy FreeSurfer license
COPY --chown=$USER_ID:$GROUP_ID docker/files/license.txt $FREESURFER_HOME/license.txt

# Set entrypoint to diffusion_pipelines
ENTRYPOINT ["/home/diffusion_pipelines/miniconda3/bin/diffusion_pipelines"]