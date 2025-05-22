# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

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

# Set HOME explicitly
ENV INSTALL_DIR="/opt"

# Create directories and set permissions
RUN mkdir -p $INSTALL_DIR/ANTS \
    $INSTALL_DIR/freesurfer \
    $INSTALL_DIR/miniconda3 \
    $INSTALL_DIR/fsl \
    $INSTALL_DIR/niflow \
    $INSTALL_DIR/Convert3D

# Create non-root user with specified name
RUN useradd -m -s /bin/bash -d /home/${USER_NAME} ${USER_NAME}

# Update HOME environment variable to use the proper user home
ENV HOME="/home/${USER_NAME}"

RUN chown -R ${USER_NAME}:${USER_NAME} $INSTALL_DIR

# Switch to non-root user
USER ${USER_NAME}

# Install ANTS
RUN cd $INSTALL_DIR/ANTS && \
    wget https://github.com/ANTsX/ANTs/releases/download/v2.4.4/ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    unzip ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    rm ants-2.4.4-ubuntu-22.04-X64-gcc.zip

# Set up environment variables for ANTs
ENV ANTSPATH="$INSTALL_DIR/ANTS/ants-2.4.4/bin"
ENV PATH="$ANTSPATH:$PATH"

# Install FreeSurfer
COPY docker/files/freesurfer7.3.2-exclude.txt $INSTALL_DIR/freesurfer/freesurfer7.3.2-exclude.txt
RUN cd $INSTALL_DIR/freesurfer
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz \
    | tar zxv --no-same-owner -C $INSTALL_DIR/freesurfer --exclude-from=$INSTALL_DIR/freesurfer/freesurfer7.3.2-exclude.txt
RUN rm $INSTALL_DIR/freesurfer/freesurfer7.3.2-exclude.txt

# Simulate SetUpFreeSurfer.sh
ENV OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="$INSTALL_DIR/freesurfer/freesurfer"
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
RUN cd $INSTALL_DIR/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $INSTALL_DIR/miniconda3/miniconda.sh && \
    bash $INSTALL_DIR/miniconda3/miniconda.sh -b -u -p $INSTALL_DIR/miniconda3 && \
    rm $INSTALL_DIR/miniconda3/miniconda.sh

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="$INSTALL_DIR/miniconda3/bin:$PATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

# Install selected FSL conda packages
COPY docker/files/fsl_deps.txt $INSTALL_DIR/fsl/fsl_deps.txt
RUN conda install --yes --file $INSTALL_DIR/fsl/fsl_deps.txt -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ -c conda-forge

# Set up environment variables for FSL
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLDIR="$INSTALL_DIR/miniconda3" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# Install niflow
RUN cd $INSTALL_DIR/niflow && \
    git clone https://github.com/niflows/nipype1-workflows.git && \
    cd nipype1-workflows/package && \
    pip install .

# Download and install Convert3D
RUN cd $INSTALL_DIR/Convert3D && \
    wget https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz && \
    tar -xzf c3d-1.0.0-Linux-x86_64.tar.gz && \
    rm c3d-1.0.0-Linux-x86_64.tar.gz
ENV PATH="$INSTALL_DIR/Convert3D/c3d-1.0.0-Linux-x86_64/bin:$PATH"

# Install workbench
RUN conda install --yes conda-forge::connectome-workbench-cli=2.0

# Install diffusion-pipelines
COPY diffusion_pipelines $INSTALL_DIR/diffusion_pipelines
RUN chown -R ${USER_NAME}:${USER_NAME} $INSTALL_DIR/diffusion_pipelines
RUN cd $INSTALL_DIR/diffusion_pipelines && \
    pip install -e . --use-pep517

# copy FreeSurfer license
COPY docker/files/license.txt $FREESURFER_HOME/license.txt

# Set entrypoint to diffusion_pipelines
ENTRYPOINT ["/opt/miniconda3/bin/diffusion_pipelines"]