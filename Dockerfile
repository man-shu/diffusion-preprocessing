# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install all deps in /home
RUN cd /home

# Install ANTS
RUN mkdir -p ANTS && \
    cd ANTS && \
    wget https://github.com/ANTsX/ANTs/releases/download/v2.4.4/ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    unzip ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    rm ants-2.4.4-ubuntu-22.04-X64-gcc.zip

# Set up environment variables for ANTs
ENV ANTSPATH="/home/ANTS/ants-2.4.4/bin" \
    PATH=$ANTSPATH:$PATH

# Install FreeSurfer
RUN mkdir -p freesurfer && \
    cd freesurfer && \
    wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz && \
    tar -vzxpf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
    --exclude='freesurfer/diffusion' \
    --exclude='freesurfer/docs' \
    --exclude='freesurfer/fsfast' \
    --exclude='freesurfer/lib/cuda' \
    --exclude='freesurfer/lib/qt' \
    --exclude='freesurfer/matlab' \
    --exclude='freesurfer/mni/share/man' \
    --exclude='freesurfer/subjects/fsaverage_sym' \
    --exclude='freesurfer/subjects/fsaverage3' \
    --exclude='freesurfer/subjects/fsaverage4' \
    --exclude='freesurfer/subjects/cvs_avg35' \
    --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
    --exclude='freesurfer/subjects/bert' \
    --exclude='freesurfer/subjects/lh.EC_average' \
    --exclude='freesurfer/subjects/rh.EC_average' \
    --exclude='freesurfer/subjects/sample-*.mgz' \
    --exclude='freesurfer/subjects/V1_average' \
    --exclude='freesurfer/trctrain'

# Clean up FreeSurfer installation
RUN rm freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz && \
    cd ..

# Simulate SetUpFreeSurfer.sh
ENV OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/home/freesurfer"
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
RUN mkdir -p miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh && \
    bash miniconda3/miniconda.sh -b -u -p miniconda3 && \
    rm miniconda3/miniconda.sh

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="/home/miniconda/bin:$PATH" \
    CPATH="/home/miniconda/include/:$CPATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

# Install FSL
RUN wget https://git.fmrib.ox.ac.uk/fsl/conda/installer/-/raw/3.13.4/fsl/installer/fslinstaller.py && \
    python fslinstaller.py --no_self_update -n -d /home/fsl -V 6.0.7.11

RUN rm fslinstaller.py
