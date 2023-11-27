FROM ubuntu:xenial-20201030

RUN apt-get update && apt-get install -y make && apt-get install -y wget && \
    apt-get install -y --no-install-recommends \
                    autoconf \
        		    unzip \
                    build-essential \
                    bzip2 \
                    ca-certificates \
                    curl \
                    cython3 \
                    git \
                    libtool \
                    lsb-release \
                    pkg-config \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
WORKDIR /preproc
#Freesurfer
RUN wget -O - https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz | tar zxv --no-same-owner -C /opt \
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


WORKDIR /preproc 

COPY preproc/test_docker/neurodebian.gpg /usr/local/etc/neurodebian.gpg

# Installing Neurodebian packages (FSL)

RUN wget -O - "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)


ENV FSL_DIR="/usr/share/fsl/5.0" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

COPY /preproc/test_docker/bin/topup /usr/share/fsl/5.0/bin/topup
COPY /preproc/test_docker/bin/imglob /usr/share/fsl/5.0/bin/imglob
COPY /preproc/test_docker/bin/eddy_openmp  /usr/lib/fsl/eddy_openmp
COPY /preproc/test_docker/lib/* /usr/lib/fsl/

#Install Workbench

WORKDIR /preproc
RUN wget https://github.com/Washington-University/workbench/archive/master.zip 
CMD unzip master.zip && \
rm master.zip  
CMD bash setup_workbench.sh
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

#Install Antspath

WORKDIR /preproc
ENV ANTSPATH=/usr/lib/ants
RUN mkdir -p $ANTSPATH
RUN wget -O - https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz | tar -zxv --strip-components=1
ENV PATH=$ANTSPATH:$PATH

'''
RUN useradd -m -s /bin/bash -G users preproc
WORKDIR /preproc
ENV HOME="/preproc"
'''

WORKDIR /preproc
#Install Minicionda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O miniconda.sh    
CMD bash Miniconda3-4.5.11-Linux-x86_64.sh -b -p /preproc/local/miniconda &&\
rm Miniconda3-4.5.11-Linux-x86_64.sh

ENV PATH="/usr/local/miniconda/bin:$PATH" \
    CPATH="/usr/local/miniconda/include/:$CPATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

RUN conda install -y -c anaconda -c conda-forge \
                     python=3.7.1 \
                     matplotlib=2.2 \
                     numpy=1.20 \
                     pip=20.3 \
                     niflow-nipype1-workflows \
        		     argparse \
                     nipype \
        		     configparser \
                     scikit-learn=0.19 


#Install Diffusion Pipelines

WORKDIR /preproc
COPY /preproc/diffusion_pipelines usr/diffusion_pipelines
WORKDIR /usr/diffusion_pipelines
RUN pip install . 