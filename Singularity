Bootstrap: docker
From: test-image

%files
    /preproc

%environment
    export PATH="/usr/local/miniconda/bin:$PATH" \
           CPATH="/usr/local/miniconda/include/:$CPATH" \
           LANG="C.UTF-8" \
           LC_ALL="C.UTF-8" \
           PYTHONNOUSERSITE=1

%labels
    Maintainer "Ziad Motagaly <ziad-ahmed-abdelmotagaly.mohamed@inria.fr>"
    Version "1.0"
    Description "Niflow diffusion docker container."

%runscript
    exec singularity run docker://test-image "$@"
