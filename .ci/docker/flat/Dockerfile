ARG UBUNTU_VERSION=focal
ARG PYTHON_VERSION=3.11
ARG PYTORCH_ROCM_ARCH=gfx942
ARG PYTORCH_URL=https://github.com/pytorch/pytorch
ARG PYTORCH_BRANCH=main
ARG ROCM_VERSION=6.1

FROM ubuntu:${UBUNTU_VERSION} as rocm
ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_VERSION
ENV ROCM_VERSION=${ROCM_VERSION}

RUN set -ex && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    curl \
    elfutils \
    libnuma1 \
    gpg \
    gpg-agent \
    ca-certificates
  
RUN set -ex && \
  mkdir --parents --mode=0755 /etc/apt/keyrings && \
  curl https://repo.radeon.com/rocm/rocm.gpg.key | \
    gpg -o /etc/apt/keyrings/rocm.gpg --dearmor && \
  . /etc/os-release && \
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/$ROCM_VERSION/ubuntu $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/amdgpu.list && \
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/rocm.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    rocm-dev \
    rocm-llvm-dev \
    rocm-libs

ENV ROCM_PATH="/opt/rocm"
ENV PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin":${PATH}

FROM rocm as conda
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV CONDA_PREFIX="/opt/conda"
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

RUN set -ex && \
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
  sha256sum miniconda.sh > miniconda.sha256 && \
  bash -exu miniconda.sh -bp ${CONDA_PREFIX} && \
  rm miniconda.sh && \
  conda init bash && \
  conda create -n pytorch -y python=${PYTHON_VERSION} intel::mkl-static intel::mkl-include
  
ENV CC=gcc
ENV CXX=g++
ENV CONDA_PYTORCH_PREFIX=${CONDA_PREFIX}/envs/pytorch
ENV CONDA_DEFAULT_ENV=pytorch
ENV PATH=${CONDA_PYTORCH_PREFIX}/bin:${PATH}
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}

FROM conda as magma
RUN set -ex && \
  git clone https://bitbucket.org/icl/magma && \
  cd magma && \
  cp make.inc-examples/make.inc.hip-gcc-mkl make.inc && \
  echo 'LIBDIR += -L$(MKLROOT)/lib' >> make.inc && \
  echo 'LIB += -Wl,--enable-new-dtags -Wl,--rpath,$(ROCM_PATH)/lib -Wl,--rpath,$(MKLROOT)/lib -Wl,--rpath,$(ROCM_PATH)/magma/lib' >> make.inc && \
  echo 'DEVCCFLAGS += --gpu-max-threads-per-block=256' >> make.inc && \
  sed -i 's/^FOPENMP/#FOPENMP/g' make.inc && \
  export MKLROOT=${CONDA_PYTORCH_PREFIX} && \
  export GPU_TARGET="$(echo $PYTORCH_ROCM_ARCH | sed -e 's/;/ /g')" && \
  export LANG=C.UTF-8 && \
  make clean && \
  make -f make.gen.hipMAGMA -j && \
  make lib/libmagma.so -j && \
  cd .. && \
  mv magma ${ROCM_PATH}

FROM magma as triton

ENV CMAKE_LIBRARY_PATH=${CONDA_PYTORCH_PREFIX}/lib
RUN set -ex && \
  ${CONDA_PYTORCH_PREFIX}/bin/python -m pip install git+https://github.com/triton-lang/triton@main#subdirectory=python --upgrade-strategy only-if-needed --verbose

FROM triton as pytorch-src

ARG PYTORCH_URL
ARG PYTORCH_BRANCH

RUN set -ex && \
  git clone --recursive ${PYTORCH_URL} -b ${PYTORCH_BRANCH} && \
  cd pytorch && \
  ${CONDA_PYTORCH_PREFIX}/bin/python -m pip install cmake && \
  ${CONDA_PYTORCH_PREFIX}/bin/python -m pip install ${ROCM_PATH}/share/amd_smi && \
  ${CONDA_PYTORCH_PREFIX}/bin/python -m pip install -r requirements.txt && \
  ${CONDA_PYTORCH_PREFIX}/bin/python tools/amd_build/build_amd.py

FROM pytorch-src as pytorch 
RUN set -ex && \
  cd pytorch && \
  ${CONDA_PYTORCH_PREFIX}/bin/python setup.py install

FROM pytorch as pytorch-avb
RUN set -ex && \
  git clone --recursive https://github.com/pytorch/vision && \
  cd vision && \
  ${CONDA_PYTORCH_PREFIX}/bin/python setup.py install && \
  cd .. && \
  git clone --recursive https://github.com/pytorch/audio && \
  cd audio && \
  ${CONDA_PYTORCH_PREFIX}/bin/python setup.py install && \
  cd .. && \
  git clone --recursive https://github.com/pytorch/benchmark && \
  cd benchmark && \
  ${CONDA_PYTORCH_PREFIX}/bin/python install.py && \
  ${CONDA_PYTORCH_PREFIX}/bin/python -m pip install -e .
