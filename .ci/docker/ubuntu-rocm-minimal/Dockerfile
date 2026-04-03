ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION} AS builder

# Match original Dockerfile ARG interface for compatibility
ARG UBUNTU_VERSION
ARG ROCM_VERSION
ARG PYTORCH_ROCM_ARCH
ARG BUILD_ENVIRONMENT
ARG GCC_VERSION
ARG NINJA_VERSION
ARG TRITON
ARG INDUCTOR_BENCHMARKS

ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_VERSION=${ROCM_VERSION} \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}

# Install base system packages (single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    gnupg \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add ROCm repository
RUN wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" \
    > /etc/apt/sources.list.d/rocm.list

# Install ONLY required ROCm libraries (NOT -complete)
RUN apt-get update && apt-get install -y --no-install-recommends \
    rocm-hip-runtime-dev \
    rocm-hip-sdk \
    rocblas-dev \
    hipblas-dev \
    rocrand-dev \
    hiprand-dev \
    hipsparse-dev \
    hipfft-dev \
    hipsolver-dev \
    rocsolver-dev \
    hipblaslt-dev \
    miopen-hip-dev \
    rocprim-dev \
    hipcub-dev \
    rocthrust-dev \
    hiprtc-dev \
    comgr \
    hsa-rocr-dev \
    rocm-smi-lib \
    rocm-cmake \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set ROCm environment
ENV ROCM_PATH=/opt/rocm \
    ROCM_HOME=/opt/rocm \
    HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Install uv (fast Python package manager) - replaces conda
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/

# Create venv and install build dependencies
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Install core Python build requirements
COPY requirements-ci.txt /tmp/requirements-ci.txt
RUN uv pip install --no-cache -r /tmp/requirements-ci.txt && \
    uv pip install --no-cache \
    "setuptools>=70.1.0,<82" \
    cmake \
    ninja \
    numpy \
    packaging \
    pyyaml \
    requests \
    six \
    "typing-extensions>=4.10.0" \
    sympy \
    filelock \
    jinja2 \
    networkx \
    fsspec && \
    rm /tmp/requirements-ci.txt

# Install GCC if version specified (matches install_gcc.sh)
RUN if [ -n "${GCC_VERSION}" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends gcc-${GCC_VERSION} g++-${GCC_VERSION} && \
      update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 && \
      update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# Install Ninja if version specified (matches install_ninja.sh)
RUN if [ -n "${NINJA_VERSION}" ]; then \
      uv pip install --no-cache ninja==${NINJA_VERSION}; \
    fi

# Install Triton if specified (matches install_triton.sh)
COPY ci_commit_pins/triton.txt /tmp/triton.txt
RUN if [ -n "${TRITON}" ]; then \
      TRITON_COMMIT=$(cat /tmp/triton.txt) && \
      uv pip install --no-cache "triton @ git+https://github.com/pytorch/triton.git@${TRITON_COMMIT}"; \
    fi && \
    rm /tmp/triton.txt

# ===== Stage 2: Test Environment =====
FROM builder AS tester

RUN uv pip install --no-cache \
    pytest \
    hypothesis \
    expecttest \
    pytest-xdist \
    pytest-rerunfailures

# ===== Stage 3: Benchmark Environment =====
FROM tester AS benchmarks

ARG INDUCTOR_BENCHMARKS

# Install inductor benchmark dependencies if specified
COPY ci_commit_pins/huggingface-requirements.txt /tmp/huggingface-requirements.txt
COPY ci_commit_pins/timm.txt /tmp/timm.txt
COPY ci_commit_pins/torchbench.txt /tmp/torchbench.txt

RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then \
      uv pip install --no-cache -r /tmp/huggingface-requirements.txt && \
      TIMM_COMMIT=$(cat /tmp/timm.txt) && \
      uv pip install --no-cache "timm @ git+https://github.com/huggingface/pytorch-image-models.git@${TIMM_COMMIT}"; \
    fi && \
    rm -f /tmp/*.txt

CMD ["bash"]
