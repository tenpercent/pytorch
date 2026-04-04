ARG UBUNTU_VERSION=24.04

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
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add ROCm repository and disable Ubuntu's ROCm packages
RUN wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
    > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: rocm* hip* roc* comgr miopen*\nPin: release o=Ubuntu\nPin-Priority: -1\n\nPackage: rocm* hip* roc* comgr miopen*\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
    > /etc/apt/preferences.d/rocm-pin

# Install ONLY required ROCm libraries (NOT -complete)
# Package names verified from ROCm 7.2 apt repository
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
    hipcub-dev \
    rocthrust-dev \
    comgr \
    rocm-llvm-dev \
    hsa-rocr-dev \
    rocm-smi-lib \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set ROCm environment
ENV ROCM_PATH=/opt/rocm \
    ROCM_HOME=/opt/rocm \
    HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Install uv (fast Python package manager) - replaces conda
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Create venv and install build dependencies
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Install core Python build requirements
# Note: requirements-ci.txt is optional, install core deps directly
RUN uv pip install --no-cache \
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
    fsspec

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

# Install Triton if specified
# Note: Triton commit pin is optional, install latest if TRITON flag set
RUN if [ -n "${TRITON}" ]; then \
      uv pip install --no-cache triton; \
    fi

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
# Note: Using latest versions instead of pinned commits
RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then \
      uv pip install --no-cache \
        transformers \
        accelerate \
        timm \
        datasets \
        tokenizers; \
    fi

CMD ["bash"]
