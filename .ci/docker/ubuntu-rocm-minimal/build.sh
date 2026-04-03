#!/bin/bash
set -ex

# Match original build script interface
UBUNTU_VERSION=${UBUNTU_VERSION:-24.04}
ROCM_VERSION=${ROCM_VERSION:-7.2}
PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx942}
BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT:-pytorch-rocm-${ROCM_VERSION}}
GCC_VERSION=${GCC_VERSION:-}  # Optional
NINJA_VERSION=${NINJA_VERSION:-}  # Optional
TRITON=${TRITON:-}  # Optional flag
INDUCTOR_BENCHMARKS=${INDUCTOR_BENCHMARKS:-}  # Optional flag
TARGET=${TARGET:-builder}  # builder, tester, or benchmarks

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build context is the repo root (for COPY requirements-ci.txt, etc.)
BUILD_CONTEXT="${SCRIPT_DIR}/../../.."

docker build \
  --target ${TARGET} \
  --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
  --build-arg ROCM_VERSION=${ROCM_VERSION} \
  --build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
  --build-arg BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT} \
  --build-arg GCC_VERSION=${GCC_VERSION} \
  --build-arg NINJA_VERSION=${NINJA_VERSION} \
  --build-arg TRITON=${TRITON} \
  --build-arg INDUCTOR_BENCHMARKS=${INDUCTOR_BENCHMARKS} \
  -t pytorch-rocm-${TARGET}:${ROCM_VERSION} \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${BUILD_CONTEXT}"

echo "Successfully built pytorch-rocm-${TARGET}:${ROCM_VERSION}"
docker images | grep pytorch-rocm-${TARGET}
