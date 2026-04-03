# Minimal ROCm Docker Image for PyTorch

Lightweight multi-stage Docker images for building and testing PyTorch with ROCm support.

## Size Comparison

| Image | Original (ubuntu-rocm) | Minimal | Reduction |
|-------|----------------------|---------|-----------|
| Builder | ~50GB | ~15-18GB | ~65% |
| Tester | ~50GB+ | ~22GB | ~55% |
| Benchmarks | ~50GB+ | ~30GB | ~40% |

## Key Optimizations

- **Selective ROCm packages**: Only required libraries instead of `-complete` meta-package
- **uv instead of Conda**: Fast Python package manager, minimal overhead
- **Multi-stage builds**: Build only what you need
- **Aggressive cleanup**: Remove apt cache and temp files after each layer
- **Build environment only**: No PyTorch source included (mount at runtime)

## Prerequisites

- Docker 20.10+
- For GPU testing: ROCm-compatible GPU (gfx942, gfx950, etc.)

## Quick Start

### Build Images

```bash
cd .ci/docker/ubuntu-rocm-minimal

# Build minimal builder (default)
./build.sh

# Build with all options
ROCM_VERSION=7.2 \
PYTORCH_ROCM_ARCH="gfx942;gfx950" \
GCC_VERSION=11 \
TRITON=1 \
./build.sh

# Build test image
TARGET=tester ./build.sh

# Build benchmark image
TARGET=benchmarks INDUCTOR_BENCHMARKS=1 ./build.sh
```

### Build PyTorch from Source

```bash
# Mount PyTorch source and build (like CI does)
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v $(pwd):/workspace \
  -w /workspace \
  pytorch-rocm-builder:7.2 \
  bash

# Inside container:
export PYTORCH_ROCM_ARCH=gfx942
pip install -e . --no-build-isolation -v
```

### Test Installation

```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  pytorch-rocm-tester:7.2 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

## Build Arguments

All original `ubuntu-rocm/Dockerfile` ARGs are supported:

| Argument | Default | Description |
|----------|---------|-------------|
| `UBUNTU_VERSION` | `22.04` | Ubuntu base version |
| `ROCM_VERSION` | `7.2` | ROCm version to install |
| `PYTORCH_ROCM_ARCH` | `gfx942` | GPU architecture(s), semicolon-separated |
| `BUILD_ENVIRONMENT` | `pytorch-rocm-${ROCM_VERSION}` | Build identifier |
| `GCC_VERSION` | (empty) | Optional: install specific GCC version |
| `NINJA_VERSION` | (empty) | Optional: install specific Ninja version |
| `TRITON` | (empty) | Set to `1` to install Triton |
| `INDUCTOR_BENCHMARKS` | (empty) | Set to `1` to install benchmark deps |

## Stages

### 1. builder (default)
**Size:** ~15-18GB
**Contains:**
- Ubuntu 22.04 base
- ROCm 7.2 development libraries
- Python 3.10 + venv
- uv package manager
- Build tools (cmake, ninja, gcc)
- Python build dependencies

**Use for:** Building PyTorch from source

### 2. tester
**Size:** ~22GB
**Contains:** builder + test dependencies (pytest, hypothesis, expecttest)

**Use for:** Running PyTorch tests

### 3. benchmarks
**Size:** ~30GB
**Contains:** tester + ML dependencies (timm, transformers, etc.)

**Use for:** Running inductor benchmarks

## Environment Variables

The following environment variables are set automatically:

```bash
ROCM_PATH=/opt/rocm
ROCM_HOME=/opt/rocm
HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/opt/venv/bin:$PATH
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

**Critical:** Set `PYTORCH_ROCM_ARCH` before building PyTorch:
```bash
export PYTORCH_ROCM_ARCH=gfx942  # or gfx950, or "gfx942;gfx950"
```

## ROCm Libraries Included

Minimal required set (per cmake/public/LoadHIP.cmake):

- ROCm HIP runtime & SDK (compiler)
- rocBLAS, hipBLAS, hipBLASLt
- rocRAND, hipRAND
- hipSPARSE
- hipFFT
- hipSOLVER, rocSOLVER
- MIOpen (deep learning primitives)
- rocPRIM, hipCUB, rocThrust
- hipRTC (runtime compilation)
- AMD compiler (comgr)
- HSA runtime
- ROCm SMI

**NOT included** (optional in PyTorch):
- MAGMA (can be enabled with `USE_MAGMA=ON`)
- RCCL (for distributed training)
- ROCm profiling tools (rocprof, roctracer)

## Publishing to Docker Hub

### Private Repository (Free Tier)

```bash
# Login
docker login

# Tag images with your username
docker tag pytorch-rocm-builder:7.2 yourusername/pytorch-rocm-builder:7.2
docker tag pytorch-rocm-tester:7.2 yourusername/pytorch-rocm-tester:7.2
docker tag pytorch-rocm-benchmarks:7.2 yourusername/pytorch-rocm-benchmarks:7.2

# Push (automatically private on free tier)
docker push yourusername/pytorch-rocm-builder:7.2
docker push yourusername/pytorch-rocm-tester:7.2
docker push yourusername/pytorch-rocm-benchmarks:7.2
```

### Alternative: GitHub Container Registry

```bash
# Login to ghcr.io
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag pytorch-rocm-builder:7.2 ghcr.io/yourusername/pytorch-rocm-builder:7.2

# Push
docker push ghcr.io/yourusername/pytorch-rocm-builder:7.2
```

## Usage in SLURM

Convert Docker image to Singularity/Apptainer:

```bash
# On a node with docker2singularity
docker run -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/output \
  quay.io/singularity/docker2singularity \
  pytorch-rocm-builder:7.2

# Or use apptainer directly
apptainer build pytorch-rocm-builder-7.2.sif docker://yourusername/pytorch-rocm-builder:7.2
```

Then in your SLURM script:
```bash
#SBATCH --container-image=/path/to/pytorch-rocm-builder-7.2.sif
```

## Troubleshooting

### Build fails: "Package X not found"

ROCm package names may vary by version. Check available packages:
```bash
docker run --rm ubuntu:22.04 bash -c "
  wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg && \
  echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 jammy main' > /etc/apt/sources.list.d/rocm.list && \
  apt-get update && \
  apt-cache search rocm | grep -E 'hip|rocblas|miopen'
"
```

### PyTorch build fails: "No GPU arch specified"

Make sure to export `PYTORCH_ROCM_ARCH` before building:
```bash
export PYTORCH_ROCM_ARCH=gfx942
```

### Runtime: "Could not load dynamic library 'libamdhip64.so'"

Mount ROCm device files:
```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video ...
```

## Differences from Original ubuntu-rocm/Dockerfile

| Feature | Original | Minimal |
|---------|----------|---------|
| Python env | Conda | uv + venv |
| ROCm install | Full `-complete` or large meta-packages | Selective packages only |
| Compilers | gcc + clang + LLVM | gcc only (optional version) |
| Size | ~50GB | ~15-18GB |
| Layers | 50+ | ~20 |
| User | jenkins | root (override with `docker run -u`) |
| Docs tools | Included | Not included |

## Contributing

When updating ROCm version or adding dependencies:

1. Check cmake/public/LoadHIP.cmake for required packages
2. Test build with `pip install -e . --no-build-isolation -v`
3. Verify tests run with `pytest test/`
4. Update this README with any changes

## License

Follows PyTorch repository license (BSD-3-Clause).
