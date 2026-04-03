# Quick Start Guide

## Build the Minimal Image

```bash
cd .ci/docker/ubuntu-rocm-minimal

# Build just the builder stage (minimal, ~15-18GB)
ROCM_VERSION=7.2 \
PYTORCH_ROCM_ARCH="gfx942;gfx950" \
./build.sh
```

## Test the Build

```bash
# Check image size
docker images | grep pytorch-rocm-builder

# Start container and build PyTorch
docker run -it --rm \
  -v $(pwd)/../../..:/workspace \
  -w /workspace \
  pytorch-rocm-builder:7.2 \
  bash

# Inside container:
export PYTORCH_ROCM_ARCH=gfx942
pip install -e . --no-build-isolation -v
```

## Common Build Options

### Minimal (no optional deps)
```bash
./build.sh
```

### With Triton
```bash
TRITON=1 ./build.sh
```

### With GCC 11
```bash
GCC_VERSION=11 ./build.sh
```

### Test image (includes pytest, etc.)
```bash
TARGET=tester ./build.sh
```

### Benchmark image (includes timm, transformers, etc.)
```bash
TARGET=benchmarks INDUCTOR_BENCHMARKS=1 ./build.sh
```

### Everything
```bash
ROCM_VERSION=7.2 \
PYTORCH_ROCM_ARCH="gfx942;gfx950" \
GCC_VERSION=11 \
TRITON=1 \
INDUCTOR_BENCHMARKS=1 \
TARGET=benchmarks \
./build.sh
```

## Push to Docker Hub (Private)

```bash
# Login
docker login

# Tag
docker tag pytorch-rocm-builder:7.2 YOUR_USERNAME/pytorch-rocm-builder:7.2

# Push (private by default)
docker push YOUR_USERNAME/pytorch-rocm-builder:7.2
```

## Convert to Singularity for SLURM

```bash
# Pull and convert
apptainer build pytorch-rocm-builder-7.2.sif \
  docker://YOUR_USERNAME/pytorch-rocm-builder:7.2

# Or from local Docker
docker save pytorch-rocm-builder:7.2 | \
  apptainer build pytorch-rocm-builder-7.2.sif docker-archive:/dev/stdin
```

## Expected Size Savings

- Original: ~50GB
- Minimal builder: ~15-18GB (65% reduction)
- Minimal tester: ~22GB (56% reduction)
- Minimal benchmarks: ~30GB (40% reduction)
