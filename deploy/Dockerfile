# DepthWizard, served on CPU.
#
# There is no GPU in this image and that is the point: measured with the GPU hidden, a
# 1024x1024 tile costs 5.0 s single-pass and 57.2 s with x8 D4 TTA plus zoom-2 fusion, so
# the product runs on an ordinary CPU container and the whole deployment is a few dollars
# a month rather than a GPU VM.
#
# torch comes from the CPU wheel index deliberately. The default PyPI wheel bundles CUDA
# and is ~2.5 GB of libraries this image can never use; the CPU build is a fraction of it
# and pulls no nvidia-* packages at all.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    DW_WORK=/tmp/depthwizard \
    OMP_NUM_THREADS=2 \
    PORT=8080

WORKDIR /app

# Wheels only -- rasterio ships manylinux wheels with GDAL inside, so no apt GDAL stack
# and no compiler in the final image.
COPY requirements-serve.txt .
RUN pip install --only-binary=:all: \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-serve.txt

COPY app/ /app/

# Fail the build, not the first request, if the baked cache is incomplete. HF_HUB_OFFLINE
# is already on, so this reaches the network never: it either finds the snapshot or dies.
RUN python -c "\
import torch, json, pathlib; \
from depthwizard.model import from_checkpoint; \
ck = torch.load('weights/best.pt', map_location='cpu', weights_only=False); \
m = from_checkpoint(ck).eval(); \
n = sum(p.numel() for p in m.parameters()); \
print(f'model built offline: {n/1e6:.1f}M params, epoch {ck.get(\"epoch\")}')"

EXPOSE 8080
CMD ["python", "tools/serve_app.py"]
