#!/usr/bin/env bash
# One-time pod setup. Run from the repo root.
# Pod: one A100 80GB, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, 100 GB container disk.
# The venv and weights stay on the container disk, because /workspace is a slow network filesystem.
# vLLM installs a CUDA 13 build of torch, so the host's driver must support CUDA 13.
set -euo pipefail
uv venv /root/venv --python 3.12 --seed
uv pip install --python /root/venv/bin/python -r requirements.txt
/root/venv/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, '|', torch.cuda.get_device_name(0))"
