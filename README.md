# `just-image-transformer`

[![CI](https://github.com/jkyl/just-image-transformer/actions/workflows/ci.yaml/badge.svg)](https://github.com/jkyl/just-image-transformer/actions/workflows/ci.yaml)
[![Weights](https://img.shields.io/badge/weights-google%20drive-blue?logo=google-drive)](https://drive.google.com/file/d/138Vvu9MM96Sm9KaxVV9bKkMVQ3VmzCmm/view?usp=sharing)

Unofficial JAX implementation of ["Back to Basics: Let Denoising Generative Models Denoise"](https://arxiv.org/abs/2511.13720) a.k.a. "Just Image Transformers" (JiT), by Tianhong Li and Kaiming He.

> [!NOTE]
> Nvidia GPUs >= Ampere are supported by default. I developed this code on a PC with a 3090 in WSL2. TPU support can be enabled by setting `implementation="xla"` in the default attention function in `model.py`. 

## Quick start

```bash
# Install dependencies
uv sync

# Train JiT-L/32 on ImageNet
./scripts/train.sh --notes="my first run"

# Generate a single image
./scripts/inference.sh model.npz output.png

# Start the inference server
./scripts/server.sh model.npz

# Lint and type check
make
```

## Inference server

`./scripts/server.sh` starts a Flask app with a simple web UI for generating images. You can adjust the seed, class label, CFG strength, number of steps, and schedule. Progress streams in real-time.


https://github.com/user-attachments/assets/4d2b9fa0-266e-43bc-8f52-34c2e9b9de51


## References

```bibtex
@article{li2025backtobasics,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```
