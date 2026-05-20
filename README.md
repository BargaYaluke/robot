# Fourier Feature Neural Image Representation

Clean PyTorch prototype for fitting a single RGB image with coordinate-based MLPs.

The codebase includes:

- Vanilla coordinate MLP.
- Random Gaussian Fourier Feature MLP.
- Single-image fitting pipeline.
- PSNR and SSIM evaluation.
- Reconstruction image saving.
- PSNR convergence curve saving.
- Configurable training parameters through command-line arguments.

Frequency curriculum learning and edge-aware sampling are intentionally not implemented yet. The current structure leaves clear extension points for them.

## Project Structure

```text
.
├── train_image.py
├── requirements.txt
└── ffnir
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── encoding.py
    ├── metrics.py
    ├── models.py
    ├── trainer.py
    └── utils.py
```

## Example Usage

Fourier Feature MLP:

```bash
python train_image.py \
  --image-path data/example.png \
  --output-dir runs/example_fourier \
  --model fourier \
  --num-steps 5000 \
  --batch-size 8192 \
  --lr 1e-4 \
  --hidden-dim 256 \
  --num-layers 4 \
  --mapping-size 256 \
  --fourier-scale 10.0
```

Vanilla coordinate MLP:

```bash
python train_image.py \
  --image-path data/example.png \
  --output-dir runs/example_vanilla \
  --model vanilla \
  --num-steps 5000 \
  --batch-size 8192
```

Optional image resizing:

```bash
python train_image.py \
  --image-path data/example.png \
  --output-dir runs/example_256 \
  --image-height 256 \
  --image-width 256
```

## Outputs

Each run writes:

- `config.json`: resolved training configuration.
- `metrics.csv`: PSNR, SSIM, and loss records.
- `psnr_curve.png`: PSNR convergence curve.
- `reconstruction_final.png`: final reconstructed image.
- `reconstruction_step_XXXXXX.png`: optional intermediate reconstructions when `--save-every` is set.

## Notes

- Pixel coordinates are normalized to `[-1, 1]` by default.
- Model outputs are passed through a sigmoid by default, so predicted RGB values are in `[0, 1]`.
- SSIM is computed with `skimage.metrics.structural_similarity` using `channel_axis=-1`.
- This repository does not include dependency installation commands; install the listed dependencies on your server as appropriate.
