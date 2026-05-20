# INR Fourier Project

This project is a clean PyTorch implementation of implicit neural
representation (INR) for single-image fitting.

An INR image model represents an image as a function:

```text
f_theta(x, y) -> (r, g, b)
```

The model takes normalized 2D pixel coordinates as input and predicts RGB
values. A vanilla coordinate MLP is included as a weak baseline, while a
Fourier Feature Network is used as the main base model for higher-frequency
image reconstruction.

## Installation

Create and activate an environment:

```bash
conda create -n inr_fourier python=3.10
conda activate inr_fourier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Put input images under:

```text
data/images/
```

Example:

```text
data/images/test.png
```

Images are loaded as RGB, resized to a square resolution, normalized to
`[0, 1]`, and converted into coordinate-RGB training pairs.

## Training

### Train Vanilla MLP

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type vanilla_mlp \
  --image_size 256 \
  --num_steps 5000 \
  --batch_size 8192 \
  --lr 1e-4 \
  --output_dir results/vanilla_test
```

### Train Fourier MLP

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type fourier_mlp \
  --image_size 256 \
  --num_steps 5000 \
  --batch_size 8192 \
  --lr 1e-4 \
  --mapping_size 256 \
  --scale 10.0 \
  --output_dir results/fourier_test
```

Useful training arguments:

- `--hidden_dim`: hidden layer width.
- `--num_layers`: number of hidden layers.
- `--eval_interval`: full-image PSNR/SSIM evaluation interval.
- `--save_interval`: intermediate reconstruction saving interval.
- `--seed`: random seed for reproducibility.

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval.py \
  --image_path data/images/test.png \
  --checkpoint_path results/fourier_test/model_checkpoint.pt \
  --model_type fourier_mlp \
  --image_size 256 \
  --output_dir results/fourier_test/eval
```

The evaluation script rebuilds the model architecture from the checkpoint
training configuration when available, so it matches the model used during
training.

## Expected Outputs

Training outputs:

```text
results/<run_name>/
+-- model_checkpoint.pt
+-- reconstructions/
|   +-- reconstruction_step_XXXXXX.png
|   +-- final_reconstruction.png
|   `-- final_comparison.png
+-- curves/
|   `-- psnr_curve.png
`-- logs/
    `-- train_log.csv
```

Evaluation outputs:

```text
results/<run_name>/eval/
+-- eval_reconstruction.png
+-- eval_comparison.png
`-- eval_metrics.txt
```

The main reported metrics are:

- PSNR: peak signal-to-noise ratio.
- SSIM: structural similarity index.

## Project Structure

```text
inr_fourier_project/
+-- train.py
+-- eval.py
+-- requirements.txt
+-- README.md
+-- configs/
|   +-- vanilla_mlp.yaml
|   `-- fourier_mlp.yaml
+-- models/
|   +-- vanilla_mlp.py
|   `-- fourier_mlp.py
+-- data/
|   +-- image_dataset.py
|   `-- images/
+-- utils/
|   +-- metrics.py
|   +-- visualization.py
|   `-- seed.py
`-- results/
    +-- reconstructions/
    +-- curves/
    `-- logs/
```

## Future Extensions

Planned research extensions:

- Frequency curriculum learning.
- Edge-aware sampling.

These modules are intentionally not implemented yet, keeping the current
baseline simple and easy to compare.
