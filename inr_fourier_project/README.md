# INR Fourier Project

This project is a clean PyTorch implementation of implicit neural
<<<<<<< HEAD
representation (INR) for single-image fitting. It supports both the first-stage
baseline pipeline and second-stage research strategies for convergence-speed
comparison.
=======
representation (INR) for single-image fitting.
>>>>>>> f610dac054b21fcc513794ac6426b207636e7b32

An INR image model represents an image as a function:

```text
f_theta(x, y) -> (r, g, b)
```

The model takes normalized 2D pixel coordinates as input and predicts RGB
values. A vanilla coordinate MLP is included as a weak baseline, while a
Fourier Feature Network is used as the main base model for higher-frequency
image reconstruction.

<<<<<<< HEAD
The second-stage code adds:

- Frequency Curriculum Learning: train first on blurred low-frequency targets,
  then gradually transition to the original image.
- Edge-Aware Sampling: after a chosen training point, sample more coordinates
  from Sobel edge and texture regions.
- Region-aware evaluation: report edge-region PSNR and smooth-region PSNR.
- Structured experiment logging for convergence and limited-budget comparison.

=======
>>>>>>> f610dac054b21fcc513794ac6426b207636e7b32
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

### Frequency Curriculum Learning

Frequency curriculum uses blurred versions of the original target image as
early supervision, then ends on the original image. Evaluation is always done
against the original image.

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type fourier_mlp \
  --image_size 256 \
  --num_steps 2000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --use_frequency_curriculum \
  --curriculum_sigmas 4.0,2.0,1.0,0.0 \
  --output_dir results/frequency_curriculum_test
```

Optional blended curriculum:

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type fourier_mlp \
  --use_frequency_curriculum \
  --use_blended_curriculum \
  --curriculum_blend_ratio 0.1 \
  --output_dir results/blended_curriculum_test
```

### Edge-Aware Sampling

Edge-aware sampling computes a Sobel edge map from the original image. Before
`edge_start_ratio * num_steps`, sampling remains uniform. After that point,
mixed sampling draws part of the batch uniformly and part from the edge-aware
probability distribution.

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type fourier_mlp \
  --image_size 256 \
  --num_steps 2000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --use_edge_sampling \
  --edge_start_ratio 0.5 \
  --edge_alpha 0.2 \
  --edge_beta 1.0 \
  --edge_ratio 0.5 \
  --edge_threshold 0.2 \
  --output_dir results/edge_sampling_test
```

### Full Method

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type fourier_mlp \
  --image_size 256 \
  --num_steps 2000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --use_frequency_curriculum \
  --use_edge_sampling \
  --output_dir results/full_method_test
```

## Standard Experiments

Run the four standard single-image settings:

1. Baseline: Fourier MLP, uniform sampling, original target.
2. Frequency Curriculum: Fourier MLP, curriculum target, uniform sampling.
3. Edge Sampling: Fourier MLP, original target, edge-aware sampling.
4. Full Method: Fourier MLP, curriculum target, edge-aware sampling.

```bash
python run_experiments.py \
  --image_path data/images/test.png \
  --image_size 256 \
  --num_steps 2000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --output_root results/standard_test \
  --seed 42
```

Summarize the four `summary.json` files into a CSV table and a markdown table:

```bash
python summarize_results.py \
  --experiment_root results/standard_test
```

Plot convergence curves:

```bash
python plot_convergence_comparison.py \
  --experiment_root results/standard_test
```

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
+-- metrics.csv
+-- metrics.json
+-- summary.json
+-- reconstructions/
|   +-- reconstruction_step_XXXXXX.png
|   +-- final_reconstruction.png
|   `-- final_comparison.png
+-- curves/
|   `-- psnr_curve.png
+-- logs/
`-- visualizations/
    +-- edge_map.png
    +-- sampling_probability.png
    `-- sampling_points_step_XXXXXX.png
```

`visualizations/` is populated when `--use_edge_sampling` is enabled.

Structured metric records are saved in both CSV and JSON format. Each
evaluation record contains:

- `step`
- `train_loss`
- `psnr`
- `ssim`
- `edge_psnr`
- `smooth_psnr`
- `current_curriculum_stage`
- `sampling_mode`
- `elapsed_time_seconds`

`summary.json` contains final metrics, best PSNR, iterations to target PSNR
values, total training time, and important hyperparameters. It is intended for
later table generation.

Evaluation outputs:

```text
results/<run_name>/eval/
+-- eval_reconstruction.png
+-- eval_comparison.png
`-- eval_metrics.txt
```

The main reported metrics are:

- PSNR: peak signal-to-noise ratio over the full image.
- SSIM: structural similarity index over the full image.
- Edge PSNR: PSNR computed only on Sobel edge-mask pixels.
- Smooth PSNR: PSNR computed only on non-edge pixels.

Convergence metrics include:

- `iter_to_30db`
- `iter_to_35db`
- `iter_to_38db`
- `iter_to_40db`

## Project Structure

```text
inr_fourier_project/
+-- train.py
+-- eval.py
+-- run_experiments.py
+-- summarize_results.py
+-- plot_convergence_comparison.py
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
+-- methods/
|   +-- frequency_curriculum.py
|   `-- edge_sampler.py
+-- utils/
|   +-- convergence.py
|   +-- metrics.py
|   +-- visualization.py
|   `-- seed.py
`-- results/
    +-- reconstructions/
    +-- curves/
    +-- logs/
    `-- visualizations/
```

## Research Notes

The baseline behavior remains unchanged when both strategy flags are disabled:
training uses the original RGB target and uniform random coordinate sampling.

During frequency curriculum training, only the training supervision target
changes. Full-image evaluation always compares the reconstruction against the
original image.

During edge-aware sampling, the Sobel edge map and edge/smooth evaluation masks
are computed from the original image, not from blurred curriculum targets.
