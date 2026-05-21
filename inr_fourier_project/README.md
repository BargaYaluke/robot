# INR Fourier Project

PyTorch implementation of implicit neural representation (INR) for single-image
fitting. An INR represents an image as a function:

```text
f_theta(x, y) -> (r, g, b)
```

The model takes normalized 2D pixel coordinates as input and predicts RGB
values.

The full method couples two strategies that share a single progress window:

- **Frequency-Band Loss Reweighting**: keep the original target intact and
  re-weight the per-pixel MSE according to the target's local frequency
  content. A Laplacian-style pyramid is computed once, and time-varying
  per-band scalars shift loss emphasis from low to high frequency.
- **Ramped Edge-Aware Sampling**: enable edge bias from step 0 with a smooth
  ramp instead of a hard step-flip.

`--coupled_schedule` forces both modules to use the same time window so that
"what to emphasize in loss" and "where to sample" move together.

## Installation

```bash
conda create -n inr_fourier python=3.10
conda activate inr_fourier
pip install -r requirements.txt
```

## Data Preparation

Put input images under `data/images/`. Images are loaded as RGB, resized to a
square resolution, normalized to `[0, 1]`, and converted into coordinate-RGB
training pairs.

## Training (Full Method)

```bash
python train.py \
  --image_path data/images/test.png \
  --model_type fourier_mlp \
  --image_size 256 \
  --num_steps 10000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --use_band_loss \
  --band_sigmas 8.0,4.0,2.0 \
  --band_w_low_start 1.0 --band_w_low_end 0.3 \
  --band_w_high_start 0.3 --band_w_high_end 1.5 \
  --use_edge_sampling \
  --edge_ratio 0.5 \
  --use_ramped_edge \
  --edge_ramp_start_ratio 0.0 \
  --edge_ramp_end_ratio 1.0 \
  --edge_ramp_mode linear \
  --coupled_schedule \
  --output_dir results/coupled_test
```

`--coupled_schedule` requires `--use_band_loss` and `--use_ramped_edge`. When
it is on, `--band_loss_start_ratio`, `--band_loss_end_ratio`, and
`--band_loss_mode` are overridden by the edge-ramp values.

## Standard Experiments

`run_experiments.py` runs the standard ablation grid (baseline, legacy blur
curriculum, legacy step-flip edge, legacy full method, band loss only, ramped
edge only, coupled) over one or more seeds.

```bash
python run_experiments.py \
  --image_path data/images/test.png \
  --num_steps 10000 \
  --seeds 42,43,44 \
  --output_root results/v2_test
```

With more than one seed, each method appears under
`<output_root>/seed_<seed>/<method>/`. With a single seed, the flat layout
`<output_root>/<method>/` is used.

Summarize and plot:

```bash
python summarize_results.py --experiment_root results/v2_test/seed_42
python plot_convergence_comparison.py --experiment_root results/v2_test/seed_42
```

Both helpers skip methods that are missing from the experiment directory.

## Evaluation

```bash
python eval.py \
  --image_path data/images/test.png \
  --checkpoint_path results/coupled_test/model_checkpoint.pt \
  --model_type fourier_mlp \
  --image_size 256 \
  --output_dir results/coupled_test/eval
```

## Outputs

```text
results/<run_name>/
+-- model_checkpoint.pt
+-- metrics.csv
+-- metrics.json
+-- summary.json
+-- reconstructions/
+-- curves/
+-- logs/
`-- visualizations/
```

Metric records contain: `step`, `train_loss`, `psnr`, `ssim`, `edge_psnr`,
`smooth_psnr`, `current_curriculum_stage`, `sampling_mode`,
`elapsed_time_seconds`, `effective_edge_ratio`, `schedule_progress`. The main
reported metrics are full-image PSNR/SSIM and Sobel-mask-based edge PSNR and
smooth PSNR.

## Project Structure

```text
inr_fourier_project/
+-- train.py
+-- eval.py
+-- run_experiments.py
+-- summarize_results.py
+-- plot_convergence_comparison.py
+-- README.md
+-- configs/
+-- models/
+-- data/
+-- methods/
|   +-- frequency_curriculum.py     (legacy blur-target curriculum)
|   +-- edge_sampler.py             (Sobel edge map + mixed sampler)
|   +-- frequency_band_loss.py      (band-weighted MSE)
|   `-- scheduler.py                (shared progress/ramp helpers)
+-- utils/
`-- results/
```
