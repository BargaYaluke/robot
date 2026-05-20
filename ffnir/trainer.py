from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import ImageFittingDataset
from .metrics import compute_psnr, compute_ssim
from .models import build_model
from .utils import (
    ensure_dir,
    save_image_tensor,
    save_json,
    save_metrics_csv,
    save_psnr_curve,
    set_seed,
)


@dataclass
class FitResult:
    final_psnr: float
    final_ssim: float
    reconstruction_path: str
    curve_path: str
    metrics_path: str


class ImageFitter:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        set_seed(config.seed)

        self.output_dir = ensure_dir(config.output_dir)
        save_json(config.to_dict(), str(self.output_dir / "config.json"))

        self.device = self._resolve_device(config.device)
        self.dataset = ImageFittingDataset(
            image_path=config.image_path,
            coordinate_range=config.coordinate_range,
            image_height=config.image_height,
            image_width=config.image_width,
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.seed)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=loader_generator,
        )

        self.model = build_model(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self) -> FitResult:
        data_iter = iter(self.loader)
        progress = tqdm(
            range(1, self.config.num_steps + 1),
            desc="Fitting image",
            dynamic_ncols=True,
        )

        last_loss = float("nan")
        for step in progress:
            try:
                coords, rgb = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                coords, rgb = next(data_iter)

            loss_value = self._train_step(coords, rgb)
            last_loss = loss_value

            if step % self.config.log_every == 0:
                progress.set_postfix(loss=f"{loss_value:.6f}")

            should_evaluate = step == 1 or step % self.config.eval_every == 0
            if should_evaluate:
                metrics = self.evaluate()
                metrics["step"] = float(step)
                metrics["loss"] = loss_value
                self.history.append(metrics)
                self._save_metric_artifacts()
                progress.set_postfix(
                    loss=f"{loss_value:.6f}",
                    psnr=f"{metrics['psnr']:.2f}",
                    ssim=f"{metrics['ssim']:.4f}",
                )

            should_save_image = self.config.save_every > 0 and step % self.config.save_every == 0
            if should_save_image:
                reconstruction = self.reconstruct()
                save_image_tensor(
                    reconstruction,
                    str(self.output_dir / f"reconstruction_step_{step:06d}.png"),
                )

        final_metrics = self.evaluate()
        final_metrics["step"] = float(self.config.num_steps)
        final_metrics["loss"] = last_loss
        self.history.append(final_metrics)

        final_reconstruction = self.reconstruct()
        reconstruction_path = self.output_dir / "reconstruction_final.png"
        save_image_tensor(final_reconstruction, str(reconstruction_path))

        self._save_metric_artifacts()

        return FitResult(
            final_psnr=final_metrics["psnr"],
            final_ssim=final_metrics["ssim"],
            reconstruction_path=str(reconstruction_path),
            curve_path=str(self.output_dir / "psnr_curve.png"),
            metrics_path=str(self.output_dir / "metrics.csv"),
        )

    def _train_step(self, coords: torch.Tensor, rgb: torch.Tensor) -> float:
        coords = coords.to(self.device, non_blocking=True)
        rgb = rgb.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.config.amp and self.device.type == "cuda"):
            pred = self.model(coords)
            loss = F.mse_loss(pred, rgb)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def reconstruct(self) -> torch.Tensor:
        self.model.eval()
        coords = self.dataset.full_coords()
        predictions = []

        for start in range(0, coords.shape[0], self.config.reconstruction_chunk_size):
            end = start + self.config.reconstruction_chunk_size
            chunk = coords[start:end].to(self.device)
            pred = self.model(chunk).detach().cpu()
            predictions.append(pred)

        image = torch.cat(predictions, dim=0)
        image = image.reshape(self.dataset.height, self.dataset.width, 3)
        self.model.train()
        return image.clamp(0.0, 1.0)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        reconstruction = self.reconstruct()
        target = self.dataset.full_image()
        psnr = compute_psnr(reconstruction, target)
        ssim = compute_ssim(reconstruction, target)
        return {"psnr": psnr, "ssim": ssim}

    def _save_metric_artifacts(self) -> None:
        metrics_path = self.output_dir / "metrics.csv"
        curve_path = self.output_dir / "psnr_curve.png"
        save_metrics_csv(self.history, str(metrics_path))
        save_psnr_curve(self.history, str(curve_path))
