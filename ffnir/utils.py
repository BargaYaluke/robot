import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(data: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def save_image_tensor(image: torch.Tensor, path: str) -> None:
    array = image.detach().cpu().float().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_metrics_csv(history: Iterable[Dict[str, float]], path: str) -> None:
    rows = list(history)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_psnr_curve(history: Iterable[Dict[str, float]], path: str) -> None:
    rows = [row for row in history if "psnr" in row]
    if not rows:
        return

    steps = [row["step"] for row in rows]
    psnr_values = [row["psnr"] for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, psnr_values, linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
