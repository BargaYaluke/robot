import argparse
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class TrainConfig:
    image_path: str
    output_dir: str = "runs/image_fit"

    model: str = "fourier"
    device: str = "auto"
    seed: int = 42

    num_steps: int = 5000
    batch_size: int = 8192
    lr: float = 1e-4
    weight_decay: float = 0.0

    hidden_dim: int = 256
    num_layers: int = 4
    activation: str = "relu"
    output_activation: str = "sigmoid"

    mapping_size: int = 256
    fourier_scale: float = 10.0
    include_input: bool = False

    coordinate_range: str = "minus_one_one"
    image_height: Optional[int] = None
    image_width: Optional[int] = None

    eval_every: int = 100
    log_every: int = 50
    save_every: int = 0
    reconstruction_chunk_size: int = 65536

    num_workers: int = 0
    amp: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Fit a single image with a coordinate MLP or Fourier Feature MLP."
    )

    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/image_fit")

    parser.add_argument("--model", type=str, default="fourier", choices=["vanilla", "fourier"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "tanh", "leaky_relu"],
    )
    parser.add_argument(
        "--output-activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "none"],
    )

    parser.add_argument("--mapping-size", type=int, default=256)
    parser.add_argument("--fourier-scale", type=float, default=10.0)
    parser.add_argument("--include-input", action="store_true")

    parser.add_argument(
        "--coordinate-range",
        type=str,
        default="minus_one_one",
        choices=["minus_one_one", "zero_one"],
    )
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)

    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate reconstructions every N steps. Set 0 to disable.",
    )
    parser.add_argument("--reconstruction-chunk-size", type=int, default=65536)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    return TrainConfig(**vars(args))
