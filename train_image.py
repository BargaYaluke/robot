from ffnir.config import parse_args
from ffnir.trainer import ImageFitter


def main() -> None:
    config = parse_args()
    fitter = ImageFitter(config)
    result = fitter.fit()

    print(f"Finished fitting image: {config.image_path}")
    print(f"Final PSNR: {result.final_psnr:.4f} dB")
    print(f"Final SSIM: {result.final_ssim:.6f}")
    print(f"Outputs saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
