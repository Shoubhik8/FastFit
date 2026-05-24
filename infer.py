import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

from app import FastFitDemo


def load_optional(path: Optional[str]) -> Optional[Image.Image]:
    return Image.open(path) if path else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastFit CLI virtual try-on. Provide a person image and at least one "
                    "garment reference (upper/lower/dress/shoes/bag)."
    )
    parser.add_argument("--person", required=True, help="Path to the person image.")
    parser.add_argument("--upper", help="Path to a top/upper-body garment image.")
    parser.add_argument("--lower", help="Path to a bottom/lower-body garment image.")
    parser.add_argument("--dress", help="Path to a dress/overall image (cannot combine with --upper or --lower).")
    parser.add_argument("--shoes", help="Path to a shoes image.")
    parser.add_argument("--bag", help="Path to a bag image.")
    parser.add_argument("--output", required=True, help="Path to write the generated image (e.g. out/result.png).")

    parser.add_argument("--ref-height", type=int, default=512, choices=[512, 768, 1024],
                        help="Reference image height. Width is auto-derived (3:4).")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=2.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--square-mask", action="store_true", help="Use a square cloth-agnostic mask.")
    parser.add_argument("--mask-part", choices=["upper", "lower", "overall", "inner", "outer"], default=None,
                        help="Mask only this garment region instead of the whole outfit. Use 'upper' for a "
                             "t-shirt-only try-on so the rest of the person is preserved. Defaults to masking "
                             "the entire outfit (multi-garment).")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose guidance.")

    parser.add_argument("--base-model-path", default="Models/FastFit-MR-1024",
                        help="Local path for the FastFit base model (auto-downloaded if missing).")
    parser.add_argument("--util-model-path", default="Models/Human-Toolkit",
                        help="Local path for the Human-Toolkit utility models (auto-downloaded if missing).")
    parser.add_argument("--mixed-precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu). Defaults to auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    demo = FastFitDemo(
        base_model_path=args.base_model_path,
        util_model_path=args.util_model_path,
        mixed_precision=args.mixed_precision,
        device=args.device,
    )

    result_img, status_key = demo.generate_image(
        person_img=load_optional(args.person),
        upper_img=load_optional(args.upper),
        lower_img=load_optional(args.lower),
        dress_img=load_optional(args.dress),
        shoe_img=load_optional(args.shoes),
        bag_img=load_optional(args.bag),
        ref_height=args.ref_height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        use_square_mask=args.square_mask,
        seed=args.seed,
        enable_pose=not args.no_pose,
        mask_part=args.mask_part,
    )

    if result_img is None:
        print(f"[FAILED] {status_key}")
        raise SystemExit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"[OK] Saved try-on result to {output_path}")


if __name__ == "__main__":
    main()
