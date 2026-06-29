"""Regenerate cached masks + VAE latents for an already-precomputed person, WITHOUT
re-running the four detectors.

The cloth-agnostic masks are a pure function of the cached parse maps
(``densepose.npy`` / ``lip.npy`` / ``atr.npy``), so after editing the mask logic in
``parse_utils/automasker.py`` we can rebuild the affected masks (and the VAE latents
that depend on them) straight from the cache. This needs only the VAE (the ``vae/``
subfolder of the base model) — not the Human-Toolkit detectors.

It mirrors the mask + latent generation in ``precompute_person_preprocessing.py``
(``process_one`` / ``_save_all_latents``), restricted to the requested categories.

Example::

    # Rebuild just the whole-outfit (multiref) mask + latents for one person:
    python regen_person_masks.py --stem Karishma_VTON_pic --categories multiref
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

from module.utils import prepare_image, prepare_mask_image
from parse_utils.automasker import cloth_agnostic_mask, multi_ref_cloth_agnostic_mask

MULTIREF = "multiref"
SINGLE_REGION = {"upper", "lower", "overall", "inner", "outer"}


def build_mask(category: str, densepose, lip, atr, square: bool) -> Image.Image:
    """Reproduce the precompute mask call for one (category, square)."""
    if category == MULTIREF:
        return multi_ref_cloth_agnostic_mask(
            densepose, lip, atr, square_cloth_mask=square, horizon_expand=True
        )
    return cloth_agnostic_mask(
        densepose, lip, atr, part=category, square_cloth_mask=square
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate cached masks + latents from cached parse maps (no detectors)."
    )
    parser.add_argument("--stem", default="Karishma_VTON_pic",
                        help="Precomputed person folder under --precompute-dir.")
    parser.add_argument("--precompute-dir", default="precomputed_person_tensors")
    parser.add_argument("--base-model-path", default="Models/FastFit-MR-1024",
                        help="Base model dir; only its vae/ subfolder is used.")
    parser.add_argument("--categories", nargs="+", default=[MULTIREF],
                        help="Mask categories to rebuild (default: multiref).")
    parser.add_argument("--mixed-precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the *_square / horizon_expand randomness (reproducibility).")
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                    "fp32": torch.float32}[args.mixed_precision]

    base = Path(args.precompute_dir) / args.stem
    masks_out = Path(args.precompute_dir) / "masks" / args.stem
    if not base.exists():
        raise SystemExit(f"No precompute folder: {base}")

    densepose = np.load(base / "densepose.npy")
    lip = np.load(base / "lip.npy")
    atr = np.load(base / "atr.npy")
    person_img = Image.open(base / "person.png").convert("RGB")
    pose_img = Image.open(base / "pose.png").convert("RGB")

    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae")
    vae.to(device, dtype=weight_dtype).eval()

    # Person/pose tensors (shared across all categories), mirroring _save_all_latents.
    with torch.no_grad():
        person_t = prepare_image(person_img, device, weight_dtype)
        pose_t = prepare_image(pose_img, device, weight_dtype, do_normalize=False)
        if pose_t.shape[-2:] != (person_img.size[1], person_img.size[0]):
            pose_t = torch.nn.functional.interpolate(
                pose_t.unsqueeze(0),
                size=(person_img.size[1], person_img.size[0]),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        for category in args.categories:
            for square in (False, True):
                sq_png = "_square" if square else ""
                sq = "square" if square else "nosquare"

                mask_img = build_mask(category, densepose, lip, atr, square)
                mask_path = masks_out / f"{category}{sq_png}.png"
                mask_img.save(mask_path)

                mask_t = prepare_mask_image(mask_img, device, weight_dtype)
                mask_latent_saved = False
                for pose_on in (True, False):
                    po = "pose" if pose_on else "nopose"
                    if pose_on:
                        masked_person = person_t * (1 - mask_t) + mask_t * pose_t
                    else:
                        masked_person = person_t * (1 - mask_t)
                    masked_person_latent = (
                        vae.encode(masked_person).latent_dist.mode()
                        * vae.config.scaling_factor
                    )
                    torch.save(
                        masked_person_latent.to("cpu", torch.bfloat16),
                        base / f"masked_person_latent_{category}_{sq}_{po}.pt",
                    )
                    if not mask_latent_saved:
                        mask_latent = torch.nn.functional.interpolate(
                            mask_t.to(dtype=torch.float32),
                            size=masked_person_latent.shape[-2:],
                            mode="nearest",
                        ).to(weight_dtype)
                        torch.save(
                            mask_latent.to("cpu", torch.bfloat16),
                            base / f"mask_latent_{category}_{sq}.pt",
                        )
                        mask_latent_saved = True
                print(f"[OK] {category} {sq}: {mask_path} + latents")

    print("Done.")


if __name__ == "__main__":
    main()
