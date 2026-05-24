"""Precompute and cache the person-side preprocessing for FastFit.

The FastFit mask is cloth-agnostic: it is a pure function of
(person image, mask_part, square_mask) and never depends on garment pixels.
The four detector passes (DWpose, DensePose, SCHP-LIP, SCHP-ATR) are the
expensive GPU work and run on the person image alone. So for a fixed person we
can run the detectors once and precompute every mask variant, then reuse them
across any garment combination at inference time.

This script populates that cache. Given a person image (or a directory of
them) it runs the four detectors and writes, per person:

    precomputed_person_tensors/
      <stem>/
        person.png       # processed person canvas (PERSON_SIZE)
        pose.png         # DWpose RGB visualization
        densepose.npy    # uint8 HxW parse index map
        lip.npy
        atr.npy
      masks/
        <stem>/
          multiref.png  multiref_square.png
          upper.png     upper_square.png
          lower.png     lower_square.png
          overall.png   overall_square.png
          inner.png     inner_square.png
          outer.png     outer_square.png

It deliberately instantiates only the four detectors (mirroring
FastFitDemo.__init__) and never loads the heavy FastFitPipeline.
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from app import PERSON_SIZE, center_crop_to_aspect_ratio
from parse_utils import (
    DWposeDetector,
    DensePose,
    SCHP,
    cloth_agnostic_mask,
    multi_ref_cloth_agnostic_mask,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Single-region parts handled by cloth_agnostic_mask. The default whole-outfit
# mask (mask_part=None at inference) is handled separately by
# multi_ref_cloth_agnostic_mask.
MASK_PARTS = ["upper", "lower", "overall", "inner", "outer"]


class PersonPreprocessor:
    """Loads only the four detectors used to build cloth-agnostic masks."""

    def __init__(self, util_model_path: str = "Models/Human-Toolkit", device: str = None):
        # Auto-download the utility models if missing (same as FastFitDemo).
        if not os.path.exists(util_model_path):
            os.makedirs(util_model_path, exist_ok=True)
            snapshot_download(
                repo_id="zhengchong/Human-Toolkit",
                local_dir=util_model_path,
                local_dir_use_symlinks=False,
            )

        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        # Mirrors app.py:209-213 — but no FastFitPipeline.
        self.dwpose_detector = DWposeDetector(
            pretrained_model_name_or_path=os.path.join(util_model_path, "DWPose"), device="cpu"
        )
        self.densepose_detector = DensePose(
            model_path=os.path.join(util_model_path, "DensePose"), device=self.device
        )
        self.schp_lip_detector = SCHP(
            ckpt_path=os.path.join(util_model_path, "SCHP", "schp-lip.pth"), device=self.device
        )
        self.schp_atr_detector = SCHP(
            ckpt_path=os.path.join(util_model_path, "SCHP", "schp-atr.pth"), device=self.device
        )

    def process_one(self, image_path: Path, output_dir: Path) -> None:
        """Run detectors + generate all mask variants for a single image."""
        stem = image_path.stem
        person_out = output_dir / stem
        masks_out = output_dir / "masks" / stem
        person_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        # Same person transform as FastFitDemo.preprocess_person_image (app.py:242-244).
        img = Image.open(image_path).convert("RGB")
        img = center_crop_to_aspect_ratio(img, 3 / 4)
        img = img.resize(PERSON_SIZE, Image.LANCZOS)
        img.save(person_out / "person.png")

        # The four detector passes — the expensive GPU work, person-only.
        pose_img = self.dwpose_detector(img)
        if not isinstance(pose_img, Image.Image):
            raise RuntimeError("Pose estimation failed")
        pose_img.save(person_out / "pose.png")

        densepose_arr = np.array(self.densepose_detector(img))
        lip_arr = np.array(self.schp_lip_detector(img))
        atr_arr = np.array(self.schp_atr_detector(img))
        np.save(person_out / "densepose.npy", densepose_arr)
        np.save(person_out / "lip.npy", lip_arr)
        np.save(person_out / "atr.npy", atr_arr)

        # 12 mask variants. NOTE: the *_square variants (and multi-ref
        # horizon_expand) involve randomness, so they are one fixed realization
        # rather than bit-reproducible. The 6 non-square variants are deterministic.
        for square in (False, True):
            suffix = "_square" if square else ""
            # Default whole-outfit mask (mask_part=None path, app.py:267-271).
            multiref = multi_ref_cloth_agnostic_mask(
                densepose_arr, lip_arr, atr_arr,
                square_cloth_mask=square, horizon_expand=True,
            )
            multiref.save(masks_out / f"multiref{suffix}.png")

            # Single-region masks (--mask-part path, app.py:262-266).
            for part in MASK_PARTS:
                mask = cloth_agnostic_mask(
                    densepose_arr, lip_arr, atr_arr,
                    part=part, square_cloth_mask=square,
                )
                mask.save(masks_out / f"{part}{suffix}.png")


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
            and not any(part.startswith(".") for part in p.relative_to(input_path).parts)
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute FastFit person preprocessing (detector outputs + mask variants)."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a person image or a directory of person images.")
    parser.add_argument("--output-dir", default="precomputed_person_tensors",
                        help="Directory to write cached tensors and masks.")
    parser.add_argument("--util-model-path", default="Models/Human-Toolkit",
                        help="Local path for the Human-Toolkit utility models (auto-downloaded if missing).")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu). Defaults to auto.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess images whose output folder already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    images = collect_images(input_path)
    if not images:
        print(f"[FAILED] No images found at {input_path}")
        raise SystemExit(1)

    preprocessor = PersonPreprocessor(util_model_path=args.util_model_path, device=args.device)

    n_ok = n_skip = n_fail = 0
    for image_path in images:
        person_out = output_dir / image_path.stem
        if person_out.exists() and not args.overwrite:
            print(f"[SKIP] {image_path} (already cached; use --overwrite to redo)")
            n_skip += 1
            continue
        try:
            preprocessor.process_one(image_path, output_dir)
            print(f"[OK]   {image_path} -> {person_out}")
            n_ok += 1
        except Exception as e:
            print(f"[FAIL] {image_path}: {e}")
            n_fail += 1

    print(f"\nDone. {n_ok} processed, {n_skip} skipped, {n_fail} failed.")
    if n_fail and not n_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
