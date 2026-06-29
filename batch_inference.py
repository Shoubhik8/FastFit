"""Batch virtual try-on driven by recommendations + the precomputed person cache.

Dresses ONE precomputed person (default: ``sumant``) in garments from
``inventory/outputs/recommendations.json`` and writes one try-on image per combo.
``--section`` chooses what to dress them in:
  * ``complementary`` (default) — each upper+lower outfit pairing, using the
    whole-outfit ``multiref`` mask.
  * ``similar`` — each single garment, using that garment's own category's
    single-region mask (upper/lower/overall) so only that region is swapped.
The relevant per-category mask + VAE latents are read from the precompute cache by
``resolve_category_cache`` (the category is data-driven per combo).

It reuses the person-side cache produced by ``precompute_person_preprocessing.py``
(``precomputed_person_tensors/<stem>/``) so the four detector passes (DWpose,
DensePose, SCHP-LIP/ATR) and mask generation are skipped entirely. With
``--use-precomputed-latents`` it additionally feeds the precomputed masked-person
VAE latents into the pipeline, skipping the masked-person encode + pose blend
(requires the ``.pt`` files, which only exist after re-running precompute with
``--overwrite`` on a CUDA machine).

Examples::

    # Validate parsing / path-resolution / cache presence WITHOUT loading any model
    # (works on a machine with no GPU / no torch-CUDA, e.g. a Mac):
    python batch_inference.py --dry-run

    # Run the batch (mode A: precomputed person preprocessing, live masked-person encode):
    python batch_inference.py --person-stem sumant

    # Full optimization (mode B: also consume the precomputed VAE latents):
    python batch_inference.py --use-precomputed-latents

    # Single-garment try-ons from the ``similar`` section (per-item category mask):
    python batch_inference.py --section similar
    python batch_inference.py --section similar --category lower   # filter to lowers

Real inference is validated on CUDA — the pipeline's __init__ uses torch.compile /
fuse_qkv_projections, so mps/cpu are best-effort only.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

# Register an AVIF opener with Pillow (side effect of the import). Garment images
# in inventory/ are mostly .avif and stock Pillow cannot decode them. We only hard
# error when an actual .avif decode is attempted, so --dry-run still works without it.
try:
    import pillow_avif  # noqa: F401
    _AVIF_OK = True
except ImportError:
    _AVIF_OK = False

_AVIF_ERR_MSG = (
    "Cannot decode .avif garment images: pillow-avif-plugin is not installed.\n"
    "Install it with:  pip install pillow-avif-plugin"
)

# Fixed reference-slot order expected by FastFitPipeline (matches REF_LABEL_MAP in
# module/pipeline_fastfit.py). prepare_reference_images() populates whichever slots
# a given combo provides; the rest become black placeholders (attention_mask=0).
REF_CATEGORIES = ["upper", "lower", "overall", "shoe", "bag"]

# Categories a `similar` single-garment item can be tried on with: each must be BOTH
# a single-region precomputed mask (MASK_PARTS in precompute_person_preprocessing.py)
# AND a ref slot (REF_LABEL_MAP). `inner`/`outer` have masks but no ref slot;
# `shoe`/`bag` have ref slots but no mask — so neither is tryable here.
TRYON_CATEGORIES = {"upper", "lower", "overall"}

# recommendations.json category -> on-disk mask/ref category (app.py maps a dress to
# the "overall" mask + ref slot).
CATEGORY_ALIAS = {"dress": "overall"}

# Where garment images actually live. recommendations.json paths drop the
# user_owned_products/ subdir, so we resolve by basename against both folders.
INVENTORY_SUBDIRS = ["inventory/new_collection", "inventory/user_owned_products"]


def normalize_category(cat: str) -> str:
    """Map a recommendations.json category to its on-disk mask/ref slot name."""
    c = (cat or "").strip().lower()
    return CATEGORY_ALIAS.get(c, c)


def slug(text: str) -> str:
    """Filesystem-safe identifier."""
    text = re.sub(r"[^\w]+", "_", str(text).strip().lower())
    return text.strip("_") or "none"


def resolve_garment_path(raw: str) -> str:
    """Resolve a recommendations.json image path to an on-disk file.

    Tries new_collection/<basename>, then user_owned_products/<basename>, then the
    raw relative path. Returns the first existing candidate, else the raw path
    (callers check existence and report).
    """
    base = os.path.basename(raw)
    for d in INVENTORY_SUBDIRS:
        cand = os.path.join(d, base)
        if os.path.exists(cand):
            return cand
    if os.path.exists(raw):
        return raw
    return raw


def parse_complementary(rec_path: str, stem: str) -> list:
    """Parse the ``complementary`` section into per-combo dicts.

    Maps each garment to its slot by the ``categories`` array (whose order varies
    between entries), NOT by position.
    """
    with open(rec_path) as f:
        data = json.load(f)

    combos = []
    for i, entry in enumerate(data.get("complementary", [])):
        cats = entry.get("categories", [])
        paths = entry.get("image_paths", [])
        names = entry.get("names", [])

        slot_path, slot_name = {}, {}
        for j, cat in enumerate(cats):
            if cat in ("upper", "lower") and j < len(paths):
                slot_path[cat] = resolve_garment_path(paths[j])
                slot_name[cat] = names[j] if j < len(names) else Path(paths[j]).stem

        upper_path = slot_path.get("upper")
        lower_path = slot_path.get("lower")
        upper_id = Path(upper_path).stem if upper_path else "none"
        lower_id = Path(lower_path).stem if lower_path else "none"

        combos.append({
            "index": i,
            "category": "multiref",  # whole-outfit mask for an upper+lower try-on
            "upper_path": upper_path,
            "lower_path": lower_path,
            "upper_name": slot_name.get("upper"),
            "lower_name": slot_name.get("lower"),
            "combo_id": f"{slug(stem)}__{slug(upper_id)}__{slug(lower_id)}",
        })
    return combos


def parse_similar(rec_path: str, stem: str, category_filter: str = None) -> tuple:
    """Parse the ``similar`` section into per-garment single-category combos.

    Each entry is one garment carrying its own ``category`` (upper/lower/...). Items
    whose category is not tryable here (no single-region mask AND ref slot, e.g.
    shoe/bag/inner/outer) are skipped. Returns ``(combos, skipped)`` where ``skipped``
    is a list of ``(index, name, raw_category, reason)`` for reporting.
    """
    with open(rec_path) as f:
        data = json.load(f)

    want = normalize_category(category_filter) if category_filter else None
    combos, skipped = [], []
    for i, item in enumerate(data.get("similar", [])):
        raw_cat = item.get("category")
        category = normalize_category(raw_cat)
        raw_path = item.get("image_path", "")
        name = item.get("name") or Path(raw_path).stem
        garment_stem = Path(raw_path).stem or "none"

        if want and category != want:
            continue
        if category not in TRYON_CATEGORIES:
            skipped.append((i, name, raw_cat,
                            f"no single-region try-on for category '{raw_cat}'"))
            continue

        combos.append({
            "index": i,
            "category": category,
            "garment_path": resolve_garment_path(raw_path),
            "name": name,
            "combo_id": f"{slug(stem)}__{slug(category)}__{slug(garment_stem)}",
        })
    return combos, skipped


def load_person_cache(precompute_dir: str, stem: str) -> dict:
    """Resolve the category-INDEPENDENT person files for one stem (no pixel I/O).

    Per-category masks/latents are resolved separately by ``resolve_category_cache``,
    since a single ``--section similar`` run mixes categories across combos.
    """
    base = Path(precompute_dir) / stem
    return {
        "person_png": base / "person.png",
        "pose_png": base / "pose.png",
        "base": base,
        "masks_dir": Path(precompute_dir) / "masks" / stem,
    }


def resolve_category_cache(person_cache: dict, category: str,
                           square: bool, pose: bool) -> dict:
    """Build the mask-png + latent paths for one (category, square, pose).

    Mirrors precompute_person_preprocessing.py naming. NOTE the two distinct square
    encodings: the PNG uses the suffix ``_square``/``""`` while the ``.pt`` files use
    the token ``square``/``nosquare``.
    """
    sq_png = "_square" if square else ""
    sq = "square" if square else "nosquare"
    po = "pose" if pose else "nopose"

    masked_pt = person_cache["base"] / f"masked_person_latent_{category}_{sq}_{po}.pt"
    mask_pt = person_cache["base"] / f"mask_latent_{category}_{sq}.pt"
    return {
        "category": category,
        "mask_png": person_cache["masks_dir"] / f"{category}{sq_png}.png",
        "masked_person_latent_pt": masked_pt,
        "mask_latent_pt": mask_pt,
        "latents_present": masked_pt.exists() and mask_pt.exists(),
    }


def prepare_reference_images(slot_img: dict, ref_height: int):
    """Standalone replica of FastFitDemo.prepare_reference_images (app.py:273-296).

    ``slot_img`` maps a ref-slot label (REF_CATEGORIES) -> PIL.Image. Slots absent (or
    with a None value) become black placeholders so the pipeline's cache loop skips
    them (attention_mask == 0)."""
    clothing_ref_size = (int(ref_height * 3 / 4), ref_height)
    accessory_ref_size = (384, 512)

    ref_images, ref_labels, ref_attention_masks = [], [], []
    for label in REF_CATEGORIES:
        size = accessory_ref_size if label in ("shoe", "bag") else clothing_ref_size
        img = slot_img.get(label)
        if img is not None:
            ref_images.append(img.convert("RGB").resize(size, Image.LANCZOS))
            ref_attention_masks.append(1)
        else:
            ref_images.append(Image.new("RGB", size, color=(0, 0, 0)))
            ref_attention_masks.append(0)
        ref_labels.append(label)
    return ref_images, ref_labels, ref_attention_masks


def load_garment(path: str) -> Image.Image:
    if path.lower().endswith(".avif") and not _AVIF_OK:
        raise RuntimeError(_AVIF_ERR_MSG)
    return Image.open(path).convert("RGB")


def resolve_device(device_arg):
    import torch
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(args, device):
    # Deferred import: keeps the --dry-run path free of torch / module / app.
    from module.pipeline_fastfit import FastFitPipeline
    return FastFitPipeline(
        base_model_path=args.base_model_path,
        device=device,
        mixed_precision=args.mixed_precision,
        allow_tf32=True,
    )


def run_combo(pipeline, person_cache, cat_cache, combo, args, device) -> Path:
    import torch

    person = Image.open(person_cache["person_png"]).convert("RGB")
    pose = Image.open(person_cache["pose_png"]).convert("RGB")
    mask = Image.open(cat_cache["mask_png"]).convert("L")

    if combo["category"] == "multiref":
        upper = load_garment(combo["upper_path"]) if combo["upper_path"] else None
        lower = load_garment(combo["lower_path"]) if combo["lower_path"] else None
        slot_img = {"upper": upper, "lower": lower}
    else:
        garment = load_garment(combo["garment_path"]) if combo["garment_path"] else None
        slot_img = {combo["category"]: garment}
    ref_images, ref_labels, ref_attention_masks = prepare_reference_images(
        slot_img, args.ref_height
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    kwargs = dict(
        person=person,
        mask=mask,
        ref_images=ref_images,
        ref_labels=ref_labels,
        ref_attention_masks=ref_attention_masks,
        pose=pose,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        return_pil=True,
    )
    if args.use_precomputed_latents:
        kwargs["masked_person_latent"] = torch.load(
            cat_cache["masked_person_latent_pt"]
        ).to(device, pipeline.weight_dtype)
        kwargs["mask_latent"] = torch.load(
            cat_cache["mask_latent_pt"]
        ).to(device, pipeline.weight_dtype)

    image = pipeline(**kwargs)
    result = image[0] if isinstance(image, list) else image

    out_path = Path(args.output_dir) / f"{combo['combo_id']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return out_path


def print_dry_run_report(args, person_cache, get_cat_cache, combos, skipped) -> None:
    line = "=" * 78
    print(line)
    print("DRY RUN — no models loaded, no images decoded")
    print(line)
    print(f"person-stem            : {args.person_stem}")
    print(f"precompute-dir         : {args.precompute_dir}")
    print(f"recommendations        : {args.recommendations}")
    print(f"section                : {args.section}"
          + (f"  (category filter: {args.category})" if args.category else ""))
    print(f"mask variant           : {'square' if args.square else 'nosquare'} / "
          f"{'pose' if args.pose else 'nopose'}")
    print(f"output-dir             : {args.output_dir}")
    print(f"ref-height/steps/gs    : {args.ref_height} / {args.steps} / {args.guidance_scale}")
    print(f"use-precomputed-latents: {args.use_precomputed_latents}")
    print(f"AVIF decoder           : "
          f"{'available (pillow_avif)' if _AVIF_OK else 'MISSING — pip install pillow-avif-plugin'}")
    print()

    print("Person cache:")
    for label, key in [("person.png", "person_png"), ("pose.png", "pose_png")]:
        p = person_cache[key]
        print(f"  [{'OK  ' if p.exists() else 'MISS'}] {label:24s} {p}")
    print()

    print("Per-category cache:")
    any_latents_missing = False
    for category in sorted({c["category"] for c in combos}):
        cc = get_cat_cache(category)
        print(f"  category '{category}':")
        for key in ("mask_png", "mask_latent_pt", "masked_person_latent_pt"):
            p = cc[key]
            print(f"    [{'OK  ' if p.exists() else 'MISS'}] {p.name:42s} {p}")
        if args.use_precomputed_latents and not cc["latents_present"]:
            any_latents_missing = True
    if any_latents_missing:
        print("  --> --use-precomputed-latents is set but some .pt latents are MISSING.")
        print("      Re-run precompute_person_preprocessing.py --overwrite on a CUDA machine,")
        print("      or drop the flag to use the fallback (live masked-person encode).")
    print()

    print(f"{args.section.capitalize()} combos ({len(combos)}):")
    for c in combos:
        print(f"  [{c['index']}] {c['combo_id']}  (category: {c['category']})")
        if c["category"] == "multiref":
            slots = [("upper", c["upper_path"], c["upper_name"]),
                     ("lower", c["lower_path"], c["lower_name"])]
        else:
            slots = [(c["category"], c["garment_path"], c["name"])]
        for slot, path, name in slots:
            if path is None:
                print(f"        {slot:7s}: (none — placeholder, attention_mask=0)")
            else:
                status = "OK  " if os.path.exists(path) else "MISS"
                print(f"        {slot:7s}: [{status}] {name}  ->  {path}")
        print(f"        output: {os.path.join(args.output_dir, c['combo_id'] + '.png')}")
    print()

    if skipped:
        print(f"Skipped {len(skipped)} item(s):")
        for i, name, raw_cat, reason in skipped:
            print(f"  [{i}] {name}: {reason}")
        print()

    print("Dry run complete. (Real inference requires CUDA.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch FastFit try-on over the complementary recommendations using a "
                    "precomputed person cache."
    )
    parser.add_argument("--person-stem", default="sumant",
                        help="Stem of the precomputed person (folder under --precompute-dir).")
    parser.add_argument("--precompute-dir", default="precomputed_person_tensors",
                        help="Directory holding the precomputed person tensors and masks.")
    parser.add_argument("--recommendations", default="inventory/outputs/recommendations.json",
                        help="Path to recommendations.json (see --section).")
    parser.add_argument("--section", choices=["complementary", "similar"], default="complementary",
                        help="Which recommendations section to process. 'complementary' = "
                             "upper+lower outfit pairings (multiref mask). 'similar' = single "
                             "garments, each tried on with its own category's mask.")
    parser.add_argument("--category", default=None,
                        help="FILTER for --section similar: only process items of this category "
                             "(does NOT override an item's own category). Ignored for "
                             "--section complementary.")
    parser.add_argument("--square", dest="square", action="store_true", default=None,
                        help="Use the *_square mask + latents. Default: square for "
                             "--section complementary, non-square for --section similar.")
    parser.add_argument("--no-square", dest="square", action="store_false",
                        help="Use the non-square mask + latents (see --square for the default).")
    parser.add_argument("--pose", dest="pose", action="store_true", default=True,
                        help="Use the pose-blended masked-person latent (default).")
    parser.add_argument("--no-pose", dest="pose", action="store_false",
                        help="Use the non-pose masked-person latent.")
    parser.add_argument("--output-dir", default="output/batch_tryon",
                        help="Where to write result PNGs.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=2.5, help="CFG scale.")
    parser.add_argument("--ref-height", type=int, default=512, choices=[512, 768, 1024],
                        help="Reference garment height (width auto-derived 3:4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (per combo).")
    parser.add_argument("--mixed-precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu). Defaults to auto.")
    parser.add_argument("--base-model-path", default="Models/FastFit-MR-1024",
                        help="Local path for the FastFit base model (auto-downloaded if missing).")
    parser.add_argument("--use-precomputed-latents", action="store_true",
                        help="Feed the precomputed masked-person/mask VAE latents to the pipeline "
                             "(skips the masked-person encode + pose blend). Requires the .pt files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many combos.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate parsing / path resolution / cache presence without loading "
                             "any model (no torch import). Useful on machines without a GPU.")
    return parser.parse_args()


def combo_garment_paths(combo) -> list:
    """The garment image paths a combo will load (for the AVIF presence check)."""
    if combo["category"] == "multiref":
        return [combo["upper_path"], combo["lower_path"]]
    return [combo["garment_path"]]


def main() -> None:
    args = parse_args()
    # Section-aware default: complementary historically used the square mask; the
    # single-region similar path hugs the garment better with the non-square mask.
    if args.square is None:
        args.square = (args.section == "complementary")

    skipped = []
    if args.section == "complementary":
        combos = parse_complementary(args.recommendations, args.person_stem)
        if args.category:
            print("[warn] --category is ignored for --section complementary")
    else:
        combos, skipped = parse_similar(
            args.recommendations, args.person_stem, args.category
        )
    if args.limit is not None:
        combos = combos[:args.limit]

    person_cache = load_person_cache(args.precompute_dir, args.person_stem)
    cat_cache_memo = {}

    def get_cat_cache(category):
        if category not in cat_cache_memo:
            cat_cache_memo[category] = resolve_category_cache(
                person_cache, category, args.square, args.pose
            )
        return cat_cache_memo[category]

    if args.dry_run:
        print_dry_run_report(args, person_cache, get_cat_cache, combos, skipped)
        return

    if not combos:
        print(f"No combos to process for --section {args.section}"
              + (f" --category {args.category}" if args.category else "") + ".")
        return

    # --- Real run: validate cache, then load the pipeline and iterate. ---
    missing = [str(person_cache[k]) for k in ("person_png", "pose_png")
               if not person_cache[k].exists()]
    latents_missing = []
    for category in sorted({c["category"] for c in combos}):
        cc = get_cat_cache(category)
        if not cc["mask_png"].exists():
            missing.append(str(cc["mask_png"]))
        if args.use_precomputed_latents and not cc["latents_present"]:
            latents_missing += [str(cc["masked_person_latent_pt"]), str(cc["mask_latent_pt"])]
    if missing:
        raise SystemExit(
            "Missing required cache files:\n  " + "\n  ".join(sorted(set(missing))) +
            f"\nRun: python precompute_person_preprocessing.py --input <{args.person_stem}>.png"
        )
    if latents_missing:
        raise SystemExit(
            "--use-precomputed-latents was set but these .pt latents are missing:\n  "
            + "\n  ".join(sorted(set(latents_missing))) +
            "\nRe-run precompute_person_preprocessing.py --overwrite on a CUDA machine to "
            "generate them, or drop --use-precomputed-latents to use the fallback path."
        )
    if not _AVIF_OK and any(
        (p or "").lower().endswith(".avif")
        for c in combos for p in combo_garment_paths(c)
    ):
        raise SystemExit(_AVIF_ERR_MSG)

    device = resolve_device(args.device)
    print(f"[device] {device}  [precision] {args.mixed_precision}  "
          f"[section] {args.section}  [mask] {'square' if args.square else 'nosquare'}  "
          f"[latents] {'on' if args.use_precomputed_latents else 'off'}  "
          f"[combos] {len(combos)}")
    pipeline = build_pipeline(args, device)

    n_ok = n_fail = 0
    durations = []  # per-combo wall-clock for the OK runs (for the avg summary)
    for combo in combos:
        t0 = time.perf_counter()
        try:
            out = run_combo(pipeline, person_cache, get_cat_cache(combo["category"]),
                            combo, args, device)
            dt = time.perf_counter() - t0
            durations.append(dt)
            print(f"[OK]   {combo['combo_id']} -> {out}  ({dt:.1f}s)")
            n_ok += 1
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"[FAIL] {combo['combo_id']}: {e}  ({dt:.1f}s)")
            n_fail += 1

    timing = ""
    if durations:
        timing = (f" Per-combo: {sum(durations) / len(durations):.1f}s avg, "
                  f"{min(durations):.1f}s min, {max(durations):.1f}s max "
                  f"(total {sum(durations):.1f}s).")
    print(f"\nDone. {n_ok} ok, {n_fail} failed"
          + (f", {len(skipped)} skipped" if skipped else "")
          + f". Outputs in {args.output_dir}." + timing)
    if n_fail and not n_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
