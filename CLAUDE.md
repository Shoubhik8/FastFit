# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastFit is a diffusion-based **multi-reference virtual try-on** framework: it dresses a single person image with multiple garment references at once (upper, lower, dress, shoes, bag). Its defining trick is **reference KV caching** — the key/value tensors for the garment/pose reference tokens are computed once and reused across all denoising steps, which is what makes inference fast. The code is adapted from HuggingFace Diffusers with Stable Diffusion v1.5 inpainting as the base.

## Environment

```bash
conda create -n fastfit python=3.10
conda activate fastfit
pip install -r requirements.txt
pip install easy-dwpose --no-dependencies   # avoids a version conflict
# if `av` fails to build: conda install -c conda-forge av
```

Dependency pins matter: `diffusers==0.32.2` and `transformers==4.49.0` are required because the vendored `module/` code targets those APIs and newer releases break against torch 2.4.x. Do not bump them casually.

## Models

Weights download automatically from HuggingFace on first run via `snapshot_download`:
- `zhengchong/FastFit-MR-1024` — the base diffusion model (VAE, UNet, scheduler subfolders).
- `zhengchong/Human-Toolkit` — preprocessing utility models: DWPose, DensePose, and SCHP (LIP + ATR checkpoints).

CLI defaults expect them under `Models/FastFit-MR-1024` and `Models/Human-Toolkit`.

## Running

```bash
# Gradio web demo (bilingual ZH/EN)
python app.py

# Single try-on from the CLI
python infer.py --person person.png --upper shirt.png --output out/result.png \
    --mask-part upper --steps 30 --guidance-scale 2.5

# Dataset inference (dresscode-mr | dresscode | viton-hd)
python infer_datasets.py --dataset viton-hd --data_dir /path/to/data \
    --batch_size 4 --num_inference_steps 50 --guidance_scale 2.5 --mixed_precision bf16 --paired

# Evaluation metrics on generated results
python eval.py --gt_folder /path/gt --pred_folder /path/pred --paired --batch_size 16 --num_workers 4
```

`--paired` toggles the paired vs. unpaired evaluation setting (omit for unpaired). There is no test suite or linter configured in this repo.

### Precompute pipeline

`precompute_person_preprocessing.py` runs the expensive person-preprocessing once per person and caches the results to `precomputed_person_tensors/<name>/` (person.png, pose.png, densepose.npy, lip.npy, atr.npy), masks under `masks/`, plus saved VAE latents. This lets repeated try-ons on the same person skip pose/parse/VAE encoding. See existing `precomputed_person_tensors/sumant/` for the expected layout.

## Architecture

The system is a thin orchestration layer over a heavily-modified Diffusers stack.

**Entry points → `FastFitDemo` (`app.py`).** `app.py`, `infer.py`, and the ComfyUI node (`__init__.py`) all funnel through the `FastFitDemo` class, which owns the four preprocessing detectors and the pipeline. Its `generate_image()` is the full flow: validate inputs → `preprocess_person_image()` (DWPose pose + DensePose + SCHP LIP/ATR parsing) → `generate_mask()` → `prepare_reference_images()` → `pipeline(...)`.

**Preprocessing (`parse_utils/`).** Three independent toolkits produce the conditioning signals:
- `dwpose.py` (`DWposeDetector`) — 2D pose skeleton.
- `densepose/` — vendored Detectron2 DensePose (dense body surface coordinates).
- `schp/` — Self-Correction Human Parsing; LIP and ATR variants give garment-region segmentation.
- `automasker.py` — turns parse maps into the cloth-agnostic inpainting mask. `cloth_agnostic_mask` (single region via `mask_part`) vs. `multi_ref_cloth_agnostic_mask` (whole outfit). `mask_part` ∈ {upper, lower, overall, inner, outer}; use `upper` for a t-shirt-only try-on that preserves the rest of the person.

**Diffusion core (`module/`).** A fork of the Diffusers UNet stack:
- `pipeline_fastfit.py` (`FastFitPipeline`) — loads VAE/UNet/DDPMScheduler from the base model and runs the denoising loop. Note the cache lifecycle: it calls the UNet **once with `cache_kv=True`** to populate the reference KV cache, then loops the denoising steps reusing it, and finally `clear_kv_cache()`.
- `attention_processor.py` — the heart of the speedup. `OptimizedAttentionCache` stores reference key/value/mask tensors; `AttnProcessor2_0` / `XFormersAttnProcessor` hold a `kv_cache` and branch on the `cache_kv` flag to either populate or reuse it. **When touching attention, preserve the cache populate/reuse/clear contract** — it threads from the pipeline through `Attention.forward(cache_kv=...)` down into the processors.
- `unet_2d_condition.py`, `unet_2d_blocks.py`, `transformer_2d.py`, `attention.py`, `resnet.py` — the modified UNet building blocks the `cache_kv` flag is threaded through.

The `cache_kv` boolean is the cross-cutting concern of this codebase: it is plumbed from `FastFitPipeline.__call__` all the way down to the attention processors. Any change to the forward signatures must keep it intact.

## Fixes

A running log of non-obvious bug fixes, what was actually wrong, and how it was repaired. Read this before touching the mask logic or the precompute cache.

### Multiref try-ons hallucinated feet/shoes (fixed)

**Symptom.** Running a whole-outfit (`complementary` / `multiref`) batch try-on — e.g. `python batch_inference.py --person-stem <name> --use-precomputed-latents` — produced people whose **feet were replaced by AI-invented footwear** (random booties, socks, pointed shoes), even though the source photo showed bare feet. Single-garment (`similar`) try-ons did not have this problem.

**Why it happened.** The inpainting pipeline regenerates *exactly* the white region of the mask and keeps everything else pixel-for-pixel from the original (the final step in `module/pipeline_fastfit.py` is literally `image = image * mask + (1 - mask) * person`). So if the feet change, the mask must be covering the feet. It was: the whole-outfit mask builder `multi_ref_cloth_agnostic_mask()` in `parse_utils/automasker.py` **deliberately included the feet** in the mask, unlike the single-region builder `cloth_agnostic_mask()` which carves the feet back out. Concretely, in the multiref function:
- the hands/feet "protect" block (which tells the mask *not* to touch hands and feet) was **commented out**, so only the face was protected;
- the feet were **explicitly added** to the masked region (`MASK_DENSE_PARTS["overall"] + ['right foot', 'left foot']`);
- there was **no re-exclusion of the protected area after the convex-hull expansion** (the single-region builder does this, which is why its `lower.png`/`overall.png` masks show clean foot-shaped holes).

This is also why it looked like it might be a "square vs. non-square mask" issue but wasn't: **both** the square and non-square multiref masks reached the bottom of the frame and covered the feet. Square vs. non-square only changes the mask's outline, not whether feet are protected.

**The fix (in `parse_utils/automasker.py`, function `multi_ref_cloth_agnostic_mask`).** Ported the exact foot/hand preservation the single-region builder already used:
1. Re-enabled the `hands_protect_area` block (DensePose hands+feet **intersected with** SCHP arms+legs) and folded it into `strong_protect_area` so feet/hands are now protected, not just the face.
2. Stopped force-adding `['right foot', 'left foot']` to the masked region.
3. After the convex-hull expansion, re-applied `mask_area = mask_area & (~strong_protect_area)` so the hull can't swallow the feet back in. (The square/bounding-box branch already subtracts `strong_protect_area`, so it benefits automatically.)

Because this lives in shared mask code, it improves **every** entry point (the live `app.py` path included). It needs no model retraining — inpainting simply paints a smaller region.

**Caveat — the `--square` mask still repaints the floor around the feet.** The square (bounding-box) variant fills a solid rectangle down to the ankles, so the *feet themselves* are now preserved (carved-out holes) but the *floor immediately around them* inside the rectangle gets regenerated (a faint rectangular patch). For the cleanest result on a barefoot, full-body photo, prefer the non-square mask: `--no-square`. The non-square multiref hugs the body and avoids the floor patch.

**Updating an already-cached person.** Masks (and the VAE latents derived from them) are baked into `precomputed_person_tensors/<name>/` at precompute time, so an existing cached person will **not** pick up this fix until its mask cache is rebuilt. Two ways:
- Full re-precompute (re-runs the four detectors, needs `Models/Human-Toolkit`): `python precompute_person_preprocessing.py --input <name>.png --overwrite`.
- Fast rebuild from the cached parse maps (no detectors, no Human-Toolkit — only needs the VAE): `python regen_person_masks.py --stem <name> --categories multiref`. This helper re-derives the masks straight from the cached `densepose.npy`/`lip.npy`/`atr.npy` and re-encodes only the affected latents, which is much faster than a full re-precompute. Use it whenever you change `automasker.py` and just need existing caches refreshed. New people precomputed after the fix need nothing extra.

### Per-combo timing in `batch_inference.py` (enhancement)

`batch_inference.py` now prints how long each try-on takes (`[OK] ... (6.7s)`) plus an avg/min/max/total summary line, measured around `run_combo` with `time.perf_counter()`. Note the **first combo is always slower** because the pipeline's one-time `torch.compile`/`fuse_qkv_projections` warmup happens on the first forward pass, not at model-load time — treat it as a warmup outlier.
