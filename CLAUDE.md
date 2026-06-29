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
