"""Fetch the model weights needed before running the FastFit pipeline.

Run from the repo root on the fastfit conda env::

    python download_models.py

``snapshot_download`` is incremental: it keeps files already present under
``local_dir`` and only downloads what's missing, so a partial
``Models/FastFit-MR-1024`` (e.g. just ``vae/``) is completed in place.
"""

from huggingface_hub import snapshot_download

# Base diffusion model (REQUIRED for batch_inference.py): vae + unet + scheduler.
# Completes the partial Models/FastFit-MR-1024 (keeps existing vae/, adds the rest).
snapshot_download(
    repo_id="zhengchong/FastFit-MR-1024",
    local_dir="Models/FastFit-MR-1024",
)
print("FastFit-MR-1024 ready.")

# Preprocessing detectors (DWPose/DensePose/SCHP) — NOT needed for this batch run,
# since person preprocessing is already cached. Uncomment only for app.py / fresh precompute.
# snapshot_download(
#     repo_id="zhengchong/Human-Toolkit",
#     local_dir="Models/Human-Toolkit",
# )
# print("Human-Toolkit ready.")
