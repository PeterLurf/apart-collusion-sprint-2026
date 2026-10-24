"""Shared Modal images, Volume and secrets (imported by every modal_app/*.py).

Local sources (`rtd`) must be the last image step, so variants branch off `_base`.
"""

import modal

from rtd.config import MODAL_HF_SECRET, MODAL_VOLUME

vol = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name(MODAL_HF_SECRET)
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(  # versions verified on Modal, Oct 2026
        "torch==2.14.1", "transformers==5.20.2", "nnsight==0.8.1",
        "accelerate>=1.0", "safetensors>=0.4", "numpy",
    )
    .env({"HF_HOME": "/vol/hf"})
)
image = _base.add_local_python_source("rtd")
probe_image = _base.pip_install("scikit-learn>=1.4").add_local_python_source("rtd")
