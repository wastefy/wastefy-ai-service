from huggingface_hub import hf_hub_download
from pathlib import Path
import os

def download_models():
    token_riimaru = os.getenv("HF_TOKEN_RIIMARU")
    token_arciii = os.getenv("HF_TOKEN_ARCIII")

    Path("model/vision").mkdir(parents=True, exist_ok=True)
    Path("model/regression").mkdir(parents=True, exist_ok=True)

    files = [
        (
            "riimaru/wastefy-models",
            "model.keras",
            "model/vision",
            token_riimaru,
        ),
        (
            "riimaru/wastefy-models",
            "model_metadata.json",
            "model/vision",
            token_riimaru,
        ),
        (
            "arciii/wastefy-models",
            "model.keras",
            "model/regression",
            token_arciii,
        ),
        (
            "arciii/wastefy-models",
            "model_metadata.json",
            "model/regression",
            token_arciii,
        ),
    ]

    for repo_id, filename, local_dir, token in files:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=token,
        )