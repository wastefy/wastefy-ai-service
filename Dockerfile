FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface-hub

RUN python <<EOF
from huggingface_hub import hf_hub_download
import os

token_riimaru = os.environ.get("HF_TOKEN_RIIMARU")
token_arciii = os.environ.get("HF_TOKEN_ARCIII")

hf_hub_download(
    repo_id="riimaru/wastefy-models",
    filename="model.keras",
    local_dir="model/vision",
    token=token_riimaru
)

hf_hub_download(
    repo_id="riimaru/wastefy-models",
    filename="model_metadata.json",
    local_dir="model/vision",
    token=token_riimaru
)

hf_hub_download(
    repo_id="arciii/wastefy-models",
    filename="model.keras",
    local_dir="model/regression",
    token=token_arciii
)

hf_hub_download(
    repo_id="arciii/wastefy-models",
    filename="model_metadata.json",
    local_dir="model/regression",
    token=token_arciii
)
EOF

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]