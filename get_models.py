import os
os.environ["HF_HUB_CACHE"] = "./hf_cache"

from huggingface_hub import snapshot_download

# This downloads the model directly into a flat folder with real files
model_path = snapshot_download(
    repo_id="ResembleAI/chatterbox",
    local_dir="./chatterbox/temp_model",
    local_dir_use_symlinks=False  # This is the "magic" flag you need
)


