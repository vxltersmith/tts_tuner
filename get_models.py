import yaml
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import sys

def load_config():
    """Load configuration from YAML file."""
    config_path = "./config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

import shutil
from pathlib import Path

def copy_dereference(src_dir: str, dst_dir: str):
    src = Path(src_dir)
    dst = Path(dst_dir)

    # Ensure destination is a clean directory
    if dst.exists() and dst.is_file():
        dst.unlink()  # remove if it's accidentally a file
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name.startswith('.') or not (item.is_symlink() or item.is_file()):
            continue
        try:
            real_src = item.resolve()
            if real_src.is_file():
                # Define the final destination path IMMEDIATELY
                final_dst_path = dst / item.name 
                
                # Perform the copy directly to the final name
                # This replaces the Copy + Rename logic
                shutil.copy2(str(real_src), str(final_dst_path))
                
                print(f"✅ Copied and dereferenced: {item.name}")
        except Exception as e:
            print(f"❌ Failed on {item.name}: {e}")


def main():
    try:
        config = load_config()
        local_dir = str(config.get('paths', {}).get('model_dir', "./data/temp_model"))
        hf_cache = str(config.get('paths', {}).get('cache_dir', "./hf_cache"))
        checkpt_dir = os.path.join(hf_cache, "chatterbox_chkpt")
        os.environ["HF_HUB_CACHE"] = hf_cache

        print(f"📥 Downloading models to cache: {checkpt_dir}")
        model_path = snapshot_download(
            repo_id="ResembleAI/chatterbox",
            local_dir=checkpt_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Model downloaded to: {model_path}")

        copy_dereference(
            checkpt_dir,
            local_dir
        )
    except Exception as e:
        print(f"❌ Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()