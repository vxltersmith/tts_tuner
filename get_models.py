import yaml
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    try:
        config = load_config()
        local_dir = str(config['model_dir'])
        os.environ["HF_HUB_CACHE"] = str(config['model_dir'] / '../../hf_cache')

        print(f"📥 Downloading model to: {local_dir}")
        model_path = snapshot_download(
            repo_id="ResembleAI/chatterbox",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Model downloaded to: {model_path}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}", file=sys.stderr)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()