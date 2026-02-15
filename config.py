from pathlib import Path
import yaml


def load_config():
    """Load configuration from YAML file relative to script parent."""
    # Matches your logic: Path(__file__).parent.parent / "config.yaml"
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
        
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
