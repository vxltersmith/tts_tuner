# TTS Tuner Development Guide

This document provides comprehensive guidelines for agents working on the TTS tuner project.

## Project Overview

TTS Tuner handles T3 (Token-To-Token) transformer-based Text-to-Speech models. The project includes tooling for training tokenizers, merging embeddings, and generating speech. The scripts are designed to be transparent, configurable, and easily extendable.

## Environment Setup

### Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.yaml              # Main configuration file
├── run_tts.py               # Generate speech from text
├── train_tokenizer.py       # Train BPE tokenizer
├── merge_embeddings.py      # Merge embeddings for extended vocabulary
├── get_models.py            # Download pre-trained model
├── merge_tokenizers.py      # Merge tokenizers (to be refactored)
├── validation_loop.py       # Validation and testing (to be refactored)
└── AGENTS.md                # This file
```

## Configuration System

All scripts load configuration from `config.yaml`:

```yaml
paths:
  model_dir: "./data/temp_model"
  tokenizer_en: "./data/temp_model/tokenizer.json"
  tokenizer_ru: "./data/en_ru_tokenizer.json"

models:
  voice_encoder: "./data/temp_model/ve.safetensors"
  t3_checkpoint: "./data/temp_model/t3_cfg.safetensors"
  s3gen_checkpoint: "./data/temp_model/s3gen.safetensors"

tts:
  text_tokens_dict_size: 786
  device: "cuda"
```

**Always use `load_config()` in scripts to access these values.**

## Code Style Guidelines

### Import Order
1. Standard library (`os`, `sys`, `pathlib`)
2. Third-party (`torch`, `yaml`, `tokenizers`)
3. Local modules (`from src.chatterbox.xxx import ...`)

### Path Operations
```python
from pathlib import Path

# Always use Path objects
config_path = Path(__file__).parent / 'config.yaml'
path = Path("./data") / "model.pth"
path.mkdir(parents=True, exist_ok=True)
```

### Type Hints
```python
def function_name(param1: str, param2: int) -> bool:
    """Docstring here."""
    pass
```

### Error Handling
```python
try:
    # Operation
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
```

### File Operations
```python
# Loading YAML
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Loading safetensors
from safetensors import safe_open
with safe_open(file_path) as f:
    data = f.get_tensor(key)
```

## Working with Scripts

### run_tts.py
- Generates speech from text using configured model
- **Entry point**: `main()`
- **Main function**: `build_model()`, `generate_tts()`
- Configuration: `config.yaml['tts']`, `config.yaml['models']`

### train_tokenizer.py
- Trains BPE tokenizer on dataset
- **Entry point**: `main()`
- Uses: `load_dataset()`, `Tokenizers`, `trainers`
- Configuration: `config.yaml['train_tokenizer']`

### merge_embeddings.py
- Merges English embeddings to support new vocabulary
- **Entry point**: `main()`
- Uses: `safetensors`, `torch`, `tokenizers`
- Configuration: `config.yaml['merge_embeddings']`

### get_models.py
- Downloads pre-trained model from HuggingFace
- **Entry point**: `main()`
- Configuration: `config.yaml['model_dir']`

## Key Implementation Details

### Configuration Pattern
Every script should:
1. Define `load_config()` function
2. Call it at script start
3. Access config values using nested dictionary notation
4. Handle errors gracefully

```python
def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    # Use config['section']['key']
```

### Error Handling Pattern
```python
try:
    # Main logic
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    sys.exit(1)
```

## Development Tasks

### Adding New Features
1. Update `config.yaml` with new parameters
2. Modify the appropriate script
3. Add proper type hints
4. Include docstrings
5. Test error handling

### Modifying Configuration
1. Update `config.yaml`
2. All scripts automatically pick up changes
3. No code modifications needed

### Working with Models
- Use `load()` functions that load from `config.yaml`
- Store model data in organized way
- Clean up after operations

## Important Notes

- **Never hardcode paths** in scripts - always use config
- **Use pathlib.Path** for all file operations
- **Always close file handles** when reading/writing
- **Log errors to stderr** with clear messages
- **Exit with code 1** on failures
- **Never make uncommitted changes** without asking

## Getting Started

For first-time usage:
1. Set up virtual environment
2. Install dependencies
3. Configure paths in `config.yaml`
4. Download model: `python get_models.py`
5. Try generating speech: `python run_tts.py`