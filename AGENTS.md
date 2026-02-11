# AGENTS.md

## Build & Setup Commands

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For ROCm support (AMD GPUs)
pip install -r requirements_rocm.txt
```

## Running Scripts

This is a Python project with no build step. Run scripts directly:

```bash
# Generate speech from text
python run_tts.py

# Train custom tokenizer
python train_tokenizer.py

# Merge tokenizers for multilingual support
python merge_embeddings.py

# Download pre-trained models
python get_models.py

# Validation loop with Whisper ASR
python validation_loop.py
```

### Running Single Scripts

To test a specific script, run it directly:
```bash
python run_tts.py  # Tests full TTS pipeline
```

No pytest/unittest suite exists. Scripts serve as integration tests.

## Code Style Guidelines

### General
- Python ≥3.10 (project requires 3.10+, project typically uses 3.8+ compatible syntax)
- Type hints: Use for function signatures, especially public APIs
- Imports: Absolute imports from project root
- Paths: Use `pathlib.Path` exclusively (not os.path strings)
- Error handling: Use try/except with specific exceptions; raise RuntimeError for recoverable errors

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: Leading underscore `_private`

### Import Organization
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import torch
import numpy as np

# Internal (group by module path alphabetically)
from .models.t3 import T3
from .models.s3gen import S3Gen
```

### Type Hints
- Annotate all function parameters and return values
- Use `torch.Tensor` for PyTorch tensors
- Use `dict`, `list`, `Path`, etc. for standard types
- For config dicts: `config: dict` or create dataclasses

### Formatting
- 4 spaces per indent (no tabs)
- Max line length: 120 characters
- Use f-strings for formatting
- Single quotes for strings (except when containing quotes)

### File Organization
- Main scripts in project root: `run_tts.py`, `train_tokenizer.py`, etc.
- Library code: `chatterbox/src/chatterbox/`
- Data/config: `data/`, `config.yaml`

### Configuration
- YAML config files for human-editable settings
- Load config via `yaml.safe_load()` from files
- Use dot notation access: `config['key']`

### Error Handling
- Validate config paths exist before use
- Use descriptive error messages with context
- GPU detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## Configuration

`config.yaml` (root) contains:
- `paths`: model_dir, tokenizer paths, output_dir, checkpoint_dir
- `models`: voice_encoder, t3_checkpoint, s3gen_checkpoint paths
- `tts`: device, text_tokens_dict_size
- `train_tokenizer`: dataset, split, vocab_size, min_frequency
- `merge_embeddings`: parameters for tokenizer merging

## ROCm/GPU Notes

- ROCm identifies as 'cuda' in code
- Use `torch.nn.attention.sdp_kernel()` for AMD performance
- Always warm up models before main generation

## Existing Rules

No Cursor or Copilot rules files found.
