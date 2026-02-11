# TTS Tuner Development Guide

This document provides comprehensive guidelines for agents working on the TTS tuner project.

## Project Overview

TTS Tuner handles T3 (Token-To-Token) transformer-based Text-to-Speech models. The project includes tooling for training tokenizers, merging embeddings, and generating speech. The scripts are designed to be transparent, configurable, and easily extendable.

## Hypothesis
1. Research Context & Model Backbone
Context: The model is a T3 (Token-To-Token) transformer-based TTS with a Llama-style architecture and an S3Gen (Flow Matching) decoder.
Code Alignment: The code handles the embedding weights (text_emb.weight, text_head.weight) for the T3 model, which are critical for token representation. The code modifies these weights to support a new script (Russian Cyrillic) by grafting embeddings.
Key Concept: The code implements a “zero-shot script extension” by expanding the vocabulary and grafting embeddings, aligning with the stated goal of avoiding extensive retraining.
2. Problem Statement & Constraint
Context: The original model is English-centric (Vorig = 704). Fine-tuning on Russian would require massive datasets and risk shifting the latent space, potentially breaking English capabilities.
Code Alignment: The code avoids extensive retraining by using a “grafting” approach. It starts with the existing English embeddings (704 tokens) and extends the vocabulary (to 786 tokens) by mapping Russian characters to Latin via transliteration.
Key Concept: The use of transliterate_ru_to_latin and harmonic_phonetic_map aligns with the idea of mapping Russian characters to English phonetic representations, preserving the latent space.
3. Methodology: SLERP and Embedding Grafting
Context: The hypothesis proposes using SLERP (Spherical Linear Interpolation) to merge high-dimensional embeddings from English subtokens to create Russian embeddings.
Code Alignment:
The slerp function implements spherical linear interpolation between two vectors.
The slerp_multiple and slerp_weighted_sequence functions recursively apply SLERP to combine multiple embeddings with weights.
The descending_linear_weights function assigns higher weights to leading subtokens, aligning with the idea that they carry stronger phonetic signals.
Key Concept: The grafting process uses SLERP to merge embeddings from the English model, ensuring that the Russian vocabulary inherits phonetic and prosodic richness.
4. Vocabulary Expansion and Tokenization
Context: The vocabulary is expanded from 704 to 786 tokens to accommodate Russian characters.
Code Alignment:
The merge_embeddings function iterates over the new tokenizer’s vocabulary (Russian) and maps each token to its Latin transliteration.
The code uses tokenizer_target.tokenizer.get_vocab() to get the new vocabulary size and token IDs.
Key Concept: The code dynamically assigns new embeddings to Russian tokens by leveraging the existing English embeddings, ensuring that the model can handle Russian text without retraining.
5. Weighting Strategies
Context: The approach uses descending linear weights to prioritize leading subtokens and bilateral weights to emphasize energy and similarity.
Code Alignment:
descending_linear_weights gives higher weights to earlier subtokens.
bilateral_harmonic_weights and optimized_bilateral_weights_with_decay use cosine similarity and energy scaling to refine the weighting.
positional_harmonic_graft applies a time bias to vectors to simulate future context, enhancing the representation.
Key Concept: These weighting strategies ensure that the grafted embeddings are more representative of the phonetic structure of Russian, especially for words with complex subtoken structures.
6. Efficiency & Zero-Shot Extension
Context: The goal is to demonstrate that this method is a viable alternative to “Vocabulary Expansion + Continued Pre-training,” significantly reducing compute.
Code Alignment:
The code avoids continued pre-training by directly grafting embeddings.
It uses efficient operations like slerp and weighted averaging to merge embeddings.
Key Concept: The method achieves zero-shot script extension with minimal additional computation, directly supporting the efficiency goal.
7. Integration with PolyIPA & Multilingual Tasks
Context: PolyIPA is a multilingual phoneme-to-grapheme conversion model, which shares the goal of handling multiple languages and scripts.
Code Alignment:
The transliteration process (transliterate_ru_to_latin) mirrors the idea of mapping between scripts.
The code’s use of harmonic_phonetic_map and slerp aligns with PolyIPA’s focus on phonetic accuracy and multilingual compatibility.
Key Concept: The approach can be seen as an extension of PolyIPA’s transliteration principles to TTS model adaptation.
8. Prosodic Transfer
Context: The hypothesis suggests that the “musicality” and emotional range of the English model will transfer to Russian.
Code Alignment:
The use of text_head.weight alongside text_emb.weight suggests attention to prosody and linguistic structure.
The positional_harmonic_graft function introduces a temporal bias that may help preserve prosodic features.
Key Concept: While not explicitly modeled, the embedding grafting process aims to preserve the prosodic richness of the original model.
Summary:
The code directly implements the core ideas of the hypothesis and research context:

SLERP-based embedding grafting to extend the model to Russian.
Efficient vocabulary expansion without extensive retraining.
Weighted merging of embeddings to prioritize phonetic signal.
Zero-shot script extension using transliteration and embedding transfer.
It aligns with the PolyIPA framework’s goals of multilingual compatibility and phonetic accuracy, while leveraging the T3 architecture’s strengths in TTS synthesis.

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