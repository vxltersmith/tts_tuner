# TTS Tuner
 ## Project Summary: 
 Phonetic Latent Grafting (PLG) for Zero-Shot Multilingual TTS
 ## Objective
   This project introduces Phonetic Latent Grafting (PLG), a novel framework for rapid, low-resource lingual adaptation of large-scale, decoder-only TTS models (e.g., Llama-3-Speech or VALL-E). PLG enables zero-shot extension to new scripts and languages by leveraging pre-trained acoustic priors and phonetic manifold alignment, without requiring extensive retraining or large-scale datasets.


## Quick Start

```bash
# Set up your environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download model weights
python get_models.py

# Generate speech from text
python run_tts.py
```

## Installation

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (AMD ROCm supported)
- 8GB+ RAM

### Setup Steps

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure paths**
   Edit `config.yaml` to set your preferred paths for models, tokenizers, and output files.

4. **Download models** (optional if already done)
   ```bash
   python get_models.py
   ```

## Tools & Usage

### 1. Generate Speech (`run_tts.py`)

Generate speech from text using the configured model.

**Basic Usage:**
```bash
python run_tts.py
```

**Customization:**
Edit the text in `run_tts.py` lines 88-89, or modify config values in `config.yaml['run_tts']`.

**Supported Features:**
- Russian language text (configurable)
- MP3 output generation
- GPU acceleration (CUDA/ROCm)
- Automatic conditioning for voice synthesis

**Output Location:**
`./data/outputs/output_rocm.mp3`

---

### 2. Train Tokenizer (`train_tokenizer.py`)

Create custom BPE tokenizers for enhanced vocabulary support.

**Purpose:**
Train tokenizers on your custom datasets for improved text-to-speech accuracy.

**Usage:**
```bash
python train_tokenizer.py
```

**Configuration:**
Edit `config.yaml['train_tokenizer']` to customize:
- `dataset`: HuggingFace dataset name or local path
- `split`: Dataset split to use (train/test)
- `vocab_size`: Target vocabulary size
- `min_frequency`: Minimum token frequency threshold

**Output Location:**
`./data/ru_bpe_2000/tokenizer.json`

---

### 3. Merge Embeddings (`merge_embeddings.py`)

Combine embeddings from multiple tokenizers for bilingual support.

**Purpose:**
Merge English and trained tokenizers to support multilingual synthesis.

**Usage:**
```bash
python merge_embeddings.py
```

**Configuration:**
Edit `config.yaml['merge_embeddings']` for custom behavior:
- `target_tokenizer`: Output tokenizer path
- `source_tokenizer`: Source tokenizer to merge
- `share_vocab_size`: Size of shared vocabulary
- Merge parameters (sigma, p, alpha)

**Output Location:**
`./data/temp_model/t3_cfg_ru_harm_bilat.safetensors`

---

### 4. Download Models (`get_models.py`)

Retrieve pre-trained model weights from HuggingFace.

**Usage:**
```bash
python get_models.py
```

**What it Downloads:**
- Voice Encoder weights
- T3 model checkpoints
- S3Gen models
- Tokenizers
- Conditionals

---

## Configuration Guide

### Project Structure

```
├── config.yaml              # Main configuration file
├── run_tts.py               # Generate speech from text
├── train_tokenizer.py       # Train custom BPE tokenizers
├── merge_embeddings.py      # Merge tokenizers for multilingual support
├── get_models.py            # Download pre-trained models
└── src/chatterbox/          # Internal model components
```

### Configuration File Overview

**Paths (`config.yaml['paths']`)**:
- `model_dir`: Directory containing model weights
- `tokenizer_en`: English tokenizer path
- `tokenizer_ru`: Russian tokenizer path
- `output_dir`: Directory for generated audio
- `checkpoint_dir`: Directory for model checkpoints

**Models (`config.yaml['models']`)**:
- `voice_encoder`: Voice encoder model path
- `t3_checkpoint`: T3 model checkpoint
- `s3gen_checkpoint`: S3Gen model checkpoint
- `voice_conditional`: Voice conditioning data
- `tokenizer_source`: Source tokenizer path

**TTS Settings (`config.yaml['tts']`)**:
- `text_tokens_dict_size`: Size of T3 text tokens dictionary
- `device`: Compute device (cuda/cpu)

### Key Configuration Parameters

```yaml
tts:
  device: "cuda"              # GPU/cuda, CPU for offline
  text_tokens_dict_size: 786  # Must match your vocabulary size

train_tokenizer:
  vocab_size: 2000            # Target vocabulary size
  min_frequency: 10           # Minimum token frequency

merge_embeddings:
  share_vocab_size: 704       # Size of shared vocabulary
  sigma_r: 0.4                # Merge similarity threshold
  p: 3.0                      # Merge parameter
  alpha: 0.6                  # Merge balance factor
```

---

## Common Workflows

### Workflow 1: Custom Text Generation

1. Edit `run_tts.py` line 88
2. Run: `python run_tts.py`
3. Check output in `./data/outputs/`

### Workflow 2: Training Custom Tokenizer

1. Prepare custom dataset (TXT or HF format)
2. Edit `config.yaml['train_tokenizer']['dataset']`
3. Update vocab size in `config.yaml['tokenizer']['vocab_size']`
4. Run: `python train_tokenizer.py`
5. Use trained tokenizer in subsequent processes

### Workflow 3: Bilingual Support

1. Train tokenizer for target language
2. Run `merge_embeddings.py` to merge
3. Update config paths to use new merged checkpoint
4. Run `run_tts.py` with multilingual text

### Workflow 4: Offline Generation (CPU)

1. Set `device: "cpu"` in `config.yaml['tts']`
2. Disable GPU features in `run_tts.py` (if needed)
3. Run: `python run_tts.py`

---

## Troubleshooting

### Audio Output Issues

**Problem**: No audio generated or empty output file
- Verify text input is valid Russian text
- Check `output_dir` has write permissions
- Ensure model files are properly downloaded

### Memory Issues

**Problem**: Out of memory during generation
- Use CPU mode (`device: "cpu"` in config)
- Reduce batch size or vocabulary size
- Close other GPU-intensive applications

### Tokenizer Errors

**Problem**: Vocabulary size mismatch
- Ensure `text_tokens_dict_size` in config matches the tokenizer vocab size
- Regenerate embeddings after vocabulary changes

### Model Loading Failures

**Problem**: Can't load model weights
- Confirm paths in `config.yaml` are correct
- Verify model files exist in `model_dir`
- Re-run `python get_models.py`

---

## Project Structure

### Core Scripts

- **`run_tts.py`**: Entry point for speech generation. Loads models, processes text, saves audio output.
- **`train_tokenizer.py`**: BPE tokenizer training on custom datasets for enhanced vocabulary.
- **`merge_embeddings.py`**: Combines multiple tokenizers for multilingual support.
- **`get_models.py`**: Downloads pre-trained model weights from HuggingFace.

### Internal Modules (`src/chatterbox/`)

Contains model implementations:
- **T3**: Transformer text encoder
- **S3Gen**: Waveform generator
- **VoiceEncoder**: Voice conditioning module
- **Tokenizers**: Text encoding/decoding

### Configuration

- **`config.yaml`**: Central configuration file for paths, models, and parameters

### Data Directories

- **`./data/temp_model`**: Contains model weights and temp files
- **`./data/outputs`**: Generated audio output
- **`./data/en_ru_tokenizer.json`**: Merged multilingual tokenizer
- **`./data/ru_bpe_2000/`**: Trained custom tokenizers
- **`./hf_cache`**: HuggingFace model cache

---

## Tips & Best Practices

### Vocabulary Management

1. Match `text_tokens_dict_size` with your tokenizer vocab size in all scripts
2. Regenerate checkpoints after changing vocabulary size
3. Keep tokenizer files in consistent locations

### Model Organization

1. Use `config.yaml` for all paths - avoid hardcoding
2. Organize model weights in dedicated directories
3. Backup custom trained tokenizers for reuse

### Performance Optimization

1. Use GPU (`cuda`) for fastest generation
2. Enable SDPA attention for AMD ROCm compatibility
3. Warm up models before generation:

```bash
# The script automatically handles warmup
```

### Language Support

1. Ensure text is formatted correctly for target language
2. Use appropriate tokenizer for language
3. Test with known-good text before production use

---

## File Locations Reference

| Component | Default Location | Configuration Key |
|-----------|------------------|--------------------|
| Models | `./data/temp_model/` | `paths.model_dir` |
| Tokenizers | `./data/en_ru_tokenizer.json` | `paths.tokenizer_en` |
| Output Audio | `./data/outputs/` | `paths.output_dir` |
| Training Data | `./data/ru_bpe_2000/` | `paths.checkpoint_dir` |
| HF Cache | `./data/` | `train_tokenizer.cache_dir` |

---

## License

NDA