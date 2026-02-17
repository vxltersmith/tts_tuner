#!/bin/bash

# 1. Environment Preparation
echo "📂 Setting up environment in chatterbox-finetuning..."
cd chatterbox-finetuning || exit
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Core ROCm Stack (The Foundation)
# We install this first so it's the "Already Satisfied" requirement later
echo "🏎️ Installing ROCm 6.2 PyTorch Stack..."
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Bootstrap Local Chatterbox (Editable Link)
# We use --no-deps to prevent it from pulling in Torch 2.6.0
echo "🔗 Linking local Chatterbox..."
.venv/bin/pip install -e ../chatterbox --no-deps

# 4. Install Secondary Requirements
# This fills in peft, librosa, silero-vad, etc.
echo "📦 Installing remaining requirements..."
.venv/bin/pip install -r requirements.txt

# 5. Model Setup
# Trigger the pretrained model downloader/setup
echo "📥 Initializing Pretrained Models..."
.venv/bin/python setup.py