import sys

import torch
import torchaudio as ta
from pathlib import Path
from safetensors.torch import load_file

# Internal Chatterbox imports
from chatterbox.tts import ChatterboxTTS, Conditionals
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.tokenizers.tokenizer import EnTokenizer

# --- 1. CONFIGURATION ---

from config import load_config

# --- 2. MODEL BUILDING ---
def get_best_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # If you have multiple GPUs, pick the one with the most total memory
    device_count = torch.cuda.device_count()
    if device_count > 1:
        best_vram = 0
        best_idx = 0
        for i in range(device_count):
            vram = torch.cuda.get_device_properties(i).total_memory
            if vram > best_vram:
                best_vram = vram
                best_idx = i
        device = torch.device(f"cuda:{best_idx}")
    else:
        device = torch.device("cuda:0")
        
    device_name = torch.cuda.get_device_name(device)
    print(f"🎯 Selected Device: {device_name}")
    if "AMD" in device_name:
        print(f"✅ ROCm Backend Active.")
        # Force MIOpen to use a specific workspace size (in MB)
        import os
        # Tell MIOpen to find and cache the best kernels immediately
        os.environ["MIOPEN_FIND_ENFORCE"] = "1"
        # Increase the limit for device memory allocation
        os.environ["MIOPEN_DEVICE_MEMORY_MAX"] = "100" # Use 100% of available slice
        os.environ["MIOPEN_DEBUG_CONVOLUTION_MAX_WORKSPACE"] = "2048" # Increase workspace
        os.environ["MIOPEN_FIND_ENFORCE"] = "1" # Force kernel search
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    return device


def build_model(config: dict):
    """Assembles the TTS model on the detected ROCm device."""
    m_cfg = config.get('merge_embeddings', {})
    
    ckpt_dir = Path(m_cfg.get('ckpt_dir', './data/temp_model'))
    tokenizer_path = m_cfg.get('target_tokenizer', './data/en_ru_tokenizer.json')
    t3_chk_name = m_cfg.get('output_checkpoint', 't3_cfg.safetensors').split('/')[-1]
    
    # ROCm identifies as 'cuda' in code
    # In your build_model function:
    device = get_best_device()

    # 1. Tokenizer & Dynamic Vocab
    tokenizer = EnTokenizer(tokenizer_path)
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    print(f"📊 Vocabulary Size: {vocab_size}")

    # 2. Configure T3
    hp = T3Config.english_only()
    hp.text_tokens_dict_size = vocab_size

    # 3. Load Components
    print(f"📦 Loading weights from {ckpt_dir}...")
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    
    
    t3 = T3(hp=hp)
    t3.load_state_dict(load_file(ckpt_dir / t3_chk_name))
    t3= t3.to(device)
    
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen = s3gen.to(device)
    
    # 4. Handle Conditionals
    conds = None
    voice_path = ckpt_dir / "conds.pt"
    if voice_path.exists():
        conds = Conditionals.load(voice_path, map_location=device)
    conds = conds.to(device)
    if conds is not None and hasattr(conds, 't3'):
        conds.t3.emotion_adv = conds.t3.emotion_adv.to(device)

    # 5. Assembly
    t3 = torch.compile(t3)
    s3gen = torch.compile(s3gen)
    ve = torch.compile(ve)
    model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds)
    return model

# --- 3. ENTRY POINT ---

def main():
    config = load_config()
    model = build_model(config)
    
    # Text for synthesis
    text = "меня зовут андрей иванов. я живу в тайланде на острове пхукет уже целый год"
    
    print("⚡ Warming up ROCm kernels...")
    # Brief warmup to avoid JIT lag during the main generation
    with torch.inference_mode():
        _ = model.generate(text="warmup")
    
    print(f"🎙️ Generating main audio...")
    # SDPA context is crucial for AMD performance
    with torch.inference_mode():
        wav = model.generate(text=text)
    
    # Save results
    output_dir = Path("./data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "output_rocm.mp3"
    
    ta.save(str(save_path), wav.cpu(), model.sr)
    print(f"🎉 Success! Audio saved to: {save_path}")

if __name__ == "__main__":
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)