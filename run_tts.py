import sys
import yaml
import torch
import torchaudio as ta
from pathlib import Path
from safetensors.torch import load_file

# Internal Chatterbox imports
from src.chatterbox.tts import ChatterboxTTS, Conditionals
from src.chatterbox.models.t3.t3 import T3
from src.chatterbox.models.t3.modules.t3_config import T3Config
from src.chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from src.chatterbox.models.s3gen import S3Gen
from src.chatterbox.models.tokenizers.tokenizer import EnTokenizer

# --- 1. CONFIGURATION ---

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

# --- 2. MODEL BUILDING ---

def build_model(config: dict):
    """Assembles the TTS model on the detected ROCm device."""
    m_cfg = config.get('merge_embeddings', {})
    
    ckpt_dir = Path(m_cfg.get('ckpt_dir', './data/temp_model'))
    tokenizer_path = m_cfg.get('target_tokenizer', './data/en_ru_tokenizer.json')
    t3_chk_name = m_cfg.get('output_checkpoint_name', 't3_cfg_ru_harm_bilat.safetensors')
    
    # ROCm identifies as 'cuda' in code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ ROCm Backend Active. Using GPU: {torch.cuda.get_device_name(0)}")

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

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)

    # 4. Handle Conditionals
    conds = None
    voice_path = ckpt_dir / "conds.pt"
    if voice_path.exists():
        conds = Conditionals.load(voice_path, map_location=device)

    # 5. Assembly
    model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds)
    model.t3.to(device).eval()
    model.s3gen.to(device).eval()
    model.voice_encoder.to(device).eval()
    
    return model

# --- 3. ENTRY POINT ---

def main():
    try:
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
        with torch.inference_mode(), torch.nn.attention.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            wav = model.generate(text=text)
        
        # Save results
        output_dir = Path("./data/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "output_rocm.mp3"
        
        ta.save(str(save_path), wav.cpu(), model.sr)
        print(f"🎉 Success! Audio saved to: {save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()