import sys
import yaml
import torch
import torchaudio as ta
import whisper
from pathlib import Path
from datasets import load_dataset
from Levenshtein import distance as lev_dist
from safetensors.torch import load_file

# Internal Chatterbox imports
from chatterbox.tts import ChatterboxTTS, Conditionals
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.tokenizers.tokenizer import EnTokenizer

# --- 1. TTS CORE LOGIC ---

def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_tts_model(config):
    m_cfg = config.get('merge_embeddings', {})
    ckpt_dir = Path(m_cfg.get('ckpt_dir', './data/temp_model'))
    tokenizer_path = m_cfg.get('target_tokenizer', './data/en_ru_tokenizer.json')
    t3_chk_name = m_cfg.get('output_checkpoint_name', 't3_cfg_ru_harm_bilat.safetensors')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = EnTokenizer(tokenizer_path)
    hp = T3Config.english_only()
    hp.text_tokens_dict_size = tokenizer.tokenizer.get_vocab_size()

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    t3 = T3(hp=hp)
    t3.load_state_dict(load_file(ckpt_dir / t3_chk_name))
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)

    conds = None
    voice_path = ckpt_dir / "conds.pt"
    if voice_path.exists():
        conds = Conditionals.load(voice_path, map_location=device)

    model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds)
    model.t3.to(device).eval()
    model.s3gen.to(device).eval()
    model.voice_encoder.to(device).eval()
    
    return model

# --- 2. METRICS & PIPELINE ---

def calculate_metrics(original, transcribed):
    # Normalize strings for fairer comparison (lowercase + strip punctuation)
    orig = original.lower().strip().replace(".", "").replace(",", "")
    tran = transcribed.lower().strip().replace(".", "").replace(",", "")
    
    d = lev_dist(orig, tran)
    cer = d / max(len(orig), 1)
    return d, cer

def run_pipeline():
    # Load Configuration and Build TTS
    config = load_config()
    tts_model = build_tts_model(config)
    
    # Initialize ASR
    print("Loading Whisper 'tiny' model...")
    asr_model = whisper.load_model("tiny")

    # Load Dataset (Adjust 'path' and 'name' as needed)
    print("Loading dataset...")
    ds = load_dataset("common_voice", "ru", split="test", streaming=True)

    results = []

    for i, item in enumerate(ds):
        input_text = item['sentence'] # 'sentence' is standard for Common Voice
        print(f"\n--- Entry {i+1} ---")
        print(f"Input: {input_text}")

        # 1. Synthesize Audio
        with torch.inference_mode(), torch.nn.attention.sdp_kernel(enable_flash=True):
            wav = tts_model.generate(text=input_text)
        
        # 2. Prepare for ASR (Whisper expects a temporary file or numpy array)
        # For simplicity and ROCm memory management, we save to a temporary buffer
        temp_wav = "temp_val.wav"
        ta.save(temp_wav, wav.cpu(), tts_model.sr)
        
        # 3. Transcribe back to text
        asr_result = asr_model.transcribe(temp_wav, language="ru")
        transcribed_text = asr_result["text"].strip()
        print(f"ASR Output: {transcribed_text}")

        # 4. Compare
        dist, cer = calculate_metrics(input_text, transcribed_text)
        print(f"Levenshtein: {dist} | CER: {cer:.2%}")
        
        results.append(cer)

        if i >= 9: # Run 10 samples for a solid average
            break

    avg_cer = sum(results) / len(results)
    print(f"\n✅ Validation Finished. Average CER: {avg_cer:.2%}")

if __name__ == "__main__":
    run_pipeline()