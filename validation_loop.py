import os
import sys
sys.modules["torchcodec"] = None
import torch
import torchaudio as ta
import whisper
import yaml
from pathlib import Path
from Levenshtein import distance as lev_dist
import torchaudio as ta

import datasets
datasets.config.USE_TORCHCODEC = False # Redundant safety

# ROCm GFX Override for NucBox (7900 XT / 780M)
if torch.version.hip:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

# --- 2. TTS CORE LOGIC ---
from run_tts import build_model

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) 

def get_dataset(config):
    # Fix: Ensure we use the full path to avoid the "Dataset scripts no longer supported" error
    validateion_conf = config.get("validation", {})
    dataset_key = validateion_conf.get("dataset_key", "Vikhrmodels/ToneSpeak")
    split_key = validateion_conf.get("split", "train")
    ds = datasets.load_dataset(dataset_key, split=split_key)
    #ds = ds.cast_column("audio", datasets.Value("string"))
    return ds

# --- 3. METRICS & PIPELINE ---

def calculate_metrics(original, transcribed):
    orig = original.lower().strip().replace(".", "").replace(",", "")
    tran = transcribed.lower().strip().replace(".", "").replace(",", "")
    d = lev_dist(orig, tran)
    cer = d / max(len(orig), 1)
    return d, cer

def run_pipeline():
    config = load_config()
    tts_model = build_model(config)
    save_path = config.get("validation", {}).get("save_path", "./data/temp.wav")
    
    
    print("Loading Whisper 'tiny' model...")
    asr_model = whisper.load_model("tiny")
    asr_model = torch.compile(asr_model)
    
    print("Loading dataset...")
    ds = get_dataset(config)
    results = []

    # Whisper expects 16,000Hz. If your model is different, we resample.
    target_sr = 16000 
    resampler = ta.transforms.Resample(orig_freq=tts_model.sr, new_freq=target_sr)

    for i, item in enumerate(ds):
        input_text = item['text']
        audio_input = item['audio']['array']
        print(f"\n--- Entry {i+1} ---")

        # 1. Synthesize Audio
        with torch.inference_mode():
            wav = tts_model.generate(text=input_text).cpu()
        
        # 2. Whisper
        tts_wav_16k = resampler(wav)
        tts_wav_16k = tts_wav_16k.squeeze().numpy()
        tts_asr_result = asr_model.transcribe(tts_wav_16k, language="ru")
        if save_path is not None:
            ta.save(save_path, wav.cpu(), tts_model.sr)
        
        audio_Wav_16k = resampler(torch.tensor(audio_input, dtype=torch.float32))
        audio_Wav_16k = audio_Wav_16k.squeeze().numpy()
        asr_result = asr_model.transcribe(audio_Wav_16k, language="ru")
        
        transcribed_text = tts_asr_result["text"].strip()
        transcribed_text_orig = asr_result["text"].strip()
        print(f"Input: {input_text}")
        print(f"TTS-ASR Output: {transcribed_text}")
        print(f"Audio-ASR Output: {transcribed_text_orig}")

        # 3. Calculate metric
        dist_rel, cer_rel = calculate_metrics(transcribed_text_orig, transcribed_text)
        dist, cer = calculate_metrics(input_text, transcribed_text)
        print(f"Levenshtein: {dist} | CER: {cer:.2%}")
        print(f"Relative:: Levenshtein: {dist_rel} | CER: {cer_rel:.2%}")
        results.append(cer)

        if i >= 9: break

    avg_cer = sum(results) / len(results)
    print(f"\n✅ Validation Finished. Average CER: {avg_cer:.2%}")

if __name__ == "__main__":
    run_pipeline()