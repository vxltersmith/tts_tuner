import torch
import torchaudio as ta
import whisper


# --- 2. TTS CORE LOGIC ---
from run_tts import build_model
from utils import get_dataset, calculate_metrics
from config import load_config

def run_pipeline():
    config = load_config()
    tts_model = build_model(config)
    save_path = config.get("validation", {}).get("save_path", "./data/temp.wav")
    
    val_conf = config.get("validation", {})
    
    whisper_ckp = val_conf.get("whisper_model")
    print(f"Loading Whisper {whisper_ckp} model...")
    asr_model = whisper.load_model(whisper_ckp)
    asr_model = torch.compile(asr_model)
    
    print("Loading dataset...")
    ds = get_dataset(config)
    
    outputs_cer = []
    outputs_wer = []
    outputs_rel_cer = []
    outputs_rel_wer = []

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
            print(f"Saved audio to {save_path}")

        audio_Wav_16k = resampler(torch.tensor(audio_input, dtype=torch.float32))
        audio_Wav_16k = audio_Wav_16k.squeeze().numpy()
        asr_result = asr_model.transcribe(audio_Wav_16k, language="ru")
        
        transcribed_text = tts_asr_result["text"].strip()
        transcribed_text_orig = asr_result["text"].strip()
        print(f"Input: {input_text}")
        print(f"TTS-ASR Output: {transcribed_text}")
        print(f"Audio-ASR Output: {transcribed_text_orig}")

        # 3. Calculate metric
        wer_rel, cer_rel = calculate_metrics(transcribed_text_orig, transcribed_text)
        wer, cer = calculate_metrics(input_text, transcribed_text)
        print(f"WER: {wer:.2%} | CER: {cer:.2%}")
        print(f"Relative WER: {wer_rel:.2%} | CER: {cer_rel:.2%}")
        outputs_cer.append(cer)
        outputs_wer.append(wer)
        outputs_rel_cer.append(cer_rel)
        outputs_rel_wer.append(wer_rel)
        
        continue

    avg_cer = sum(results) / len(results)
    print(f"\n✅ Validation Finished. Average CER: {avg_cer:.2%}")

if __name__ == "__main__":
    run_pipeline()