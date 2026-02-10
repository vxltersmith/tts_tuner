import whisper
from datasets import load_dataset
from Levenshtein import distance as lev_dist

def calculate_metrics(original, transcribed):
    # Levenshtein Distance
    d = lev_dist(original, transcribed)
    # CER (Character Error Rate)
    cer = d / max(len(original), 1)
    return d, cer

def run_pipeline():   
    # 2. Initialize ASR (Tiny model for least dependencies/resources)
    print("Loading Whisper 'tiny' model...")
    asr_model = whisper.load_model("tiny")

    # 3. Iterate over dataset
    for i, item in enumerate(ds):
        input_text = item['text'] # Adjust key if dataset schema varies
        print(f"\n--- Entry {i+1} ---")
        print(f"Input: {input_text}")

        # --- INJECTION POINT: YOUR TTS ---
        # audio_path = your_tts_module.synthesize(input_text)
        # For now, we assume your TTS saves to "temp_output.wav"
        audio_path = "temp_output.wav" 
        
        # 4. ASR Integration (Convert back to text)
        # Whisper handles the audio loading internally via ffmpeg
        result = asr_model.transcribe(audio_path)
        transcribed_text = result["text"].strip()
        print(f"ASR Output: {transcribed_text}")

        # 5. Compare strings
        dist, cer = calculate_metrics(input_text, transcribed_text)
        print(f"Levenshtein: {dist} | CER: {cer:.2%}")

        if i >= 4: # Limit for testing
            break

if __name__ == "__main__":
    run_pipeline()