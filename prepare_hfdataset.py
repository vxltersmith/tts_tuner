import os
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
import torch

# --- Configuration ---
DATASET_NAME = "Vikhrmodels/ToneSpeak"
SPLIT = "train"
OUTPUT_DIR = "./chatterbox-finetuning/MyTTSDataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
TARGET_SR = 16000  # Chatterbox standard

# Create folders
os.makedirs(WAV_DIR, exist_ok=True)

# 1. Load HF Dataset
print(f"Loading {DATASET_NAME}...")
ds = load_dataset(DATASET_NAME, split=SPLIT)

# 2. Process and Save
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(ds, desc="Converting to LJSpeech")):
        # Get data (adjust keys based on your specific HF dataset card)
        audio_array = item["audio"]["array"]
        orig_sr = item["audio"]["sampling_rate"]
        text = item["text"].replace("\n", " ") # Remove newlines for CSV safety
        
        file_id = f"tonespeak_{i:06d}"
        wav_filename = f"{file_id}.wav"
        wav_path = os.path.join(WAV_DIR, wav_filename)
        text_path = os.path.join(WAV_DIR, f"{file_id}.txt")

        # Convert to Tensor and Resample
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
        if orig_sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
            waveform = resampler(waveform)

        # Save WAV
        torchaudio.save(wav_path, waveform, TARGET_SR)
        with open(text_path, "w") as tf:
            tf.write(text)

        # Write Metadata (format: ID|Text|Text)
        f.write(f"{file_id}|{text}|{text}\n")

print(f"Done! Dataset ready at {OUTPUT_DIR}")