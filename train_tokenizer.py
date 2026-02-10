from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
import os
import tempfile

# -------------------------------
# 1. Load the dataset
# -------------------------------

print("📥 Loading the dataset")

try:
    dataset = load_dataset("Vikhrmodels/ToneSpeak", split="train")
except Exception as e:
    raise RuntimeError(f"Failed to load 'ToneSpeak' dataset. Make sure you're logged in to Hugging Face. Error: {e}")

print(f"✅ Loaded {len(dataset)} examples.")

# Extract text 

def extract_text(example, name="text"):
    return {"text": example[name]}

# Apply mapping to extract text
print("🧹 Extracting text with tashkeel...")
texts = dataset.map(extract_text, num_proc=4, remove_columns=dataset.column_names)


# Save to temporary file for tokenizer training
temp_dir = tempfile.mkdtemp()
temp_file = os.path.join(temp_dir, "ru.txt")

print("🧹 Saving Russian text to file...")

with open(temp_file, "w", encoding="utf-8") as f:
    for i in range(len(texts)):
        example = texts[i]  # This ensures we get a dict
        text = example["text"].strip()
        if text:
            f.write(text + "\n")
        if i % 5000 == 0:
            print(f"📝 Written {i} lines")

print(f"✅ Text saved to {temp_file}")

# -------------------------------
# 2. Initialize BPE Tokenizer
# -------------------------------

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# -------------------------------
# 3. Configure BPE Trainer (vocab_size = 2000)
# -------------------------------

trainer = trainers.BpeTrainer(
    vocab_size=2000,
    min_frequency=10,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    show_progress=True,
)

# -------------------------------
# 4. Train the Tokenizer
# -------------------------------

print("🚀 Training BPE tokenizer on Russian text ")

tokenizer.train(files=[temp_file], trainer=trainer)

# -------------------------------
# 5. Save the Tokenizer
# -------------------------------

output_dir = "./data/ru_bpe_2000"
os.makedirs(output_dir, exist_ok=True)
tokenizer.save(f"{output_dir}/tokenizer.json")
tokenizer.model.save(output_dir)  # Saves vocab.json and merges.txt

print(f"✅ Tokenizer trained and saved to '{output_dir}'")
print(f"📊 Final vocabulary size: {tokenizer.get_vocab_size()}")

# -------------------------------
# 7. Cleanup
# -------------------------------

import shutil
shutil.rmtree(temp_dir)
print(f"🧹 Temporary data directory removed.")