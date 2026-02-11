from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
from pathlib import Path
import os
import yaml
import tempfile
import sys


def load_config(config_path  = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_text_data(dataset, name="text"):
    """Extract text field from dataset examples."""
    return {"text": dataset[name]}


def train_bpe_tokenizer(config):
    """Train BPE tokenizer on Russian text dataset."""
    dataset_name = config['train_tokenizer']['dataset']
    split_name = config['train_tokenizer']['split']
    min_frequency = config['train_tokenizer']['min_frequency']
    temp_dir = config['train_tokenizer']['temp_dir']
    output_dir = Path(config['train_tokenizer']['output_dir'])

    print(f"📥 Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split=split_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load '{dataset_name}' dataset. "
                          f"Make sure you're logged in to Hugging Face. Error: {e}")

    print(f"✅ Loaded {len(dataset)} examples")

    print(f"🧹 Extracting text from dataset...")
    texts = dataset.map(
        lambda x: extract_text_data(x),
        num_proc=4,
        remove_columns=dataset.column_names
    )

    print(f"🧹 Saving Russian text to temporary file...")
    temp_file = Path(temp_dir) / "ru.txt"
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_file, "w", encoding='utf-8') as f:
        for i, example in enumerate(texts):
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
            if i % 5000 == 0 and i > 0:
                print(f"📝 Written {i} lines")

    print(f"✅ Text saved to {temp_file}")

    print(f"🚀 Training BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = config['tokenizer']['special_tokens']
    vocab_size = config['tokenizer']['vocab_size']

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=[str(temp_file)], trainer=trainer)

    print(f"📊 Saving tokenizer to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    tokenizer.model.save(str(output_dir))

    final_vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Tokenizer trained and saved. Final vocabulary size: {final_vocab_size}")

    if temp_file.exists():
        temp_file.unlink()
    print(f"🧹 Temporary file removed")


def main():
    try:
        config_path = "./config.yaml"
        config = load_config(config_path)
        train_bpe_tokenizer(config)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()