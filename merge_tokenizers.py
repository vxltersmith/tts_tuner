from chatterbox.models.tokenizers.tokenizer import EnTokenizer
from tokenizers import Tokenizer
import json

# Configuration
CONFIG = {
    "tokenizer_paths": {
        "ru_tokenizer": "./data/ru_bpe_2000/tokenizer.json",
        "en_tokenizer": "./data/temp_model/tokenizer.json",
        "output_tokenizer": "./data/en_ru_tokenizer.json"
    },
    "shared_vocab_size": 704,
    "test_string": "привет"
}

def load_tokenizer(path):
    """Load tokenizer from file."""
    return Tokenizer.from_file(path)

def merge_bpe_merges(merges1, merges2):
    """
    Merge two BPE merge lists:
    - merges1: list of strings like "a b"
    - merges2: list of lists like ['a', 'b']
    Returns: list of strings in format "a b", no duplicates, order preserved.
    """
    seen = set()
    merged = []

    # Add merges1
    for merge in merges1:
        if merge not in seen:
            seen.add(merge)
            merged.append(merge)

    # Add merges2 (convert ['a', 'b'] -> "a b")
    for pair in merges2:
        merge_str = " ".join(pair)
        if merge_str not in seen:
            seen.add(merge_str)
            merged.append(merge_str)

    return merged

def merge_vocabs(vocab1, vocab2, offset):
    """Merge two vocabularies, preserving order and avoiding duplicates."""
    new_vocab = {}
    next_id = 0

    # Add all tokens from vocab1 first
    for token in vocab1:
        if token not in new_vocab:
            new_vocab[token] = next_id
            next_id += 1

    # Add tokens from vocab2, skipping duplicates
    for token in vocab2:
        if token not in new_vocab:
            new_vocab[token] = next_id
            next_id += 1

    return new_vocab

def main():
    # Load tokenizers
    ru_tokenizer = load_tokenizer(CONFIG["tokenizer_paths"]["ru_tokenizer"])
    en_tokenizer = EnTokenizer(CONFIG["tokenizer_paths"]["en_tokenizer"])

    print("✅ Tokenizers loaded successfully!")
    print(f"📊 Vocabulary sizes: {ru_tokenizer.get_vocab_size()} and {en_tokenizer.tokenizer.get_vocab_size()}")

    # Test encoding/decoding
    ids = ru_tokenizer.encode(CONFIG["test_string"]).ids
    print(f"Test string '{CONFIG['test_string']}' encoded as: {ids}")
    print(f"Decoded back: {ru_tokenizer.decode(ids)}")

    # Load tokenizer states
    with open(CONFIG["tokenizer_paths"]["en_tokenizer"], "r") as f:
        state = json.load(f)
    state2 = json.loads(ru_tokenizer.__getstate__())

    # Get vocabularies and merges
    vocab = state["model"]["vocab"]
    vocab2 = state2["model"]["vocab"]
    merges1 = state["model"]["merges"]
    merges2 = state2["model"]["merges"]

    # Adjust indices for second vocabulary
    for key in vocab2.keys():
        vocab2[key] += CONFIG["shared_vocab_size"]

    # Merge vocabularies and merges
    state["model"]["vocab"] = merge_vocabs(vocab, vocab2, CONFIG["shared_vocab_size"])
    state["model"]["merges"] = merge_bpe_merges(merges1, merges2)

    # Save merged tokenizer
    with open(CONFIG["tokenizer_paths"]["output_tokenizer"], "w") as f:
        json.dump(state, f)

    print(f"✅ Merged tokenizer saved to {CONFIG['tokenizer_paths']['output_tokenizer']}")
    
    tokenizer_new = EnTokenizer(CONFIG['tokenizer_paths']['output_tokenizer'])
    print(tokenizer_new.tokenizer.get_vocab())

if __name__ == "__main__":
    main()