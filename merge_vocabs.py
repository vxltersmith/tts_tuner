from chatterbox.models.tokenizers.tokenizer import EnTokenizer
from tokenizers import Tokenizer
import json
from config import load_config

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
        new_vocab[token] = next_id
        next_id += 1

    for token in vocab2:
        if token in new_vocab: continue
        
        new_vocab[token] = next_id
        next_id += 1

    return new_vocab

def main():
    config = load_config()
    merge_vocabs_conf = config.get("merge_vocabs")
    ru_tokenizer = merge_vocabs_conf["tokenizer_ru"]
    en_tokenizer_path = merge_vocabs_conf["tokenizer_en"]
    output_tokenizer = merge_vocabs_conf["tokenizer_output"]
    cut_vocab_offset = merge_vocabs_conf.get("cut_vocab_offset", None)

    # Load tokenizers
    ru_tokenizer = load_tokenizer(ru_tokenizer)
    en_tokenizer = EnTokenizer(en_tokenizer_path)
    
    shared_vocab_size = en_tokenizer.tokenizer.get_vocab_size()

    print("✅ Tokenizers loaded successfully!")
    print(f"📊 Vocabulary sizes: {ru_tokenizer.get_vocab_size()} and {en_tokenizer.tokenizer.get_vocab_size()}")

    # Test encoding/decoding
    test_string = merge_vocabs_conf["test_string"]
    test_ids = ru_tokenizer.encode(test_string).ids
    print(f"Ru tokenizer: Test string '{merge_vocabs_conf['test_string']}' encoded as: {test_ids}")
    print(f"Decoded back: {ru_tokenizer.decode(test_ids)}")

    # Load tokenizer states
    with open(en_tokenizer_path, "r") as f:
        state = json.load(f)
    state2 = json.loads(ru_tokenizer.__getstate__())

    # Get vocabularies and merges
    vocab = state["model"]["vocab"]
    vocab2 = state2["model"]["vocab"]
    
    merges1 = state["model"]["merges"]
    merges2 = state2["model"]["merges"]

    if cut_vocab_offset is not None:
        new_tokens_list = list(vocab2.keys())[cut_vocab_offset:]
        # If you need to map them back to their IDs for the merger loop:
        vocab2 = {k: vocab2[k] for k in new_tokens_list}
    
    # Merge vocabularies and merges
    state["model"]["vocab"] = merge_vocabs(vocab, vocab2, shared_vocab_size)
    state["model"]["merges"] = merge_bpe_merges(merges1, merges2)

    # Save merged tokenizer
    with open(output_tokenizer, "w") as f:
        json.dump(state, f)

    print(f"✅ Merged tokenizer saved to {output_tokenizer}")
    
    tokenizer_new = EnTokenizer(output_tokenizer)
    print(tokenizer_new.tokenizer.get_vocab())
    
    # Check a few Russian words manually
    test_words = ['привет', 'ближай', 'ручья', 'синхрофазотрон']
    for word in test_words:
        idx = tokenizer_new.encode(word)
        print(f"{word} tokenized as : {idx}")
        
    test_string_decoded = tokenizer_new.decode(test_ids)
    test_new_ids = tokenizer_new.encode(test_string)
    print(f"test_ids {test_ids} from ru tokenizer decoded be new tokenizer as: {test_string_decoded}")
    print(f"test string '{test_string}' encoded as :{test_new_ids}")
    
    print("✅ Done!")

if __name__ == "__main__":
    main()