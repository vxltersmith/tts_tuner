import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
import numpy as np
import json
import math
import re

# -----------------------------------------------------------------------------
# 1. Positional Encoding
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# -----------------------------------------------------------------------------
# 2. Simple Tokenizer Wrapper (No Transformers Dependency)
# -----------------------------------------------------------------------------
class SimpleTokenizer:
    """
    Simple tokenizer that works with vocab.json directly.
    Supports basic encoding/decoding without transformers library.
    """
    def __init__(self, vocab_path="./data/temp_model/tokenizer.json"):
        with open(vocab_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.vocab = state["model"]["vocab"]
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.unk_token = "<unk>"
        self.unk_id = self.vocab.get(self.unk_token, 0)

    def encode(self, text):
        """Encode text to token IDs using longest-match greedy approach."""
        ids = []
        i = 0
        sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)

        while i < len(text):
            matched = False
            for token in sorted_tokens:
                if text[i:i+len(token)] == token:
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(self.unk_id)
                i += 1
        return ids

    def decode(self, ids, clean_up_tokenization_spaces=False):
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(idx, self.unk_token) for idx in ids]
        text = ''.join(tokens)
        if clean_up_tokenization_spaces:
            text = text.replace(' ', '')
        return text

# -----------------------------------------------------------------------------
# 3. FOURIER MERGER (LEARNABLE LOW-PASS + SPECTRAL DOWNSAMPLING)
# -----------------------------------------------------------------------------

def gaussian_low_pass(num_freqs, sigma=1.0):
    k = torch.arange(num_freqs).float()
    mask = torch.exp(-(k**2) / (2 * (sigma**2)))
    return mask / mask.max() # Normalize to 1.0

class FourierMerger(nn.Module):
    """
    Merges subtoken embeddings using Fourier Domain Downsampling.
    1. Transforms sequence to frequency domain (FFT).
    2. Applies learnable Low-Pass Filter (attenuates high freq noise).
    3. Downsamples by keeping top-K frequencies.
    4. Projects back to embedding dimension.
    """
    def __init__(self, dim, max_seq_len=8, keep_freqs=4, ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.keep_freqs = keep_freqs

        # FFT of real signal of length L returns L//2 + 1 complex frequencies
        self.num_freqs = max_seq_len // 2 + 1

        # 1. LEARNABLE LOW-PASS MASK
        # Initialize with decay (1.0 for DC, decreasing for higher freqs)
        init_mask = torch.linspace(1.0, 0.1, self.num_freqs)
        self.spectral_mask = nn.Parameter(init_mask)

        # 2. LEARNABLE PROJECTION
        # Input: (keep_freqs * 2) * dim  (Real + Imag parts of kept frequencies)
        # Output: dim
        input_features = self.keep_freqs * 2 * dim
        self.proj = nn.Linear(input_features, dim, bias=False)

        # Initialize projection to preserve magnitude roughly
        nn.init.xavier_uniform_(self.proj.weight)

        # LayerNorm for stability after projection
        self.norm = nn.LayerNorm(dim)
        
    def set_projection_weights(self, W):
        # W is (8192, 1024)
        with torch.no_grad():
            self.proj.weight.copy_(W.t()) # Transpose to (1024, 8192)

    def forward(self, subtoken_embs, return_features=False):
        n = len(subtoken_embs)
        if n == 0:
            return torch.zeros(self.dim, device=subtoken_embs[0].device if subtoken_embs else 'cpu')
        if n == 1:
            return self.norm(self.proj(subtoken_embs[0].unsqueeze(0).repeat(self.keep_freqs * 2, 1).view(1, -1))).squeeze(0)

        # 1. Stack and Pad to max_seq_len for consistent FFT
        seq = torch.stack(subtoken_embs)  # (n, dim)
        device = seq.device

        # Padding
        if n < self.max_seq_len:
            pad_len = self.max_seq_len - n
            padding = torch.zeros(pad_len, self.dim, device=device)
            seq = torch.cat([seq, padding], dim=0)  # (max_seq_len, dim)

        # 2. FFT along Sequence Dimension (dim=0)
        # Result shape: (num_freqs, dim) complex
        seq_fft = torch.fft.rfft(seq, dim=0)

        # 3. Apply Learnable Low-Pass Filter
        # Expand mask to match (num_freqs, dim)
        mask = self.spectral_mask.unsqueeze(-1).to(device)
        mask = gaussian_low_pass(seq_fft.shape[0], 1).unsqueeze(-1).to(device)
        seq_fft_filtered = seq_fft * mask

        # 4. Spectral Downsampling (Keep Top-K Frequencies)
        # We take the first 'keep_freqs' frequencies (Lowest frequencies)
        if self.keep_freqs < self.num_freqs:
            seq_fft_downsampled = seq_fft_filtered[:self.keep_freqs, :]
        else:
            seq_fft_downsampled = seq_fft_filtered

        # 5. Flatten Real and Imaginary parts for Projection
        real_part = seq_fft_downsampled.real
        imag_part = seq_fft_downsampled.imag

        # Concatenate features
        features = torch.cat([real_part, imag_part], dim=0) # (keep_freqs * 2, dim)
        features = features.flatten().unsqueeze(0) # (1, keep_freqs * 2 * dim)
        if return_features:
            return features
        
        # 6. Project back to original dimension
        merged = self.proj(features) # (1, dim)

        # 7. Normalize
        merged = self.norm(merged)

        return merged.squeeze(0)

# -----------------------------------------------------------------------------
# 4. SLERP (Fallback)
# -----------------------------------------------------------------------------
def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)
    norm_v0 = torch.norm(v0)
    norm_v1 = torch.norm(v1)
    epsilon = 1e-10
    if norm_v0 < epsilon or norm_v1 < epsilon:
        return lerp(t, v0, v1)
    u0 = v0 / norm_v0
    u1 = v1 / norm_v1
    dot = torch.clamp(torch.sum(u0 * u1), -1.0, 1.0)
    if torch.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0, v1)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    if sin_theta_0 < epsilon:
        return lerp(t, v0, v1)
    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    return s0 * v0 + s1 * v1

def slerp_recursive(vectors):
    if len(vectors) == 0:
        return None
    if len(vectors) == 1:
        return vectors[0]
    if len(vectors) == 2:
        return slerp(0.5, vectors[0], vectors[1])
    merged = []
    for i in range(0, len(vectors), 2):
        if i + 1 < len(vectors):
            merged.append(slerp(0.5, vectors[i], vectors[i+1]))
        else:
            merged.append(vectors[i])
    return slerp_recursive(merged)

# -----------------------------------------------------------------------------
# 5. Load Vocab
# -----------------------------------------------------------------------------
def load_vocab_from_json(tokenizer_path="./data/temp_model/tokenizer.json"):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    vocab = state["model"]["vocab"]
    return vocab

# -----------------------------------------------------------------------------
# 6. Transliteration
# -----------------------------------------------------------------------------
def transliterate_ru_to_latin(text: str) -> str:
    """
    Transliterates Russian Cyrillic to English-friendly phonetic spelling.
    Uses a two-stage mapping: Russian → IPA → English Phonetic

    This provides more accurate pronunciation representation for tokenizer compatibility.
    """

    # Stage 1: Russian Cyrillic → IPA (International Phonetic Alphabet)
    cyrillic_to_ipa = {
        # Vowels
        'а': 'a',      # /a/
        'о': 'o',      # /o/
        'у': 'u',      # /u/
        'э': 'e',      # /e/
        'ы': 'i',      # /ɨ/ → simplified to /i/
        'и': 'i',      # /i/
        'е': 'ye',     # /je/ after consonants, /e/ elsewhere
        'ё': 'yo',     # /jo/
        'ю': 'yu',     # /ju/
        'я': 'ya',     # /ja/

        # Consonants - Voiced
        'б': 'b',      # /b/
        'в': 'v',      # /v/
        'г': 'g',      # /g/
        'д': 'd',      # /d/
        'ж': 'zh',     # /ʒ/
        'з': 'z',      # /z/
        'л': 'l',      # /l/
        'м': 'm',      # /m/
        'н': 'n',      # /n/
        'р': 'r',      # /r/

        # Consonants - Voiceless
        'п': 'p',      # /p/
        'ф': 'f',      # /f/
        'к': 'k',      # /k/
        'т': 't',      # /t/
        'ш': 'sh',     # /ʃ/
        'щ': 'shh',    # /ɕː/ → long sh
        'с': 's',      # /s/
        'х': 'kh',     # /x/ → guttural h
        'ц': 'ts',     # /ts/
        'ч': 'ch',     # /tʃ/

        # Semi-vowels
        'й': 'y',      # /j/

        # Silent markers
        'ъ': '',       # hard sign (silent)
        'ь': '',       # soft sign (palatalizes previous, silent)
    }

    # Uppercase mapping
    cyrillic_to_ipa_upper = {k.upper(): v.capitalize() if v else '' 
                              for k, v in cyrillic_to_ipa.items()}
    cyrillic_to_ipa.update(cyrillic_to_ipa_upper)

    # Stage 2: IPA → English Phonetic Optimizations
    # These rules make the IPA more compatible with English tokenizers
    ipa_to_english = {
        'zh': 'zh',    # kept (as in "pleasure")
        'sh': 'sh',    # kept (as in "ship")
        'shh': 'sh',   # simplified (щ → sh, not shh)
        'kh': 'h',     # simplified (х → h, not kh, for English compatibility)
        'ch': 'ch',    # kept (as in "church")
        'ts': 'ts',    # kept (as in "tsunami")
        'ye': 'ye',    # kept (as in "yes")
        'yo': 'yo',    # kept (as in "yonder")
        'yu': 'yu',    # kept (as in "yule")
        'ya': 'ya',    # kept (as in "yard")
        'y': 'y',      # kept
    }

    # Special context rules (apply before main mapping)
    result = []
    i = 0
    text_lower = text.lower()

    while i < len(text):
        char = text[i]

        # Handle special cases
        if char in 'ЕеЁёЮюЯя':
            # These vowels are 'ye', 'yo', 'yu', 'ya' at start of word or after vowels/soft signs
            if i == 0 or (i > 0 and text[i-1] in 'ъьЪЬ'):
                if char in 'Ее':
                    result.append('Ye' if char.isupper() else 'ye')
                elif char in 'Ёё':
                    result.append('Yo' if char.isupper() else 'yo')
                elif char in 'Юю':
                    result.append('Yu' if char.isupper() else 'yu')
                elif char in 'Яя':
                    result.append('Ya' if char.isupper() else 'ya')
                i += 1
                continue
            else:
                # After consonants: е→e, ё→o, ю→u, я→a (with palatalization implied)
                if char in 'Ее':
                    result.append('E' if char.isupper() else 'e')
                elif char in 'Ёё':
                    result.append('O' if char.isupper() else 'o')
                elif char in 'Юю':
                    result.append('U' if char.isupper() else 'u')
                elif char in 'Яя':
                    result.append('A' if char.isupper() else 'a')
                i += 1
                continue

        # Handle ы (special Russian vowel, no English equivalent)
        if char in 'ыЫ':
            result.append('I' if char.isupper() else 'i')
            i += 1
            continue

        # Handle х (Russian guttural, map to 'h' for English compatibility)
        if char in 'хХ':
            result.append('H' if char.isupper() else 'h')
            i += 1
            continue

        # Handle щ (long sh, simplify to 'sh')
        if char in 'щЩ':
            result.append('Sh' if char.isupper() else 'sh')
            i += 1
            continue

        # Standard mapping
        if char in cyrillic_to_ipa:
            mapped = cyrillic_to_ipa[char]
            # Apply IPA to English optimizations
            for ipa, eng in ipa_to_english.items():
                if mapped.lower() == ipa.lower():
                    if char.isupper():
                        mapped = eng.capitalize()
                    else:
                        mapped = eng
                    break
            result.append(mapped)
        else:
            # Keep unchanged (numbers, punctuation, etc.)
            result.append(char)

        i += 1

    return ''.join(result)

# -----------------------------------------------------------------------------
# Test Function
# -----------------------------------------------------------------------------
def test_transliteration():
    """Test the improved transliteration with common Russian words."""
    test_cases = [
        # Russian → Expected English Phonetic
        ('привет', 'privet'),           # hello
        ('мир', 'mir'),                 # world/peace
        ('Россия', 'Rossiya'),          # Russia
        ('Москва', 'Moskva'),           # Moscow
        ('Санкт-Петербург', 'Sankt-Peterburg'),
        ('хорошо', 'horosho'),          # good/well
        ('спасибо', 'spasibo'),         # thank you
        ('да', 'da'),                   # yes
        ('нет', 'net'),                 # no
        ('пожалуйста', 'pozhalusta'),   # please
        ('щёлка', 'shchelka'),          # click
        ('чай', 'chay'),                # tea
        ('щука', 'shchuka'),            # pike (fish)
        ('йогурт', 'yogurt'),           # yogurt
        ('подъезд', 'podiezd'),         # entrance
    ]

    print("\n=== TRANSLITERATION TEST ===")
    for russian, expected in test_cases:
        result = transliterate_ru_to_latin(russian)
        status = "✓" if result.lower() == expected.lower() else "✗"
        print(f"{status} '{russian}' → '{result}' (expected: '{expected}')")
    print("=== END TEST ===\n")

# -----------------------------------------------------------------------------
# 7. Training Logic
# -----------------------------------------------------------------------------
def find_training_pairs(tokenizer_old, embed_matrix, vocab, max_pairs=500):
    inputs = []
    targets = []
    count = 0
    max_valid_idx = embed_matrix.shape[0] - 1
    text_to_id = {token: idx for token, idx in vocab.items() if idx <= max_valid_idx}

    print(f"Scanning OLD tokenizer vocabulary (max valid index: {max_valid_idx})...")

    for token_str, token_id in vocab.items():
        if count >= max_pairs:
            break
        if token_id > max_valid_idx:
            continue
        try:
            text = token_str
            ids_split = tokenizer_old.encode(text)
            if len(ids_split) == 1 and ids_split[0] == token_id:
                ids_split = tokenizer_old.encode(' ' + text)
            valid_ids = [i for i in ids_split if i <= max_valid_idx]
            if len(valid_ids) > 1:
                reconstructed = tokenizer_old.decode(valid_ids, clean_up_tokenization_spaces=False)
                if text in reconstructed or reconstructed.replace(' ', '') == text.replace(' ', ''):
                    sub_embs = [embed_matrix[i] for i in valid_ids]
                    target_emb = embed_matrix[token_id]
                    inputs.append(sub_embs)
                    targets.append(target_emb)
                    count += 1
        except:
            continue

    if count < max_pairs:
        print("  Trying character-level decomposition...")
        for token_str, token_id in vocab.items():
            if count >= max_pairs:
                break
            if token_id > max_valid_idx:
                continue
            text = token_str
            if len(text) < 2 or len(text) > 4:
                continue
            char_ids = []
            for char in text:
                if char in text_to_id:
                    char_id = text_to_id[char]
                    if char_id <= max_valid_idx:
                        char_ids.append(char_id)
            if len(char_ids) >= 2:
                sub_embs = [embed_matrix[i] for i in char_ids]
                target_emb = embed_matrix[token_id]
                inputs.append(sub_embs)
                targets.append(target_emb)
                count += 1

    print(f"Found {len(inputs)} training pairs.")
    return inputs, targets

def cosine_similarity_loss(pred, target):
    pred_norm = F.normalize(pred, p=2, dim=0)
    target_norm = F.normalize(target, p=2, dim=0)
    cosine_sim = torch.dot(pred_norm, target_norm)
    return 1.0 - cosine_sim

def combined_loss(pred, target, mse_weight=1.0, cosine_weight=0.0):
    mse = F.mse_loss(pred, target)
    cosine = cosine_similarity_loss(pred, target)
    return mse_weight * mse + cosine_weight * cosine

def train_merger(merger, inputs, targets, epochs=50, lr=5e-4, weight_decay=0.1):
    optimizer = torch.optim.AdamW(merger.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = combined_loss

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None

    print(f"Training Fourier Merger with {len(inputs)} pairs, {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        indices = torch.randperm(len(inputs))
        batch_size = min(32, len(inputs))

        for i in range(0, len(inputs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0

            for idx in batch_indices:
                sub_embs = inputs[idx]
                target = targets[idx]
                pred = merger(sub_embs)
                loss = criterion(pred, target)
                loss.backward()
                batch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(merger.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss

        scheduler.step()
        avg_loss = total_loss / len(inputs)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in merger.state_dict().items()}

        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

    if best_state:
        merger.load_state_dict(best_state)
        print(f"Restored best weights (loss: {best_loss:.6f})")

    return best_loss


def compare_metrics(X, Y, W, lmbda):
    with torch.no_grad():
        # 1. Residual (Accuracy)
        prediction = X @ W
        residual = torch.mean((Y - prediction) ** 2) 
        
        # 2. L2 Loss (Complexity)
        # Note: L2 is usually defined as the squared norm
        l2_loss = torch.norm(W) ** 2
        
        # 3. The Penalty term
        penalty = lmbda * l2_loss
        
    print(f"Residual (Accuracy Error): {residual.item():.6f}")
    print(f"L2 Penalty (Complexity):    {penalty.item():.6f}")
    print(f"Ratio (Res/Penalty):       {residual.item() / (penalty.item() + 1e-8):.2f}")
    return l2_loss

def solve_with_l2(X, Y, lmbda=0.1):
    Xt = X.t() 
    XtX = Xt @ X  # Result is (8192, 8192)
    
    # Adding lambda to the diagonal is the L2 regularization
    regularizer = lmbda * torch.eye(XtX.size(0), device=X.device)
    
    # Solve (XtX + λI)W = XtY
    W = torch.linalg.solve(XtX + regularizer, Xt @ Y)
    return W

def solve_merger(merger, inputs, targets, lmbda=1e-4):   
    indices = torch.randperm(len(inputs))
    batch_size = min(32, len(inputs))

    X =[]
    Y=[]
    for i in range(0, len(inputs), batch_size):
        batch_indices = indices[i:i+batch_size]

        for idx in batch_indices:
            sub_embs = inputs[idx]
            target = targets[idx]
            features = merger(sub_embs, return_features = True)
            X.append(features)
            Y.append(target)
            
    # X is (967, 8192), Y is (967, 1024)
    X = torch.cat(X, dim=0)
    Y = torch.stack(Y, dim=0)
    
    W  = solve_with_l2(X, Y, lmbda)
    merger.set_projection_weights(W)
    
    l2_loss = compare_metrics(X, Y, W, lmbda)
    return l2_loss

# -----------------------------------------------------------------------------
# 8. Main Integration Script
# -----------------------------------------------------------------------------

def load_safetensors_to_torch(file_path: str, device: str = "cpu") -> dict:
    print(f"Loading safetensors file: {file_path}")
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    print(f"Successfully loaded {len(state_dict)} tensors.")
    return state_dict


def main():
    # --- CONFIGURATION ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MERGER_TYPE = "fourier"  # Options: "fourier", "slerp"
    TRAIN_MERGER = True
    

    # --- LOAD MODEL ---
    chk = load_safetensors_to_torch("./data/temp_model/t3_cfg.safetensors")
    embed = chk["text_emb.weight"].to(DEVICE)
    embed2 = chk["text_head.weight"].to(DEVICE)

    print(f"Embedding shape: {embed.shape}")
    dim = embed.shape[1]
    max_old_idx = embed.shape[0] - 1

    # --- LOAD VOCABS ---
    print("\nLoading vocabulary from tokenizer.json files...")
    vocab_old = load_vocab_from_json("./data/temp_model/tokenizer.json")
    vocab_new = load_vocab_from_json("./data/en_ru_tokenizer.json")
    print(f"Loaded {len(vocab_old)} old tokens, {len(vocab_new)} new tokens")

    # --- INIT SIMPLE TOKENIZER (No Transformers) ---
    print("\nInitializing simple tokenizer from vocab.json...")
    tokenizer_orig = SimpleTokenizer("./data/temp_model/tokenizer.json")
    print("Tokenizer ready.")

    # --- DEBUG ---
    print("\n--- DEBUG: Test direct mapping ---")
    test_chars = ['В', 'Д', 'З', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р']
    for char in test_chars:
        latin = transliterate_ru_to_latin(char)
        if latin in vocab_old:
            old_id = vocab_old[latin]
            if old_id <= max_old_idx:
                print(f"'{char}' → '{latin}' ✓ ID {old_id}")
    print("--- END DEBUG ---\n")

    # --- 1. INITIALIZE & TRAIN MERGER ---
    print(f"Initializing {MERGER_TYPE} merger...")

    if MERGER_TYPE == "fourier":
        merger = FourierMerger(dim, max_seq_len=8, keep_freqs=4).to(DEVICE)

        if TRAIN_MERGER:
            train_inputs, train_targets = find_training_pairs(tokenizer_orig, embed, vocab_old, max_pairs=10000)

            if len(train_inputs) > 50:
                print("\n=== TRAINING FOURIER MERGER ===")
                #best_loss = train_merger(merger, train_inputs, train_targets, epochs=50, lr=5e-4, weight_decay=0.1)
                best_loss = solve_merger(merger, train_inputs, train_targets)
                print(f"Training complete. Best loss: {best_loss:.6f}")

                print("\n=== EVALUATION ===")
                merger.eval()
                with torch.no_grad():
                    for i in range(min(5, len(train_inputs))):
                        pred = merger(train_inputs[i])
                        target = train_targets[i]
                        cos_sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
                        print(f"  Sample {i+1}: Cosine similarity = {cos_sim:.4f}")
            else:
                print("WARNING: Not enough training pairs. Using baseline weights.")
                TRAIN_MERGER = False
        merger.eval()

    elif MERGER_TYPE == "slerp":
        merger = None
        print("Using SLERP merger (no training)")
    else:
        merger = None
        print("Using deterministic merger (no training)")

    # --- 2. GENERATE NEW EMBEDDINGS ---
    new_vocab_size = max(vocab_new.values()) + 1
    new_embed = torch.zeros(new_vocab_size, dim, device=DEVICE)
    new_embed2 = torch.zeros(new_vocab_size, dim, device=DEVICE)

    shared_count = min(705, embed.shape[0])
    new_embed[:shared_count] = embed[:shared_count].clone()
    new_embed2[:shared_count] = embed2[:shared_count].clone()

    tot = 0
    success_count = 0
    direct_map_count = 0
    merged_count = 0
    failed_tokens = []
    print("\nGenerating new embeddings for Russian tokens...")

    for token_id in range(shared_count, new_vocab_size):
        try:
            token_text = None
            for text, idx in vocab_new.items():
                if idx == token_id:
                    token_text = text
                    break

            if token_text is None:
                raise ValueError(f"Token ID {token_id} not found in vocab_new")

            latin_text = transliterate_ru_to_latin(token_text)

            # DIRECT MAPPING
            if latin_text in vocab_old:
                direct_id = vocab_old[latin_text]
                if direct_id <= max_old_idx:
                    new_embed[token_id] = embed[direct_id]
                    new_embed2[token_id] = embed2[direct_id]
                    direct_map_count += 1
                    success_count += 1
                    continue

            # ENCODE & MERGE
            subtoken_ids = tokenizer_orig.encode(latin_text)
            valid_subtokens = [(i, embed[i]) for i in subtoken_ids if i <= max_old_idx]

            if len(valid_subtokens) == 0:
                for char in latin_text:
                    if char in vocab_old:
                        char_id = vocab_old[char]
                        if char_id <= max_old_idx:
                            valid_subtokens.append((char_id, embed[char_id]))

            if len(valid_subtokens) == 0:
                raise ValueError("Cannot decompose token")

            emb_list = [emb for _, emb in valid_subtokens]

            if len(emb_list) == 1:
                new_embed[token_id] = emb_list[0]
                new_embed2[token_id] = emb_list[0]
                direct_map_count += 1
            else:
                if MERGER_TYPE == "slerp" or merger is None:
                    merged_emb = slerp_recursive(emb_list)
                    merged_emb2 = slerp_recursive(emb_list)
                else:
                    with torch.no_grad():
                        merged_emb = merger(emb_list)
                        merged_emb2 = merger(emb_list)

                new_embed[token_id] = merged_emb
                new_embed2[token_id] = merged_emb2
                merged_count += 1
            success_count += 1
            if success_count % 50 == 0:
                print(f"  Mapped {success_count} tokens (direct: {direct_map_count}, merged: {merged_count})...")
        except Exception as e:
            new_embed[token_id] = torch.randn(dim, device=DEVICE) * 0.02
            new_embed2[token_id] = torch.randn(dim, device=DEVICE) * 0.02
            tot += 1
            failed_tokens.append((token_id, str(e)))

    print(f"\n=== SUMMARY ===")
    print(f"Successfully mapped: {success_count} tokens")
    print(f"  - Direct mapping: {direct_map_count} tokens")
    print(f"  - Merged: {merged_count} tokens")
    print(f"Failed: {tot} tokens")

    if failed_tokens[:10]:
        print("\nFirst 10 failures:")
        for tid, err in failed_tokens[:10]:
            token_text = None
            for text, idx in vocab_new.items():
                if idx == tid:
                    token_text = text
                    break
            print(f"  Token {tid} '{token_text}': {err}")

    # --- 3. SAVE ---
    chk["text_emb.weight"] = new_embed.cpu()
    chk["text_head.weight"] = new_embed2.cpu()
    fn = "./data/temp_model/t3_ru_fft_cfg.safetensors"
    save_file(chk, fn)
    print(f"\n Saved to {fn}")

if __name__ == "__main__":
    main()