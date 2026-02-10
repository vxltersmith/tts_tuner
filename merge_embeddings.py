import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
import torch
import numpy as np
from src.chatterbox.models.tokenizers.tokenizer import EnTokenizer


def harmonic_phonetic_map(text: str) -> str:
    # Optimized for English BPE latent triggers
    mapping = {
        'а': 'a',   'б': 'b',  'в': 'v',  'г': 'g',  'д': 'd',
        'е': 'ye',  'ё': 'yo', 'ж': 'zh', 'з': 'z',  'и': 'ee',
        'й': 'y',   'к': 'k',  'л': 'l',  'м': 'm',  'н': 'n',
        'о': 'o',   'п': 'p',  'р': 'r',  'с': 's',  'т': 't',
        'у': 'oo',  'ф': 'f',  'х': 'kh', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh',  'щ': 'sh', 'ъ': '',   'ы': 'ui', 'ь': 'i', 
        'э': 'e',   'ю': 'yu', 'я': 'ya'
    }
    # Note: 'щ' is mapped to 'sh' instead of 'shh' to keep the vector 'sharp'
    # 'ь' is mapped to 'i' to trigger palatalization in the preceding consonant
    return "".join(mapping.get(c.lower(), c) for c in text)

def transliterate_ru_to_latin(text: str) -> str:
    """
    Transliterates Russian Cyrillic text to Latin using GOST 7.79-2000 (System B).
    
    Args:
        text (str): Input string in Russian (Cyrillic).
    
    Returns:
        str: Transliterated string in Latin characters.
    """
    mapping = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
        'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
        'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh', 'щ': 'shh', 'ъ': '', 'ы': 'y', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
        'Е': 'E', 'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I',
        'Й': 'J', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
        'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
        'У': 'U', 'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch',
        'Ш': 'Sh', 'Щ': 'Shh', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
        'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
    }

    return ''.join(mapping.get(char, char) for char in text)


def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation between two 1D tensors (vectors)."""
    # Ensure float32 for stability
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    norm_v0 = torch.norm(v0)
    norm_v1 = torch.norm(v1)

    epsilon = 1e-10
    if norm_v0 < epsilon or norm_v1 < epsilon:
        # Fall back to lerp if either vector is near zero
        return lerp(t, v0, v1)

    # Normalize
    u0 = v0 / norm_v0
    u1 = v1 / norm_v1

    dot = torch.sum(u0 * u1)
    dot = torch.clamp(dot, -1.0, 1.0)  # Clamp for numerical safety

    if torch.abs(dot) > DOT_THRESHOLD:
        # Vectors are nearly parallel → use lerp
        return lerp(t, v0, v1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    if sin_theta_0 < epsilon:
        return lerp(t, v0, v1)

    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return s0 * v0 + s1 * v1

def harmonic_energy_weights(emb_list, p=2.0):
    # Calculate magnitudes (L2 norms)
    norms = torch.stack([torch.norm(v) for v in emb_list])
    
    # Power-scaling to emphasize "sharp" acoustic features
    weights = torch.pow(norms, p)
    
    # Normalize weights so they sum to 1
    return weights / weights.sum()

import torch.nn.functional as F

def bilateral_harmonic_weights(emb_list, sigma_r=0.5, p=2.0):
    if not emb_list: return torch.tensor([])
    
    anchor = emb_list[0].view(1, -1) # Ensure 2D for F.cosine_similarity
    
    norms = torch.stack([torch.norm(v) for v in emb_list])
    energy_weights = torch.pow(norms, p) # Shape: (n,)
    
    # Calculate similarities and force to 1D
    sims = []
    for v in emb_list:
        sim = F.cosine_similarity(anchor, v.view(1, -1))
        sims.append(sim)
    range_weights = torch.exp(-((1.0 - torch.cat(sims)) ** 2) / (2 * sigma_r ** 2)) # Shape: (n,)
    
    final_weights = energy_weights * range_weights
    return final_weights / (final_weights.sum() + 1e-10)

def optimized_bilateral_weights_with_decay(emb_list, token_ids=None, tokenizer=None, 
        sigma_r=0.4, p=3.0, alpha=0.6):
    if not emb_list: return torch.tensor([])
    n = len(emb_list)
    if n == 1: return torch.tensor([1.0])

    all_weight_sets = []
    decayed_scores = []

    for i in range(n):
        anchor = emb_list[i].view(1, -1)
        norms = torch.stack([torch.norm(v) for v in emb_list])
        energy_weights = torch.pow(norms, p)
        
        sims = torch.cat([F.cosine_similarity(anchor, v.view(1, -1)) for v in emb_list])
        range_weights = torch.exp(-((1.0 - sims)**2) / (2 * sigma_r**2))
        
        current_weights = energy_weights * range_weights
        all_weight_sets.append(current_weights)
        
        # APPLY DECAY: Reduce the 'election power' of tokens further down the string
        positional_decay = 1.0 / (1.0 + alpha * i)
        raw_score = current_weights.sum().item()
        decayed_scores.append(raw_score * positional_decay)

    best_anchor_idx = int(np.argmax(decayed_scores))
    
    # Optional Print for Debugging the "Election"
    if token_ids is not None and tokenizer is not None:
        chosen_id = token_ids[best_anchor_idx]
        token_str = tokenizer.tokenizer.id_to_token(chosen_id)
        print(f"  [Decay Centroid] Winner: '{token_str}' (Pos: {best_anchor_idx}) | DecayScore: {decayed_scores[best_anchor_idx]:.4f}")

    final_weights = all_weight_sets[best_anchor_idx]
    return final_weights / (final_weights.sum() + 1e-10)

def slerp_multiple(vectors):
    """
    Slerp a list of 1D tensors (same shape) into a single representative vector.
    Uses recursive pairwise slerp with equal weighting.
    """
    if not vectors:
        raise ValueError("Empty vector list")
    if len(vectors) == 1:
        return vectors[0].clone()

    result = vectors[0]
    for i in range(1, len(vectors)):
        # Weight: interpolate current result with next vector
        # At step i (0-indexed), we have i+1 vectors total so far.
        # To give equal weight, use t = 1/(i+1)
        t = 1.0 / (i + 1)
        result = slerp(t, result, vectors[i])
    return result
# Load model

def slerp_weighted_sequence(vectors, weights=None):
    """
    Combine vectors in order with optional weights (e.g., positional decay).
    If no weights, uses uniform weighting via recursive slerp.
    """
    if weights is None:
        weights = torch.ones(len(vectors))
    else:
        weights = weights.clone().detach()
    
    # Normalize weights
    weights = weights / weights.sum()

    result = vectors[0]
    cum_weight = weights[0]
    for i in range(1, len(vectors)):
        next_weight = weights[i]
        t = next_weight / (cum_weight + next_weight)  # interpolation ratio
        result = slerp(t, result, vectors[i])
        result = result + 0.01* slerp(t, vectors[i-1], vectors[i])
        cum_weight += next_weight
    return result

def positional_harmonic_graft(emb_list, p=3.0):
    """
    Returns a LIST of tensors where each tensor has been 
    independently nudged and energy-scaled.
    """
    if not emb_list: return []
    
    dim = emb_list[0].shape[0]
    n = len(emb_list)
    nudged_list = []
    
    for i, vec in enumerate(emb_list):
        # Apply the 'Future Nudge'
        time_bias = torch.randn(dim).to(vec.device) * (i / n) * 0.01 
        
        # We apply the energy scaling (p) but keep the vector separate
        # This preserves the 'identity' of the sub-token for the Bilateral Filter
        nudged_vec = (vec + time_bias) * torch.pow(torch.norm(vec), p)
        nudged_list.append(nudged_vec)
        
    return nudged_list

def descending_linear_weights(n):
    """
    Generate normalized descending linear weights for n elements.
    Example: n=4 → [4, 3, 2, 1] → normalized to [0.4, 0.3, 0.2, 0.1]
    """
    if n <= 0:
        raise ValueError("n must be positive")
    weights = torch.arange(n, 0, -1, dtype=torch.float32)  # [n, n-1, ..., 1]
    return weights / weights.sum()  # normalize to sum=1
    
def load_safetensors_to_torch(file_path: str, device: str = "cpu") -> dict:
    print(f"Loading safetensors file: {file_path}")
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    print(f"Successfully loaded {len(state_dict)} tensors.")
    return state_dict

def merge_embeddings(tokenizer_source, tokenizer_target, source_emb_weight, source_head_weight, 
    law="harm"):
    # 3. Determine EXACT size from the NEW tokenizer
    # get_vocab_size() is the most reliable way to get 911
    new_vocab_size = tokenizer_target.tokenizer.get_vocab_size() 
    dim = source_emb_weight.shape[1]

    new_embed = torch.zeros(new_vocab_size, dim)
    new_embed2 = torch.zeros(new_vocab_size, dim)

    # 4. Copy Backbone (0-703)
    new_embed[:704] = source_emb_weight[:704].clone()
    new_embed2[:704] = source_head_weight[:704].clone()

    tot_failed = 0
    # 5. Graft Russian (704 to 911)
    for token, token_id in tokenizer_target.tokenizer.get_vocab().items():
        if token_id < 704:
            continue
        print(f"{token}")
        word = transliterate_ru_to_latin(token)
        word = harmonic_phonetic_map(token)
        print(f"{word}\n")
        
        # FIX: Handle empty transliterations (ъ, ь)
        if not word.strip():
            word = " " 

        try:
            ids = tokenizer_source.encode(word)
            if not ids:
                # Fallback to a space if encoding is still empty
                ids = tokenizer_source.encode(" ")
            
            n = len(ids) # linear weights
            weights = descending_linear_weights(n)
            
            emb_list = [source_emb_weight[i] for i in ids]
            emb_list = positional_harmonic_graft(emb_list)
            emb2_list = [source_head_weight[i] for i in ids]
            emb2_list = positional_harmonic_graft(emb2_list)
            
            
            weights_embed = weights_head = weights
            weights_embed = optimized_bilateral_weights_with_decay(emb_list, ids, tokenizer_source, alpha=0)
            weights_head = optimized_bilateral_weights_with_decay(emb2_list, ids, tokenizer_source, alpha=0)
            
            new_embed[token_id] = slerp_weighted_sequence(emb_list, weights_embed)
            new_embed2[token_id] = slerp_weighted_sequence(emb2_list, weights_head)

        except Exception as e:
            print(f"Failed token {token} (ID: {token_id}): {e}")
            tot_failed += 1

    print(f"Grafting complete. Failed: {tot_failed}. New shape: {new_embed.shape}")
    return new_embed, new_embed2


def main(config:dict):
    # 1. Load Tokenizers
    tokenizer_new = EnTokenizer(config["target_tokenizer_path"])
    tokenizer_orig = EnTokenizer(config["source_tokenizer_path"])

    # 2. Load Weights
    chk = load_safetensors_to_torch(config["checkpoint_path"])
    embed = chk["text_emb.weight"]
    embed2 = chk["text_head.weight"]

    new_embed, new_embed2 = merge_embeddings(tokenizer_source=tokenizer_orig, 
        tokenizer_target=tokenizer_new, source_emb_weight=embed, source_head_weight=embed2
    )

    # 6. Save to NEW file to avoid corruption
    new_chk = {k: v for k, v in chk.items()} # Copy other tensors
    new_chk["text_emb.weight"] = new_embed
    new_chk["text_head.weight"] = new_embed2
    
    save_file(new_chk, config["output_checkpoint_path"])
    print(f"✅ Saved merged weights to {config['output_checkpoint_path']}")

if __name__ == "__main__":
    config = {
        "target_tokenizer_path": "./data/en_ru_tokenizer.json",
        "source_tokenizer_path": "./data/temp_model/tokenizer.json",
        "checkpoint_path": "./data/temp_model/t3_cfg.safetensors",
        "output_checkpoint_path": "./data/temp_model/t3_cfg_ru_harm_bilat.safetensors", # USE A NEW NAME
        "shared_vocab_size": 704,
    }
    main(config=config)