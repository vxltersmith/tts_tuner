import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from safetensors import safe_open
from safetensors.torch import save_file

# Assuming Chatterbox is installed in your .venv
from chatterbox.models.tokenizers.tokenizer import EnTokenizer

# --- 1. CONFIG & UTILS ---

def load_config(config_path="./config.yaml"):
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def harmonic_phonetic_map(text: str) -> str:
    """
    Map Russian Cyrillic to English phonetic clusters.
    FIX: 'ъ' (Hard Sign) now maps to a space to ensure a valid token is returned.
    """
    mapping = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
        'е': 'ye', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'ee',
        'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'у': 'oo', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh', 'щ': 'sh', 
        'ъ': ' ',  # Separator: mapped to space to prevent empty tensor error
        'ы': 'ui', 'ь': 'i',
        'э': 'e', 'ю': 'yu', 'я': 'ya'
    }
    return ''.join(mapping.get(char.lower(), char) for char in text)

def load_safetensors(file_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    print(f"📂 Loading weights: {file_path}")
    state_dict = {}
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

# --- 2. THE MATHEMATICAL GRAFT (SLERP & BILATERAL) ---

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """Spherical linear interpolation for high-dimensional embedding vectors."""
    v0_n = v0 / (torch.norm(v0) + 1e-10)
    v1_n = v1 / (torch.norm(v1) + 1e-10)
    dot = torch.clamp(torch.sum(v0_n * v1_n), -1.0, 1.0)

    if torch.abs(dot) > dot_threshold:
        return (1 - t) * v0 + t * v1

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    return s0 * v0 + s1 * v1

def calculate_bilateral_weights(emb_list: List[torch.Tensor], sigma_r: float, p: float, alpha: float) -> torch.Tensor:
    """Calculates weights based on energy (p) and similarity to the sequence start (sigma_r)."""
    stack = torch.stack(emb_list)
    anchor = emb_list[0].view(1, -1)
    
    # 1. Range Weight: How similar is this part of the phoneme to the start?
    sims = torch.nn.functional.cosine_similarity(anchor, stack)
    range_weights = torch.exp(-((1.0 - sims)**2) / (2 * sigma_r**2))
    
    # 2. Energy Weight: Favor vectors with higher 'magnitude' (more information)
    energy_weights = torch.pow(torch.norm(stack, dim=1), p)
    
    # 3. Positional Decay: Favor the beginning of the phonetic sequence
    decay = 1.0 / (1.0 + alpha * torch.arange(len(emb_list), device=stack.device))
    
    final_scores = range_weights * energy_weights * decay
    return final_scores / (final_scores.sum() + 1e-10)

def graft_sequence(vectors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Recursive Slerp through a phonetic sequence using calculated weights."""
    res = vectors[0]
    cum_w = weights[0]
    for i in range(1, len(vectors)):
        t = weights[i] / (cum_w + weights[i] + 1e-10)
        res = slerp(t.item(), res, vectors[i])
        cum_w += weights[i]
    return res

# --- 3. MAIN GRAFTING LOGIC ---

def merge_token_embeddings(tokenizer_src, tokenizer_tgt, src_emb, src_head, config):
    new_size = tokenizer_tgt.tokenizer.get_vocab_size()
    dim = src_emb.shape[1]
    
    # Initialize new tensors
    new_emb = torch.zeros(new_size, dim)
    new_head = torch.zeros(new_size, dim)
    
    # Copy Shared Backbone (Special tokens/ASCII)
    m_cfg = config['merge_embeddings']
    backbone_size = m_cfg.get('share_vocab_size', 704)
    new_emb[:backbone_size] = src_emb[:backbone_size].clone()
    new_head[:backbone_size] = src_head[:backbone_size].clone()
    
    print(f"🏗️ Grafting {new_size - backbone_size} Russian tokens...")

    for token, token_id in tokenizer_tgt.tokenizer.get_vocab().items():
        if token_id < backbone_size: continue
        
        # 1. Phonetic Mapping
        word = harmonic_phonetic_map(token)
        ids = tokenizer_src.tokenizer.encode(word).ids
        
        # SAFETY GUARD: If mapping is empty, fallback to space
        if not ids or len(ids) == 0:
            print(f"⚠️ Warning: Skipping empty mapping for '{token}' -> fallback to space.")
            ids = tokenizer_src.tokenizer.encode(" ").ids

        # 2. Extract and weight vectors
        e_list = [src_emb[i] for i in ids]
        h_list = [src_head[i] for i in ids]
        
        w = calculate_bilateral_weights(e_list, m_cfg['sigma_r'], m_cfg['p'], m_cfg['alpha'])
        
        # 3. Sequence Interpolation
        new_emb[token_id] = graft_sequence(e_list, w)
        new_head[token_id] = graft_sequence(h_list, w)

    return new_emb, new_head

def main():
    try:
        cfg = load_config()
        m_cfg = cfg['merge_embeddings']
        
        t_src = EnTokenizer(m_cfg['source_tokenizer'])
        t_tgt = EnTokenizer(m_cfg['target_tokenizer'])
        
        checkpoint = load_safetensors(m_cfg['checkpoint'])
        
        new_emb, new_head = merge_token_embeddings(
            t_src, t_tgt, checkpoint['text_emb.weight'], checkpoint['text_head.weight'], cfg
        )
        
        checkpoint['text_emb.weight'] = new_emb
        checkpoint['text_head.weight'] = new_head
        
        out_path = Path(m_cfg['output_checkpoint'])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(checkpoint, str(out_path))
        print(f"✨ Success! Merged weights saved to: {out_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()