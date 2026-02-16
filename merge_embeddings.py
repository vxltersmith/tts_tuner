import sys
import yaml

import torch
import math

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from safetensors import safe_open
from safetensors.torch import save_file

# Assuming Chatterbox is installed in .venv
from chatterbox.models.tokenizers.tokenizer import EnTokenizer
from g2p_transliterator import get_mapping, get_aligner


# --- 1. CONFIG & UTILS ---

def load_config(config_path="./config.yaml"):
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_safetensors(file_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    print(f"📂 Loading weights: {file_path}")
    state_dict = {}
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

# --- 2. THE MATHEMATICAL GRAFT (SLERP & BILATERAL) ---

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """Spherical linear interpolation for high-dimensional embedding vectors with numerical stability."""
    
    # Normalize vectors with numeric stability
    norm0 = torch.norm(v0)
    norm1 = torch.norm(v1)
    
    if norm0 < 1e-12:
        print(f"⚠️ Warning: Zero norm vector encountered in SLERP, using v1")
        return v1
    if norm1 < 1e-12:
        print(f"⚠️ Warning: Zero norm vector encountered in SLERP, using v0")
        return v0
    
    v0_n = v0 / norm0
    v1_n = v1 / norm1
    
    # Clamp dot product for numerical stability
    dot = torch.sum(v0_n * v1_n)
    dot = torch.clamp(dot, -1.0, 1.0)

    if torch.abs(dot) > dot_threshold:
        # Linear interpolation for near-identical vectors
        result = (1 - t) * v0 + t * v1
        result = result / (torch.norm(result) + 1e-10)
        return result

    theta_0 = torch.acos(dot)
    theta_t = theta_0 * t
    
    # Handle edge cases for near-zero angles
    if abs(theta_0) < 1e-8 or abs(theta_t) < 1e-8:
        # Linear interpolation for near-zero angles
        result = (1 - t) * v0 + t * v1
        result = result / (torch.norm(result) + 1e-10)
        return result
    
    sin_theta_0 = torch.sin(theta_0)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    
    result = s0 * v0 + s1 * v1
    
    # Final normalization for numerical safety
    result = result / (torch.norm(result) + 1e-10)
    
    return result

def calculate_bilateral_weights(anchor, emb_list: List[torch.Tensor], sigma_r: float, p: float, alpha: float) -> torch.Tensor:
    """Calculates weights based on energy (p) and similarity to the sequence start (sigma_r)."""
    stack = torch.stack(emb_list)
    if anchor is None:
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
    
    # Validation checks
    if src_emb.shape[0] != 704:
        print(f"⚠️ Warning: Expected backbone size of 704, got {src_emb.shape[0]}")
    if dim != src_head.shape[1]:
        print(f"❌ Error: Embedding and head dimensions mismatch: {dim} != {src_head.shape[1]}")
        raise ValueError("Dimension mismatch between text_emb and text_head")
    
    new_emb = torch.zeros(new_size, dim)
    new_head = torch.zeros(new_size, dim)
    
    m_cfg = config['merge_embeddings']
    backbone_size = m_cfg.get('share_vocab_size', 704)
    ipa_conf = m_cfg.get("ipa", {})
    tokenizer_path = ipa_conf.get('tokenizer_path', None)
    ipa_aligner = get_aligner(tokenizer_path)
    
    # Copy backbone embeddings
    if new_emb[:backbone_size].shape == src_emb[:backbone_size].shape:
        new_emb[:backbone_size] = src_emb[:backbone_size].clone()
        new_head[:backbone_size] = src_head[:backbone_size].clone()
    else:
        print(f"❌ Error: Shape mismatch when copying backbone embeddings")
        raise ValueError("Backbone embedding shapes don't match")
    
    graft_count = 0
    skip_count = 0
    error_count = 0
    
    print(f" Starting grafting {new_size - backbone_size} new tokens...")
    print(f" Embedding dimension: {dim}, Backbone size: {backbone_size}, New vocab: {new_size}")

    
    unmapped_tokens = []
    for token, token_id in tokenizer_tgt.tokenizer.get_vocab().items():
        if token_id < backbone_size: continue
        if "[" in token: continue 
            
        # 1. Phonetic Mapping
        word = get_mapping(token, aligner=ipa_aligner)
        print(f"Mapped '{token}' to '{word}'")
        
        clean_word = word.strip() if word != ' ' else word
        if len(clean_word) == 0: 
            word = get_mapping(token) # trying to fallback to hardcoded mapping
            print(f"projected '{token}' to '{word}' via fallback mapping")
            if len(word) == 0:
                print(f"Fallback failed, Skipping token '{token}'")
                unmapped_tokens.append(token)
                continue
            clean_word = word
        ids = tokenizer_src.tokenizer.encode(clean_word).ids
        if not ids or len(ids) == 0:
            ids = tokenizer_src.tokenizer.encode(" ").ids
        
        if len(ids) == 0:
            ids = [tokenizer_src.tokenizer.get_vocab()['[UNK]']]

        # 2. Extract vectors with validation
        try:
            e_list = [src_emb[i] for i in ids]
            h_list = [src_head[i] for i in ids]
            
            if len(e_list) != len(ids) or len(h_list) != len(ids):
                error_count += 1
                skip_count += 1
                print(f"⚠️ Warning: ID out of bounds for '{token}' ({token_id}). Using fallback.")
                continue
        except IndexError as e:
            print(f"⚠️ Warning: Index out of bounds for '{token}' ({token_id}): {e}")
            error_count += 1
            skip_count += 1
            continue
        
        # 3. Bilateral Weighting 
        try:
            anchor = e_list[0].view(1, -1)
            w = calculate_bilateral_weights(anchor, e_list, m_cfg['sigma_r'], m_cfg['p'], m_cfg['alpha'])
        except Exception as e:
            print(f"⚠️ Warning: Failed to calculate weights for '{token}': {e}")
            error_count += 1
            skip_count += 1
            continue
            
        # 4. Sequence Interpolation
        try:
            new_emb[token_id] = graft_sequence(e_list, w)
            new_head[token_id] = graft_sequence(h_list, w)
            graft_count += 1
        except Exception as e:
            print(f"⚠️ Warning: Failed to graft '{token}': {e}")
            error_count += 1
            skip_count += 1
            continue
        
        # Progress reporting every 100 tokens
        if graft_count % 100 == 0:
            print(f"{graft_count}/{new_size - backbone_size} tokens grafted")

    skip_count = len(unmapped_tokens) 
    print(f"✅ Grafting complete! {graft_count}/{new_size - backbone_size} tokens successfully grafted")
    print(f"⚠️ Stats: Skipped {skip_count}, Errors {error_count}")
    print(f"New vocab size: {new_size}, New backbone size: {backbone_size}")
    return new_emb, new_head

def main():
    print("="*60)
    print("INITIALIZING EMBEDDING GRAFTING PIPELINE")
    print("="*60)
    
    cfg = load_config()
    m_cfg = cfg['merge_embeddings']
    
    print(f"Configuration loaded successfully")
    print(f"   Source tokenizer: {m_cfg['source_tokenizer']}")
    print(f"   Target tokenizer: {m_cfg['target_tokenizer']}")
    print(f"   Checkpoint: {m_cfg['checkpoint']}")
    print(f"   Output path: {m_cfg['output_checkpoint']}")
    print(f"   Backbone vocab size: {m_cfg.get('share_vocab_size', 704)}")
    
    print()
    print("Initializing tokenizers...")
    t_src = EnTokenizer(m_cfg['source_tokenizer'])
    
    print(f"Source tokenizer loaded ({t_src.tokenizer.get_vocab_size()} tokens)")
    
    print(f"Loading target tokenizer...")
    t_tgt = EnTokenizer(m_cfg['target_tokenizer'])
    
    print(f"Target tokenizer loaded ({t_tgt.tokenizer.get_vocab_size()} tokens)")
    
    print()
    print("oading model checkpoint...")
    checkpoint = load_safetensors(m_cfg['checkpoint'])
    
    # Validate required tensors exist
    if 'text_emb.weight' not in checkpoint:
        raise KeyError("text_emb.weight not found in checkpoint")
    if 'text_head.weight' not in checkpoint:
        raise KeyError("text_head.weight not found in checkpoint")
    
    print(f"Checkpoint loaded: {len(checkpoint)} tensors")
    
    # Validate weights exist
    required_keys = ['text_emb.weight', 'text_head.weight']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Missing required tensor: {key}")
        print(f"   Found: {key} with shape {checkpoint[key].shape}")
    
    # Validate config parameters
    if 'sigma_r' not in m_cfg:
        raise ValueError("Config missing required parameter: sigma_r")
    if 'p' not in m_cfg:
        raise ValueError("Config missing required parameter: p")
    if 'alpha' not in m_cfg:
        raise ValueError("Config missing required parameter: alpha")
    
    # Start of logic ##############
    print()
    print("Starting embedding grafting...")
    new_emb, new_head = merge_token_embeddings(
        t_src, t_tgt, checkpoint['text_emb.weight'], checkpoint['text_head.weight'], cfg
    )
    
    print()
    print("Saving merged checkpoint...")
    
    # Update checkpoint with new embeddings
    checkpoint['text_emb.weight'] = new_emb
    checkpoint['text_head.weight'] = new_head
    
    out_path = Path(m_cfg['output_checkpoint'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_file(checkpoint, str(out_path))
    
    print()
    print("="*60)
    print("SUCCESS")
    print("="*60)
    print(f"Output saved to: {out_path}")
    print(f"Checkpoint size: checkpoint loaded, {len(checkpoint)} tensors")
    print("="*60)
        

if __name__ == "__main__":
    try:
        main()
        print("Finish!")
        
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)