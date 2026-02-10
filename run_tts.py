import torch
from pathlib import Path
import torchaudio as ta
from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.models.t3.t3 import T3
from src.chatterbox.models.t3.modules.t3_config import T3Config
from safetensors.torch import load_file
from src.chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from src.chatterbox.models.s3gen import S3Gen
from src.chatterbox.models.tokenizers.tokenizer import EnTokenizer
from src.chatterbox.tts import Conditionals





def build_model(
        ckpt_dir="./data/temp_model", 
        tokenizer_path = "./data/en_ru_tokenizer.json", 
        t3_chk_name = "t3_cfg.safetensors",
        text_tokens_dict_size = 786, map_location= 'cpu'):
    # 1. Define our fixed T3 initialization
    hp_fixed = T3Config.english_only()
    hp_fixed.text_tokens_dict_size = text_tokens_dict_size

    # 2. Manually load the model since from_local is rigid
    ckpt_dir = Path(ckpt_dir)
    device = "cuda"

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    ve.to(device).eval()

    t3 = T3(hp=hp_fixed)

    t3_state = load_file(ckpt_dir / t3_chk_name)
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()

    tokenizer = EnTokenizer(tokenizer_path)

    conds = None
    map_location = torch.device(map_location)
    if (builtin_voice := ckpt_dir / "conds.pt").exists():
        conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

    # Create the final class instance
    model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds)
    
    return model

if __name__ == "__main__":
    ckpt_dir="./data/temp_model"
    tokenizer_path = "./data/en_ru_tokenizer.json"
    t3_chk_name = 't3_cfg_ru_harm_bilat.safetensors'
    text_tokens_dict_size = 2687 # new dict size could be calculated automatically
    
    model = build_model(ckpt_dir, tokenizer_path=tokenizer_path, text_tokens_dict_size=text_tokens_dict_size,
        t3_chk_name = t3_chk_name)
    
    #
    text = "меня зовут андрей иванов. я живу в тайланде на острове пхукет уже целый год"
    #text = "моя киса, мой котёнок!"
    wav = model.generate(text=text)
    
    fname = f"./data/{text}.mp3"
    ta.save(fname, wav, model.sr)
    print(f"Saved: {fname}")