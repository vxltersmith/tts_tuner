from Levenshtein import distance as lev_dist
import datasets
datasets.config.USE_TORCHCODEC = False # Redundant safety



def get_dataset(config):
    # Fix: Ensure we use the full path to avoid the "Dataset scripts no longer supported" error
    validateion_conf = config.get("validation", {})
    dataset_key = validateion_conf.get("dataset_key", "Vikhrmodels/ToneSpeak")
    split_key = validateion_conf.get("split", "train")
    ds = datasets.load_dataset(dataset_key, split=split_key)
    #ds = ds.cast_column("audio", datasets.Value("string"))
    return ds

# --- 3. METRICS & PIPELINE ---

def calculate_metrics(original, transcribed):
    orig = original.lower().strip().replace(".", "").replace(",", "")
    tran = transcribed.lower().strip().replace(".", "").replace(",", "")
    d = lev_dist(orig, tran)
    cer = d / max(len(orig), 1)
    return d, cer