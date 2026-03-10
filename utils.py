import jiwer
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

def get_standard_transformation():
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    
def calculate_metrics(original, transcribed, transformation=None):
    if transformation is None:
        transformation = get_standard_transformation()

    wer = calculate_advanced_wer(transcribed, original, transformation=transformation)
    cer = calculate_advanced_cer(transcribed, original, transformation=transformation)
    return wer, cer


def calculate_advanced_wer(hypothesis_text, reference_text, transformation):
    """
    Standard deterministic WER implementation.
    1. Transcribes audio via ASR
    2. Normalizes both texts (removes punctuation/casing)
    3. Calculates Levenshtein-based WER
    """
    transform = jiwer.Compose([
            transformation,
            jiwer.ReduceToListOfListOfWords() # Crucial: wraps words in nested lists
        ])

    # 3. Apply deterministic calculation
    wer_score = jiwer.wer(
        reference_text, 
        hypothesis_text, 
        reference_transform=transform, 
        hypothesis_transform=transform
    )
    
    return wer_score
    
def calculate_advanced_cer(hypothesis_text, reference_text, transformation):
    """
    Standard deterministic WER implementation.
    1. Transcribes audio via ASR
    2. Normalizes both texts (removes punctuation/casing)
    3. Calculates Levenshtein-based WER
    """
    transform = jiwer.Compose([
        transformation,
        jiwer.ReduceToListOfListOfChars()
    ])
    # 3. Apply deterministic calculation
    wer_score = jiwer.cer(
        reference_text, 
        hypothesis_text, 
        reference_transform=transform, 
        hypothesis_transform=transform
    )
    
    return wer_score