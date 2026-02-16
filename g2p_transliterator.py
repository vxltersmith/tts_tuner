import epitran
import panphon.distance
from chatterbox.models.tokenizers.tokenizer import EnTokenizer

DEFAULT_MAPPING = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'ye', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'ee',
    'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'oo', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
    'ш': 'sh', 'щ': 'sh', 
    'ъ': ' ', 
    'ы': 'ui', 'ь': 'i',
    'э': 'e', 'ю': 'yu', 'я': 'ya'
}

class TokenizerAligner:
    def __init__(self, en_tokenizer_path):
        # Initialize phonetic engines
        self.epi_ru = epitran.Epitran('rus-Cyrl')
        self.epi_en = epitran.Epitran('eng-Latn')
        self.dst = panphon.distance.Distance()
        
        # Load the source English tokenizer
        self.en_tok = EnTokenizer(en_tokenizer_path)
        self.en_vocab = self.en_tok.tokenizer.get_vocab()
        
        # Pre-cache English IPA footprints to make the search fast
        print("Pre-calculating English phonetic manifold...")
        self.build_ipa_cache()
        
    def build_ipa_cache(self):
            en_ipa_cache = {}
            # Simple fallback for tokens Epitran rejects
            simple_map = str.maketrans("abcdefghijklmnopqrstuvwxyz", "abkdefɡhijklmnopqrstuvwxyz")

            for token, _ in self.en_vocab.items():
                clean = token.replace('##', '').replace(' ', '').strip().lower()
                if not clean: continue

                try:
                    # Path A: SOTA Epitran
                    ipa = self.epi_en.transliterate(clean)
                    if ipa:
                        en_ipa_cache[token] = ipa
                except:
                    # Path B: Fallback (Literal phonetic approximation)
                    # This ensures we don't drop 90% of the vocab
                    en_ipa_cache[token] = clean.translate(simple_map)

            self.en_ipa_cache = en_ipa_cache
            print(f"✅ Manifold expanded to {len(en_ipa_cache)} English tokens.")

    def find_closest_english_tokens(self, ru_text, top_k=3):
        # 1. Project Russian to IPA
        target_ipa = self.epi_ru.transliterate(ru_text)
        
        matches = []
        for en_token, en_ipa in self.en_ipa_cache.items():
            # FILTER: Skip tokens that are strictly suffixes (start with ##) 
            # or are too short to have a stable phonetic signature.
            if en_token.startswith('##') or len(en_token) < 3:
                continue
                
            score = self.dst.feature_edit_distance(target_ipa, en_ipa)
            matches.append((en_token, score))
        
        matches.sort(key=lambda x: x[1])
        # Now you are more likely to get 'maga', 'margarita', or 'mama' 
        # instead of 'ing' and 'oy'.
        return [m[0] for m in matches[:top_k]]
    
    def find_phonetic_composites(self, ru_text):
        """
        Instead of one match for the whole word, find matches for phonetic segments.
        """
        ru_ipa = self.epi_ru.transliterate(ru_text)
        
        # Simple bigram split of the IPA string (e.g., "maɡa" -> ["ma", "aɡ", "ɡa"])
        windows = [ru_ipa[i:i+2] for i in range(len(ru_ipa)-1)]
        
        final_composition = []
        for window in windows:
            matches = []
            for en_token, en_ipa in self.en_ipa_cache.items():
                # We want English tokens that CONTAIN or ARE this sound
                if window in en_ipa:
                    # Calculate distance only for the relevant segment
                    score = self.dst.feature_edit_distance(window, en_ipa)
                    matches.append((en_token, score))
            
            matches.sort(key=lambda x: x[1])
            if matches:
                final_composition.append(matches[0][0]) # Take the best match for this slice
                
        return ''.join(final_composition) # Unique English building blocks

def phonetic_map(text: str, mapping: dict) -> str:
    """
    Map Russian Cyrillic to English phonetic clusters.
    FIX: 'ъ' (Hard Sign) now maps to a space to ensure a valid token is returned.
    """
    return ''.join(mapping.get(char.lower(), char) for char in text)


def get_aligner(tokenizer_path=None):
    if tokenizer_path is None: return tokenizer_path
    
    tokenizer_aligner = TokenizerAligner(tokenizer_path)
    return tokenizer_aligner

def get_mapping(text:str, aligner=None, tokenizer_path=None) :
    if aligner is None:
        if tokenizer_path is None: 
            return phonetic_map(text, DEFAULT_MAPPING)
        tokenizer_aligner = get_aligner(tokenizer_path)
    else:
         tokenizer_aligner = aligner
    discoveries = tokenizer_aligner.find_phonetic_composites(text)
    return discoveries

if __name__ == "__main__":
    input_text = "мага"
    tokenizer_path ="./data/temp_model/tokenizer.json"
    
    discoveries = get_mapping(input_text, tokenizer_path=tokenizer_path)
    
    print("Input Text:", input_text)
    print("Discovered Phonetic Clusters:", discoveries)  # Output: ['привет', 'привет']
    print(discoveries)
    print('-----finished---------')