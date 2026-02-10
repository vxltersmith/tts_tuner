1. Research Context & Model Backbone
Context: The model is a T3 (Token-To-Token) transformer-based TTS with a Llama-style architecture and an S3Gen (Flow Matching) decoder.
Code Alignment: The code handles the embedding weights (text_emb.weight, text_head.weight) for the T3 model, which are critical for token representation. The code modifies these weights to support a new script (Russian Cyrillic) by grafting embeddings.
Key Concept: The code implements a “zero-shot script extension” by expanding the vocabulary and grafting embeddings, aligning with the stated goal of avoiding extensive retraining.
2. Problem Statement & Constraint
Context: The original model is English-centric (Vorig = 704). Fine-tuning on Russian would require massive datasets and risk shifting the latent space, potentially breaking English capabilities.
Code Alignment: The code avoids extensive retraining by using a “grafting” approach. It starts with the existing English embeddings (704 tokens) and extends the vocabulary (to 786 tokens) by mapping Russian characters to Latin via transliteration.
Key Concept: The use of transliterate_ru_to_latin and harmonic_phonetic_map aligns with the idea of mapping Russian characters to English phonetic representations, preserving the latent space.
3. Methodology: SLERP and Embedding Grafting
Context: The hypothesis proposes using SLERP (Spherical Linear Interpolation) to merge high-dimensional embeddings from English subtokens to create Russian embeddings.
Code Alignment:
The slerp function implements spherical linear interpolation between two vectors.
The slerp_multiple and slerp_weighted_sequence functions recursively apply SLERP to combine multiple embeddings with weights.
The descending_linear_weights function assigns higher weights to leading subtokens, aligning with the idea that they carry stronger phonetic signals.
Key Concept: The grafting process uses SLERP to merge embeddings from the English model, ensuring that the Russian vocabulary inherits phonetic and prosodic richness.
4. Vocabulary Expansion and Tokenization
Context: The vocabulary is expanded from 704 to 786 tokens to accommodate Russian characters.
Code Alignment:
The merge_embeddings function iterates over the new tokenizer’s vocabulary (Russian) and maps each token to its Latin transliteration.
The code uses tokenizer_target.tokenizer.get_vocab() to get the new vocabulary size and token IDs.
Key Concept: The code dynamically assigns new embeddings to Russian tokens by leveraging the existing English embeddings, ensuring that the model can handle Russian text without retraining.
5. Weighting Strategies
Context: The approach uses descending linear weights to prioritize leading subtokens and bilateral weights to emphasize energy and similarity.
Code Alignment:
descending_linear_weights gives higher weights to earlier subtokens.
bilateral_harmonic_weights and optimized_bilateral_weights_with_decay use cosine similarity and energy scaling to refine the weighting.
positional_harmonic_graft applies a time bias to vectors to simulate future context, enhancing the representation.
Key Concept: These weighting strategies ensure that the grafted embeddings are more representative of the phonetic structure of Russian, especially for words with complex subtoken structures.
6. Efficiency & Zero-Shot Extension
Context: The goal is to demonstrate that this method is a viable alternative to “Vocabulary Expansion + Continued Pre-training,” significantly reducing compute.
Code Alignment:
The code avoids continued pre-training by directly grafting embeddings.
It uses efficient operations like slerp and weighted averaging to merge embeddings.
Key Concept: The method achieves zero-shot script extension with minimal additional computation, directly supporting the efficiency goal.
7. Integration with PolyIPA & Multilingual Tasks
Context: PolyIPA is a multilingual phoneme-to-grapheme conversion model, which shares the goal of handling multiple languages and scripts.
Code Alignment:
The transliteration process (transliterate_ru_to_latin) mirrors the idea of mapping between scripts.
The code’s use of harmonic_phonetic_map and slerp aligns with PolyIPA’s focus on phonetic accuracy and multilingual compatibility.
Key Concept: The approach can be seen as an extension of PolyIPA’s transliteration principles to TTS model adaptation.
8. Prosodic Transfer
Context: The hypothesis suggests that the “musicality” and emotional range of the English model will transfer to Russian.
Code Alignment:
The use of text_head.weight alongside text_emb.weight suggests attention to prosody and linguistic structure.
The positional_harmonic_graft function introduces a temporal bias that may help preserve prosodic features.
Key Concept: While not explicitly modeled, the embedding grafting process aims to preserve the prosodic richness of the original model.
Summary:
The code directly implements the core ideas of the hypothesis and research context:

SLERP-based embedding grafting to extend the model to Russian.
Efficient vocabulary expansion without extensive retraining.
Weighted merging of embeddings to prioritize phonetic signal.
Zero-shot script extension using transliteration and embedding transfer.
It aligns with the PolyIPA framework’s goals of multilingual compatibility and phonetic accuracy, while leveraging the T3 architecture’s strengths in TTS synthesis.