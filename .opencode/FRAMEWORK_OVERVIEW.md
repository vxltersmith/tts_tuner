### **Project Summary: Phonetic Latent Grafting (PLG) for Zero-Shot Multilingual TTS**

#### **Objective**
This project introduces **Phonetic Latent Grafting (PLG)**, a novel framework for rapid, low-resource lingual adaptation of large-scale, decoder-only TTS models (e.g., Llama-3-Speech or VALL-E). PLG enables zero-shot extension to new scripts and languages by leveraging pre-trained acoustic priors and phonetic manifold alignment, without requiring extensive retraining or large-scale datasets.

---

### **Framework Overview**

PLG is composed of two core phases:

---

### **Phase I: Input Space Alignment – Phonetic Embedding Propagation**

**Goal:** Align the input space of the target language (e.g., Russian) with the pre-trained English model’s latent space.

**Mechanism:**
- **Transliteration:** Russian Cyrillic characters are mapped to phonetically equivalent Latin characters using a **harmonic phonetic mapping** (e.g., `а` → `a`, `щ` → `sh`).
- **Embedding Grafting:** Each new Russian token is initialized by interpolating its constituent English subtokens using **Spherical Linear Interpolation (SLERP)**.
  - **Weighting Strategy:** Bilateral weights are calculated based on:
    - **Energy** (magnitude of embeddings, using power scaling).
    - **Similarity** (cosine similarity to the sequence start).
    - **Positional Decay** (prioritizing earlier phonetic components).
- **Result:** New Russian embeddings occupy meaningful coordinates in the English latent space, avoiding the "cold start" problem.

**Key Innovation:**
- Unlike traditional vocabulary expansion or fine-tuning, PLG uses **SLERP-based interpolation** to ensure that the new embeddings are not just random but are **geometrically aligned** with the source model’s phonetic manifold.

---

### **Phase II: Manifold Refinement – Prosodic LoRA Adaptation**

**Goal:** Correct prosodic mismatches (stress, rhythm, intonation) that persist after phonetic alignment.

**Mechanism:**
- **Frozen Backbone:** The pre-trained transformer layers are frozen to prevent catastrophic forgetting.
- **Low-Rank Adaptation (LoRA):** Trainable LoRA matrices (A, B) are injected into the **Query and Value (Q, V)** attention layers.
  - These learn language-specific prosodic features:
    - Palatalization (soft signs like `ь`).
    - Lexical stress patterns.
    - Rhythmic nuances.
- **Training Data Efficiency:** LoRA modules can be trained on **small, diverse datasets** (e.g., Common Voice), bypassing the need for large, professional-grade audio.

**Scientific Rationale:**
- **Efficiency:** Avoids full model retraining, reducing compute costs.
- **Data Democratization:** Allows training on in-the-wild data, democratizing access to high-quality multilingual TTS.
- **Modularity:** The same foundation model can support dozens of language-specific LoRA adapters, enabling **modular, plug-and-play language switching** at inference time.

---

### **Construction Methods for Embedding Initialization**

1. **Centroid Initialization (Naive):**
   - A simple mean of k-nearest neighbors in the embedding space.

2. **Featurized IPA Mapping:**
   - Weights are derived from IPA feature vectors (Place, Manner, Voicing) using similarity metrics like Levenshtein or Jaro-Winkler.

3. **Cross-Modal Neural Projection:**
   - Uses self-supervised models (e.g., Wav2Vec 2.0 or HuBERT) to measure acoustic distance between English and Russian phoneme clusters, guiding the mixing weights.

---

### **Alignment with Research Contexts**

- **PolyIPA & Multilingual G2P:** PLG builds on PolyIPA’s phonetic mapping and transliteration techniques, extending them to TTS embedding adaptation.
- **Cross-Lingual Transfer in TTS:** Aligns with works like Huang et al. (2022) and Do et al. (2023), which emphasize phoneme embedding alignment and cross-lingual adaptation.
- **Zero-Shot and Low-Resource TTS:** PLG supports zero-shot script extension and low-resource language adaptation, consistent with recent advancements in multilingual TTS (e.g., VALL-E, DiaMoE-TTS).
- **Efficiency & Scalability:** The modular LoRA approach aligns with parameter-efficient fine-tuning (PEFT) trends, offering scalable and compute-efficient solutions.

---

### **Conclusion**

**Phonetic Latent Grafting (PLG)** represents a significant step forward in **zero-shot, multilingual TTS adaptation**. By combining:
- **SLERP-based embedding interpolation** for phonetic alignment,
- **Bilateral weighting strategies** for robust token mapping,
- **LoRA-based prosodic refinement** for linguistic nuance,

PLG enables **rapid, efficient, and modular** extension of TTS models to new languages and scripts — all without catastrophic forgetting or extensive retraining.