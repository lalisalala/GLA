import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------
# CONFIG
# ----------------------

ABLATION_PATHS = {
    "key_original": "data/key_original.json",
    "key_nlp": "data/key_nlp.json",
    "key_llm": "data/key_llm.json",
    "desc_original": "data/desc_original.json",
    "desc_llm": "data/desc_llm.json", 
    "full_OG": "data/full_OG.json",
    "full_nlp": "data/full_nlp.json",
    "full_llm": "data/full_llm.json",
    "onlykey_original": "data/onlykey_original.json",
    "onlykey_nlp": "data/onlykey_nlp.json",
    "onlykey_llm": "data/onlykey_llm.json",
    "onlytopic_original": "data/onlytopic_original.json",
    "onlytopic_nlp": "data/onlytopic_nlp.json",
    "onlytopic_llm": "data/onlytopic_llm.json",
    "onlytitle": "data/onlytitle.json",
}

EMBEDDING_MODEL = "BAAI/bge-base-en"
INDEX_SAVE_DIR = "models/faiss_indices"
os.makedirs(INDEX_SAVE_DIR, exist_ok=True)

# ----------------------
# LOAD MODEL
# ----------------------

print(f"🔍 Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# ----------------------
# BUILD INDEX FUNCTION
# ----------------------

def build_faiss_index(texts):
    embeddings = model.encode(
        [f"Represent this document for retrieval: {text}" for text in texts],
        show_progress_bar=True,
        normalize_embeddings=False
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity
    index.add(embeddings)
    return index, embeddings

# ----------------------
# MAIN
# ----------------------

def process_ablation(ablation_name, filepath):
    print(f"\n⚙️ Processing ablation: {ablation_name}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        title = item.get("title", "")
        keywords = " ".join(item.get("keywords", []))
        topics = " ".join(item.get("topics", []))
        description = item.get("description", "")

        if "desc" in ablation_name:
            text = description
        elif "onlykey" in ablation_name:
            text = keywords
        elif "onlytopic" in ablation_name:
            text = topics
        elif "onlytitle" in ablation_name: 
            text = title
        elif "key" in ablation_name:
            text = f"{keywords} {topics}"
        elif "full" in ablation_name or "combined" in ablation_name:
            text = f"{keywords} {topics}. {description}"
        else:
            raise ValueError(f"Unknown ablation type for: {ablation_name}")
        
        texts.append(text.strip())

    index, embeddings = build_faiss_index(texts)

    # Save FAISS index and ID mapping
    faiss.write_index(index, os.path.join(INDEX_SAVE_DIR, f"{ablation_name}.index"))
    with open(os.path.join(INDEX_SAVE_DIR, f"{ablation_name}_ids.json"), "w") as f:
        json.dump([item["id"] for item in data], f)

    print(f"✅ Saved FAISS index and ID mapping for {ablation_name}")


if __name__ == "__main__":
    for ablation, path in ABLATION_PATHS.items():
        process_ablation(ablation, path)
