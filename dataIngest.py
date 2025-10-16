import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "./web_pages"

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Containers
documents = []
file_names = []

# Read all .txt files
for file_name in os.listdir(DATA_DIR):
    if file_name.endswith(".txt"):
        file_path = os.path.join(DATA_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append(content)
                file_names.append(file_name)

print(f"📄 Loaded {len(documents)} documents into memory.")

# Compute embeddings
embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
print("✅ Embeddings generated:", embeddings.shape)

# Initialize FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

print(f"✅ FAISS index created with {index.ntotal} vectors.")

# Save FAISS index and metadata
faiss.write_index(index, "canonical_index.faiss")

with open("metadata.txt", "w", encoding="utf-8") as f:
    for i, name in enumerate(file_names):
        f.write(f"{i}\t{name}\n")

print("💾 Saved FAISS index and metadata successfully.")
