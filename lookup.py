import polars as pl
import numpy as np
import faiss

df = pl.read_csv("DATASET")
df = df.with_row_index()

def parse_embedding(embedding_str):
    return np.array([float(x) for x in embedding_str.strip("[]").split(', ')])

image_embeddings = np.vstack([parse_embedding(x) for x in df["Image Embedding"].to_list()]).astype(np.float32)
text_embeddings = np.vstack([parse_embedding(x) for x in df["Text Embedding"].to_list()]).astype(np.float32)

d = image_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(image_embeddings)

def predict(x):
    query_embedding = x.reshape(1, -1)
    D, I = index.search(query_embedding, 10)
    
    top_10_images = df[I[0]]["index"]
    
    return top_10_images
