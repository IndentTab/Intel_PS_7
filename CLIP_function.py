import torch
from transformers import CLIPTokenizer, CLIPModel

# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text):
    """Takes a text query and returns its CLIP embedding as a NumPy array."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy().flatten()

    return text_embedding

# Example usage
query = "A beautiful sunset over the mountains."
embedding = get_text_embedding(query)

print(embedding[:5]) #prints only first 5 embeddings 
