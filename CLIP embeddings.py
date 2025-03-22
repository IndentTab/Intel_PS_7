import torch
import pandas as pd
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset in streaming mode to avoid long download times
dataset = load_dataset("primecai/dsd_data", split="train", streaming=True)

# Limit the number of samples to process (avoid Kaggle crash)
max_samples = 50000
data = []
count = 0

for item in tqdm(dataset, total=max_samples):
    if count >= max_samples:
        break  # Stop after processing max_samples

    try:
        # Extract image and text
        image = item["conditioning"]
        text = item["caption"]

        # Process image
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embedding = model.get_image_features(**image_inputs).cpu().numpy().flatten()

        # Process text
        text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**text_inputs).cpu().numpy().flatten()

        # Append results
        data.append({
            "Text Description": text,
            "Image Embedding": image_embedding.tolist(),
            "Text Embedding": text_embedding.tolist()
        })

        count += 1  # Increase count only when successfully processed

    except Exception as e:
        print(f"Skipping sample due to error: {e}")

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = "/kaggle/working/dsd_embeddings.csv"
df.to_csv(output_path, index=False)

print(f"Embeddings saved to {output_path}")
