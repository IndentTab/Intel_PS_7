# Visual Search using VLM
We used CLIP to generate image and text embeddings and FAISS to search through a dataset of 50k images to provide visual search.

## Running
Run `Gradio.ipynb`, change the constants pointing to where the images are stored. Note, `IMAGE_FOLDER_PATH` also needs to be updated after the images have been extracted.

## Datasets
Two datasets are required for this project. DSD 50k is a collected of the first 50000 images from [this](https://huggingface.co/datasets/primecai/dsd_data) dataset of AI generated images named by their index for convenience. DSD Embeddings are the image embeddings of those images and text embeddings of their description also as generated by CLIP.

DSD 50k: https://www.kaggle.com/datasets/meltqx/dsd-50k/data

DSD Embeddings: https://www.kaggle.com/datasets/meltqx/primecai

If using DSD 50k, disable the initial code to download and extract the zip file contain the images in the ipynb file.

## Demo


https://github.com/user-attachments/assets/f25acb4c-bf5b-4d74-81a5-5f697ce5e94a

