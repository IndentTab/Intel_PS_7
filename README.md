# Visual Search using VLM
We used CLIP to generate image and text embeddings and FAISS to search through a dataset of 50k images to provide visual search.
Dataset: We've chosen 50000 images from [this](https://huggingface.co/datasets/primecai/dsd_data) dataset, consisting of AI generated images with the prompts used to enhance them.

## Running
Run `Gradio.ipynb`, change the constants pointing to where the images are stored. Note, `IMAGE_FOLDER_PATH` also needs to be updated after the images have been extracted. A live link is also generated to ease the convinence of access with a click which is active while the server is live. Click [here](https://5cbe13885a733d66d0.gradio.live/) and have a go!
