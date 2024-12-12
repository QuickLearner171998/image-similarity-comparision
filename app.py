import os
import torch
from transformers import CLIPProcessor, CLIPModel
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader, Dataset
import gradio as gr
from tqdm import tqdm
from PIL import Image

# Constants
K = 5  # Default number of top matches
BATCH_SIZE = 16  # Batch size for DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the HuggingFace CLIP model and processor
MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Download and preprocess the dataset
print("Downloading dataset...")
dataset, info = tfds.load('caltech101', split='test', with_info=True, as_supervised=True)

# Obtain class names
class_names = info.features['label'].names

# Filter the dataset if needed
data = [(image, label) for image, label in tfds.as_numpy(dataset)]

class CustomDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = Image.fromarray(image)
        processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return processed_image, label

# Create dataset and dataloader
dataset = CustomDataset(data, PROCESSOR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def compute_embeddings(dataloader):
    all_features = []
    all_labels = []
    all_images = []
    for images, labels in tqdm(dataloader):
        with torch.no_grad():
            image_features = MODEL.get_image_features(pixel_values=images.to(DEVICE))
        all_features.append(image_features.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_images = torch.cat(all_images, dim=0)
    return all_features, all_labels, all_images

# Compute embeddings for train set (support set)
train_image_features, train_labels, train_images = compute_embeddings(dataloader)

# Gradio Interface
def find_top_k_matches(input_image=None, input_text=None, top_k=K):
    if input_image is None and input_text is None:
        raise ValueError("Either input_image or input_text (or both) must be provided.")
    
    if input_image is not None:
        processed_image = PROCESSOR(images=input_image, return_tensors="pt").pixel_values.squeeze(0).unsqueeze(0)
        with torch.no_grad():
            image_features = MODEL.get_image_features(pixel_values=processed_image.to(DEVICE))
    else:
        image_features = None

    if input_text is not None:
        processed_text = PROCESSOR(text=input_text, return_tensors="pt").input_ids
        with torch.no_grad():
            text_features = MODEL.get_text_features(input_ids=processed_text.to(DEVICE)).float()
    else:
        text_features = None

    if image_features is not None and text_features is not None:
        # Normalize and combine features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        combined_features = (image_features + text_features) / 2
    elif image_features is not None:
        combined_features = image_features
    else:
        combined_features = text_features

    similarity = torch.nn.functional.cosine_similarity(combined_features, train_image_features)
    top_k_indices = similarity.topk(top_k).indices
    top_k_results = list(zip(top_k_indices.tolist(), similarity[top_k_indices].tolist()))
    top_k_images = [Image.fromarray(train_images[idx].permute(1, 2, 0).mul(255).byte().cpu().numpy()) for idx, _ in top_k_results]
    top_k_labels = [class_names[train_labels[idx].item()] for idx, _ in top_k_results]
    return top_k_images + top_k_labels

def setup_gradio_interface():
    image_input = gr.Image(type='pil', label="Input Image (optional)")
    text_input = gr.Textbox(label="Input Text (optional)")
    k_input = gr.Number(label="Top-K Matches", value=K)
    image_outputs = gr.Gallery(label="Top Matches").style(grid=(K, 2))
    label_outputs = gr.Textbox(num_lines=K, label="Labels for Top Matches")
    
    return gr.Interface(
        fn=find_top_k_matches,
        inputs=[image_input, text_input, k_input],
        outputs=[image_outputs, label_outputs],
        live=True
    )

gradio_interface = setup_gradio_interface()
gradio_interface.launch()