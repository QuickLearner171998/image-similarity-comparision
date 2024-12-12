import os
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import tensorflow_datasets as tfds

# Constants
K = 5  # Default number of top matches
SIMILARITY_THRESHOLD = 0.55  # Default similarity threshold to filter results
BATCH_SIZE = 16  # Batch size for DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (224, 224)  # Resize images to this size
NUM_CLASSES = 10  # Number of classes to use from the dataset

# Load CLIP model and processor
MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the dataset
print("Downloading dataset...")
dataset, info = tfds.load("caltech101", split="test", with_info=True, as_supervised=True)

# Obtain class names
class_names = info.features["label"].names
print(f"Classes in the dataset: {class_names}")

# Select classes (animals and common objects)
selected_class_names = ['airplanes', 'wild_cat', 'butterfly', 'camera', 'kangaroo', 'elephant', 'motorbikes', 'chair', 'watch']
selected_class_indices = [class_names.index(class_name) for class_name in selected_class_names]

# Filter the dataset to include only the selected classes
data = [(image, label) for image, label in tfds.as_numpy(dataset) if label in selected_class_indices]

print(f"Selected classes: {selected_class_names}")

class CustomDataset(Dataset):
    def __init__(self, data, processor, image_size):
        self.data = data
        self.processor = processor
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image_pil = Image.fromarray(image).resize(self.image_size)
        
        # Convert to tensor and normalize using the processor
        processed_image = self.processor(images=image_pil, return_tensors="pt").pixel_values.squeeze(0).to(DEVICE)
        
        return processed_image, np.array(image_pil), label

# Create the dataset and dataloader
dataset = CustomDataset(data, PROCESSOR, IMAGE_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def clear_mem():
    """ Utility function to clear memory. """
    torch.cuda.empty_cache()
    gc.collect()

def compute_embeddings_in_chunks(dataloader):
    all_features = []
    all_labels = []
    all_original_images = []
    
    for idx, (processed_images, original_images_batch, labels) in enumerate(tqdm(dataloader)):
        processed_images = processed_images.to(DEVICE)
        
        with torch.no_grad():
            image_features = MODEL.get_image_features(pixel_values=processed_images)

        all_features.append(image_features.cpu())
        all_labels.append(labels.cpu())
        
        # Ensure original images are converted to numpy arrays
        original_images_batch_np = [np.array(img) for img in original_images_batch]
        all_original_images.extend(original_images_batch_np)
        
        # Clear memory
        clear_mem()
        
        # Save intermediate results to disk to avoid memory crashes
        if idx % 100 == 0 and idx > 0:
            torch.save((torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), all_original_images), f'embeddings_chunk_{idx//100}.pt')
            all_features = []
            all_labels = []
            all_original_images = []
    
    if len(all_features) > 0:
        torch.save((torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), all_original_images), f'embeddings_chunk_final.pt')
    
    return

# Compute embeddings for the train (support) set
compute_embeddings_in_chunks(dataloader)

def load_all_embeddings():
    all_features = []
    all_labels = []
    all_original_images = []
    
    for filename in os.listdir('.'):
        if filename.startswith('embeddings_chunk_') and filename.endswith('.pt'):
            features, labels, original_images = torch.load(filename)
            all_features.append(features)
            all_labels.append(labels)
            all_original_images.extend(original_images)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels, all_original_images

train_image_features, train_labels, original_images = load_all_embeddings()

def find_top_k_matches(input_image=None, input_text=None, top_k=K, similarity_threshold=SIMILARITY_THRESHOLD):
    if input_image is None and input_text is None:
        raise ValueError("Either input_image or input_text (or both) must be provided.")
    
    if input_image is not None:
        input_image_resized = input_image.resize(IMAGE_SIZE)
        processed_image = PROCESSOR(images=input_image_resized, return_tensors="pt").pixel_values.squeeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            image_features = MODEL.get_image_features(pixel_values=processed_image)
    else:
        image_features = None

    if input_text is not None:
        processed_text = PROCESSOR(text=input_text, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            text_features = MODEL.get_text_features(input_ids=processed_text)
    else:
        text_features = None

    if image_features is not None and text_features is not None:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        combined_features = (image_features + text_features) / 2
    elif image_features is not None:
        combined_features = image_features
    else:
        combined_features = text_features

    combined_features = combined_features.to("cpu")
    similarity = torch.nn.functional.cosine_similarity(combined_features, train_image_features)
    
    # Apply the similarity threshold to filter results
    valid_indices = similarity >= similarity_threshold
    
    top_k_indices = similarity.topk(min(top_k, valid_indices.sum().item())).indices
    top_k_images = [Image.fromarray(original_images[idx]) for idx in top_k_indices]
    top_k_labels = [selected_class_names[selected_class_indices.index(train_labels[idx].item())] for idx in top_k_indices]
    # log topk similarity values
    print(f"Top-K similarity values: {similarity[top_k_indices]}")

    
    # Combine images and labels into a list of tuples
    results = [(img, label) for img, label in zip(top_k_images, top_k_labels)]
    
    return results

def get_example_image_and_label():
    # Get an example image belonging to the 'elephant' class (index 5 in selected_class_names)
    elephant_class_idx = selected_class_indices[selected_class_names.index("elephant")]
    for image_array, label in data:
        if label == elephant_class_idx:
            return Image.fromarray(image_array), label
    return None, None

def setup_gradio_interface():
    image_input = gr.Image(type="pil", label="Input Image (optional)")
    text_input = gr.Textbox(label="Input Text (optional)")
    k_input = gr.Number(label="Top-K Matches", value=K)
    gallery_output = gr.Gallery(label="Top Matches")
    
    example_image, example_label = get_example_image_and_label()
    example_text = f"A picture of an {selected_class_names[selected_class_indices.index(example_label)]}"
    examples = [[example_image, example_text, 5]]
    
    description = f"""
    Find top-K similar images from the Caltech-101 dataset using CLIP. 
    This application is limited to the following classes:
    {', '.join(selected_class_names)}
    """
    
    return gr.Interface(
        fn=find_top_k_matches,
        inputs=[image_input, text_input, k_input],
        outputs=[gallery_output],
        title="Caltech-101 Top-K Image Matcher for Selected Classes",
        description=description,
        examples=examples
    )

gradio_interface = setup_gradio_interface()
gradio_interface.launch(server_name="0.0.0.0", debug=True)  # Adjust this for your specific hosting environment (e.g., Colab)

clear_mem()