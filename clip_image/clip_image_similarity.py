import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle
from pathlib import Path

# Few Shot Parameters
K_WAY = 10  # Classes in support set
N_SHOT = 5  # Samples per class in support set
N_CLASS = 100  # Number of classes in dataset

CACHE_DIR = Path('./cache')

# Ensure the cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

class CIFAR100Subset(Dataset):
    def __init__(self, root, train, download=False, few_shot_set="support"):
        super(CIFAR100Subset, self).__init__()
        self.cifar100 = datasets.CIFAR100(root, train=train, download=download)
        
        self.class_to_indices_support = {i: [] for i in range(N_CLASS - K_WAY, N_CLASS)}
        self.class_to_indices_query = {i: [] for i in range(N_CLASS - K_WAY, N_CLASS)}

        for idx, (_, class_idx) in enumerate(self.cifar100):
            if class_idx in self.class_to_indices_support and len(self.class_to_indices_support[class_idx]) < N_SHOT:
                self.class_to_indices_support[class_idx].append(idx)
            elif class_idx in self.class_to_indices_support:
                self.class_to_indices_query[class_idx].append(idx)

        if few_shot_set == "support":
            self.indices = [idx for indices in self.class_to_indices_support.values() for idx in indices]
        elif few_shot_set == "query":
            self.indices = [idx for indices in self.class_to_indices_query.values() for idx in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.cifar100[self.indices[idx]]
        return image, label

class ClipEvaluator:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def compute_embeddings(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.cpu()

    def cache_embeddings(self, dataloader, cache_file):
        if cache_file.exists():
            print(f"- Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        embeddings_dict = {}
        for images, labels in tqdm(dataloader, desc="Caching Embeddings"):
            for image, label in zip(images, labels):
                key = str(label)
                embeddings = self.compute_embeddings([image])

                if key not in embeddings_dict:
                    embeddings_dict[key] = []

                embeddings_dict[key].append(embeddings)

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)

        return embeddings_dict

    def top_k_accuracy(self, query_loader, support_loader, k=5):
        print("- Caching support set embeddings")
        support_embeddings = self.cache_embeddings(support_loader, CACHE_DIR / 'support_embeddings.pkl')
        
        correct = 0
        total = 0
    
        print("- Computing query set embeddings")
        for query_images, query_labels in tqdm(query_loader, desc="Query Set Evaluation"):
            for query_image, query_label in zip(query_images, query_labels):
                query_embedding = self.compute_embeddings([query_image])

                distances = []

                for class_label, class_embs in support_embeddings.items():
                    mean_class_emb = torch.mean(torch.cat(class_embs), dim=0)
                    dist = torch.mean(1 - torch.nn.functional.cosine_similarity(query_embedding, mean_class_emb.unsqueeze(0))).item()
                    distances.append((dist, int(class_label)))

                distances.sort()

                top_k_classes = [label for _, label in distances[:k]]
                if query_label in top_k_classes:
                    correct += 1
                total += 1

        accuracy = 100.0 * correct / total
        print('Top {:d} Accuracy: {:.2f}%'.format(k, accuracy))
        return accuracy

def load_or_create_dataloader(few_shot_set="support", train=False, batch_size=1):
    dataset = CIFAR100Subset('./data', train=train, download=True, few_shot_set=few_shot_set)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    return loader

def main():
    support_loader = load_or_create_dataloader(few_shot_set="support", train=False, batch_size=1)
    query_loader = load_or_create_dataloader(few_shot_set="query", train=False, batch_size=1000)

    evaluator = ClipEvaluator()
    evaluator.top_k_accuracy(query_loader, support_loader, k=5)

if __name__ == "__main__":
    main()