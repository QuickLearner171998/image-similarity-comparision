import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt

class CIFAR100Pairs(Dataset):
    def __init__(self, root, train=True, transform=None, download=True, n_train_classes=80, seed=42):
        self.root = root
        self.train = train
        self.transform = transform

        # Set the seed for reproducibility
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        
        # Load the CIFAR-100 dataset
        self.cifar100 = datasets.CIFAR100(root=self.root, train=True, download=download, transform=transform)
        
        self.n_train_classes = n_train_classes
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        labels = np.array(self.cifar100.targets)
        self.train_data, self.query_data = [], []
        self.train_labels, self.query_labels = [], []

        unique_classes = np.unique(labels)
        self.random_state.shuffle(unique_classes)  # Shuffle to pick random classes for training

        train_classes = unique_classes[:self.n_train_classes]
        query_classes = unique_classes[self.n_train_classes:]
        
        # Separate the training classes and test (query) classes
        for c in train_classes:
            indices = np.where(labels == c)[0]
            self.train_data.extend(self.cifar100.data[indices])
            self.train_labels.extend([c] * len(indices))

        for c in query_classes:
            indices = np.where(labels == c)[0]
            self.query_data.extend(self.cifar100.data[indices])
            self.query_labels.extend([c] * len(indices))

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                while True:
                    index2 = random.randint(0, len(self.train_data) - 1)
                    if self.train_labels[index2] == label1:
                        break
            else:
                while True:
                    index2 = random.randint(0, len(self.train_data) - 1)
                    if self.train_labels[index2] != label1:
                        break
            img2, label2 = self.train_data[index2], self.train_labels[index2]
            label = torch.tensor([1.0], dtype=torch.float32) if should_get_same_class else torch.tensor([0.0], dtype=torch.float32)
        else:
            img1, label1 = self.query_data[index], self.query_labels[index]
            img2 = img1  # Only used as a placeholder, won't be returned or used
            label = torch.tensor([label1], dtype=torch.int64)  # Evaluation will use separate handling
        img1 = self._transform(img1)
        img2 = self._transform(img2) if self.train else img2
        if self.train:
            return img1, img2, label
        else:
            return img1, label

    def _transform(self, img):
        img = img.astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.train_data) if self.train else len(self.query_data)
    
    def visualize_sample(self, index):
        if not self.train:
            raise ValueError("Visualization is only available for the training dataset.")

        img1, img2, label = self.__getitem__(index)
        
        img1_np = img1.permute(1, 2, 0).numpy()
        img2_np = img2.permute(1, 2, 0).numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow((img1_np * 255).astype(np.uint8))
        axes[0].set_title('Anchor')
        axes[0].axis('off')

        axes[1].imshow((img2_np * 255).astype(np.uint8))
        axes[1].set_title('Positive' if label.item() == 1.0 else 'Negative')
        axes[1].axis('off')

        plt.show()

# Test the dataloader with visualization
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the dataset
    train_dataset = CIFAR100Pairs(root='./data', train=True, transform=transform, n_train_classes=80)
    test_dataset = CIFAR100Pairs(root='./data', train=False, transform=transform, n_train_classes=80)
    
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Visualize a sample
    index = random.randint(0, len(train_dataset) - 1)
    train_dataset.visualize_sample(index)