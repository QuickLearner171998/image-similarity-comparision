import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import logging
from scipy.spatial import distance
import numpy as np

from cifar_data_loader import CIFAR100Pairs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 128)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_network(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (anchor, positive, negative, anchor_label) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = loss_fn(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)

def evaluate_network(model, test_loader, loss_fn, device, top_k=5):
    model.eval()
    running_loss = 0.0
    correct_top_k = 0
    total = 0
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative, anchor_label) in enumerate(test_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = loss_fn(anchor_out, positive_out, negative_out)
            running_loss += loss.item()

            # Store embeddings and corresponding labels
            all_embeddings.append(anchor_out.cpu().numpy())
            all_labels.append(anchor_label.numpy())

        # Concatenate all embeddings and labels
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.hstack(all_labels)

        for i in range(len(all_embeddings)):
            query_embedding = all_embeddings[i]
            true_label = all_labels[i]
            distances = distance.cdist([query_embedding], all_embeddings, "cosine")[0]
            sorted_indices = np.argsort(distances)

            # Extract top_k indices
            top_k_indices = sorted_indices[1:top_k+1]  # Avoid the first as it's the same image
            top_k_labels = all_labels[top_k_indices]

            # Check if true label is among top_k labels
            if all(label == true_label for label in top_k_labels):
                correct_top_k += 1
            total += 1

    accuracy_top_k = 100 * correct_top_k / total
    avg_loss = running_loss / len(test_loader)

    return avg_loss, accuracy_top_k

def main():
    # Data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Datasets and DataLoaders
    train_dataset = CIFAR100Pairs(root='./data', train=True, transform=transform, n_train_classes=80)
    test_dataset = CIFAR100Pairs(root='./data', train=False, transform=transform, n_train_classes=80)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss = train_network(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        # Evaluation after each epoch
        test_loss, test_accuracy_top_k = evaluate_network(model, test_loader, criterion, device, top_k=5)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Top-5 Accuracy: {test_accuracy_top_k:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'siamese_network.pth')
    print("Model saved to siamese_network.pth")

if __name__ == '__main__':
    main()