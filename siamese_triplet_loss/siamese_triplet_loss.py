# -*- coding: utf-8 -*-
"""Siamese-triplet-loss

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/siamese-triplet-loss-99132570-9878-4a84-aeb9-cdde057fdb15.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20241212/auto/storage/goog4_request%26X-Goog-Date%3D20241212T103622Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D6b33c2173896df726aaaf43b4fa5dfc3aa468261f2a9366afba3994cf678d12a491fd8f37bf2568e382eae7b4f60181112ea4a2e489aaa00eae4303618a4c36f120732d9edcc7302b3b5fc2f608db8b3fe848df888c101a6411228b068c3171289ddc8b018193ac324211c3080fac1f4889e3f5173531ccc480bac817fdaebec74a5e3ca21805d15181920f72f390b67392da035bf09d30ca16861102d8b7cbdc78a41dae277a61b377ed957314b9e8b8e346bbe03ab23f6d19988f7d089dd918fcbc89b59453fd36b5646591780928aeff2de4f9c0eb5d9b24adf4836d4450704dc797d46aa741f09bd11e0951e92840d7d8fa0a6bd5a811fdf961108073d8e

**Libraries:**
"""

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T, models, utils
from matplotlib import pyplot as plt

"""**Path where to save and reload the model**"""

PATH = './'

"""**FEW SHOT PARAMETERS:**"""

K_WAY = 10 # Means the support set has K classes, this classes are unseen during training, in this case We exclude K_WAY classes from the original dataset
N_SHOT = 5 # Means every class has N sampes
N_CLASS = 100 # for cifar100 = 100, cifar10 = 10 etc... if K_WAY == 0 the model learns every class

"""**Select Triplet Loss Function (Cosine Distance or Euclidean Distance):**"""

TRIPLET_COSINE = True #True cosine, False Euclidean

"""**Preprocessing dataset:**"""

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

"""**Custom Arguments class:**"""

#It's inconvenient to switch to the command line for colab, so I created my own args class
class Arguments:
    def __init__(self, batch_size=64, test_batch_size=1000, epochs=14, lr=0.001,
                 no_cuda=False, no_mps=False, dry_run=False, seed=1,
                 log_interval=10, save_model=True):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.no_cuda = no_cuda
        self.no_mps = no_mps
        self.dry_run = dry_run
        self.seed = seed
        self.log_interval = log_interval
        self.save_model = save_model

"""**ResNet optimized for cifar10/cifar100:**"""

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [5, 5, 5]):
        super(SiameseNetwork, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        output = out.view(out.size()[0], -1)

        return output

"""**Loss Custom Implementation:**"""

class TripletLoss_Cosine(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss_Cosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cos_sim_pos = F.cosine_similarity(anchor, positive)
        cos_sim_neg = F.cosine_similarity(anchor, negative)
        loss = torch.relu(cos_sim_neg - cos_sim_pos + self.margin)
        return loss.mean()

class TripletLoss_Euclidean(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss_Euclidean, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

"""**Custom Matcher for binary test (img1, img2, label) (label = 1/0):**"""

class APP_MATCHER_BINARY(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER_BINARY, self).__init__()
        self.dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        self.data = torch.stack([img for img, _ in self.dataset], dim=0)
        self.group_examples()

    def group_examples(self):
        np_arr = np.array(self.dataset.targets)
        self.grouped_examples = {}
        for i in range(0, N_CLASS-K_WAY):
            self.grouped_examples[i] = np.where((np_arr==i))[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        selected_class = random.randint(0, N_CLASS-K_WAY-1)
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        index_1 = self.grouped_examples[selected_class][random_index_1]
        image_1 = self.data[index_1].clone()

        if index % 2 == 0:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            index_2 = self.grouped_examples[selected_class][random_index_2]
            image_2 = self.data[index_2].clone()
            target = torch.tensor(1, dtype=torch.float)
        else:
            other_selected_class = random.randint(0, N_CLASS-K_WAY-1)
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, N_CLASS-K_WAY-1)
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)
            index_2 = self.grouped_examples[other_selected_class][random_index_2]
            image_2 = self.data[index_2].clone()
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target

"""**Custom Matcher for triplet logic during train (Anchor, Positive, Negative)**:"""

class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()
        self.dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        self.data = torch.stack([img for img, _ in self.dataset], dim=0)
        self.group_examples()

    def group_examples(self):
        np_arr = np.array(self.dataset.targets)
        self.grouped_examples = {}
        for i in range(0, N_CLASS-K_WAY):
            self.grouped_examples[i] = np.where((np_arr==i))[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        selected_class = random.randint(0, N_CLASS-K_WAY-1)
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        index_1 = self.grouped_examples[selected_class][random_index_1]
        anchor = self.data[index_1].clone()

        random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        while random_index_2 == random_index_1:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        index_2 = self.grouped_examples[selected_class][random_index_2]
        positive = self.data[index_2].clone()

        other_selected_class = random.randint(0, N_CLASS-K_WAY-1)
        while other_selected_class == selected_class:
            other_selected_class = random.randint(0, N_CLASS-K_WAY-1)
        random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)
        index_3 = self.grouped_examples[other_selected_class][random_index_2]
        negative = self.data[index_3].clone()

        return anchor, positive, negative

"""**Custom Matcher for support_set and query_set:**"""

class CIFAR100Subset(Dataset):
    def __init__(self, root, train, download=False, few_shot_set="support"):
        super(CIFAR100Subset, self).__init__()
        self.cifar100 = datasets.CIFAR100(root, train=train, download=download, transform = transform)

        self.class_to_indices_support = {i: [] for i in range(N_CLASS-K_WAY, N_CLASS)}
        self.class_to_indices_query = {i: [] for i in range(N_CLASS-K_WAY, N_CLASS)}

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

"""**Train loop:**"""

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    if TRIPLET_COSINE:
        criterion = TripletLoss_Cosine()
    else:
        criterion = TripletLoss_Euclidean()
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

# Top-k accuracy evaluation:
def top_k_accuracy(model, device, query_loader, support_loader, k=5):
    class_embeddings = {}
    model.eval()

    # Compute embeddings for support set
    with torch.no_grad():
        for images, labels in support_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            for emb, label in zip(embeddings, labels):
                if label.item() not in class_embeddings:
                    class_embeddings[label.item()] = []
                class_embeddings[label.item()].append(emb)

    class_embeddings = {key: torch.stack(class_embeddings[key]) for key in class_embeddings}

    correct = 0
    total = 0

    # compute embeddings for query set
    with torch.no_grad():
        for images, labels in query_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            for emb, label in zip(embeddings, labels):
                distances = []
                for class_label, class_embs in class_embeddings.items():
                    if TRIPLET_COSINE:
                        dist = torch.mean(1 - F.cosine_similarity(emb.unsqueeze(0), class_embs)).item()
                    else:
                        dist = torch.mean(torch.norm(emb.unsqueeze(0) - class_embs, dim=1)).item()
                    distances.append((dist, class_label))

                distances.sort()

                top_k_classes = [label for _, label in distances[:k]]
                if label.item() in top_k_classes:
                    correct += 1
                total += 1

    accuracy = 100.0 * correct / total
    print('Top {:d} Accuracy: {:.2f}%'.format(k, accuracy))
    return accuracy

# Load datasets:
args = Arguments(batch_size=128, test_batch_size=1000, epochs=150, lr=0.001,
                 no_cuda=False, no_mps=False, dry_run=False, seed=1,
                 log_interval=10, save_model=True)

use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if(K_WAY > 0):
    support_dataset = CIFAR100Subset('../data', train=False, download=True, few_shot_set = "support")
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1)

    query_dataset = CIFAR100Subset('../data', train=False, download=True, few_shot_set = "query")
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=args.test_batch_size, shuffle=True)

train_dataset = APP_MATCHER('../data', train=True, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

model = SiameseNetwork(num_blocks=[9, 9, 9]).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
n_epoch = 5

if os.path.exists(PATH+"siamese_network.pth"):
    checkpoint = torch.load(PATH+"siamese_network.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    n_epoch = checkpoint['epoch']
    print("- Checkpoint found, I resume training")
else:
    print("- A pre-trained model was not found, I proceed with new training.")


# Let's Train:
best_accuracy = 0
patience = 10
trigger_times = 0

for epoch in range(n_epoch, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)

    accuracy = top_k_accuracy(model, device, query_loader, support_loader, k=5)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = PATH + "best_siamese_network.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, best_model_path)
        print(f"- Best model saved with accuracy: {best_accuracy:.2f}%")
        trigger_times = 0
    else:
        trigger_times += 1
        print(f"- Early stopping trigger times: {trigger_times}/{patience}")

    if trigger_times >= patience:
        print("- Early stopping")
        break

    if args.save_model:
        path = PATH + f"siamese_network_{epoch}.pth"
        path_fast_load = PATH + f"siamese_network.pth"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(checkpoint, path)
        torch.save(checkpoint, path_fast_load)
        print("- Checkpoint saved successfully")

