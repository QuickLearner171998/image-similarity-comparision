import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T
from matplotlib import pyplot as plt

"""**Path where to save and reload the model**"""

PATH = './'

"""**FEW SHOT PARAMETERS:**"""

K_WAY = 10 # Means the support set has K classes, this classes are unseen during training, in this case We exclude K_WAY classes from the original dataset
N_SHOT = 5 # Means every class has N sampes
N_CLASS = 100 # for cifar100 = 100, cifar10 = 10 etc... if K_WAY == 0 the model learns every class


CONTRASTIVE_COSINE = True #True cosine, False Euclidean

"""**Preprocessing dataset:**"""

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

"""**Custom Arguments class:**"""

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

"""**Loss Custom Implementation (Contrastive Loss):**"""

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        if CONTRASTIVE_COSINE:
            distances = 1 - F.cosine_similarity(output1, output2)
        else:
            distances = torch.norm(output1 - output2, dim=1)
        loss_pos = label * distances.pow(2)
        loss_neg = (1 - label) * F.relu(self.margin - distances).pow(2)
        loss = loss_pos + loss_neg
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

"""**Custom Matcher for support_set and query_set:**"""

class CIFAR100Subset(Dataset):
    def __init__(self, root, train, download=False, few_shot_set="support"):
        super(CIFAR100Subset, self).__init__()
        self.cifar100 = datasets.CIFAR100(root, train=train, download=download, transform=transform)

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
    criterion = ContrastiveLoss()
    for batch_idx, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output1 = model(img1)
        output2 = model(img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img1), len(train_loader.dataset),
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
                    if CONTRASTIVE_COSINE:
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

train_dataset = APP_MATCHER_BINARY('../data', train=True, download=True)
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