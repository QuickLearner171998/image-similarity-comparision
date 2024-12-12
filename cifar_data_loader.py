from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
import numpy as np
import random

class CIFAR100Pairs(Dataset):
    def __init__(self, root, train=True, transform=None, n_train_classes=80):
        self.cifar100 = CIFAR100(root=root, train=train, download=True)
        self.train = train
        self.transform = transform
        self.n_train_classes = n_train_classes

        self.data = self.cifar100.data
        self.targets = np.array(self.cifar100.targets)

        self.data_by_class = {}
        for idx in range(len(self.data)):
            target = self.targets[idx]
            if target not in self.data_by_class:
                self.data_by_class[target] = []
            self.data_by_class[target].append(self.data[idx])

    def __getitem__(self, index):
        if self.train:
            anchor_label = random.choice(range(self.n_train_classes))
            positive_label = anchor_label
            negative_label = random.choice(range(self.n_train_classes, 100))
        else:
            anchor_label = random.choice(range(self.n_train_classes, 100))
            positive_label = anchor_label
            negative_label = random.choice(range(self.n_train_classes))

        positive_pair = random.choice(self.data_by_class[positive_label])
        negative_pair = random.choice(self.data_by_class[negative_label])

        anchor_img = random.choice(self.data_by_class[anchor_label])

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_pair = self.transform(positive_pair)
            negative_pair = self.transform(negative_pair)

        return anchor_img, positive_pair, negative_pair, anchor_label

    def __len__(self):
        return len(self.data)