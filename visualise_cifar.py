import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_cifar100_batch(file):
    batch = unpickle(file)
    images = batch['data']
    fine_labels = batch['fine_labels']
    images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1)
    return images, fine_labels

def load_cifar100(data_dir):
    meta_data = unpickle(os.path.join(data_dir, 'meta'))
    fine_label_names = meta_data['fine_label_names']
    
    train_data, train_labels = load_cifar100_batch(os.path.join(data_dir, 'train'))
    test_data, test_labels = load_cifar100_batch(os.path.join(data_dir, 'test'))
    
    return (train_data, train_labels), (test_data, test_labels), fine_label_names

def visualize_cifar100(images, labels, label_names, num_images=12):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(num_images):
        img = images[i]
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(label_names[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    data_dir = './data/cifar-100-python'  # Specify the path where CIFAR-100 is extracted
    (train_images, train_labels), (test_images, test_labels), label_names = load_cifar100(data_dir)
    
    # Visualize Train Images
    visualize_cifar100(train_images, train_labels, label_names)

if __name__ == '__main__':
    main()