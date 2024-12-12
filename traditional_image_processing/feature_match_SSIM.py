import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor

# Few Shot Parameters
K_WAY = 10  # Classes in support set
N_SHOT = 5  # Samples per class in support set
N_CLASS = 100  # Number of classes in dataset
IMAGE_SIZE = 224
NUM_WORKERS = 4  # Number of parallel processes

class CIFAR100Subset(Dataset):
    def __init__(self, root, train, download=False, few_shot_set="support"):
        super(CIFAR100Subset, self).__init__()
        self.transform = Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.cifar100 = datasets.CIFAR100(root, train=train, download=download, transform=ToTensor())
        
        # Establish class to indices mapping for the support and query sets
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
        image = self.transform(image)
        return image, label


class TraditionalImageSimilarity:
    def __init__(self):
        self.support_images, self.support_labels = self.download_and_prepare_dataset("support")
        self.query_images, self.query_labels = self.download_and_prepare_dataset("query")

    def download_and_prepare_dataset(self, few_shot_set):
        dataset = CIFAR100Subset('./data', train=False, download=True, few_shot_set=few_shot_set)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
        images, labels = [], []
        for batch in dataloader:
            image, label = batch[0]
            image = (image.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            images.append(image)
            labels.append(label)
        return images, labels

    def compute_histogram(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compute_ssim(self, imageA, imageB):
        imageA = cv2.cvtColor(np.array(imageA), cv2.COLOR_BGR2GRAY)
        imageB = cv2.cvtColor(np.array(imageB), cv2.COLOR_BGR2GRAY)
        score, _ = ssim(imageA, imageB, full=True)
        return score

    def feature_matching(self, img1, img2):
        img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        score = sum([match.distance for match in matches]) / len(matches) if matches else float('inf')
        return score

    def compute_distances(self, query_image, query_hist):
        distances = []
        for img, label in zip(self.support_images, self.support_labels):
            hist = self.compute_histogram(img)
            hist_dist = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)
            ssim_dist = self.compute_ssim(query_image, img)
            feature_dist = self.feature_matching(query_image, img)
            combined_dist = (hist_dist + ssim_dist - feature_dist) / 3
            distances.append((combined_dist, label))
        return sorted(distances, reverse=True)

    def top_k_accuracy(self, query_images, query_labels, k=5):
        correct = 0
        total = 0

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for query_image, query_label in zip(query_images, query_labels):
                query_hist = self.compute_histogram(query_image)
                futures.append(executor.submit(self.compute_distances, query_image, query_hist))
                
            for future, query_label in tqdm(zip(futures, query_labels), total=len(futures), desc="Query Set Evaluation"):
                distances = future.result()
                top_k_classes = [label for _, label in distances[:k]]
                if query_label in top_k_classes:
                    correct += 1
                total += 1

        accuracy = 100.0 * correct / total
        print('Top {:d} Accuracy: {:.2f}%'.format(k, accuracy))
        return accuracy


def main():
    # Initialize TraditionalImageSimilarity class
    traditional_evaluator = TraditionalImageSimilarity()
    
    # Evaluate using traditional methods
    traditional_evaluator.top_k_accuracy(traditional_evaluator.query_images, traditional_evaluator.query_labels, k=5)

if __name__ == "__main__":
    main()