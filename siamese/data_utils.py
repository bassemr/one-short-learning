# Import libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
import random
import copy

def get_full_cifar100(root="data"):
    """
    Load the full CIFAR-100 dataset (train + test) as a single dataset.
    Note:
        - No transforms are applied here. You should apply preprocessing
          (e.g., ToTensor, normalization, augmentations) after splitting 
          into seen/unseen classes or train/val/test.

    Args:
        root (str): Path where the CIFAR-100 dataset will be stored or loaded from.

    Returns:
        torch.utils.data.ConcatDataset: Combined dataset containing both training and test images.
    """

    # Paths to check if the dataset is already downloaded
    train_folder = os.path.join(root, "cifar-100-python", "train")
    test_folder = os.path.join(root, "cifar-100-python", "test")

    # If the dataset files are missing, set download flag to True
    download_flag = not (os.path.exists(train_folder) and os.path.exists(test_folder))
    

    # Load CIFAR-100 training data (no transform yet)
    train_data = datasets.CIFAR100(
        root=root,
        train=True,
        download=download_flag,
        transform=None
    )

    # Load CIFAR-100 test data (no transform yet)
    test_data = datasets.CIFAR100(
        root=root,
        train=False,
        download=download_flag,
        transform=None
    )

    # Merge train and test into one large dataset
    full_dataset = torch.utils.data.ConcatDataset([train_data, test_data])

    return full_dataset


def get_seen_unseen_datasets(full_dataset, num_seen=90):
    """
    Split a dataset (e.g., CIFAR-100) into 'seen' and 'unseen' subsets based on class labels.

    Args:
        full_dataset (torch.utils.data.Dataset or ConcatDataset):
            Dataset containing (image, label) pairs, such as CIFAR-100.
        num_seen (int, default=90):
            Number of 'seen' classes. The remaining classes are considered 'unseen'.

    Returns:
        seen_dataset (torch.utils.data.Subset):
            Subset of full_dataset containing only samples from seen classes.
        unseen_dataset (torch.utils.data.Subset):
            Subset of full_dataset containing only samples from unseen classes.
    """
    num_classes_total = 100  # CIFAR-100 has 100 classes total

    # -----------------------------
    # Reproducibility
    # -----------------------------
    torch.manual_seed(42)

    # Randomly permute all classes (0–99)
    perm = torch.randperm(num_classes_total)

    # First `num_seen` are "seen", remaining are "unseen"
    seen_classes = perm[:num_seen].tolist()
    unseen_classes = perm[num_seen:].tolist()

    # -----------------------------
    # Collect dataset indices by class
    # -----------------------------
    seen_indices = [
        i for i, (_, label) in enumerate(full_dataset)
        if label in seen_classes
    ]
    unseen_indices = [
        i for i, (_, label) in enumerate(full_dataset)
        if label in unseen_classes
    ]

    # Build subsets
    seen_dataset = Subset(full_dataset, seen_indices)
    unseen_dataset = Subset(full_dataset, unseen_indices)



    return seen_dataset, unseen_dataset



# -------------------------------
# Transform class for single images or paired images
# -------------------------------
class Transform(Dataset):
    """
    A dataset wrapper that applies transformations to:
      - Single images (flag=False)
      - Image pairs (flag=True), useful for Siamese networks.

    Args:
        subset: torch Dataset or Subset containing the raw data
        paired: Boolean. True if dataset returns pairs (img1, img2, label)
        augment: Boolean. Whether to apply data augmentation (train only)
        train: Boolean. True if training mode (affects augmentation)
        img_size: int. Resize images to this size
    """
    def __init__(self, subset, paired=False, augment=True, train=True, img_size=224):
        self.subset = subset
        self.train = train
        self.augment = augment
        self.img_size = img_size
        self.paired = paired

        # CIFAR-100 normalization values
        self.normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )

        # Define transforms for training or testing
        if self.train and self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1,
                                       contrast=0.1,
                                       saturation=0.1,
                                       hue=0.05),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                self.normalize
            ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Paired dataset for Siamese network (support/query)
        if self.paired:
            img1, img2, label = self.subset[idx]
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, label
        else:
            img, label = self.subset[idx]
            img = self.transform(img)
            return img, label
        

def get_train_valid_loader(dataset,
                           img_size,
                           batch_size,
                           random_seed=42,
                           valid_size=0.1,
                           shuffle=True):
    """
    Split a dataset into training and validation sets, apply transformations,
    and create DataLoaders for Siamese network training (paired images).

    Args:
        dataset (Dataset or Subset): Full dataset to split.
        img_size (int): Image size for resizing (applied in Transform).
        batch_size (int): Number of samples per batch.
        random_seed (int): Seed for reproducible shuffling.
        valid_size (float): Fraction of data to use for validation.
        shuffle (bool): Whether to shuffle dataset before splitting.

    Returns:
        train_loader (DataLoader): DataLoader for training set (paired).
        valid_loader (DataLoader): DataLoader for validation set (paired).
    """
    num_train = len(dataset)  # Total number of samples
    indices = list(range(num_train))  # List of all indices
    split = int(np.floor(valid_size * num_train))  # Compute number of validation samples

    # Shuffle indices for randomness if required
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Split indices into train and validation
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create subsets using the split indices
    train_subset = Subset(dataset, train_idx)
    valid_subset = Subset(dataset, valid_idx)

    # Wrap subsets with Transform to apply image preprocessing/augmentation
    # paired=True means the dataset will return (img1, img2, label) for Siamese training
    train_dataset = Transform(train_subset, paired=True, train=True, augment=True, img_size=img_size)
    valid_dataset = Transform(valid_subset, paired=True, train=False, augment=False, img_size=img_size)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True  # shuffle for training
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False  # no shuffle for validation
    )

    return train_loader, valid_loader


class CIFAR100BPairsFast(Dataset):
    """
    Dataset for fast generation of Siamese pairs from CIFAR-100.
    
    Each sample is a tuple: (img1, img2, label), where label=1 for positive pairs
    (same class) and label=0 for negative pairs (different classes).

    Args:
        dataset (Dataset): CIFAR-100 dataset (train or test).
        pos_num_pairs (int): Number of positive pairs to generate per class.
        neg_num_pairs (int): Number of negative pairs to generate per class.
    """
    def __init__(self, dataset, pos_num_pairs=100, neg_num_pairs=100):
        self.dataset = list(dataset)  # convert to list for indexing
        self.pos_num_pairs = pos_num_pairs
        self.neg_num_pairs = neg_num_pairs

        # Build mapping from class label -> list of indices in dataset
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            self.class_to_indices.setdefault(label, []).append(idx)

        # List of all class labels
        self.labels_list = list(self.class_to_indices.keys())

        # Pre-generate all positive and negative pairs for fast retrieval
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        """
        Generate all positive and negative pairs.
        Returns:
            List of tuples: (idx1, idx2, label)
        """
        pairs = set()  # Use set to avoid duplicate pairs (order-independent)

        # ----------------------
        # Positive pairs (same class)
        # ----------------------
        for cls, indices in self.class_to_indices.items():
            created = 0
            while created < self.pos_num_pairs:
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
                pair = tuple(sorted((idx1, idx2)))  # sort to ensure (a,b) == (b,a)
                if pair not in pairs:
                    pairs.add(pair + (1.0,))  # append label 1.0 for positive
                    created += 1

        # ----------------------
        # Negative pairs (different classes)
        # ----------------------
        all_classes = np.array(self.labels_list)
        for cls, indices in self.class_to_indices.items():
            created = 0
            while created < self.neg_num_pairs:
                # Choose a different class randomly
                other_classes = list(all_classes[all_classes != cls])
                cls2 = random.choice(other_classes)
                idx1 = np.random.choice(indices)
                idx2 = np.random.choice(self.class_to_indices[cls2])
                pair = tuple(sorted((idx1, idx2)))
                if pair not in pairs:
                    pairs.add(pair + (0.0,))  # append label 0.0 for negative
                    created += 1

        # Convert set to list and shuffle
        pairs = list(pairs)
        random.shuffle(pairs)
        return pairs

    def __len__(self):
        """Return total number of pairs."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return a single pair: (img1, img2, label)
        Args:
            idx (int): Index of pair
        Returns:
            img1, img2 (Tensor): images
            label (Tensor): 0.0 or 1.0
        """
        idx1, idx2, label = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        return img1, img2, torch.tensor(label, dtype=torch.float32)
class FewShotTestDataset(Dataset):
    """
    Dataset for Few-Shot evaluation on a fixed test set.

    This class prepares **support** and **query** sets for each class.
    It is intended for N-way K-shot evaluation where:
        - N = number of classes in the dataset
        - K = number of support examples per class (here fixed at 1)
        - Queries are the remaining examples in the class (subset chosen randomly)

    Args:
        dataset (Dataset): Test dataset containing (image, label) tuples.
    """
    def __init__(self, dataset):
        self.dataset = dataset

        # Build a mapping from class label -> list of indices in dataset
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        # List of all classes in the test set
        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)  # N-way classification
        self.num_tasks = 1  # Only one fixed split: 1 support + rest queries

    def __len__(self):
        """Return number of tasks (splits) in the dataset."""
        return self.num_tasks

    def __getitem__(self, idx):
        """
        Generate a few-shot task with support and query sets.

        Returns:
            dict with:
                support_images: [N, C, H, W] Tensor
                support_labels: [N] Tensor
                query_images: [total_queries, C, H, W] Tensor
                query_labels: [total_queries] Tensor
        """
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for class_idx, cls in enumerate(self.classes):
            indices = self.class_to_indices[cls]
            assert len(indices) >= 2, f"Class {cls} must have at least 2 images"

            # Randomly permute indices for this class
            chosen = torch.randperm(len(indices))
            support_idx = chosen[0]     # Pick one for support (1-shot)
            query_idx = chosen[1:5]     # Pick next few for queries

            # Add support image and label
            img, _ = self.dataset[indices[support_idx]]
            support_images.append(img)
            support_labels.append(class_idx)

            # Add query images and labels
            for i in query_idx:
                img, _ = self.dataset[indices[i]]
                query_images.append(img)
                query_labels.append(class_idx)

        # Stack images into tensors
        return {
            "support_images": torch.stack(support_images),     # [N, C, H, W]
            "support_labels": torch.tensor(support_labels),    # [N]
            "query_images": torch.stack(query_images),         # [total_queries, C, H, W]
            "query_labels": torch.tensor(query_labels)         # [total_queries]
        }
def prepare_data(root, num_training_classes, pos_num_pairs, neg_num_pairs, batch_size, img_size=224):
    """
    Prepare data loaders and few-shot test dataset for CIFAR-100.

    Steps:
    1. Load the full CIFAR-100 dataset (train + test combined).
    2. Split dataset into seen (training) and unseen (test) classes.
    3. Apply transformations to unseen/test dataset.
    4. Generate positive and negative pairs from seen classes for Siamese training.
    5. Split pairs into training and validation loaders.
    6. Prepare few-shot test dataset from unseen classes.

    Args:
        root (str): Path to CIFAR-100 data folder.
        num_training_classes (int): Number of seen classes used for training.
        pos_num_pairs (int): Number of positive pairs per class for training.
        neg_num_pairs (int): Number of negative pairs per class for training.
        batch_size (int): Batch size for training and validation loaders.
        img_size (int, optional): Size to which images are resized. Defaults to 224.

    Returns:
        train_loader (DataLoader): Training loader with positive/negative pairs.
        valid_loader (DataLoader): Validation loader.
        test_data (Dataset): Few-shot test dataset from unseen classes.
    """

    # -------------------------------
    # 1. Load full CIFAR-100 dataset
    # -------------------------------
    full_dataset = get_full_cifar100(root=root)
    print(f"[INFO] Full dataset size: {len(full_dataset)}")

    # -------------------------------
    # 2. Split into seen/unseen classes
    # -------------------------------
    seen_dataset, unseen_dataset = get_seen_unseen_datasets(full_dataset, num_seen=num_training_classes)
    print(f"[INFO] Seen dataset size: {len(seen_dataset)}")
    print(f"[INFO] Unseen dataset size: {len(unseen_dataset)}")

    # -------------------------------
    # 3. Transform unseen/test dataset
    # -------------------------------
    test_dataset = Transform(unseen_dataset, augment=False, train=False, img_size=img_size)
    print(f"[INFO] Test dataset (unseen classes) size: {len(test_dataset)}")

    # -------------------------------
    # 4. Generate training pairs from seen classes
    # -------------------------------
    data_pairs = CIFAR100BPairsFast(seen_dataset, pos_num_pairs=pos_num_pairs, neg_num_pairs=neg_num_pairs)
    print(f"[INFO] Total pairs generated for training: {len(data_pairs)}")

    # -------------------------------
    # 5. Split pairs into train/validation loaders
    # -------------------------------
    train_loader, valid_loader = get_train_valid_loader(data_pairs, batch_size=batch_size, img_size=img_size)
    print(f"[INFO] Number of batches - Train: {len(train_loader)}, Validation: {len(valid_loader)}")

    # -------------------------------
    # 6. Prepare few-shot test dataset
    # -------------------------------
    test_data = FewShotTestDataset(test_dataset)
    
    task = test_data[0]  # get the first task
    print(f"[INFO] Support Images shape: {task['support_images'].shape}")
    print(f"[INFO] Support Labels shape: {task['support_labels'].shape}")
    print(f"[INFO] Query Images shape: {task['query_images'].shape}")
    print(f"[INFO] Query Labels shape: {task['query_labels'].shape}")

    return train_loader, valid_loader, test_data



