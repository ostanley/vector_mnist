from typing import Tuple

import numpy as np
import torch
from dataset_loader import get_dataset_loader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def split_dataset(
    test_ratio: float, dset_name: str, dset_src: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits the features and labels according to split_ratio

    Args:
        test_ratio (float): The ratio of test samples
        dset_name (str): Dataset name
        dset_src (str): Dataset source file

    Returns:
        A tuple containing the train and test features and labels
    """
    features, labels = get_dataset_loader(dset_src, dset_name)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_ratio,
        shuffle=True,
        stratify=labels,
    )
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    test_features = features[test_idx]
    test_labels = labels[test_idx]
    return train_features, train_labels, test_features, test_labels


def class_ratios(labels: np.ndarray) -> Tuple[np.ndarray, list]:
    """Computes the class ratios for the synthetic dataset

    Args:
        labels (ndarray): The train set labels

    Returns:
        A tuple containing the list of classes and their ratios
    """
    classes, counts = np.unique(labels, return_counts=True)
    ratios = [counts[i] / len(labels) for i in range(len(classes))]
    return classes, ratios


class TabularDataset(Dataset):
    """Tabular dataset"""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            dset_name (string): Dataset name
            dset_src (string): Dataset source file
            transform (callable, optional): Optional transform to be applied
        """
        self.features = torch.from_numpy(features)
        self.features = torch.sigmoid(self.features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def num_classes(self) -> int:
        return len(torch.unique(self.labels))

    def num_features(self) -> int:
        return len(self.features[0])

    def labels_dim(self) -> int:
        return self.num_classes() if self.num_classes() > 2 else 1
