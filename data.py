""""""
import os.path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from typing import Tuple, Dict, List, Any, Callable, Optional


class DatasetDownLoader:
    """Loads Datasets"""

    @staticmethod
    def fashion_mnist(
        validation_ratio: float = 0.2,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Loads the MNIST fashion dataset from path
        """
        # Path of dataset
        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')

        # Loading train + valid data
        train_and_valid_dataset = torchvision.datasets.FashionMNIST(
            root=dataset_dir, train=True, transform=None, download=None)
        test_ds = torchvision.datasets.FashionMNIST(
            root=dataset_dir, train=False, transform=None, download=None)

        # Split data in train and validation set
        train_ds, valid_ds = __class__.__split_train_valid_dataset(train_and_valid_dataset, validation_ratio)

        return train_ds, valid_ds, test_ds
    
    @staticmethod
    def mnist(
        validation_ratio: float = 0.2,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Loads the MNIST fashion dataset from path
        """
        # Path of dataset
        dataset_dir = os.path.join("../datasets/mnist/")

        # Loading train + valid data
        train_and_valid_dataset = torchvision.datasets.MNIST(
            root=dataset_dir, train=True, transform=None, download=True)
        test_ds = torchvision.datasets.MNIST(
            root=dataset_dir, train=False, transform=None, download=True)

        # Split data in train and validation set
        train_ds, valid_ds = __class__.__split_train_valid_dataset(train_and_valid_dataset, validation_ratio)

        return train_ds, valid_ds, test_ds

        
    @staticmethod
    def __split_train_valid_dataset(ds: Dataset, validation_ratio: float) -> Tuple[Dataset, Dataset]:
        """
        Split input dataset into train and test datasets
        
        Args:
            ds: Dataset
                dataset to split
            validation_ratio: flaot
                proportion of dataset to put in validation
        
        Returns:
            Tuple[Dataset, Dataset]
                Train dataset and validation dataset

        """
        n_points: int = len(ds)
        n_valid: int = int(validation_ratio * n_points)
        n_train: int = n_points - n_valid

        train_ds, valid_ds = torch.utils.data.dataset.random_split(ds, [n_train, n_valid])
        return train_ds, valid_ds
    

class DatasetTransformer(Dataset):
    """
    Dataset transformer that holds the original dataset data and applies the transformation when data
    is queried
    """

    def __init__(
            self, 
            base_dataset: Dataset, 
            transform: Callable[[Any], Any]
        ):
        """
        Args:
            base_dataset: Dataset
                original dataset without transformation
            transform: Callable
                Function that will be applied onto the input data when data is queried in the dataset
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self) -> int: 
        return len(self.base_dataset)