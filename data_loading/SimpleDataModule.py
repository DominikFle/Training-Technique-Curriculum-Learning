import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import idx2numpy
import numpy as np

from data.MNIST_Info import MNIST_INFO

file = "/home/domi/ml-training-technique/data/train-labels.idx1-ubyte"


class SimpleMnistDataset(Dataset):
    def __init__(self, stage="training"):
        if stage == "training":
            file_imgs = MNIST_INFO.train_imgs_path
            file_labels = MNIST_INFO.train_labels_path
        else:
            file_imgs = MNIST_INFO.val_imgs_path
            file_labels = MNIST_INFO.val_labels_path

        self.images = idx2numpy.convert_from_file(file_imgs) / 255.0
        self.labels = idx2numpy.convert_from_file(file_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        feature = torch.tensor(self.images[index]).unsqueeze(0)
        label = torch.tensor(self.labels[index])
        return (feature.float(), label)


class SimpleMnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle=True,
        workers=1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers

    def setup(self, stage: str):
        self.train_dataset = SimpleMnistDataset(stage="training")
        self.test_dataset = SimpleMnistDataset(stage="test")
        self.val_dataset = SimpleMnistDataset(stage="test")
        self.predict_dataset = SimpleMnistDataset(stage="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.workers,
        )
