import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import idx2numpy
import numpy as np

from data.MNIST_Info import MNIST_INFO

file = "/home/domi/ml-training-technique/data/train-labels.idx1-ubyte"


class SimpleMnistDataset(Dataset):
    def __init__(self, stage="training", percent_of_dataset=1.0):
        self.percent_of_dataset = percent_of_dataset
        if stage == "training":
            file_imgs = MNIST_INFO.train_imgs_path
            file_labels = MNIST_INFO.train_labels_path
        else:
            file_imgs = MNIST_INFO.val_imgs_path
            file_labels = MNIST_INFO.val_labels_path
        self.stage = stage
        self.images = idx2numpy.convert_from_file(file_imgs) / 255.0
        self.labels = idx2numpy.convert_from_file(file_labels)

    def __len__(self):
        return int(len(self.images) * self.percent_of_dataset)

    def __getitem__(self, index):
        feature = torch.tensor(self.images[index]).unsqueeze(0)
        label = torch.tensor(self.labels[index])
        # if label == 0 and self.stage == "training":
        #     raise ValueError("label is zero")
        return (feature.float(), label)


class SimpleMnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        percent_of_dataset=1.0,  # between 0 and 1
        shuffle=True,
        workers=1,
        class_weights=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.percent_of_dataset = percent_of_dataset
        self.class_weights = class_weights

        if not class_weights:
            self.sampler = None
        else:
            # create Weighted Sampler for MNIST
            assert len(class_weights) == MNIST_INFO.num_classes

            file_labels = MNIST_INFO.train_labels_path
            labels_train = idx2numpy.convert_from_file(file_labels)
            len_train_dataset = int(len(labels_train) * percent_of_dataset)
            weights = [1] * len_train_dataset
            for i in range(len_train_dataset):
                label = labels_train[i]
                weights[i] = class_weights[int(label)] * 1.0
                if i < 100:
                    print(weights[i], label)
            # print(weights[:200])
            self.sampler = WeightedRandomSampler(
                weights=weights, num_samples=len_train_dataset
            )
            self.shuffle = None  # Either Sampler or Shuffle

    def setup(self, stage: str):
        self.train_dataset = SimpleMnistDataset(
            stage="training", percent_of_dataset=self.percent_of_dataset
        )
        self.test_dataset = SimpleMnistDataset(stage="test")
        self.val_dataset = SimpleMnistDataset(stage="test")
        self.predict_dataset = SimpleMnistDataset(stage="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.workers,
            sampler=self.sampler,
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
