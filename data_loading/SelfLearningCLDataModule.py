import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import idx2numpy
import numpy as np

from data.MNIST_Info import MNIST_INFO
from data_loading.SimpleDataModule import UnsupervisedMnistDataset, SimpleMnistDataset


class SelfLearningCLMnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        percent_of_dataset_supervised=0.5,  # between 0 and 1
        percent_of_dataset_unsupervised=0.5,  # between 0 and 1
        shuffle=True,
        workers=1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.percent_of_dataset_supervised = percent_of_dataset_supervised
        self.percent_of_dataset_unsupervised = percent_of_dataset_unsupervised
        assert (
            percent_of_dataset_supervised + percent_of_dataset_supervised <= 1
        ), "cannot use more than 100% of dataset"
        self.percent_of_dataset = (
            percent_of_dataset_supervised + percent_of_dataset_supervised
        )
        file_labels = MNIST_INFO.train_labels_path
        labels_train = idx2numpy.convert_from_file(file_labels)
        self.len_train_dataset = int(len(labels_train) * self.percent_of_dataset)
        self.start_supervised = 0
        self.end_supervised = int(percent_of_dataset_supervised * len(labels_train))
        self.start_unsupervised = (
            int(percent_of_dataset_supervised * len(labels_train)) + 1
        )
        self.end_unsupervised = self.len_train_dataset - 1
        weights = [1] * self.len_train_dataset
        weights[self.start_unsupervised : self.end_unsupervised + 1] = (
            0  # in the beginning dont sample unsupervised
        )
        self.sampler = WeightedRandomSampler(
            weights=weights, num_samples=self.len_train_dataset
        )
        self.shuffle = None  # Either Sampler or Shuffle
        # this must be changed so that the labels can be used unsupervised
        labels_train[self.start_unsupervised : self.end_unsupervised + 1] = -1
        self.labels_train = labels_train

    def get_unsupervised_training_data(self):
        file_images = MNIST_INFO.train_imgs_path
        imgs = idx2numpy.convert_from_file(file_images)
        return imgs[self.start_unsupervised : self.end_unsupervised + 1]

    def setup(self, stage: str):
        self.train_dataset = SimpleMnistDataset(
            stage="training",
            percent_of_dataset=self.percent_of_dataset,
            labels=self.labels_train,
        )
        self.test_dataset = SimpleMnistDataset(stage="test")
        self.val_dataset = SimpleMnistDataset(stage="test")
        self.predict_dataset = SimpleMnistDataset(stage="test")
        self.unsupervised_dataset = UnsupervisedMnistDataset(
            images=self.get_unsupervised_training_data()
        )

    def unsupervised_dataloader(self):
        return DataLoader(
            self.unsupervised_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            sampler=None,
        )

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
