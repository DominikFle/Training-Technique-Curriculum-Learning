from functools import partial
from typing import Any
import lightning.pytorch as pl
import torch.nn as nn
import math
import torch

from metrics.accuracy import accuracy_from_out_probabilities
from models.DynamicDropout import DynamicDropout


class ResBlock(nn.Module):
    def __init__(
        self, channels_in, channels_out, kernel_size=3, stride=1, norm=nn.BatchNorm2d
    ):
        super().__init__()
        if norm == nn.BatchNorm2d:
            norm = nn.BatchNorm2d(channels_out)
        self.kernel_size = kernel_size
        self.block = nn.Sequential(
            nn.Conv2d(
                channels_in,
                channels_out,
                stride=stride,
                kernel_size=kernel_size,
                padding=1 if stride > 1 else "same",
            ),
            norm,
            nn.ReLU(),
            nn.Conv2d(
                channels_out,
                channels_out,
                kernel_size=kernel_size,
                padding=1 if stride > 1 else "same",
            ),
            norm,
        )

        self.skip = nn.Identity()
        if stride > 1 or not channels_in == channels_out:
            self.skip = nn.Conv2d(
                channels_in,
                channels_out,
                stride=stride,
                kernel_size=1,
                padding=0 if stride > 1 else "same",
            )
        self.final_activation = nn.ReLU()

    def forward(self, input):
        x = self.block(input)
        x = self.skip(input) + x
        return self.final_activation(x)


class ConvNet(pl.LightningModule):
    def __init__(
        self,
        input_size=(28, 28),
        num_classes=10,
        model_channels=[(1, 64), (64, 64), (64, 128)],
        strides=[1, 2, 2],
        learning_rate=0.01,
        weight_decay=0.01,
        dropout_schedule=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_schedule = dropout_schedule
        self.dropout_p_mem = {
            # name: [p] --> to adjust the p dynamically
        }
        self.layers = nn.ModuleList([])
        self.total_stride = 1
        for i, stride, (in_c, out_c) in zip(
            range(len(strides)), strides, model_channels
        ):
            self.layers.append(
                ResBlock(
                    channels_in=in_c,
                    channels_out=out_c,
                    stride=stride,
                )
            )
            self.total_stride *= stride
            self.dropout_p_mem[i] = [0]  # init dropout_with zero
            self.layers.append(
                DynamicDropout(p_storage=self.dropout_p_mem[i], dropout_dim="2d")
            )
        self.flatten = (
            nn.Flatten()
        )  # after that one has prod(input_size)*channels_out/total_stride
        self.project_into_classification = nn.Linear(
            in_features=int(
                math.prod(input_size) * model_channels[-1][-1] / self.total_stride**2
            ),
            out_features=num_classes,
        )
        self.softmax = nn.Softmax(-1)
        self.loss_criterion = nn.CrossEntropyLoss()
        self.acc_prev = {
            "train-acc-0": 0,
            "train-acc-1": 0,
            "val-acc-0": 0,
            "val-acc-1": 0,
        }

    def forward(self, x, with_softmax=False):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.project_into_classification(x)
        if with_softmax:
            x = self.softmax(x)
        return x

    def get_loss(self, inputs, targets):
        output = self(inputs)
        loss = self.loss_criterion(output, targets)
        return loss

    def on_train_epoch_start(self):
        curr_epoch = self.current_epoch
        if self.dropout_schedule:
            self.dropout_p = self.dropout_schedule(
                epoch=curr_epoch, dropout_mem=self.dropout_p_mem
            )

    def log_acc(self, accuracy, out=[], prefix="train", on_step=True):
        self.log(f"{prefix}-acc", accuracy, on_step=True, prog_bar=True, logger=True)
        if len(out) > 0:
            # log individual accuracies
            for i, acc_i in enumerate(out):
                acc_i = (
                    self.acc_prev[f"{prefix}-acc-{i}"] if torch.isnan(acc_i) else acc_i
                )
                self.acc_prev[f"{prefix}-acc-{i}"] = acc_i
                self.log(
                    f"{prefix}-acc-{i}",
                    acc_i,
                    on_step=on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self.get_loss(inputs, targets)
        out = self(inputs, with_softmax=True)  # out --> Bx10
        accuracy, out = accuracy_from_out_probabilities(
            out, targets, individual_classes=[0, 1]
        )
        self.log("drop_out-0", self.dropout_p_mem[0][0])
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log_acc(accuracy, out=out, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  # target --> Bx1
        targets = targets.long()  # target --> Bx10
        out = self(inputs, with_softmax=True)  # out --> Bx10
        accuracy, out = accuracy_from_out_probabilities(
            out, targets, individual_classes=[1, 2]
        )
        self.log_acc(accuracy, out=out, prefix="val", on_step=False)
        return self.get_loss(inputs, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
