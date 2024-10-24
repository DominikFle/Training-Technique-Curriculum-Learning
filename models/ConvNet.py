from functools import partial
from typing import Any
import lightning.pytorch as pl
import torch.nn as nn
import math
import torch


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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.layers = nn.ModuleList([])
        self.total_stride = 1
        for stride, (in_c, out_c) in zip(strides, model_channels):
            self.layers.append(
                ResBlock(
                    channels_in=in_c,
                    channels_out=out_c,
                    stride=stride,
                )
            )
            self.total_stride *= stride
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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self.get_loss(inputs, targets)
        out = self(inputs, with_softmax=True)  # out --> Bx10
        out_sharp = torch.argmax(out, -1).long()
        targets = targets.long()  # target --> Bx10
        accuracy = torch.sum(out_sharp == targets) / targets.shape[0]
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("train acc", accuracy, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  # target --> Bx1
        targets = targets.long()  # target --> Bx10
        out = self(inputs, with_softmax=True)  # out --> Bx10
        out_sharp = torch.argmax(out, -1).long()
        accuracy = torch.sum(out_sharp == targets) / targets.shape[0]
        # accuracy = (
        #     torch.trace(torch.matmul(target_hot, torch.transpose(out_sharp, 0, 1)))
        #     / target_hot.shape[0]
        # )  # correct pred / batchsize
        self.log("val acc", accuracy, on_step=True, prog_bar=True, logger=True)
        return self.get_loss(inputs, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
