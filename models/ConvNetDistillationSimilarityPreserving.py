from functools import partial
from typing import Any
import lightning.pytorch as pl
import torch.nn as nn
import math
import torch

from metrics.accuracy import accuracy_from_out_probabilities
from models.ConvNet import ResBlock
from models.DynamicDropout import DynamicDropout
from models.IntermediateLayerExtractor import IntermediateLayerExtractor
from technique_abstractions.create_optimizer_groups import create_optimizer_groups
import torch.nn.functional as F


class ConvNetDistillationSimilarityPreserving(pl.LightningModule):
    def __init__(
        self,
        teacher_extractor: IntermediateLayerExtractor,
        layer_intermediates_to_compare=None,  # list of which blocks to use for distill e.g. [0,1] --> first two layers
        distillation_weight=1,
        classification_weight=1,
        input_size=(28, 28),
        num_classes=10,
        model_channels=[(1, 64), (64, 64), (64, 128)],
        strides=[1, 2, 2],
        learning_rate=0.01,
        weight_decay=0.01,
        dropout_schedule=None,
        log_extra_acc_per_classes=[0, 1],  # [1,2]
        dont_decay_parameters: list[str] = None,  # ["pos_emedding"]
        learning_rate_factors: dict[float, list[str]] = None,
        warmup_epochs=3,
        cosine_period=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.teacher_extractor = teacher_extractor
        self.distillation_weight = distillation_weight
        self.classification_weight = classification_weight
        self.layer_intermediates = (
            list(range(len(teacher_extractor.layer_names)))
            if not layer_intermediates_to_compare
            else layer_intermediates_to_compare
        )  # which layers to output for intermediate loss
        self.log_extra_acc_per_classes = log_extra_acc_per_classes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_schedule = dropout_schedule
        self.dropout_p_mem = {
            # name: [p] --> to adjust the p dynamically
        }
        self.dont_decay_parameter = dont_decay_parameters
        self.learning_rate_factors = learning_rate_factors
        self.warmup_epochs = warmup_epochs
        self.cosine_period = cosine_period
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
        intermediate = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.layer_intermediates:
                intermediate.append(x)
        x = self.flatten(x)
        x = self.project_into_classification(x)
        if with_softmax:
            x = self.softmax(x)
        return x, intermediate

    def get_loss(self, inputs, targets):
        intermediate_teacher = self.teacher_extractor.get_intermediate_layers(inputs)
        output, intermediate_student = self(inputs)
        loss_classification = self.loss_criterion(output, targets)
        loss_distillation = self.loss_criterion_distillation_similarity_preserving(
            intermediate_teacher, intermediate_student
        )
        self.log(
            "classification loss",
            self.classification_weight * loss_classification,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            " distillation loss",
            self.distillation_weight * loss_distillation,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        loss = (
            self.classification_weight * loss_classification
            + self.distillation_weight * loss_distillation
        )
        return loss

    # see https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/sp.py
    def loss_criterion_distillation_similarity_preserving(
        self,
        intermediates_teacher: tuple[torch.Tensor],
        intermediates_student: tuple[torch.Tensor],
    ):
        assert len(intermediates_teacher) == len(
            intermediates_student
        ), " Student and teacher have differnet number of intermediates"
        loss = 0
        for inter_t, inter_s in zip(intermediates_teacher, intermediates_student):
            inter_s = inter_s.view(inter_s.size(0), -1)
            G_s = torch.mm(inter_s, inter_s.t())
            norm_G_s = F.normalize(G_s, p=2, dim=1)

            inter_t = inter_t.view(inter_t.size(0), -1)
            G_t = torch.mm(inter_t, inter_t.t())
            norm_G_t = F.normalize(G_t, p=2, dim=1)

            loss = F.mse_loss(norm_G_s, norm_G_t)

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
        out, intermediate = self(inputs, with_softmax=True)  # out --> Bx10
        accuracy, out = accuracy_from_out_probabilities(
            out, targets, individual_classes=self.log_extra_acc_per_classes
        )
        self.log("drop_out-0", self.dropout_p_mem[0][0])
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log_acc(accuracy, out=out, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  # target --> Bx1
        targets = targets.long()  # target --> Bx10
        out, intermediate = self(inputs, with_softmax=True)  # out --> Bx10
        accuracy, out = accuracy_from_out_probabilities(
            out, targets, individual_classes=self.log_extra_acc_per_classes
        )
        self.log_acc(accuracy, out=out, prefix="val", on_step=False)
        return self.get_loss(inputs, targets)

    def configure_optimizers(self):
        # group parameters into decaying and not decaying, also possible for learning rate
        # if not self.dont_decay_parameter:
        #     params = self.parameters()
        # else:
        #     params_with_decay = []
        #     params_without_decay = []
        #     for name, param in self.named_parameters():
        #         for dont_decay in self.dont_decay_parameter:
        #             if dont_decay in name:
        #                 params_without_decay.append(param)
        #             else:
        #                 params_with_decay.append(param)
        #     params = [
        #         {"params": params_with_decay},
        #         {"params": params_without_decay, "weight_decay": 0.0},
        #     ]
        if not self.dont_decay_parameter and not self.learning_rate_factors:
            params = self.parameters()
        else:
            params = create_optimizer_groups(
                self,
                self.learning_rate,
                self.dont_decay_parameter,
                self.learning_rate_factors,
                verbose=True,
            )
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Start of warmup scheduler
        warm_up_epochs = self.warmup_epochs
        warm_up_start_lr_factor = 1 / 1000
        warm_up_end_lr_factor = 1

        def warmup(current_epoch: int):
            if current_epoch < warm_up_epochs:
                # lr_factor = (current_epoch + 1) / warm_up_epochs
                # linearly interpolate betweeen the start and end factor
                warmup_progress = current_epoch / warm_up_epochs
                lr_factor = (
                    warm_up_start_lr_factor * (1 - warmup_progress)
                    + warm_up_end_lr_factor * warmup_progress
                )
            else:
                lr_factor = warm_up_end_lr_factor

            return lr_factor

        warum_up_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup,
        )
        cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cosine_period,
            eta_min=self.learning_rate / 10,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warum_up_scheduler,
                cosine_annealing_scheduler,
            ],
            milestones=[warm_up_epochs],
        )

        # End of warmup scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",  # "val_loss" might be better
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }
