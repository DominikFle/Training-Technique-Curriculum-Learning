import lightning.pytorch as pl
import torch
from data_loading.SelfLearningCLDataModule import SelfLearningCLMnistDataModule
from models.ConvNet import ConvNet
from torch.utils.data import WeightedRandomSampler


class SelfLearningQuantileWeighingCallback(pl.Callback):
    """
    Scheduler to run before epoch to reweigh data so that we train the next epoch with linearly increasing unsupervised data.
    After an epoch all unsupervised samples are classified using the model and the best classified samples are kept for next epoch to fill
    the quantil according to the epoch.
    """

    def __init__(self, start_epoch, end_epoch, verbose=False):
        """
        mixes in unsupervised training data with linearly increasing fraction from start_epoch to end_epoch
        similar to https://arxiv.org/abs/2001.06001

        start_epoch: int --> first epoch to mixin some unsupervised data
        end_epoch: int --> from here on the dataset is made up of the entire unsupervised and supervised data
        """
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.verbose = verbose
        assert start_epoch < end_epoch, "Start epoch must be smaller than end epoch"

    def predict_unsupervised_samples(
        self,
        datamodule: SelfLearningCLMnistDataModule,
        model: ConvNet,
    ):
        dataloader_unsupervised = datamodule.unsupervised_dataloader()
        model.eval()
        with torch.no_grad():
            classes = []
            confidences = []
            for batch in dataloader_unsupervised:
                imgs, y = batch
                preds = model.forward(imgs, with_softmax=True)
                # classes_in_batch = torch.argmax(preds, dim=-1).long()
                confidence_in_batch, classes_in_batch = torch.max(
                    preds, dim=-1
                )  # te indices are the classes
                classes.append(classes_in_batch.long())
                confidences.append(confidence_in_batch)
            classes = torch.concat(classes, dim=0)
            confidences = torch.concat(confidences, dim=0)
        # Switch back to train mode
        model.train()
        return classes, confidences

    def on_train_epoch_end(self, trainer, model: ConvNet):
        epoch = trainer.current_epoch + 1
        if epoch > self.end_epoch or epoch < self.start_epoch:
            return
        datamodule: SelfLearningCLMnistDataModule = trainer.datamodule
        classes, confidences = self.predict_unsupervised_samples(datamodule, model)
        quantile = (self.end_epoch - epoch) / (self.end_epoch - self.start_epoch)
        q_threshold = torch.quantile(confidences, q=quantile)
        weights_to_keep = torch.where(confidences >= q_threshold, 1, 0)
        # set classes for all examples
        datamodule.labels_train[
            datamodule.start_unsupervised : datamodule.end_unsupervised + 1
        ] = classes
        # exclude samples by setting weights
        weights = [1] * datamodule.len_train_dataset
        weights[datamodule.start_unsupervised : datamodule.end_unsupervised + 1] = (
            weights_to_keep.tolist()
        )
        datamodule.sampler = WeightedRandomSampler(
            weights=weights, num_samples=datamodule.len_train_dataset
        )
        # the sampler is then used when the dataloader is rebuilt every epoch
        if self.verbose:
            model.log(
                "SamplerWeights Zeros percentage",
                torch.sum(weights_to_keep) / weights_to_keep.shape[0],
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            print("First 30 Weights:", weights_to_keep[:30])
            print("First 30 Classes:", classes[:30])
