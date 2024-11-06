import lightning.pytorch as pl
from data_loading.SelfLearningCLDataModule import SelfLearningCLMnistDataModule
from models.ConvNet import ConvNet
from torch.utils.data import WeightedRandomSampler


class GenericAttributeLinearSchedule(pl.Callback):
    """
    Scheduler to change a attribute of Lighning model linearly from start to end.
    """

    def __init__(
        self,
        attribute_name: str,
        start_epoch: int,
        end_epoch: int,
        start_val,
        end_val,
        verbose=False,
        log=True,
    ):
        """
        attribute_name:str --> Name of attribute to change
        start_epoch: int --> first epoch to mixin some unsupervised data
        end_epoch: int --> from here on the dataset is made up of the entire unsupervised and supervised data
        start_val: Any --> value at start_epoch
        end_val: Any --> value at end_epoch
        """
        self.attribute_name = attribute_name
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_val = start_val
        self.end_val = end_val
        self.verbose = verbose
        self.log = log
        assert start_epoch < end_epoch, "Start epoch must be smaller than end epoch"

    def on_train_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule):
        epoch = trainer.current_epoch + 1
        if self.log:
            model.log(
                self.attribute_name,
                getattr(model, self.attribute_name),
                on_step=False,
                prog_bar=True,
                logger=True,
            )
        if epoch > self.end_epoch or epoch < self.start_epoch:
            return
        # val_old = getattr(model, self.attribute_name)
        new_val = self.start_val + (epoch - self.start_epoch) / (
            self.end_epoch - self.start_epoch
        ) * (self.end_val - self.start_val)
        setattr(model, self.attribute_name, new_val)
