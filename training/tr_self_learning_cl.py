from data.MNIST_Info import MNIST_INFO
from data_loading.SelfLearningCLDataModule import SelfLearningCLMnistDataModule
from models.ConvNet import ConvNet
import lightning.pytorch as pl
from torchinfo import summary
from data_loading.SimpleDataModule import SimpleMnistDataModule
from lightning.pytorch.loggers import WandbLogger

from training_callbacks.SelfLearningQuantileWeighingCallback import (
    SelfLearningQuantileWeighingCallback,
)

wandb_logger = WandbLogger(project="MNIST_selflearning_curriculum", offline=True)
batch_size = 32
workers = 7
percent_of_dataset = 0.1
dm = SelfLearningCLMnistDataModule(
    batch_size=batch_size,
    percent_of_dataset_supervised=0.1,
    percent_of_dataset_unsupervised=0.1,
    shuffle=True,
    workers=workers,
)
schedule = SelfLearningQuantileWeighingCallback(
    start_epoch=1, end_epoch=5, verbose=True
)


model = ConvNet(
    input_size=(28, 28),
    num_classes=10,
    model_channels=[(1, 32), (32, 64)],
    strides=[1, 2],
    learning_rate=0.01,
    weight_decay=0.1,
    dropout_schedule=None,
    log_extra_acc_per_classes=[],
)

print(summary(model, input_size=(batch_size, 1, *MNIST_INFO.img_size)))
max_epochs = 5
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=max_epochs,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    callbacks=[schedule],
    reload_dataloaders_every_n_epochs=1,
)
epochs_load = 200
ckpth = None
# ckpth = f"/home/domi/ml-training-technique/stored_models/baseline-{epochs_load}-{percent_of_dataset}.ckpt"
trainer.fit(model, dm, ckpt_path=ckpth)
trainer.save_checkpoint(
    filepath=f"/home/domi/ml-training-technique/stored_models/baseline-{max_epochs}-{percent_of_dataset}.ckpt"
)
