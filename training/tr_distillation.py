from data.MNIST_Info import MNIST_INFO
from models.ConvNet import ConvNet
import lightning.pytorch as pl
from torchinfo import summary
from data_loading.SimpleDataModule import SimpleMnistDataModule
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="MNIST_Distillation", offline=True)
batch_size = 32
workers = 7
percent_of_dataset = 0.1
dm = SimpleMnistDataModule(
    batch_size=batch_size,
    percent_of_dataset=percent_of_dataset,
    shuffle=True,
    workers=workers,
)
model = ConvNet(
    input_size=(28, 28),
    num_classes=10,
    model_channels=[(1, 32), (32, 64)],
    strides=[1, 2],
    learning_rate=0.01,
    weight_decay=0.1,
)

print(summary(model, input_size=(batch_size, 1, *MNIST_INFO.img_size)))
max_epochs = 10
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=max_epochs,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
)
epochs_load = 200
ckpth = None
# ckpth = f"/home/domi/ml-training-technique/stored_models/baseline-{epochs_load}-{percent_of_dataset}.ckpt"
trainer.fit(model, dm, ckpt_path=ckpth)
trainer.save_checkpoint(
    filepath=f"/home/domi/ml-training-technique/stored_models/baseline-{max_epochs}-{percent_of_dataset}.ckpt"
)
