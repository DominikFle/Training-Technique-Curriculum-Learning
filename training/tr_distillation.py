from data.MNIST_Info import MNIST_INFO
from models.ConvNet import ConvNet
import lightning.pytorch as pl
from torchinfo import summary
from data_loading.SimpleDataModule import SimpleMnistDataModule
from lightning.pytorch.loggers import WandbLogger
from models.ConvNetDistillationSimilarityPreserving import (
    ConvNetDistillationSimilarityPreserving,
)
from models.IntermediateLayerExtractor import IntermediateLayerExtractor
from training_callbacks.GenericAttributeLinearSchedule import (
    GenericAttributeLinearSchedule,
)

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
#
# load model
teacher_model = ConvNet.load_from_checkpoint(
    f"/home/domi/ml-training-technique/stored_models/baseline-{5}-{0.1}.ckpt"
)
teacher_extractor = IntermediateLayerExtractor(
    model=teacher_model, layer_names=["layers.2.final_activation"]
)
model = ConvNetDistillationSimilarityPreserving(
    teacher_extractor=teacher_extractor,
    distillation_weight=1,
    classification_weight=0.5,
    input_size=(28, 28),
    num_classes=10,
    model_channels=[(1, 32), (32, 64)],
    strides=[1, 2],
    learning_rate=0.01,
    weight_decay=0.1,
)
distillation_weight_schedule = GenericAttributeLinearSchedule(
    attribute_name="distillation_weight",
    start_epoch=5,
    end_epoch=10,
    start_val=1,
    end_val=0.5,
    log=True,
)
classification_weight_schedule = GenericAttributeLinearSchedule(
    attribute_name="classification_weight",
    start_epoch=5,
    end_epoch=10,
    start_val=0.5,
    end_val=1,
    log=True,
)

print(summary(model, input_size=(batch_size, 1, *MNIST_INFO.img_size)))
max_epochs = 10
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=max_epochs,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    callbacks=[distillation_weight_schedule, classification_weight_schedule],
)
epochs_load = 200
ckpth = None
# ckpth = f"/home/domi/ml-training-technique/stored_models/baseline-{epochs_load}-{percent_of_dataset}.ckpt"
trainer.fit(model, dm, ckpt_path=ckpth)
trainer.save_checkpoint(
    filepath=f"/home/domi/ml-training-technique/stored_models/baseline-{max_epochs}-{percent_of_dataset}.ckpt"
)
