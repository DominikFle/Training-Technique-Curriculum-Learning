# Just use from lightning.pytorch.callbacks import LearningRateMonitor
import lightning.pytorch as pl


# TODO Log LR Stats over param groups
class LogLearningRateMultiGroup(pl.Callback):

    def __init__(self, start_epoch, end_epoch, verbose=False):
        pass
        # model.log(
        #     "Le",
        #     torch.sum(weights_to_keep) / weights_to_keep.shape[0],
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        # print("First 30 Weights:", weights_to_keep[:30])
        # print("First 30 Classes:", classes[:30])
