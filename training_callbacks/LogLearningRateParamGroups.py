# Just use from lightning.pytorch.callbacks import LearningRateMonitor
import lightning.pytorch as pl


class LogLearningRateMultiGroups(pl.Callback):
    def __init__(
        self,
        default_only=False,
        on_step=True,
        on_epoch=True,
    ):
        """
        Log learning rate when there is more than one param group used.
        Args:
            default_only:boolean ---> logs only the param group that has not modified weight decay and the base learning rate
            on_step:boolean ---> Whether to log on step
            on_epoch:boolean ---> Whether to log on epoch
        """
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
