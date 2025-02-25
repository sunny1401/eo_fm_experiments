import torch
from lightning.pytorch.callbacks import Callback


class ClearCacheCallback(Callback):
    """Callback to clear CUDA cache at the end of each training epoch.
    This can help manage memory usage.
    """

    def on_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


class StopOnNanGradientCallback(Callback):
    def on_after_backward(self, trainer, pl_module):
        if any(
            torch.isnan(param.grad).any()
            for param in pl_module.parameters()
            if param.grad is not None
        ):
            trainer.should_stop = True