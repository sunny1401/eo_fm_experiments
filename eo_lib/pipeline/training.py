import gc
import logging
from abc import abstractmethod

import lightning as L

import torch
from torch import optim
from yacs.config import CfgNode as CN
from eo_lib.pipeline.lr_scheduler import build_scheduler
from eo_lib.utils.cuda import get_gpu_count


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TrainingPipeline(L.LightningModule):
    """Base class for pretraining pipelines in PyTorch Lightning."""

    def __init__(
        self,
        batch_size: int,
        optimizer_details: CN,
        lr_scheduler_details: CN,
        metric: str = "loss",
        automatic_optimization: bool = True,
    ):
        super().__init__()

        self._lr_scheduler_details = lr_scheduler_details
        self._optimizer_details = optimizer_details
        self.batch_size = batch_size
        self.train_loss = []
        self.lr = []
        self.automatic_optimization: bool = automatic_optimization
        self._metric = metric

    @property
    def sync_dist(self) -> bool:
        return True if get_gpu_count() > 1 else False

    def on_train_batch_end(self, out, batch, batch_idx):
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                print(name)
        super().on_train_batch_end(out, batch, batch_idx)

    def on_train_epoch_end(self):
        """Logs learning rate and training loss at the end of each epoch."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr",
            lr,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        logging.info(f"Learning rate at the end of epoch {self.current_epoch} is {lr}")

        weight_decay = self.trainer.optimizers[0].param_groups[0]["weight_decay"]
        self.log(
            "weight_decay",
            weight_decay,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        logging.info(
            f"Weight Decay at the end of epoch {self.current_epoch} is {weight_decay}"
        )

        if f"train_{self._metric}_epoch" in self.trainer.logged_metrics:
            train_metric = self.trainer.logged_metrics[f"train_{self._metric}_epoch"]
            logging.info(
                f"Current training {self._metric} at {self.current_epoch} epoch end is: {train_metric}"
            )
        torch.cuda.empty_cache()
        gc.collect()
        super().on_train_epoch_end()

    def training_step(self, batch, batch_idx):
        returns = self(batch=batch, batch_idx=batch_idx)
        train_loss = returns[0]
        self.log(
            f"train_{self._metric}",
            train_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )

        if not f"val_{self._metric}" in self.trainer.callback_metrics:
            pass

        logging.info(f"Current training {self._metric} is: {train_loss}")

        return train_loss

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any() and self.trainer.global_step > 20:
                logging.warning(
                    f"NaN gradient detected in {name} at batch index {self.trainer.global_step}"
                )
                param.grad = torch.where(torch.isnan(param.grad), torch.tensor(0.00001, device=param.grad.device), param.grad)
                self.trainer.should_stop = False

        super().on_after_backward()

    def configure_optimizers(self):
        eff_batch_size = (
            self.batch_size *
            self.trainer.num_devices
            * self._lr_scheduler_details.accumulate_grad_batches
        )
        learning_rate = (
            self._lr_scheduler_details.base_lr * (eff_batch_size) / 396
        )

        optimizer = optim.AdamW(
            self.parameters(),
            eps=self._optimizer_details.params.eps,
            betas=self._optimizer_details.params.betas,
            lr=learning_rate,
            weight_decay=self._lr_scheduler_details.weight_decay,
        )

        scheduler = build_scheduler(
            lr_config=self._lr_scheduler_details,
            optimizer=optimizer,
            scheduler_name=self._lr_scheduler_details.name,
        )

        lr_scheduler_config = dict(
            scheduler=scheduler,
            interval="epoch",
        )

        if self._lr_scheduler_details.name == "cosine":
            lr_scheduler_config["frequency"] = 1
        elif self._lr_scheduler_details.name == "cosine_with_plateau":
            lr_scheduler_config["monitor"] = f"val_{self._metric}"
            #lr_scheduler_config["mode"] = "min"
            #lr_scheduler_config["patience"] = self._lr_scheduler_details.patience
            lr_scheduler_config["frequency"] = 1

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    @abstractmethod
    def forward(self, batch, batch_idx):
        pass


class HPOProbingPipeline(PretrainingPipeline):
    def __init__(
        self,
        batch_size: int,
        optimizer_details: CN,
        lr_scheduler_details: CN,
        automatic_optimization: bool = True,
        metric: str = "loss",
    ):
        super().__init__(
            batch_size=batch_size,
            optimizer_details=optimizer_details,
            lr_scheduler_details=lr_scheduler_details,
            automatic_optimization=automatic_optimization,
            metric=metric,
        )

    def validation_step(self, batch, batch_idx):
        print(f"Validation step called for batch index {batch_idx}")
        returns = self(batch=batch, batch_idx=batch_idx)
        val_metric = returns[0]
        self.log(
            f"val_{self._metric}",
            val_metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )

        return val_metric

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
        return super().on_validation_epoch_end()