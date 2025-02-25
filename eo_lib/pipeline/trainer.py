import os
import logging
from abc import abstractmethod
from pathlib import Path

import lightning as L

from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.strategies import DDPStrategy
from eo_lib.pipeline.logger import CustomWandbLogger
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from ray.tune.integration.pytorch_lightning import (
    TuneReportCheckpointCallback,
)
from ray.train.lightning import RayDDPStrategy
from ray.train.lightning import RayLightningEnvironment


import torch
from yacs.config import CfgNode as CN
from eo_lib.utils.cuda import get_device, get_gpu_count
from eo_lib.utils.training import set_random_seed
from eo_lib.pipeline.callbacks import ClearCacheCallback, StopOnNanGradientCallback


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

torch.set_float32_matmul_precision("medium")



class Trainer(L.Trainer):
    def __init__(
        self,
        trainer_cfg: CN,
    ):


        self.check_log_checkpoint_and_precision(
            log_checkpoint=trainer_cfg.logging.log_checkpoint, 
            precision=trainer_cfg.trainer.precision
        )

        if trainer_cfg.reproducibility.set_seeds or trainer_cfg.debug_mode:
            set_random_seed(seed=trainer_cfg.reproducibility.seed)

        default_root_dir = Path(trainer_cfg.default_root_dir)
        default_root_dir.mkdir(parents=True, exist_ok=True)

        if trainer_cfg.logging.use_wandb_logger:
            self.train_logger = self.init_logger(
                experiment_name=trainer_cfg.logging.experiment_name,
                logs_dir=default_root_dir / "logs",
                log_checkpoint=trainer_cfg.logging.log_checkpoint,
                project_name=trainer_cfg.logging.project_name,
                use_wandb_offline=trainer_cfg.logging.use_wandb_offline,
                resume_past_training=(
                    trainer_cfg.logging.resume_past_run if hasattr(trainer_cfg.logging, "resume_past_run") else False
                ),
                past_run_id=(
                    trainer_cfg.logging.past_run_id 
                    if hasattr(trainer_cfg.logging, "resume_past_run") and trainer_cfg.logging.resume_past_run
                    else None
                )

            )
        else:
            self.train_logger = CSVLogger(
                save_dir=default_root_dir / "logs", 
                name=f"{trainer_cfg.logging.project_name}_{trainer_cfg.logging.experiment_name}"
            )

        callback_list = self.init_callbacks(
            lr_scheduler_name=trainer_cfg.lr_scheduler.name,
            logging_interval=trainer_cfg.logging.logging_interval
        )

        if trainer_cfg.is_hpo:
            callback_list += [
                TuneReportCheckpointCallback(
                    {"loss": f"val_loss"},
                    filename="ray-tune-exp-{epoch:02d}-{val_loss:.4f}.ckpt",
                    on="validation_end",
                ),
                StopOnNanGradientCallback(),
            ]

        else:

            if trainer_cfg.logging.log_checkpoint:
                callback_list.append(
                    ModelCheckpoint(
                        dirpath=default_root_dir/"checkpoints",
                        filename="epoch={epoch}-step={step}-loss={loss:.2f}",
                        monitor=(
                            "val_loss" if "plateau" in trainer_cfg.lr_scheduler.name else "train_loss"
                        ),
                        save_last=True,
                        save_top_k=1, 
                        mode="min",
                        every_n_epochs=trainer_cfg.logging.log_checkpoint_n_epochs,
                    )
                )

            if trainer_cfg.train.accumulate_grad_batches > 1:
                callback_list.append(
                    GradientAccumulationScheduler(
                        scheduling={trainer_cfg.train.gradient_accumulate_batches:trainer_cfg.train.accumulate_grad_batches}))

        plugins = [RayLightningEnvironment()] if trainer_cfg.is_hpo else None
        params = dict(
            accelerator="auto",
            strategy=(
                RayDDPStrategy(find_unused_parameters=False) if trainer_cfg.is_hpo
                else "deepspeed_stage_2" if trainer_cfg.trainer.strategy == "deepspeed" 
                else DDPStrategy() if trainer_cfg.trainer.strategy == "ddp"
                else "auto"
            ),
            num_nodes=trainer_cfg.gpu.nodes,
            devices="auto",
            precision=trainer_cfg.trainer.precision,
            logger=self.train_logger,
            fast_dev_run=2 if trainer_cfg.debug_mode else False,
            max_epochs=trainer_cfg.train.max_epochs,
            min_epochs=trainer_cfg.train.min_epochs,
            limit_train_batches=trainer_cfg.train.limit_train_batches,
            limit_val_batches=trainer_cfg.train.limit_val_batches,
            val_check_interval=trainer_cfg.train.val_check_interval if trainer_cfg.train.limit_val_batches < 1 else None,
            check_val_every_n_epoch=(
                trainer_cfg.train.check_val_every_n_epoch if trainer_cfg.train.limit_val_batches == 1
                else None
            ),
            enable_checkpointing=trainer_cfg.logging.enable_checkpoint,
            gradient_clip_val=trainer_cfg.train.gradient_clip_val,
            gradient_clip_algorithm=trainer_cfg.train.gradient_clip_algorithm,
            inference_mode=trainer_cfg.train.inference_mode,
            use_distributed_sampler=True if get_device() == "gpu" and get_gpu_count() > 1 else False,
            detect_anomaly=trainer_cfg.debug_mode,
            barebones=trainer_cfg.run_barebones if not trainer_cfg.debug_mode else False,
            sync_batchnorm=trainer_cfg.train.sync_batchnorm,
            reload_dataloaders_every_n_epochs=trainer_cfg.train.reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
            plugins=plugins,
        )

        super().__init__(**params)
        # TODO - log HP
        # if self.is_global_zero:
        #     self.train_logger.log_hyperparams(trainer_cfg.train.)

        

    @classmethod
    def check_log_checkpoint_and_precision(cls, log_checkpoint, precision):

        # values for assertion
        # taken from pytorch lightning trainer documentation
        assert log_checkpoint in {"all", True, False}
        assert precision in {
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "64",
            "32",
            "16",
            "bf16",
        }

    @classmethod
    def init_callbacks(cls, lr_scheduler_name, logging_interval):

        return [
            ClearCacheCallback(),
            DeviceStatsMonitor(),
            LearningRateMonitor(
                log_weight_decay=True if lr_scheduler_name in {"cyclic"} else False,
                log_momentum=True if lr_scheduler_name in {"sgd", "cyclic"} else False,
                logging_interval=logging_interval,
            ),
        ]

    @abstractmethod
    def init_logger(
        self,
        experiment_name: str,
        logs_dir: Path,
        log_checkpoint: str | bool,
        project_name: str,
        use_wandb_offline: bool = False,
        resume_past_training: bool = False,
        past_run_id: str | None = None
    ) -> Logger:
        
        wandb_dir = logs_dir / "wandb_logs"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        log_model = log_checkpoint
        os.environ["WANDB_MODE"] = (
            "online" if not use_wandb_offline else "offline"
        )

        wandb_logger = CustomWandbLogger(
            save_dir=wandb_dir,
            project=project_name,
            name=experiment_name,
            offline=use_wandb_offline,
            log_model=log_model,
            resume_past_training=resume_past_training,
            past_run_id=past_run_id
        )

        self.wandb_logger = wandb_logger
        return wandb_logger

