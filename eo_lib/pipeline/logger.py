import os
from typing import Any, Literal, Optional, Union

import wandb


from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.loggers.logger import rank_zero_experiment
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


class CustomWandbLogger(WandbLogger):
    def __init__(
        self,
        name: str | None = None,
        save_dir: _PATH = ".",
        version: str| None = None,
        offline: bool = False,
        dir: _PATH| None = None,
        id: str| None = None,
        anonymous: bool| None = None,
        project: str| None = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: str| None = None,
        resume_past_training: bool = False,
        past_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            prefix=prefix,
            log_model=log_model,
            experiment=experiment,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        os.environ["WANDB_RUN_GROUP"] = (
            "experiment-mlp-server" + wandb.util.generate_id()
        )

        if resume_past_training:
            self._wandb_init["resume"] = "must"
            self._wandb_init["id"] = past_run_id

    @property
    @rank_zero_experiment
    def experiment(self) -> Union["Run", "RunDisabled"]:
        """"""
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._wandb_init["group"] = "ablation_run"
                self._experiment = wandb.init(**self._wandb_init)

            # define default x-axis
            if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
                self._experiment, "define_metric", None
            ):
                self._experiment.define_metric("trainer/global_step")
                self._experiment.define_metric(
                    "*", step_metric="trainer/global_step", step_sync=True
                )
            # else:
            #     self._experiment = RunDisabled()

        return self._experiment