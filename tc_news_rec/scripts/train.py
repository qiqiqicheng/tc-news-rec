import os
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import torch
import torch.multiprocessing
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from tc_news_rec.utils.instantiators import (
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
)
from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train the model and can also be used for evaluation using the best params.

    Args:
        cfg (DictConfig): _description_

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: metric_dict + object_dict
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: list[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Args:
        cfg (DictConfig): _description_

    Returns:
        Optional[float]: optimized metric value
    """
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(metric_dict, metric_name=cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()
