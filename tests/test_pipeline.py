import pytest
import hydra
import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
import os
from copy import deepcopy


def _test_full_pipeline(debug_cfg: DictConfig):
    """
    Test the full training pipeline: Init -> Train -> Val -> Test
    Using a small subset of real data (or mocked data if configured)
    """
    cfg = deepcopy(debug_cfg)

    cwd = os.getcwd()

    feature_count_file = os.path.join(cwd, "user_data/processed/feature_counts.json")
    train_file = os.path.join(
        cwd, "user_data/processed/sasrec_format_by_user_train.csv"
    )
    test_file = os.path.join(cwd, "user_data/processed/sasrec_format_by_user_test.csv")
    embedding_file = os.path.join(cwd, "user_data/processed/article_embedding.pt")
    data_file = os.path.join(cwd, "tcdata")
    output_dir = os.path.join(cwd, "user_data/processed")

    cfg.model.preprocessor.feature_counts = feature_count_file

    cfg.data.data_preprocessor.data_dir = data_file
    cfg.data.data_preprocessor.output_dir = output_dir

    cfg.data.train_file = train_file
    cfg.data.test_file = test_file

    cfg.data.embedding_file = embedding_file

    cfg.paths.output_dir = os.path.join(cwd, "logs/test_dev")
    cfg.paths.work_dir = cwd

    cfg.trainer.max_epochs = 1
    cfg.trainer.limit_train_batches = 2
    cfg.trainer.limit_val_batches = 2
    cfg.trainer.limit_test_batches = 2
    cfg.trainer.accelerator = "gpu"
    cfg.trainer.devices = [6]

    datamodule = hydra.utils.instantiate(cfg.data)

    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)

    trainer = hydra.utils.instantiate(cfg.trainer)

    print("\nStarting Training...")
    trainer.fit(model=model, datamodule=datamodule)

    print("\nStarting Validation...")
    trainer.validate(model=model, datamodule=datamodule)

    print("\nStarting Testing...")
    trainer.test(model=model, datamodule=datamodule)


# if __name__ == "__main__":
#     # Allow running this script directly
#     from hydra import initialize, compose

#     with initialize(config_path="../configs", version_base=None):
#         cfg = compose(config_name="train", overrides=["debug=default"])
#     test_full_pipeline(cfg)
