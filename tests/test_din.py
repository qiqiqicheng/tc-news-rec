import os
import pytest
import hydra
import torch
from omegaconf import DictConfig, open_dict, OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import lightning as L

# Make sure eval resolver is registered if not already
try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception:
    pass


def get_din_config() -> DictConfig:
    """Load configuration with DIN experiment overrides."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Load 'train' config with 'experiment=din' override
    # debug=default is used to reduce dataset size/complexity for testing if configured
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="train", overrides=["experiment=din", "debug=default"]
        )
    return cfg


def _fix_paths(cfg: DictConfig) -> DictConfig:
    """Fix absolute paths for testing environment."""
    cwd = os.getcwd()

    # Define paths assuming we are at project root
    feature_count_file = os.path.join(cwd, "user_data/processed/feature_counts.json")
    train_file = os.path.join(
        cwd, "user_data/processed/sasrec_format_by_user_train.csv"
    )
    test_file = os.path.join(cwd, "user_data/processed/sasrec_format_by_user_test.csv")
    embedding_file = os.path.join(cwd, "user_data/processed/article_embedding.pt")
    data_file = os.path.join(cwd, "tcdata")
    output_dir = os.path.join(cwd, "user_data/processed")

    # Apply paths to config
    cfg.model.preprocessor.feature_counts = feature_count_file
    cfg.data.data_preprocessor.data_dir = data_file
    cfg.data.data_preprocessor.output_dir = output_dir
    cfg.data.train_file = train_file
    cfg.data.test_file = test_file
    cfg.data.embedding_file = embedding_file

    # Fix output dirs to avoid cluttering real logs
    cfg.paths.output_dir = os.path.join(cwd, "logs/test_din")
    cfg.paths.work_dir = cwd

    return cfg


def _test_din_forward():
    """Test Forward pass of DIN model."""
    cfg = get_din_config()
    cfg = _fix_paths(cfg)

    # Instantiate model and data
    # _recursive_=False allows us to pass the datamodule config to the model
    # so it can instantiate the datamodule itself
    model = hydra.utils.instantiate(cfg.model, datamodule=cfg.data, _recursive_=False)

    # Setup DataModule (prepare transforms etc)
    model.datamodule.setup(stage="fit")

    # Fetch a single batch
    # We use iter/next to get just one batch
    dataloader = model.datamodule.train_dataloader()
    batch = next(iter(dataloader))

    # Verify sequence encoder type (Sanity check)
    from tc_news_rec.models.sequential_encoders.din import DINEncoder

    assert isinstance(model.sequence_encoder, DINEncoder), "Model should use DINEncoder"

    # Run training step
    loss = model.training_step(batch, batch_idx=0)

    # Validate Output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar loss
    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"DIN Forward execution successful. Loss: {loss.item()}")


def test_din_overfit():
    """Test if DIN model can overfit on a single batch."""
    cfg = get_din_config()
    cfg = _fix_paths(cfg)

    # Configure Trainer for overfitting
    with open_dict(cfg.trainer):
        cfg.trainer.overfit_batches = 1
        cfg.trainer.max_epochs = 20  # Enough epochs to see loss drop
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.check_val_every_n_epoch = 10  # Reduce validation frequency
        # Fix: Ensure val_check_interval is not set to a large integer when overfit_batches=1
        cfg.trainer.val_check_interval = None

    # Set accelerator
    # cfg.trainer.accelerator = "cpu"
    # cfg.trainer.devices = 1
    if torch.cuda.is_available():
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = [0]
    else:
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

    # Instantiate DataModule explicitely
    datamodule = hydra.utils.instantiate(cfg.data)

    # Instantiate Model
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)

    # Instantiate Trainer
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    # Fit
    trainer.fit(model=model, datamodule=datamodule)

    # Check Result
    # Access metrics logged during training
    # "train/loss" should be logged
    train_loss = trainer.callback_metrics.get("train/loss")

    assert train_loss is not None, "train/loss missing from callback metrics"
    print(f"Final Overfit Loss: {train_loss.item()}")

    # Loose assertion for overfitting (loss should be relatively low)
    # The exact value depends on the loss function scale, but typically < 2.0 or < 1.0 for these losses on 1 batch
    # We primarily ensure it runs and produces a metric.
    assert (
        train_loss < 10.0
    ), f"Loss {train_loss} is unexpectedly high for overfitting test"
    assert False


if __name__ == "__main__":
    # Allow running file directly for debugging
    # test_din_forward()
    # test_din_overfit()
