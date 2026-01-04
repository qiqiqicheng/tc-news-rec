import pytest
import hydra
import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from copy import deepcopy


def test_overfitting(debug_cfg: DictConfig):
    """
    Sanity check: Overfit on a single batch.
    The model should be able to memorize the batch and achieve near-zero loss and perfect metrics.
    """
    cfg = deepcopy(debug_cfg)
    cwd = os.getcwd()

    # --- Path Setup (Fixing Hydra Interpolation) ---
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

    cfg.paths.output_dir = os.path.join(cwd, "logs/test_overfit")
    cfg.paths.work_dir = cwd
    

    # --- Overfitting Configuration ---
    # Use a single batch for training and validation
    with open_dict(cfg.trainer):
        cfg.trainer.overfit_batches = 1
        cfg.trainer.log_every_n_steps = 1

    # Train long enough to converge
    cfg.trainer.max_epochs = 100
    cfg.model.scheduler.total_steps = 100

    # Check validation frequently
    cfg.trainer.check_val_every_n_epoch = 10

    # Disable other limits to avoid conflicts
    cfg.trainer.limit_train_batches = None
    cfg.trainer.limit_val_batches = None
    cfg.trainer.limit_test_batches = None

    # Ensure we are on CPU for this small test (or GPU if available, but let's stick to config)
    cfg.trainer.accelerator = "gpu"
    cfg.trainer.devices = [6]
    
    # # --- AdamW Optimizer Configuration ---
    # # Use AdamW with a high learning rate for overfitting
    # cfg.model.optimizer = {
    #     "_target_": "torch.optim.AdamW",
    #     "lr": 0.001,
    #     "weight_decay": 0.0,
    # }

    # --- Instantiation ---
    L.seed_everything(42)  # Fix seed for reproducibility

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # --- Execution ---
    print("\nStarting Overfitting Test...")
    trainer.fit(model=model, datamodule=datamodule)

    # --- Assertions ---
    metrics = trainer.callback_metrics
    print(f"\nFinal Metrics: {metrics}")

    # 1. Loss should be low
    train_loss = metrics.get("train/loss")
    assert train_loss is not None, "train/loss not found in metrics"
    assert (
        train_loss < 0.5
    ), f"Training loss {train_loss} is too high. Model failed to overfit."

    # 2. Hit Rate should be high (ideally 1.0)
    # Note: We use val/hr@5 because that's what we monitor
    val_hr = metrics.get("val/hr@5")
    assert val_hr is not None, "val/hr@5 not found in metrics"
    assert (
        val_hr > 0.8
    ), f"Validation HR@5 {val_hr} is too low. Model failed to memorize the batch."

    # 3. NDCG should be high
    val_ndcg = metrics.get("val/ndcg@5")
    assert val_ndcg is not None, "val/ndcg@5 not found in metrics"
    assert val_ndcg > 0.8, f"Validation NDCG@5 {val_ndcg} is too low."


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["debug=default"])
    test_overfitting(cfg)
