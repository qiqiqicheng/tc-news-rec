import pytest
import hydra
import torch
import os
from omegaconf import DictConfig, OmegaConf
from tc_news_rec.data.tc_dataset import TCDataModule, TCDataset
from copy import deepcopy
from hydra import initialize, compose

def _test_datamodule_setup_and_batch(debug_cfg: DictConfig):
    cwd = os.getcwd()
    datamodule_cfg = deepcopy(debug_cfg.data)

    train_file = os.path.join(cwd, "user_data/processed/sasrec_format_by_user_train.csv")
    test_file = os.path.join(cwd, "user_data/processed/sasrec_format_by_user_test.csv")    
    embedding_file = os.path.join(cwd, "user_data/processed/article_embedding.pt")
    data_dir = os.path.join(cwd, "tcdata")
    output_dir = os.path.join(cwd, "user_data/processed")
    
    datamodule_cfg.train_file = train_file
    datamodule_cfg.test_file = test_file
    datamodule_cfg.embedding_file = embedding_file
    datamodule_cfg.data_preprocessor.data_dir = data_dir
    datamodule_cfg.data_preprocessor.output_dir = output_dir
        
    # Instantiate DataModule
    datamodule: TCDataModule = hydra.utils.instantiate(datamodule_cfg)

    print("Datamodule instantiated successfully.")
    # Setup
    datamodule.setup(stage="fit")

    assert datamodule.train_dataset is not None

    # Get Dataloader
    train_loader = datamodule.train_dataloader()

    # Fetch one batch
    batch = next(iter(train_loader))
    
    B = debug_cfg.data.batch_size
    N = debug_cfg.data.max_seq_length
    D = debug_cfg.model.item_embedding_dim

    batch_eg = {
        "user_id": torch.randint(0, 100, (B,)),
        "historical_item_ids": torch.randint(1, 1000, (B, N)),
        "historical_item_embeddings": torch.randn(B, N, 250),
        "historical_item_click_times": torch.randint(0, 100000, (B, N)),
        "historical_item_category_ids": torch.randint(0, 50, (B, N)),
        "historical_item_created_ats": torch.randint(0, 100000, (B, N)),
        "historical_item_words_counts": torch.randint(0, 1000, (B, N)),
        "historical_item_ages": torch.randint(0, 100, (B, N)),
        "historical_item_hours": torch.randint(0, 24, (B, N)),
        "historical_item_days": torch.randint(0, 7, (B, N)),
        # User features
        "environment": torch.randint(0, 5, (B,)),
        "deviceGroup": torch.randint(0, 5, (B,)),
        "os": torch.randint(0, 5, (B,)),
        "country": torch.randint(0, 5, (B,)),
        "region": torch.randint(0, 5, (B,)),
        "referrer_type": torch.randint(0, 5, (B,)),
        "target_item_id": torch.randint(1, 1000, (B,)),
        "target_item_embedding": torch.randn(B, 250),
        "target_item_click_time": torch.randint(0, 100000, (B,)),
        "history_len": torch.randint(1, N + 1, (B,)),
    }
    
    expected_batch_keys = [key for key in batch_eg.keys()]

    for key in expected_batch_keys:
        assert key in batch, f"Key {key} missing in batch"
        assert batch[key].shape == batch_eg[key].shape, f"Shape mismatch for key {key}"

# if __name__ == "__main__":
#     def debug_cfg() -> DictConfig:
#         with initialize(config_path="../configs", version_base=None):
#             cfg = compose(config_name="train", overrides=["debug=default"])

#         return cfg
    
#     cfg = debug_cfg()
#     test_datamodule_setup_and_batch(cfg)