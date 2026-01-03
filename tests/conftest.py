import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose


@pytest.fixture(scope="session")
def debug_cfg() -> DictConfig:
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["debug=default"])

    return cfg


@pytest.fixture
def fake_batch(debug_cfg: DictConfig) -> dict:
    B = debug_cfg.data.batch_size
    N = debug_cfg.data.max_seq_length
    D = debug_cfg.model.item_embedding_dim

    batch = {
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
    return batch
