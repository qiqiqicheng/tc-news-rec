import os
from copy import deepcopy
from typing import Dict, Optional

import hydra
import torch
from omegaconf import DictConfig

from tc_news_rec.models.utils.features import (
    SequentialFeatures,
    get_sequential_features,
)

# OmegaConf.register_new_resolver("eval", eval)


def _test_get_sequential_features(debug_cfg: DictConfig, fake_batch: Dict):
    device = torch.device("cpu")
    max_output_length = debug_cfg.model.gr_output_length + 1

    seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

    user_keys = [
        "environment",
        "deviceGroup",
        "os",
        "country",
        "region",
        "referrer_type",
    ]

    historical_keys = [
        "content_embedding",
        "category_id",
        "created_at",
        "words_count",
        "age",
        "hour_of_day",
        "day_of_week",
    ]

    assert isinstance(seq_features, SequentialFeatures)
    assert seq_features.past_lens.shape == (debug_cfg.data.batch_size,)
    # assert seq_features.past_lens.shape[0] == debug_cfg.data.batch_size
    assert seq_features.past_ids.shape[0] == debug_cfg.data.batch_size
    assert seq_features.past_ids.shape[1] == debug_cfg.data.max_seq_length + max_output_length

    for key in user_keys:
        assert key in seq_features.past_payloads
        feature = seq_features.past_payloads[key]
        assert feature.ndim == 1
        assert feature.shape[0] == debug_cfg.data.batch_size

    for key in historical_keys:
        assert key in seq_features.past_payloads
        feature = seq_features.past_payloads[key]
        assert feature.shape[0] == debug_cfg.data.batch_size
        assert feature.shape[1] == debug_cfg.data.max_seq_length + max_output_length

    assert target_ids.ndim == 2
    assert target_ids.shape[0] == debug_cfg.data.batch_size
    assert target_ids.shape[1] == 1


def _test_preprocessor(debug_cfg: DictConfig, fake_batch: Dict):
    device = torch.device("cpu")
    max_output_length = debug_cfg.model.gr_output_length + 1
    B = debug_cfg.data.batch_size
    N = debug_cfg.data.max_seq_length
    D = debug_cfg.model.item_embedding_dim

    seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

    seq_features.past_ids.scatter_(
        dim=1,
        index=seq_features.past_lens.view(-1, 1),
        src=target_ids.view(-1, 1),
    )
    preprocessor_config = deepcopy(debug_cfg.model.preprocessor)
    import os

    preprocessor_config.feature_counts = os.path.join(os.getcwd(), "user_data/processed/feature_counts.json")

    preprocessor = hydra.utils.instantiate(preprocessor_config, _recursive_=False)
    past_lens, seq_emeddings, valid_mask, aux_mask = preprocessor(
        past_lens=seq_features.past_lens,
        past_ids=seq_features.past_ids,
        past_payloads=seq_features.past_payloads,
    )

    print(f"past_lens shape: {past_lens.shape}")
    print(f"seq_embeddings shape: {seq_emeddings.shape}")
    print(f"valid_mask shape: {valid_mask.shape}")
    print(f"aux_mask shape: {aux_mask.shape}")

    assert past_lens.shape == (B,)
    assert seq_emeddings.shape == (B, N + max_output_length, D)
    assert valid_mask.shape == (B, N + max_output_length, 1)
    assert aux_mask.shape == (B, N + max_output_length)

    print(f"past_lens sample: {past_lens}")
    print(f"valid mask sample: {valid_mask[0]}")
    print(f"aux mask sample: {aux_mask[0]}")


def _test_model_forward(debug_cfg: DictConfig, fake_batch: Optional[Dict] = None):
    model_config = deepcopy(debug_cfg.model)
    data_config = deepcopy(debug_cfg.data)

    # from pprint import pprint
    # print("Model config:")
    # pprint(model_config)
    # print()
    # print("Data config:")
    # pprint(data_config)

    feature_count_file = os.path.join(os.getcwd(), "user_data/processed/feature_counts.json")
    train_file = os.path.join(os.getcwd(), "user_data/processed/sasrec_format_by_user_train.csv")
    test_file = os.path.join(os.getcwd(), "user_data/processed/sasrec_format_by_user_test.csv")
    embedding_file = os.path.join(os.getcwd(), "user_data/processed/article_embedding.pt")
    data_file = os.path.join(os.getcwd(), "tcdata")
    output_dir = os.path.join(os.getcwd(), "user_data/processed")

    model_config.preprocessor.feature_counts = feature_count_file

    data_config.data_preprocessor.data_dir = data_file
    data_config.data_preprocessor.output_dir = output_dir

    data_config.train_file = train_file
    data_config.test_file = test_file

    data_config.embedding_file = embedding_file

    retrieve_model = hydra.utils.instantiate(model_config, datamodule=data_config, _recursive_=False)

    # print(f"type of datamodule: {type(retrieve_model.datamodule)}")
    # print(f"type of train_dataset: {type(retrieve_model.datamodule.train_dataset)}")

    retrieve_model.datamodule.setup(stage="fit")
    batch = retrieve_model.datamodule.train_dataloader().__iter__().__next__()

    # print(f"Batch keys: {list(batch.keys())}")
    # print("Details of one batch:")
    # for key, value in batch.items():
    #     print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    loss = retrieve_model.training_step(batch, 0)
    print(f"Training step loss: {loss.item()}")
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


# if __name__ == "__main__":
#     def debug_cfg() -> DictConfig:
#         with initialize(config_path="../configs", version_base=None):
#             cfg = compose(config_name="train", overrides=["debug=default"])

#         return cfg
#     cfg = debug_cfg()
#     test_model_forward(cfg)
