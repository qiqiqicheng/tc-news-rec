import json
import os

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata, SCMode
from omegaconf.nodes import ValueNode

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Register PyTorch safe globals for OmegaConf
torch.serialization.add_safe_globals([
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Metadata,
    SCMode,
    ValueNode,
])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def predict(cfg: DictConfig):
    # Set seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[], logger=[])

    log.info("Starting prediction!")
    # Get the checkpoint path (best model from training)
    # You might need to specify the exact path if it's not in the config
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        # Fallback to searching in the default log dir or user must provide it
        # Assuming the user provides it via command line: ckpt_path=...
        raise ValueError("Please provide a checkpoint path via ckpt_path=...")

    log.info(f"Using checkpoint: {ckpt_path}")

    # Run prediction
    # model.predict_step returns a dict with top_k_ids, top_k_scores, target_ids, user_ids
    # Trainer keeps the order if shuffle=False (which is default for predict_dataloader)
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    # Process predictions
    # After on_predict_epoch_end, predictions is a list (one element per dataloader)
    # Each element is a dict with keys: top_k_ids, top_k_scores, target_ids, user_ids
    # Each value is a concatenated tensor or list of tensors

    output_dir = os.path.join(cfg.paths.root_dir, "prediction_results")
    os.makedirs(output_dir, exist_ok=True)

    # Check what we got
    if not predictions:
        log.warning("No predictions returned.")
        return

    log.info(
        f"Predictions type: {type(predictions)}, length: {len(predictions) if isinstance(predictions, list) else 'N/A'}"
    )

    # Check mapping files
    data_dir = cfg.data.data_preprocessor.data_dir
    processed_dir = cfg.data.data_preprocessor.output_dir

    # Load item mapping (mapped_id -> original_id)
    with open(os.path.join(processed_dir, "item_id_mapping.json")) as f:
        item_map_reverse = json.load(f)
    # Convert keys to int
    item_map_reverse = {int(k): int(v) for k, v in item_map_reverse.items()}

    # Reconstruct user mapping
    # Logic from preprocessor.py:
    # train_df = pd.read_csv("train_click_log.csv")
    # test_df = pd.read_csv("testA_click_log.csv")
    # all_users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()
    # user_map = {u: i + 1 for i, u in enumerate(sorted(all_users))}
    log.info("Reconstructing user mapping...")
    train_df = pd.read_csv(os.path.join(data_dir, "train_click_log.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "testA_click_log.csv"))
    all_users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()
    sorted_users = sorted(all_users)
    user_map_reverse = {i + 1: u for i, u in enumerate(sorted_users)}

    submission_data = []

    # Handle predictions structure
    # Depending on Lightning version and on_predict_epoch_end implementation,
    # predictions can be either:
    # 1. A dict directly (single dataloader case, after on_predict_epoch_end modification)
    # 2. A list containing dicts (one per dataloader)

    top_k_ids_list = None
    user_ids_list = None

    if isinstance(predictions, dict):
        # Direct dict - predictions is already the concatenated result
        log.info(f"Detected direct dict predictions with keys: {predictions.keys()}")

        top_k_ids_val = predictions["top_k_ids"]
        user_ids_val = predictions["user_ids"]

        # If it's already a tensor (from on_predict_epoch_end concatenation)
        if isinstance(top_k_ids_val, torch.Tensor):
            log.info("Detected concatenated tensor predictions.")
            top_k_ids_list = top_k_ids_val
            user_ids_list = user_ids_val
        # If it's a list of tensors, concatenate them
        elif isinstance(top_k_ids_val, list) and len(top_k_ids_val) > 0:
            if isinstance(top_k_ids_val[0], torch.Tensor):
                log.info("Detected list of tensor predictions, concatenating...")
                top_k_ids_list = torch.cat(top_k_ids_val)
                user_ids_list = torch.cat(user_ids_val)
            else:
                log.error(f"Unexpected prediction value type in list: {type(top_k_ids_val[0])}")
                return
        else:
            log.error(f"Unexpected prediction value type: {type(top_k_ids_val)}")
            return

    elif isinstance(predictions, list) and len(predictions) > 0:
        # List of dicts - get first dataloader's predictions
        first_pred = predictions[0]
        log.info(
            f"Detected list predictions. First element type: {type(first_pred)}, keys: {first_pred.keys() if isinstance(first_pred, dict) else 'N/A'}"
        )

        if isinstance(first_pred, dict):
            # Check if values are already concatenated tensors or lists of tensors
            top_k_ids_val = first_pred["top_k_ids"]
            user_ids_val = first_pred["user_ids"]

            # If it's already a tensor (from on_predict_epoch_end concatenation)
            if isinstance(top_k_ids_val, torch.Tensor):
                log.info("Detected concatenated tensor predictions.")
                top_k_ids_list = top_k_ids_val
                user_ids_list = user_ids_val
            # If it's a list of tensors, concatenate them
            elif (
                isinstance(top_k_ids_val, list)
                and len(top_k_ids_val) > 0
                and isinstance(top_k_ids_val[0], torch.Tensor)
            ):
                log.info("Detected list of tensor predictions, concatenating...")
                top_k_ids_list = torch.cat(top_k_ids_val)
                user_ids_list = torch.cat(user_ids_val)
            else:
                log.error(f"Unexpected prediction value type: {type(top_k_ids_val)}")
                return
        else:
            log.error(f"Expected dict but got: {type(first_pred)}")
            return
    else:
        log.error(f"Unexpected predictions type or empty: {type(predictions)}")
        return

    # Validate that we successfully extracted the data
    if top_k_ids_list is None or user_ids_list is None:
        log.error("Failed to extract top_k_ids and user_ids from predictions")
        return

    # Process tensors
    top_k_ids_np = top_k_ids_list.cpu().numpy()
    user_ids_np = user_ids_list.cpu().numpy()

    log.info(f"Processing {len(user_ids_np)} predictions...")

    for i in range(len(user_ids_np)):
        mapped_user_id = int(user_ids_np[i])
        mapped_item_ids = top_k_ids_np[i]

        # Map back to original IDs
        if mapped_user_id not in user_map_reverse:
            log.warning(f"Mapped user ID {mapped_user_id} not found in reverse map.")
            original_user_id = mapped_user_id  # Fallback
        else:
            original_user_id = user_map_reverse[mapped_user_id]

        original_item_ids = []
        for mid in mapped_item_ids:
            mid = int(mid)
            if mid in item_map_reverse:
                original_item_ids.append(item_map_reverse[mid])
            else:
                original_item_ids.append(mid)  # Fallback

        # We need top 5. The model returns k=5 (set in predict_step) or self.k
        # Ensure we have at least 5
        top_5 = original_item_ids[:5]
        while len(top_5) < 5:
            top_5.append(0)  # Padding if needed

        row = [original_user_id] + top_5
        submission_data.append(row)

    # Create DataFrame
    columns = [
        "user_id",
        "article_1",
        "article_2",
        "article_3",
        "article_4",
        "article_5",
    ]
    submission_df = pd.DataFrame(submission_data, columns=columns)

    # Save
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    log.info(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    predict()
