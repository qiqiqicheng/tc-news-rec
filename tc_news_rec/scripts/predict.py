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
        log.info("ckpt_path not provided. Searching for the latest checkpoint in logs/train/runs...")
        import glob

        # Use cfg.paths.log_dir if available, otherwise assume 'logs' relative to root
        root_dir = cfg.paths.root_dir
        # The user specifically mentioned logs/train/runs
        # We try to construct the path robustly
        search_dir = os.path.join(root_dir, "logs", "train", "runs")

        if not os.path.exists(search_dir):
            raise ValueError(f"Cannot find search directory: {search_dir}. Please provide ckpt_path=...")

        # Find all run directories
        run_dirs = sorted(
            [d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))],
            reverse=True,
        )

        found_ckpt = None
        for run_dir in run_dirs:
            ckpt_dir = os.path.join(search_dir, run_dir, "checkpoints")
            if os.path.exists(ckpt_dir):
                # Look for .ckpt files
                ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
                if ckpts:
                    best_ckpts = [c for c in ckpts if "best_model" in os.path.basename(c)]
                    epoch_ckpts = [
                        c for c in ckpts if "epoch=" in os.path.basename(c) or "epoch_" in os.path.basename(c)
                    ]

                    if best_ckpts:
                        found_ckpt = best_ckpts[0]
                    elif epoch_ckpts:
                        # Sort by epoch number if possible, or name
                        # Assuming name format epoch_016.ckpt -> sort reverse
                        epoch_ckpts.sort(reverse=True)
                        found_ckpt = epoch_ckpts[0]
                    else:
                        # Fallback to any ckpt, sorted by time
                        ckpts.sort(key=os.path.getmtime, reverse=True)
                        found_ckpt = ckpts[0]

                    log.info(f"Found latest checkpoint in {run_dir}: {found_ckpt}")
                    break

        if found_ckpt:
            ckpt_path = found_ckpt
        else:
            raise ValueError(f"No checkpoints found in {search_dir}. Please provide ckpt_path=...")

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

    log.info("Loading user mapping... SKIPPED (Using original user IDs from dataset)")
    # No user mapping needed as we use original IDs directly

    submission_data = []

    # Handle predictions structure
    top_k_ids_list = None
    user_ids_list = None

    def extract_tensors(pred_dict):
        top_k = pred_dict["top_k_ids"]
        u_ids = pred_dict["user_ids"]

        # Handle list of tensors vs tensor
        if isinstance(top_k, list):
            top_k = torch.cat(top_k)
            u_ids = torch.cat(u_ids)

        return top_k, u_ids

    if isinstance(predictions, dict):
        top_k_ids_list, user_ids_list = extract_tensors(predictions)
    elif isinstance(predictions, list) and len(predictions) > 0:
        # Assuming single dataloader for now
        top_k_ids_list, user_ids_list = extract_tensors(predictions[0])

    # Validation
    if top_k_ids_list is None or user_ids_list is None:
        log.error("Failed to extract top_k_ids and user_ids from predictions")
        return

    # Process tensors
    top_k_ids_np = top_k_ids_list.cpu().numpy()
    user_ids_np = user_ids_list.cpu().numpy()

    log.info(f"Processing {len(user_ids_np)} predictions...")

    for i in range(len(user_ids_np)):
        # user_ids_np contains original user IDs now
        original_user_id = int(user_ids_np[i])
        mapped_item_ids = top_k_ids_np[i]

        original_item_ids = []
        for mid in mapped_item_ids:
            mid = int(mid)
            if mid in item_map_reverse:
                original_item_ids.append(item_map_reverse[mid])
            else:
                original_item_ids.append(mid)  # Fallback

        # We need top 5.
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
