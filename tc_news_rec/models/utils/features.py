from typing import NamedTuple, Dict, Tuple
import torch

class SequentialFeatures(NamedTuple):
    past_lens: torch.Tensor  # [B,]
    past_ids: torch.Tensor  # [B, N]
    past_payloads: Dict[str, torch.Tensor]
    
def get_sequential_features(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    max_output_length: int
) -> Tuple[SequentialFeatures, torch.Tensor]:
    past_lens = batch["history_len"].to(device)  # [B,]
    past_ids = batch["historical_item_ids"].to(device)  # [B, N]
    user_keys_map = {
        'environment': 'environment',
        'deviceGroup': 'deviceGroup',
        'os': 'os',
        'country': 'country',
        'region': 'region',
        'referrer_type': 'referrer_type',
    }
    historial_keys_map = {
        'historical_item_embeddings': 'content_embedding', 
        # 'historical_item_click_times': 'click_time', 
        'historical_item_category_ids': 'category_id',
        'historical_item_created_ats': 'created_at',
        'historical_item_words_counts': 'words_count',
        'historical_item_ages': 'age',
        'historical_item_hours': 'hour_of_day',
        'historical_item_days': 'day_of_week',
    }
    payloads = {}
    
    if max_output_length > 0:
        B = past_ids.size(0)
        past_ids = torch.concat(
            [
                past_ids,
                torch.zeros(B, max_output_length, device=device, dtype=past_ids.dtype)
            ],
            dim=1
        )  # [B, N + max_output_length]
        for key, new_key in historial_keys_map.items():
            payload = batch[key].to(device)
            padding_payload = torch.zeros(
                B,
                max_output_length,
                *payload.size()[2:],
                device=device,
                dtype=payload.dtype
            )
            payloads[new_key] = torch.concat(
                [payload, padding_payload],
                dim=1
            )  # [B, N + max_output_length, ...]
    else:
        for key, new_key in historial_keys_map.items():
            payloads[new_key] = batch[key].to(device)        
    
    for key, new_key in user_keys_map.items():
        payloads[new_key] = batch[key].to(device)
    
    target_ids = batch["target_item_id"].to(device).unsqueeze(1)  # [B,]

    seq_features = SequentialFeatures(
        past_lens=past_lens,
        past_ids=past_ids,
        past_payloads=payloads
    )
    del batch
    
    return seq_features, target_ids