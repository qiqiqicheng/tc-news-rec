import abc
from typing import Tuple

import torch


class TopKModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings_t: torch.Tensor,
        item_ids: torch.Tensor,
        k: int,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings (torch.Tensor): [B, D]
            item_embeddings_t (torch.Tensor): transposed [D, X]
            item_ids (torch.Tensor): [1, X]
            k (int): _description_
            sorted (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - topk_scores (torch.Tensor): [B, k]
                - topk_item_ids (torch.Tensor): [B, k]
        """
        pass


class MIPSBruteTopK(TopKModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings_t: torch.Tensor,
        item_ids: torch.Tensor,
        k: int,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings (torch.Tensor): [B, D]
            item_embeddings_t (torch.Tensor): transposed [D, X]
            item_ids (torch.Tensor): [1, X]
            k (int): _description_
            sorted (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - topk_scores (torch.Tensor): [B, k]
                - topk_item_ids (torch.Tensor): [B, k]
        """
        all_logits = torch.mm(query_embeddings.to(item_embeddings_t.device), item_embeddings_t)  # [B, X]

        # Safety check for NaN/Inf in logits
        if torch.isnan(all_logits).any() or torch.isinf(all_logits).any():
            # Replace NaN/Inf with very negative values to avoid topk errors
            all_logits = torch.nan_to_num(all_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        topk_logits, topk_indices = torch.topk(all_logits, k=k, dim=1, largest=True, sorted=sorted)  # [B, k], [B, k]

        # Safety: Clamp indices to valid range
        max_idx = item_ids.size(1) - 1
        topk_indices = torch.clamp(topk_indices, 0, max_idx)

        return topk_logits, item_ids.squeeze(0)[topk_indices]
