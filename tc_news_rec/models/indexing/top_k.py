import abc
import torch
from typing import Tuple

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
        sorted: bool = True
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
        sorted: bool = True
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
        all_logits = torch.mm(query_embeddings, item_embeddings_t)  # [B, X]
        topk_logits, topk_indices = torch.topk(
            all_logits, 
            k=k, 
            dim=1, 
            largest=True, 
            sorted=sorted
        )  # [B, k], [B, k]
        
        return topk_logits, item_ids.squeeze(0)[topk_indices]