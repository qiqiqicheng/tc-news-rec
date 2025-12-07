import torch
from typing import Optional
import torch.nn.functional as F
import abc

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)

class NDPModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    @abc.abstractmethod
    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings (torch.Tensor): [B, D]
            item_embeddings (torch.Tensor): [1, X, D] (global) or [B, X, D] (local)

        Returns:
            torch.Tensor: [B, X] scores
        """
        pass

class DotProductSimilarity(NDPModule):
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings (torch.Tensor): [T, D]
            item_embeddings (torch.Tensor): [1, X, D] (global) or [T, X, D] (local)

        Returns:
            torch.Tensor: [T, X] scores
        """
        assert input_embeddings.dim() == 2  # [T, D]
        if item_embeddings.size(0) == 1:
            # global item embeddings
            scores = torch.matmul(
                input_embeddings.unsqueeze(1),  # [B, 1, D]
                item_embeddings.transpose(1, 2)  # [1, D, X]
            ).squeeze(1)  # [B, X]
        else:
            # local item embeddings
            scores = torch.bmm(
                input_embeddings.unsqueeze(1),  # [B, 1, D]
                item_embeddings.transpose(1, 2)  # [B, D, X]
            ).squeeze(1)  # [B, X]
        return scores
