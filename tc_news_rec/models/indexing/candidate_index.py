from typing import Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig

from tc_news_rec.models.indexing.top_k import TopKModule


class CandidateIndex(torch.nn.Module):
    def __init__(
        self,
        ids: torch.Tensor,
        top_k_module: TopKModule | DictConfig,
        embeddings: Optional[torch.Tensor] = None,
        invalid_ids: Optional[torch.Tensor] = None,
        debug_path: Optional[str] = None,
    ):
        # TODO: finish the signature
        """
        Args:
            ids (torch.Tensor): [1, X] all items from datamodule
            top_k_module (TopKModule): _description_
            embeddings (Optional[torch.Tensor], optional): [1, X, D]. Defaults to None.
            invalid_ids (Optional[torch.Tensor], optional): _description_. Defaults to None.
            debug_path (Optional[str], optional): _description_. Defaults to None.
        """
        super().__init__()
        assert ids.dim() == 2 and ids.size(0) == 1, "ids should be of shape [1, X]"
        assert embeddings is None or (
            embeddings.dim() == 3 and embeddings.size(0) == 1 and embeddings.size(1) == ids.size(1)
        ), "embeddings should be of shape [1, X, D] if provided"
        self.register_buffer("_ids", torch.as_tensor(ids))  # [1, X]
        self._invalid_ids = invalid_ids
        self._debug_path = debug_path
        self.update_embeddings(embeddings)
        self._top_k_module = None
        self._init_modules(top_k_module)

    def _init_modules(self, top_k_module: TopKModule | DictConfig) -> None:
        if isinstance(top_k_module, DictConfig):
            self._top_k_module = hydra.utils.instantiate(top_k_module)
        elif isinstance(top_k_module, TopKModule):
            self._top_k_module = top_k_module
        else:
            raise ValueError(f"top_k_module should be of type TopKModule or DictConfig, got {type(top_k_module)}")

    def update_embeddings(self, embeddings: Optional[torch.Tensor]) -> None:
        if embeddings is not None:
            self._embedding_t = embeddings.permute(2, 1, 0).squeeze(2)  # [D, X]
        else:
            self._embedding_t = None

    @property
    def ids(self) -> torch.Tensor:
        """
        Returns:
            [1, X], where valid ids are positive integers.
        """
        return self._ids  # type: ignore

    @property
    def num_objects(self) -> int:
        return self._ids.size(1)  # type: ignore

    @property
    def embeddings(self) -> Optional[torch.Tensor]:
        # TODO: probably some mistake with [B, X, D] case
        """
        Returns:
            [1, X, D] with the same shape as `ids'.
        """
        if self._embedding_t is None:
            return None
        return self._embedding_t.unsqueeze(2).permute(2, 1, 0).squeeze(2)

    def get_top_k_outputs(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        invalid_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings (torch.Tensor): [B, D]
            k (int): _description_
            invalid_ids (Optional[torch.Tensor]): [B, N] or None. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - topk_ids (torch.Tensor): [B, k]
                - topk_scores (torch.Tensor): [B, k]
        """
        max_invalid_id_num = 0
        if invalid_ids is not None:
            max_invalid_id_num = invalid_ids.size(1)

        # expand k to avoid invalid ids in the results
        k_prime = min(k + max_invalid_id_num, self.num_objects)

        top_k_prime_scores, top_k_prime_ids = self._top_k_module(
            query_embeddings=query_embeddings,
            item_embeddings_t=self._embedding_t,  # [D, X]
            item_ids=self._ids,  # [1, X]
            k=k_prime,
            sorted=True,
        )  # [B, k'], [B, k']

        # TODO: understand the vectorized invalid id removal and torch functions used
        if invalid_ids is not None:
            # Check for invalid ids: [B, K', N] -> [B, K']
            is_invalid = (top_k_prime_ids.unsqueeze(2) == invalid_ids.unsqueeze(1)).any(dim=2)

            # Mask scores of invalid items with -inf
            masked_scores = top_k_prime_scores.clone()
            masked_scores[is_invalid] = float("-inf")

            # Re-select top-k from the masked prime list
            # If valid items < k, we will select items with -inf score (which are technically invalid)
            # This maintains the shape [B, k] without crashing
            top_k_scores, indices = torch.topk(masked_scores, k=k, dim=1)

            # Gather corresponding IDs
            top_k_ids = torch.gather(top_k_prime_ids, dim=1, index=indices)

        else:
            top_k_ids = top_k_prime_ids[:, :k]  # [B, k]
            top_k_scores = top_k_prime_scores[:, :k]  # [B, k]

        return top_k_ids, top_k_scores

    # TODO: implement other methods
