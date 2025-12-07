from typing import List, Tuple, Optional
import torch

from tc_news_rec.models.indexing.top_k import TopKModule, MIPSBruteTopK
import tc_news_rec.models.utils.ops as ops

class CandidateIndex(torch.nn.Module):
    def __init__(
        self,
        ids: torch.Tensor,
        top_k_module: TopKModule,
        embeddings: Optional[torch.Tensor] = None,
        invalid_ids: Optional[torch.Tensor] = None,
        debug_path: Optional[str] = None
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
        assert embeddings is None or (embeddings.dim() == 3 and embeddings.size(0) == 1 and embeddings.size(1) == ids.size(1)), \
            "embeddings should be of shape [1, X, D] if provided"
        self.register_buffer("_ids", torch.as_tensor(ids))  # [1, X]
        self._top_k_module = top_k_module
        self._invalid_ids = invalid_ids
        self._debug_path = debug_path
        self.update_embeddings(embeddings)
        
    def update_embeddings(
        self,
        embeddings: Optional[torch.Tensor]
    ) -> None:
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
    def embeddings(self) -> torch.Tensor:
        # TODO: probably some mistake with [B, X, D] case
        """
        Returns:
            [1, X, D] with the same shape as `ids'.
        """
        return self._embeddings_t.unsqueeze(2).permute(2, 1, 0).squeeze(2)  # type: ignore
    
    def get_top_k_outputs(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        invalid_ids: Optional[torch.Tensor] = None
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
            sorted=True
        )  # [B, k'], [B, k']
        
        
        # TODO: understand the vectorized invalid id removal and torch functions used
        if invalid_ids is not None:
            id_is_valid = ~(
                (
                    top_k_prime_ids.unsqueeze(2) == invalid_ids.unsqueeze(1)  # [B, K, 1], [B, 1, N] -> [B, K, N]
                ).max(2)[0]  # [B, K]
            )
            id_is_valid = torch.logical_and(
                id_is_valid,  # [B, K]
                torch.cumsum(id_is_valid.int(), dim=1) <= k  # [B, K]
            )
            
            top_k_rowwise_offsets = torch.nonzero(
                id_is_valid,
                as_tuple=True
            )[1].view(-1, k)  # [B, k]
            
            top_k_scores = torch.gather(
                top_k_prime_ids,
                dim=1,
                index=top_k_rowwise_offsets
            )
            top_k_ids = torch.gather(
                top_k_prime_ids,
                dim=1,
                index=top_k_prime_ids
            )
        else:
            top_k_ids = top_k_prime_ids[:, :k]  # [B, k]
            top_k_scores = top_k_prime_scores[:, :k]  # [B, k]
            
        return top_k_ids, top_k_scores
        
    # TODO: implement other methods

    