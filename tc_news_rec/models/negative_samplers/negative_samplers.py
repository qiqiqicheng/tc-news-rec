import abc
from typing import List, Tuple

import torch

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)


class NegativeSampler(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=1e-8)

    @abc.abstractmethod
    def forward(self, positive_item_ids: torch.Tensor, num_to_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def forward_with_query(
        self,
        positive_item_ids: torch.Tensor,
        query_embeddings: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with query embeddings for hard negative mining.
        Default implementation falls back to standard forward (no hard negatives).

        Args:
            positive_item_ids (torch.Tensor): [T,] positive item ids
            query_embeddings (torch.Tensor): [T, D] query embeddings for similarity computation
            num_to_sample (int): number of negative samples to draw

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - negative_item_ids (torch.Tensor): [T, num_to_sample]
                - negative_item_embeddings (torch.Tensor): [T, num_to_sample, D]
        """
        return self.forward(positive_item_ids, num_to_sample)


class GlobalNegativeSampler(NegativeSampler):
    def __init__(
        self,
        l2_normalize: bool = True,
        *args,
        **kwargs,
    ) -> None:
        log.info(f"passing args: {args}, kwargs: {kwargs} to GlobalNegativeSampler")
        super().__init__()
        self._l2_normalize = l2_normalize
        self._item_emb: torch.nn.Embedding | None = None
        # self._all_item_ids: torch.Tensor | None = None

    def set_item_embedding(self, item_emb: torch.nn.Embedding) -> None:
        self._item_emb = item_emb

    def set_all_item_ids(self, all_item_ids: List[int], device: torch.device) -> None:
        self._num_items = len(all_item_ids)
        # TODO: understand register_buffer
        if not hasattr(self, "_all_item_ids"):
            self.register_buffer(
                "_all_item_ids",
                torch.tensor(all_item_ids, dtype=torch.long, device=device),
            )
        else:
            self._all_item_ids = torch.tensor(all_item_ids, dtype=torch.long).to(self._all_item_ids.device)

    def forward(self, positive_item_ids: torch.Tensor, num_to_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positive_item_ids (torch.Tensor): [B,] or [B, L] (transformer)
            num_to_sample (int): _number of negative samples to draw

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - negative_item_ids (torch.Tensor): [B, num_to_sample] or [B, L, num_to_sample]
                - negative_item_embeddings (torch.Tensor): [B, num_to_sample, D] or [B, L, num_to_sample, D]
        """
        if self._item_emb is None:
            raise ValueError("Item embedding is not set for GlobalNegativeSampler")
        if self._all_item_ids is None:
            raise ValueError("All item ids are not set for GlobalNegativeSampler")

        output_shape = positive_item_ids.shape + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=positive_item_ids.dtype,
            device=positive_item_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)  # type: ignore

        if self._l2_normalize:
            negative_item_embeddings = self.normalize_embeddings(self._item_emb(sampled_ids))
        else:
            negative_item_embeddings = self._item_emb(sampled_ids)

        return sampled_ids, negative_item_embeddings


class HardNegativeSampler(NegativeSampler):
    """
    Hard Negative Mining Sampler.

    Samples negative items based on similarity to query embeddings.
    A portion of negatives are "hard" (high similarity to query),
    while the rest are random to maintain diversity.

    Hard negatives force the model to learn finer distinctions between
    similar items, which is crucial for recommendation quality.
    """

    def __init__(
        self,
        l2_normalize: bool = True,
        hard_negative_ratio: float = 0.5,
        candidate_pool_size: int = 4096,
    ) -> None:
        """
        Args:
            l2_normalize: Whether to L2 normalize embeddings
            hard_negative_ratio: Fraction of negatives that should be hard (0.0-1.0)
            candidate_pool_size: Size of candidate pool to sample hard negatives from
        """
        super().__init__()
        self._l2_normalize = l2_normalize
        self._hard_negative_ratio = hard_negative_ratio
        self._candidate_pool_size = candidate_pool_size
        self._item_emb: torch.nn.Embedding | None = None

    def set_item_embedding(self, item_emb: torch.nn.Embedding) -> None:
        self._item_emb = item_emb

    def set_all_item_ids(self, all_item_ids: List[int], device: torch.device) -> None:
        self._num_items = len(all_item_ids)
        if not hasattr(self, "_all_item_ids"):
            self.register_buffer(
                "_all_item_ids",
                torch.tensor(all_item_ids, dtype=torch.long, device=device),
            )
        else:
            self._all_item_ids = torch.tensor(all_item_ids, dtype=torch.long).to(self._all_item_ids.device)

    def forward(self, positive_item_ids: torch.Tensor, num_to_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fallback to random sampling when query embeddings are not provided.
        """
        if self._item_emb is None:
            raise ValueError("Item embedding is not set for HardNegativeSampler")
        if self._all_item_ids is None:
            raise ValueError("All item ids are not set for HardNegativeSampler")

        output_shape = positive_item_ids.shape + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=positive_item_ids.dtype,
            device=positive_item_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)

        if self._l2_normalize:
            negative_item_embeddings = self.normalize_embeddings(self._item_emb(sampled_ids))
        else:
            negative_item_embeddings = self._item_emb(sampled_ids)

        return sampled_ids, negative_item_embeddings

    def forward_with_query(
        self,
        positive_item_ids: torch.Tensor,
        query_embeddings: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negatives with hard negative mining based on query similarity.

        Algorithm:
        1. Sample a large candidate pool from all items
        2. Compute similarity between query and candidates
        3. Select top-k most similar as hard negatives
        4. Fill remaining slots with random samples for diversity

        Args:
            positive_item_ids (torch.Tensor): [T,] positive item ids
            query_embeddings (torch.Tensor): [T, D] query embeddings (L2 normalized)
            num_to_sample (int): total number of negative samples

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - negative_item_ids (torch.Tensor): [T, num_to_sample]
                - negative_item_embeddings (torch.Tensor): [T, num_to_sample, D]
        """
        if self._item_emb is None:
            raise ValueError("Item embedding is not set for HardNegativeSampler")
        if self._all_item_ids is None:
            raise ValueError("All item ids are not set for HardNegativeSampler")

        T = positive_item_ids.size(0)
        device = positive_item_ids.device

        # Calculate number of hard vs random negatives
        num_hard = int(num_to_sample * self._hard_negative_ratio)
        num_random = num_to_sample - num_hard

        # Ensure candidate pool is large enough for hard negative selection
        effective_pool_size = max(self._candidate_pool_size, num_hard * 2)
        effective_pool_size = min(effective_pool_size, self._num_items)

        # Step 1: Sample candidate pool (shared across all queries for efficiency)
        # Using a shared pool reduces computation while still providing diversity
        candidate_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=(effective_pool_size,),
            dtype=torch.long,
            device=device,
        )
        candidate_ids = self._all_item_ids[candidate_offsets]  # [pool_size,]
        candidate_embeddings = self._item_emb(candidate_ids)  # [pool_size, D]

        if self._l2_normalize:
            candidate_embeddings = self.normalize_embeddings(candidate_embeddings)

        # Step 2: Compute similarity between queries and candidates
        # query_embeddings: [T, D], candidate_embeddings: [pool_size, D]
        # similarity: [T, pool_size]
        similarity = torch.matmul(query_embeddings, candidate_embeddings.t())

        # Mask out positive items from candidates to avoid sampling them as negatives
        # positive_item_ids: [T,], candidate_ids: [pool_size,]
        positive_mask = positive_item_ids.unsqueeze(1) == candidate_ids.unsqueeze(0)  # [T, pool_size]
        similarity = similarity.masked_fill(positive_mask, float("-inf"))

        # Step 3: Select top-k hard negatives (highest similarity)
        if num_hard > 0:
            # Get indices of top-k most similar candidates for each query
            _, hard_indices = torch.topk(similarity, k=min(num_hard, effective_pool_size), dim=1)  # [T, num_hard]
            hard_ids = candidate_ids[hard_indices]  # [T, num_hard]
            hard_embeddings = candidate_embeddings[hard_indices]  # [T, num_hard, D]
        else:
            hard_ids = torch.empty((T, 0), dtype=torch.long, device=device)
            hard_embeddings = torch.empty((T, 0, query_embeddings.size(-1)), device=device)

        # Step 4: Sample random negatives for diversity
        if num_random > 0:
            random_offsets = torch.randint(
                low=0,
                high=self._num_items,
                size=(T, num_random),
                dtype=torch.long,
                device=device,
            )
            random_ids = self._all_item_ids[random_offsets.view(-1)].reshape(T, num_random)
            random_embeddings = self._item_emb(random_ids)  # [T, num_random, D]

            if self._l2_normalize:
                random_embeddings = self.normalize_embeddings(random_embeddings)
        else:
            random_ids = torch.empty((T, 0), dtype=torch.long, device=device)
            random_embeddings = torch.empty((T, 0, query_embeddings.size(-1)), device=device)

        # Step 5: Concatenate hard and random negatives
        negative_ids = torch.cat([hard_ids, random_ids], dim=1)  # [T, num_to_sample]
        negative_embeddings = torch.cat([hard_embeddings, random_embeddings], dim=1)  # [T, num_to_sample, D]

        return negative_ids, negative_embeddings


class InBatchNegativesSampler(NegativeSampler):
    def __init__(self, l2_normalize: bool = True, ddup: bool = True) -> None:
        super().__init__()
        self._l2_normalize = l2_normalize
        self._ddup = ddup

    def process_batch(self, ids: torch.Tensor, valid_mask: torch.Tensor, embeddings: torch.Tensor) -> None:
        """
        Args:
            ids (torch.Tensor): [N,]
            valid_mask (torch.Tensor): same as ids
            embeddings (torch.Tensor): [N, D]
        """
        assert ids.dim() == 1, "ids should be 1-D tensor"
        assert ids.size(0) == embeddings.size(0), "ids and embeddings should have the same batch size"
        assert valid_mask.shape == ids.shape, "valid_mask should have the same shape as ids"

        if self._dedup_embeddings:
            valid_ids = ids[valid_mask]  # [N,]
            unique_ids, unique_ids_inverse_indices = torch.unique(input=valid_ids, sorted=False, return_inverse=True)
            device = unique_ids.device
            unique_embedding_offsets = torch.empty(
                (unique_ids.numel(),),
                dtype=torch.int64,
                device=device,
            )
            unique_embedding_offsets[unique_ids_inverse_indices] = torch.arange(
                valid_ids.numel(), dtype=torch.int64, device=device
            )
            unique_embeddings = embeddings[valid_mask][unique_embedding_offsets, :]
            self._cached_embeddings = self.normalize_embeddings(unique_embeddings)  # [N, D]
            self._cached_ids = unique_ids  # [N,]
        else:
            self._cached_embeddings = self.normalize_embeddings(embeddings[valid_mask])
            self._cached_ids = ids[valid_mask]

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def forward(self, postive_ids: torch.Tensor, num_to_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            postive_ids (torch.Tensor): [B,]
            num_to_sample (int): k

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - sampled_ids: [B, k]
                - sampled_embeddings: [B, k, D]
        """
        num_ids = self._cached_ids.size(0)
        # TODO: understand the offset
        sampled_offsets = torch.randint(
            low=0,
            high=num_ids,
            size=postive_ids.shape + (num_to_sample,),  # [B, k]
            dtype=postive_ids.dtype,
            device=postive_ids.device,
        )
        return (
            self._cached_ids[sampled_offsets],
            self._cached_embeddings[sampled_offsets],
        )
