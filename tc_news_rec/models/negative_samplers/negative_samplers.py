import torch
from typing import List, Optional, Tuple
import abc

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)


class NegativeSampler(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=1e-8)

    @abc.abstractmethod
    def forward(
        self, postive_item_ids: torch.Tensor, num_to_sample: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class GlobalNegativeSampler(NegativeSampler):
    def __init__(
        self,
        l2_normalize: bool = True,
    ) -> None:
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
                "_all_item_ids", torch.tensor(all_item_ids, dtype=torch.long, device=device)
            )
        else:
            self._all_item_ids = torch.tensor(all_item_ids, dtype=torch.long).to(
                self._all_item_ids.device
            )

    def forward(
        self, postive_item_ids: torch.Tensor, num_to_sample: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            postive_item_ids (torch.Tensor): [B,] or [B, L] (transformer)
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

        output_shape = postive_item_ids.shape + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=postive_item_ids.dtype,
            device=postive_item_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)  # type: ignore

        if self._l2_normalize:
            negative_item_embeddings = self.normalize_embeddings(
                self._item_emb(sampled_ids)
            )
        else:
            negative_item_embeddings = self._item_emb(sampled_ids)

        return sampled_ids, negative_item_embeddings


class InBatchNegativesSampler(NegativeSampler):
    def __init__(self, l2_normalize: bool = True, ddup: bool = True) -> None:
        super().__init__()
        self._l2_normalize = l2_normalize
        self._ddup = ddup

    def process_batch(
        self, ids: torch.Tensor, valid_mask: torch.Tensor, embeddings: torch.Tensor
    ) -> None:
        """
        Args:
            ids (torch.Tensor): [N,]
            valid_mask (torch.Tensor): same as ids
            embeddings (torch.Tensor): [N, D]
        """
        assert ids.dim() == 1, "ids should be 1-D tensor"
        assert ids.size(0) == embeddings.size(
            0
        ), "ids and embeddings should have the same batch size"
        assert (
            valid_mask.shape == ids.shape
        ), "valid_mask should have the same shape as ids"

        if self._dedup_embeddings:
            valid_ids = ids[valid_mask]  # [N,]
            unique_ids, unique_ids_inverse_indices = torch.unique(
                input=valid_ids, sorted=False, return_inverse=True
            )
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
            self._cached_embeddings = self.normalize_embeddings(
                unique_embeddings
            )  # [N, D]
            self._cached_ids = unique_ids  # [N,]
        else:
            self._cached_embeddings = self.normalize_embeddings(embeddings[valid_mask])
            self._cached_ids = ids[valid_mask]

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def forward(
        self, postive_ids: torch.Tensor, num_to_sample: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
