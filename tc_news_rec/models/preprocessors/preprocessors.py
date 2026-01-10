import abc
import math
from typing import Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig

from tc_news_rec.models.embeddings import EmbeddingModule
from tc_news_rec.models.utils.initialization import truncated_normal
from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)


def instantiate_embedding_module(
    embedding_module: EmbeddingModule | DictConfig, num_items: int, emb_dim: int
) -> EmbeddingModule:
    if isinstance(embedding_module, DictConfig):
        embedding_module = embedding_module.copy()
        if "num_items" not in embedding_module:
            embedding_module.num_items = num_items
        if "emb_dim" not in embedding_module:
            embedding_module.emb_dim = emb_dim
        log.info(
            f"Instantiating embedding module from config: {embedding_module} with num_items={num_items}, emb_dim={emb_dim}"
        )
        embedding_module = hydra.utils.instantiate(embedding_module)
    else:
        log.warning("Using the provided EmbeddingModule instance directly without instantiation.")
    assert isinstance(embedding_module, EmbeddingModule)
    return embedding_module


class InputPreprocessor(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self._item_id_embedding: EmbeddingModule = None  # type: ignore

    @abc.abstractmethod
    def get_item_id_embedding_module(self) -> torch.nn.Embedding:
        pass

    @abc.abstractmethod
    def get_embedding_by_id(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_all_item_ids(self) -> List[int]:
        pass


class AllEmbeddingsInputPreprocessor(InputPreprocessor):
    def __init__(
        self,
        embedding_module: EmbeddingModule | DictConfig,
        feature_counts: Dict[str, int] | str,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        payload_selection: List[str] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(embedding_module, DictConfig):
            emb_dim = embedding_module["emb_dim"]
        elif isinstance(embedding_module, EmbeddingModule):
            emb_dim = embedding_module.item_embedding_dim
        self._embedding_dim = emb_dim
        self._max_seq_len = max_seq_len

        if isinstance(feature_counts, str):
            import json

            assert feature_counts.endswith(".json"), "feature_counts file should be a json file"
            with open(feature_counts) as f:
                feature_counts = json.load(f)

        self._feature_counts = feature_counts
        self._dropout = torch.nn.Dropout(dropout_rate)
        self._position_embeddings = torch.nn.Embedding(max_seq_len, emb_dim)

        self._content_emb_linear = torch.nn.Linear(250, emb_dim)

        self._item_id_embedding = instantiate_embedding_module(
            embedding_module=embedding_module,
            num_items=feature_counts["item_id"] + 1,
            emb_dim=emb_dim,
        )

        self._embedding_features = (
            "category_id",
            "created_at",
            "words_count",
            "age",
            "hour_of_day",
            "day_of_week",
            "environment",
            "deviceGroup",
            "os",
            "country",
            "region",
            "referrer_type",
        )
        self._aux_features = (
            "environment",
            "deviceGroup",
            "os",
            "country",
            "region",
            "referrer_type",
        )

        self._embedding_module_dict = torch.nn.ModuleDict()
        for feature_name in self._embedding_features:
            if (payload_selection is None) or (feature_name in payload_selection):
                num_items = feature_counts[feature_name]
                if num_items > 0:
                    emb_module = instantiate_embedding_module(
                        embedding_module=embedding_module,
                        num_items=num_items + 1,
                        emb_dim=emb_dim,
                    )
                    self._embedding_module_dict[feature_name] = emb_module
                    log.info(
                        f"Created embedding module for feature '{feature_name}' with num_items={num_items + 1}, emb_dim={emb_dim}"
                    )

        self.reset_state()

    def reset_state(self) -> None:
        truncated_normal(
            self._position_embeddings.weight,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_item_id_embedding_module(self) -> torch.nn.Embedding:
        return self._item_id_embedding._item_embedding  # type: ignore

    def get_embedding_by_id(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_id_embedding.get_item_embeddings(item_ids)

    def get_all_item_ids(self) -> List[int]:
        return list(range(1, self._feature_counts["item_id"] + 1))

    def forward(
        self,
        past_lens: torch.Tensor,  # [B,]
        past_ids: torch.Tensor,  # [B, N]
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            past_lens (torch.Tensor): [B,]
            past_ids (torch.Tensor): [B, N]
            past_payloads (Dict[str, torch.Tensor]): various payloads with shape [B, N] or [B,]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - past_lens: [B,]
                - seq_embeddings: [B, N, D]
                - valid_mask: [B, N, 1]
                - aux_mask: [B, N] (exclude the first token for attention)
        """
        B, N = past_ids.size()

        # Item ID Embeddings
        seq_embeddings = self._item_id_embedding(past_ids)
        self.seq_embeddings = seq_embeddings
        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]

        # Add sequential feature embeddings and content embeddings
        for feature in set(self._embedding_features) - set(self._aux_features):
            feature_id = past_payloads[feature]  # [B, N]
            if feature in self._embedding_module_dict:
                feature_emb_module = self._embedding_module_dict[feature]
                feature_embeddings = feature_emb_module.get_item_embeddings(feature_id)  # type: ignore # [B, N, D]
                seq_embeddings += feature_embeddings
            else:
                raise ValueError(f"Embedding module for feature '{feature}' not found.")
        content_embeddings = past_payloads["content_embedding"]  # [B, N, 250]
        seq_embeddings += self._content_emb_linear(content_embeddings)

        # Add aux embedding token at the first position
        aux_embeddings = torch.zeros(B, self._embedding_dim, device=seq_embeddings.device)
        for feature in self._aux_features:
            feature_id = past_payloads[feature]
            if feature_id.dim() == 2:
                feature_id = feature_id[:, 0]  # [B,]
            if feature in self._embedding_module_dict:
                feature_emb_module = self._embedding_module_dict[feature]
                feature_embeddings = feature_emb_module.get_item_embeddings(feature_id)  # type: ignore # [B, D]
                aux_embeddings += feature_embeddings
            else:
                raise ValueError(f"Embedding module for feature '{feature}' not found.")

        aux_embeddings = aux_embeddings.unsqueeze(1)  # [B, 1, D]
        seq_embeddings = torch.cat(
            [
                aux_embeddings * (self._embedding_dim**0.5),
                seq_embeddings * (self._embedding_dim**0.5),
            ],
            dim=1,
        )  # [B, N + 1, D]
        valid_mask = torch.cat([torch.ones(B, 1, 1, device=valid_mask.device), valid_mask], dim=1)  # [B, N + 1, 1]

        # Remove the last token
        seq_embeddings = seq_embeddings[:, :-1, :]  # [B, N, D]
        valid_mask = valid_mask[:, :-1, :]  # [B, N, 1]

        # position embedding
        seq_embeddings = seq_embeddings + self._position_embeddings(
            torch.arange(
                start=0,
                end=N,
                step=1,
                dtype=torch.long,
                device=seq_embeddings.device,
            )
        ).unsqueeze(0)  # [B, N, D]
        seq_embeddings = self._dropout(seq_embeddings)
        seq_embeddings = seq_embeddings * valid_mask

        # set aux_mask (exclude the first token for attention)
        aux_mask = torch.arange(N, device=past_lens.device).unsqueeze(0) < (past_lens + 1).unsqueeze(1)  # [B, N]
        # TODO: consider the aux_mask and the first aux token
        # aux_mask[:, 0] = False  # [B, N]

        return past_lens + 1, seq_embeddings, valid_mask, aux_mask
