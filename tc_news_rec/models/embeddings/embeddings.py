import abc
import torch

from tc_news_rec.models.utils.initialization import truncated_normal, init_mlp_xavier_weights_zero_bias
from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)

class EmbeddingModule(torch.nn.Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass
    
    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass
    
class LocalEmbeddingModule(EmbeddingModule):
    def __init__(self, num_items: int, emb_dim: int) -> None:
        super().__init__()
        self._num_items = num_items
        self._emb_dim = emb_dim
        self._item_embedding = torch.nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.reset_params()
        
    def reset_params(self) -> None:
        for name, p in self.named_parameters():
            if "_item_emb" in name:
                truncated_normal(p, mean=0.0, std=0.02)
    
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_embedding(item_ids)
    
    @property
    def item_embedding_dim(self) -> int:
        return self._emb_dim
    
    