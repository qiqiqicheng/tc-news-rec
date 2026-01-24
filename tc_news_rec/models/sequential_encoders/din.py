import torch
import torch.nn as nn

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger()


class DINEncoder(nn.Module):
    """
    Deep Interest Network (DIN) inspired Sequence Encoder.
    Adapts the Local Activation Unit from DIN for sequential/autoregressive recommendation.

    At each step t, the item at t is used as the 'Target' (Query) to attend over the history items (0...t).
    This computes a user representation relevant to the current item context.

    References:
        Zhou et al. "Deep Interest Network for Click-Through Rate Prediction". KDD 2018.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.0, *args, **kwargs):
        super().__init__()
        log.info(f"Initializing DINEncoder with args: {args}; kwargs {kwargs}")
        self.embedding_dim = embedding_dim

        # DIN Local Activation Unit
        # Input: [Query, Key, Query-Key, Query*Key]
        # Dimension: 4 * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.PReLU(),  # DIN typically uses PReLU or Dice
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, past_lengths, user_embeddings, valid_mask=None, past_payloads=None):
        """
        Args:
           past_lengths: [B] (Not used directly if valid_mask is provided)
           user_embeddings: [B, N, D] Sequence of item embeddings
           valid_mask: [B, N] Boolean/Float mask where 1 is valid, 0 is padding
        Returns:
           encoded_embeddings: [B, N, D]
           cached_states: None
        """
        B, N, D = user_embeddings.size()

        # Q: Query items (Target) -> For autoregressive, use current item as proxy target
        # [B, N, 1, D]
        Q = user_embeddings.unsqueeze(2).expand(B, N, N, D)

        # K: Key items (History)
        # [B, 1, N, D]
        K = user_embeddings.unsqueeze(1).expand(B, N, N, D)

        # Causal Mask: Ensure we only attend to history (j <= i)
        # mask[i, j] = 1 if j <= i else 0
        # Shape: [1, N, N, 1]
        ones = torch.ones(N, N, device=user_embeddings.device)
        causal_mask = torch.tril(ones).unsqueeze(0).unsqueeze(-1)

        # Compute interactions for Activation Unit
        # Input: [Q, K, Q-K, Q*K]
        # Shape: [B, N, N, 4D]
        inp = torch.cat([Q, K, Q - K, Q * K], dim=-1)

        # Apply MLP to get attention scores
        # Shape: [B, N, N, 1]
        # We flatten batch dimensions for efficiency
        inp_flat = inp.view(-1, 4 * D)
        scores = self.mlp(inp_flat).view(B, N, N, 1)

        # Apply Causal Mask
        # We multiply by mask. Invalid positions (future) become 0.
        # DIN uses raw weights (not softmax), so 0 means no contribution.
        scores = scores * causal_mask

        # Apply Padding Mask (valid_mask)
        if valid_mask is not None:
            # valid_mask: [B, N] or [B, N, 1]
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(-1)  # [B, N, 1]

            # Mask out invalid Keys
            # valid_mask [B, N, 1] -> unsqueeze(1) -> [B, 1, N, 1]
            k_mask = valid_mask.unsqueeze(1)
            scores = scores * k_mask

            # Mask out invalid Queries (optional, but keeps output clean)
            # valid_mask [B, N, 1] -> unsqueeze(2) -> [B, N, 1, 1]
            q_mask = valid_mask.unsqueeze(2)
        # Compute output as weighted sum of Keys (Values)
        # Output[b, i] = Sum_j (scores[b, i, j] * K[b, j])
        # scores: [B, N, N, 1]
        # user_embeddings: [B, N, D]

        # [B, N, N] x [B, N, D] -> [B, N, D]
        scores_squeeze = scores.squeeze(-1)
        encoded_embeddings = torch.matmul(scores_squeeze, user_embeddings)

        if self.dropout.p > 0:
            encoded_embeddings = self.dropout(encoded_embeddings)

        return encoded_embeddings, None
