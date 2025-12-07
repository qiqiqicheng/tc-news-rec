import torch
import torch.nn.functional as F
import abc

from tc_news_rec.utils.logger import RankedLogger
from tc_news_rec.models.similarity.similarity import NDPModule
from tc_news_rec.models.negative_samplers.negative_samplers import NegativeSampler

log = RankedLogger(__name__)


class AutoregressiveLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negative_sampler: NegativeSampler,
        similarity_module: NDPModule,
    ) -> torch.Tensor:
        pass


class BCELoss(AutoregressiveLoss):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self._temperature = temperature

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negative_sampler: NegativeSampler,
        similarity_module: NDPModule,
    ) -> torch.Tensor:
        """
        Args:
            output_embeddings (torch.Tensor): [B, D]
            supervision_ids (torch.Tensor): [B,]
            supervision_embeddings (torch.Tensor): target id, [B, D]
            supervision_weights (torch.Tensor): reweighting or masking tensor, [B,]
            negative_sampler (NegativeSampler): _description_
            similarity_module (NDPModule): _description_

        Returns:
            torch.Tensor: _loss value
        """
        negative_ids, negative_embeddings = negative_sampler(
            positve_item_ids=supervision_ids, num_to_sample=1
        )  # [B, num_to_sample], [B, num_to_sample, D]

        postive_logits = (
            similarity_module(
                input_embeddings=output_embeddings,
                item_embeddings=supervision_embeddings.unsqueeze(1),  # [B, 1, D]
            ).squeeze(1)
            / self._temperature
        )  # [B,]

        negative_logits = (
            similarity_module(
                input_embeddings=output_embeddings,
                item_embeddings=negative_embeddings,  # [B, num_to_sample, D]
            ).squeeze(1)
            / self._temperature
        )  # [B,]

        sampled_negative_valid_mask = (
            supervision_ids != negative_ids.squeeze(1)
        ).float()  # [B,]

        loss_weights = supervision_weights * sampled_negative_valid_mask  # [B,]

        weighted_losses = (
            (
                F.binary_cross_entropy_with_logits(
                    postive_logits, torch.ones_like(postive_logits), reduction="none"
                )
                + F.binary_cross_entropy_with_logits(
                    negative_logits, torch.zeros_like(negative_logits), reduction="none"
                )
            )
            * loss_weights
            * 0.5
        )  # [B,]
        loss = weighted_losses.sum() / (loss_weights.sum() + 1e-8)
        return loss


class SampledSoftmaxLoss(AutoregressiveLoss):
    def __init__(self, num_to_sample: int, softmax_temperature: float):
        super().__init__()
        self._num_to_sample = num_to_sample
        self._softmax_temperature = softmax_temperature

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negative_sampler: NegativeSampler,
        similarity_module: NDPModule,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output_embeddings (torch.Tensor): [T, D]
            supervision_ids (torch.Tensor): [T]
            supervision_embeddings (torch.Tensor): target id, [T, D]
            supervision_weights (torch.Tensor): reweighting or masking tensor, [T,]
            negative_sampler (NegativeSampler): _description_
            similarity_module (NDPModule): _description_

        Returns:
            torch.Tensor: _loss value
        """
        negative_ids, negative_embeddings = negative_sampler(
            postive_item_ids=supervision_ids, num_to_sample=self._num_to_sample
        )  # [T, num_to_sample], [T, num_to_sample, D]

        positive_logits = (
            similarity_module(
                input_embeddings=output_embeddings,  # [T, D]
                item_embeddings=supervision_embeddings.unsqueeze(1),  # [T, 1, D]
            )  # [T, 1]
            .squeeze(1)  # [T,]
            / self._softmax_temperature  # [T,]
        )

        negative_logits = similarity_module(
            input_embeddings=output_embeddings,  # [T, D]
            item_embeddings=negative_embeddings,  # [T, num_to_sample, D]
        )  # [T, num_to_sample]
        negative_logits = torch.where(
            supervision_ids.unsqueeze(1) == negative_ids,
            torch.tensor(
                -5e4, device=negative_logits.device, dtype=negative_logits.dtype
            ),
            negative_logits / self._softmax_temperature,
        )  # [T, num_to_sample]

        logits = torch.cat(
            [positive_logits.unsqueeze(1), negative_logits], dim=1
        )  # [T, 1 + num_to_sample]

        # labels = torch.zeros(
        #     logits.size(0),
        #     dtype=torch.long,
        #     device=logits.device
        # )  # [B,], all zeros since positive is the first class

        loss_weights = supervision_weights  # [T,]

        weighted_losses = -F.log_softmax(logits, dim=1)[:, 0] * loss_weights  # [T,]

        loss = weighted_losses.sum() / (loss_weights.sum() + 1e-8)
        return loss
