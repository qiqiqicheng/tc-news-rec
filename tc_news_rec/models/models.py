from typing import Optional, Any, Dict, Literal, Tuple
import hydra
from omegaconf import DictConfig
import torch
import torchmetrics
import torch.nn.functional as F
import lightning as L
import abc

from tc_news_rec.utils.logger import RankedLogger
from tc_news_rec.data.tc_dataset import TCDataset, TCDataModule
from tc_news_rec.models.embeddings import LocalEmbeddingModule, EmbeddingModule
from tc_news_rec.models.similarity.similarity import NDPModule, DotProductSimilarity
from tc_news_rec.models.losses.losses import AutoregressiveLoss, SampledSoftmaxLoss
from tc_news_rec.models.negative_samplers.negative_samplers import (
    GlobalNegativeSampler,
    NegativeSampler,
    InBatchNegativesSampler,
)
from tc_news_rec.models.sequential_encoders.hstu import HSTU
from tc_news_rec.models.preprocessors.preprocessors import (
    InputPreprocessor,
    AllEmbeddingsInputPreprocessor,
)
from tc_news_rec.models.postprocessors.postprocessors import (
    OutputPostprocessorModule,
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from tc_news_rec.models.utils.features import (
    SequentialFeatures,
    get_sequential_features,
)
import tc_news_rec.models.utils.ops as ops
from tc_news_rec.models.indexing.candidate_index import CandidateIndex
from tc_news_rec.models.metrics.retrieval_metrics import RetrievalMetrics

log = RankedLogger(__name__)


class BaseRecommender(L.LightningModule):
    def __init__(
        self,
        k: int,
        datamodule: TCDataModule | DictConfig,
        # embeddiings: EmbeddingModule | DictConfig,  # TODO: check if we need this
        preprocessor: InputPreprocessor | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negative_sampler: NegativeSampler | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
        optimizer: torch.optim.Optimizer | DictConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | DictConfig,
        configure_optimizer_params: DictConfig,
        gr_output_length: int,
        item_embedding_dim: int,
        compile_model: bool,
    ) -> None:
        super().__init__()

        self.k = k

        self.optimizer: torch.optim.Optimizer = (
            hydra.utils.instantiate(optimizer)
            if isinstance(optimizer, DictConfig)
            else optimizer
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = (
            hydra.utils.instantiate(scheduler)
            if isinstance(scheduler, DictConfig)
            else scheduler
        )

        self.configure_optimizer_params: DictConfig = configure_optimizer_params

        self.gr_output_length: int = gr_output_length
        self.item_embedding_dim: int = item_embedding_dim
        self.compile_model: bool = compile_model

        self._hydra_init_modules(
            datamodule,
            preprocessor,
            sequence_encoder,
            postprocessor,
            similarity,
            negative_sampler,
            loss,
            candidate_index,
            metrics,
        )

    def _hydra_init_modules(
        self,
        datamodule: TCDataModule | DictConfig,
        preprocessor: InputPreprocessor | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negative_sampler: NegativeSampler | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
    ) -> None:
        # TODO: check it carefully later
        self.datamodule: TCDataModule = (
            hydra.utils.instantiate(datamodule, _recursive_=False)
            if isinstance(datamodule, DictConfig)
            else datamodule
        )

        self.preprocessor: InputPreprocessor = (
            hydra.utils.instantiate(preprocessor, _recursive_=False)
            if isinstance(preprocessor, DictConfig)
            else preprocessor
        )

        self.sequence_encoder: torch.nn.Module = (
            hydra.utils.instantiate(sequence_encoder, _recursive_=False)
            if isinstance(sequence_encoder, DictConfig)
            else sequence_encoder
        )

        self.postprocessor: OutputPostprocessorModule = (
            hydra.utils.instantiate(postprocessor, _recursive_=False)
            if isinstance(postprocessor, DictConfig)
            else postprocessor
        )

        self.similarity: NDPModule = (
            hydra.utils.instantiate(similarity)
            if isinstance(similarity, DictConfig)
            else similarity
        )

        self.negative_sampler: NegativeSampler = (
            hydra.utils.instantiate(negative_sampler, _recursive_=False)
            if isinstance(negative_sampler, DictConfig)
            else negative_sampler
        )

        self.loss_fn: AutoregressiveLoss = (
            hydra.utils.instantiate(loss, _recursive_=False)
            if isinstance(loss, DictConfig)
            else loss
        )

        if isinstance(candidate_index, DictConfig):
            all_ids = (
                torch.Tensor(self.preprocessor.get_all_item_ids()).view(1, -1).long()
            )
            self.candidate_index = hydra.utils.instantiate(
                candidate_index, ids=all_ids, _recursive_=False
            )
        else:
            raise ValueError("candidate_index must be a DictConfig")

        self.metrics: torchmetrics.Metric = (
            hydra.utils.instantiate(metrics, _recursive_=False)
            if isinstance(metrics, DictConfig)
            else metrics
        )

    def setup(
        self, stage: Literal["fit", "validate", "test", "predict"] | None = None
    ) -> None:
        # TODO: understand this
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            dict[str, Any]: A dict containing the configured optimizers and learning-rate
                schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.parameters())  # type: ignore
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)  # type: ignore
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.configure_optimizer_params,
                },
            }
        return {"optimizer": optimizer}

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Call the superclass's state_dict method to get the full state dictionary
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars  # type: ignore
        )

        # List of module names you don't want to save
        modules_to_exclude = [
            "similarity",
            "negatives_sampler",
            "candidate_index",
            "loss",
            "metrics",
        ]

        # Remove the keys corresponding to the modules to exclude
        keys_to_remove = [
            key
            for key in state_dict.keys()
            for module_name in modules_to_exclude
            if key.startswith(prefix + module_name)
        ]
        for key in keys_to_remove:
            del state_dict[key]

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # since we removed some keys from the state_dict, we need to set strict=False
        super().load_state_dict(state_dict, strict=False)

    @abc.abstractmethod
    def forward(
        self, seq_features: SequentialFeatures
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lightning calls this inside the training loop

        Args:
            seq_features (SequentialFeatures): past_lens, past_ids, past_payloads with batch size B

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - encoded_embeddings: [B, D]
                - cached_states: returned from encoder
        """
        pass

    @abc.abstractmethod
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Args:
            batch (Dict[str, torch.Tensor]): _description_
            batch_idx (int): _description_

        Returns:
            torch.Tensor: loss value
        """
        pass

    @abc.abstractmethod
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    @abc.abstractmethod
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    @abc.abstractmethod
    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Any:
        pass

    def dense_to_jagged(
        self, lengths: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Convert dense tensor to jagged tensor.

        Args:
            lengths (torch.Tensor): The lengths tensor.
            **kwargs: The dict with the dense tensor to be converted.

        Returns:
            dict[str, torch.Tensor]: The jagged tensor.
        """
        jagged_id_offsets = ops.asynchronous_complete_cumsum(lengths)
        output = {}
        if "supervision_ids" in kwargs:
            output["supervision_ids"] = (
                ops.dense_to_jagged(
                    kwargs.pop("supervision_ids").unsqueeze(-1).float(),
                    jagged_id_offsets,
                )
                .squeeze(1)
                .long()
            )

        if "supervision_weights" in kwargs:
            output["supervision_weights"] = ops.dense_to_jagged(
                kwargs.pop("supervision_weights").unsqueeze(-1), jagged_id_offsets
            ).squeeze(1)
        for key, value in kwargs.items():
            output[key] = ops.dense_to_jagged(value, jagged_id_offsets)
        return output


class RetreivalModel(BaseRecommender):
    def forward(
        self, seq_features: SequentialFeatures
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lightning calls this inside the training loop

        Args:
            seq_features (SequentialFeatures): past_lens, past_ids, past_payloads with batch size B

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - encoded_embeddings: [B, N, D] encoded sequence embeddings
                - valid_lengths: [B,] valid lengths after preprocessing (includes aux token)
                - cached_states: returned from encoder
        """
        # preprocessor returns past_lens + 1 (adding aux token)
        past_lens_with_aux, seq_embeddings, valid_mask, aux_mask = self.preprocessor(
            past_lens=seq_features.past_lens,
            past_ids=seq_features.past_ids,
            past_payloads=seq_features.past_payloads,
        )

        encoded_embeddings, cached_states = self.sequence_encoder(
            past_lengths=past_lens_with_aux,
            user_embeddings=seq_embeddings,
            valid_mask=valid_mask,
            past_payloads=seq_features.past_payloads,
        )

        # Apply aux_mask to filter valid positions and get the actual valid lengths
        # valid_lengths will be used for downstream jagged operations
        valid_lengths = past_lens_with_aux  # default: past_lens + 1
        if aux_mask is not None:
            encoded_embeddings, valid_lengths = ops.mask_dense_by_aux_mask(
                encoded_embeddings,
                aux_mask,
                past_lens_with_aux,
                max_lengths=seq_features.past_ids.size(1),
            )

        encoded_embeddings = self.postprocessor(encoded_embeddings)
        return encoded_embeddings, valid_lengths, cached_states

    def _update_candidate_embeddings(self) -> None:
        item_emb_module = self.preprocessor.get_item_id_embedding_module()
        embeddings = item_emb_module.weight[1:].unsqueeze(0)  # [1, N, D]

        self.candidate_index.update_embeddings(
            self.negative_sampler.normalize_embeddings(embeddings)
        )

    @torch.inference_mode()
    def retrieve(
        self,
        seq_features: SequentialFeatures,
        k,
        filter_past_ids: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_features (SequentialFeatures): _description_
            filter_past_ids (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - topk_ids: [B, K]
                - topk_scores: [B, K]
        """
        encoded_embeddings, valid_lengths, _ = self.forward(seq_features)
        # Use valid_lengths returned from forward() for correct position indexing
        current_embeddings = ops.get_current_embeddings(
            valid_lengths, encoded_embeddings
        )

        if self.candidate_index.embeddings is None:
            log.info(
                "Initializing candidate index embeddings with current item embeddings"
            )
            self._update_candidate_embeddings()

        top_k_ids, top_k_scores = self.candidate_index.get_top_k_outputs(
            query_embeddings=current_embeddings,
            k=k,
            invalid_ids=(seq_features.past_ids if filter_past_ids else None),
        )
        return top_k_ids, top_k_scores

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        seq_features, target_ids = get_sequential_features(
            batch, device=self.device, max_output_length=self.gr_output_length + 1
        )

        # Place target_id at the correct position in past_ids
        # After preprocessor adds aux token at position 0, the sequence becomes:
        # [aux, item1, item2, ..., itemN, 0, 0, ...]
        # We want to place target at position (past_lens), which after aux shift
        # corresponds to the position right after the last historical item
        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lens.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        # Forward pass returns valid_lengths which accounts for aux token
        encoded_embeddings, valid_lengths, _ = self.forward(seq_features)

        if isinstance(self.negative_sampler, InBatchNegativesSampler):
            in_batch_ids = seq_features.past_ids.view(-1)
            self.negative_sampler.process_batch(
                ids=in_batch_ids,
                valid_mask=(in_batch_ids != 0).unsqueeze(-1).float(),
                embeddings=self.preprocessor.get_embedding_by_id(in_batch_ids),
            )
        elif isinstance(self.negative_sampler, GlobalNegativeSampler):
            self.negative_sampler.set_item_embedding(
                self.preprocessor.get_item_id_embedding_module()
            )
            self.negative_sampler.set_all_item_ids(
                self.preprocessor.get_all_item_ids(), device=encoded_embeddings.device
            )

        # Construct autoregressive supervision signals:
        # For autoregressive training, output at position i should predict item at position i+1
        # After preprocessor, sequence is: [aux, item1, item2, ..., itemN, target, 0, ...]
        # Output[0] (aux position) -> predicts item1
        # Output[1] (item1 position) -> predicts item2
        # ...
        # Output[N] (itemN position) -> predicts target
        #
        # So supervision_ids should be the NEXT item for each position:
        # supervision_ids = past_ids shifted left by 1 (i.e., past_ids[:, 1:])
        all_ids = seq_features.past_ids

        # Shift supervision_ids by 1 to get next-item targets
        # output_embeddings[i] should predict supervision_ids[i] = all_ids[i+1]
        supervision_ids = all_ids[
            :, 1:
        ]  # [B, N-1] - the target for each output position
        output_embeddings_for_loss = encoded_embeddings[
            :, :-1, :
        ]  # [B, N-1, D] - exclude last position

        # Adjust valid_lengths: we lose one position due to the shift
        # valid_lengths was (past_lens + 1), now we use (past_lens + 1 - 1) = past_lens
        # But we need to be careful: valid_lengths already accounts for aux token
        supervision_lengths = valid_lengths - 1  # [B,]
        supervision_lengths = torch.clamp(
            supervision_lengths, min=0
        )  # ensure non-negative

        # supervision_weights: mask out padding positions
        supervision_weights = (supervision_ids != 0).float()  # [B, N-1]

        jagged_features = self.dense_to_jagged(
            lengths=supervision_lengths,
            output_embeddings=output_embeddings_for_loss,  # [B, N-1, D] -> [T, D]
            supervision_ids=supervision_ids,  # [B, N-1] -> [T,]
            supervision_embeddings=self.preprocessor.get_embedding_by_id(
                supervision_ids
            ),  # [B, N-1, D] -> [T, D]
            supervision_weights=supervision_weights,  # [B, N-1] -> [T,]
        )

        loss = self.loss_fn.jagged_forward(
            negative_sampler=self.negative_sampler,
            similarity_module=self.similarity,
            **jagged_features,
        )

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the validation epoch."""
        self.metrics.reset()
        self._update_candidate_embeddings()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        seq_features, target_ids = get_sequential_features(
            batch, device=self.device, max_output_length=self.gr_output_length + 1
        )

        top_k_ids, top_k_scores = self.retrieve(
            seq_features=seq_features, k=self.k, filter_past_ids=True
        )

        self.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

    def on_validation_epoch_end(self) -> None:
        """
        Lightning calls this at the end of the validation epoch.

        Args:
            outputs (list[torch.Tensor]): A list of the outputs from each validation step.
        """
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def on_test_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the test epoch."""
        self.metrics.reset()
        self._update_candidate_embeddings()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """
        Lightning calls this at the end of the validation epoch.

        Args:
            outputs (list[torch.Tensor]): A list of the outputs from each validation step.
        """
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        if "monitor" in self.configure_optimizer_params:
            return results[self.configure_optimizer_params["monitor"].split("/")[1]]

    def on_predict_epoch_start(self) -> None:
        """Lightning calls this at the beginning of the predict epoch."""
        self._update_candidate_embeddings()

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Any:
        seq_feature, target_ids = get_sequential_features(
            batch, device=self.device, max_output_length=self.gr_output_length + 1
        )
        top_k_ids, top_k_scores = self.retrieve(
            seq_features=seq_feature, k=self.k, filter_past_ids=True
        )
        return {
            "top_k_ids": top_k_ids,
            "top_k_scores": top_k_scores,
            "target_ids": target_ids,
        }

    def on_predict_epoch_end(self) -> None:
        """Lightning calls this at the end of the predict epoch."""
        # Convert predictions from list of dicts to dict of lists
        # TODO: NOTE the python skills here
        for i, predictions in enumerate(
            self.trainer.predict_loop._predictions
        ):  # NOTE: loop fo different dataloaders
            if predictions and isinstance(
                predictions[0], dict
            ):  # NOTE: the predictions are List[Dict[str, Tensor]] of all batches
                keys = predictions[0].keys()
                converted_predictions = {
                    key: sum((pred[key] for pred in predictions), []) for key in keys
                }
                self.trainer.predict_loop._predictions[i] = converted_predictions  # type: ignore
                # final results -> Dict[str, List[Tensor]], normally the final results after prediction
