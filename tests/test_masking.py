"""
Test suite for masking operations in the recommendation model.

This module tests:
1. Preprocessor aux_mask generation and validity
2. mask_dense_by_aux_mask operation correctness
3. valid_mask behavior with padding
4. Autoregressive supervision signal alignment after masking
5. Edge cases: empty sequences, full sequences, variable lengths
"""

import pytest
import torch
import hydra
import os
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf

from tc_news_rec.models.utils.features import (
    get_sequential_features,
    SequentialFeatures,
)
from tc_news_rec.models.utils import ops

OmegaConf.register_new_resolver("eval", eval, replace=True)


# =============================================================================
# Test: ops.mask_dense_by_aux_mask basic functionality
# =============================================================================


class TestMaskDenseByAuxMask:
    """Test the mask_dense_by_aux_mask operation."""

    def _test_basic_masking(self):
        """Test basic masking with simple inputs."""
        B, N, D = 2, 5, 4

        # Create dense tensor with known values
        dense_tensor = torch.arange(B * N * D).float().reshape(B, N, D)

        # Create aux_mask: first sample has 3 valid, second has 4 valid
        aux_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
            ]
        )
        lengths = torch.tensor([3, 4])

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=N
        )

        # Check new_lengths
        assert torch.equal(
            new_lengths, torch.tensor([3, 4])
        ), f"Expected lengths [3, 4], got {new_lengths}"

        # Check output shape
        assert masked_tensor.shape == (
            B,
            N,
            D,
        ), f"Expected shape {(B, N, D)}, got {masked_tensor.shape}"

        # Check that valid positions are preserved correctly
        # For sample 0: positions 0, 1, 2 should be preserved
        assert torch.allclose(
            masked_tensor[0, :3], dense_tensor[0, :3]
        ), "Sample 0 valid positions not preserved correctly"

        # For sample 1: positions 0, 1, 2, 3 should be preserved
        assert torch.allclose(
            masked_tensor[1, :4], dense_tensor[1, :4]
        ), "Sample 1 valid positions not preserved correctly"

    def _test_masking_with_gaps(self):
        """Test masking when valid positions have gaps (non-contiguous)."""
        B, N, D = 1, 6, 2

        dense_tensor = torch.arange(B * N * D).float().reshape(B, N, D)
        # Only positions 0, 2, 4 are valid (gaps at 1, 3, 5)
        aux_mask = torch.tensor([[True, False, True, False, True, False]])
        lengths = torch.tensor([6])

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=N
        )

        assert new_lengths.item() == 3, f"Expected length 3, got {new_lengths.item()}"

        # The masked tensor should have values from positions 0, 2, 4
        # packed into positions 0, 1, 2
        expected_values = dense_tensor[0, [0, 2, 4], :]
        assert torch.allclose(
            masked_tensor[0, :3], expected_values
        ), "Masked values not correctly packed"

    def _test_empty_mask(self):
        """Test behavior when mask is all False."""
        B, N, D = 1, 4, 2

        dense_tensor = torch.randn(B, N, D)
        aux_mask = torch.zeros(B, N, dtype=torch.bool)
        lengths = torch.tensor([0])

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=N
        )

        assert new_lengths.item() == 0, "Empty mask should result in length 0"

    def _test_full_mask(self):
        """Test behavior when mask is all True."""
        B, N, D = 2, 4, 3

        dense_tensor = torch.randn(B, N, D)
        aux_mask = torch.ones(B, N, dtype=torch.bool)
        lengths = torch.tensor([N, N])

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=N
        )

        assert torch.equal(
            new_lengths, lengths
        ), "Full mask should preserve all lengths"
        assert torch.allclose(
            masked_tensor, dense_tensor
        ), "Full mask should preserve all values"


# =============================================================================
# Test: Preprocessor mask generation
# =============================================================================


class TestPreprocessorMasking:
    """Test mask generation in the preprocessor."""

    @pytest.fixture
    def preprocessor(self, debug_cfg: DictConfig):
        """Create preprocessor instance."""
        preprocessor_config = deepcopy(debug_cfg.model.preprocessor)
        preprocessor_config.feature_counts = os.path.join(
            os.getcwd(), "user_data/processed/feature_counts.json"
        )
        return hydra.utils.instantiate(preprocessor_config, _recursive_=False)

    def _test_aux_mask_shape(self, debug_cfg: DictConfig, preprocessor):
        """Test that aux_mask has correct shape."""
        B = 4
        N = debug_cfg.data.max_seq_length
        max_output_length = debug_cfg.model.gr_output_length + 1
        total_len = N + max_output_length

        # Create fake inputs
        past_ids = torch.randint(1, 100, (B, total_len))
        past_lens = torch.tensor([10, 20, 30, N])  # various lengths

        # Create fake payloads
        past_payloads = self._create_fake_payloads(B, total_len)

        past_lens_out, seq_emb, valid_mask, aux_mask = preprocessor(
            past_lens=past_lens,
            past_ids=past_ids,
            past_payloads=past_payloads,
        )

        assert aux_mask.shape == (
            B,
            total_len,
        ), f"aux_mask shape mismatch: expected {(B, total_len)}, got {aux_mask.shape}"
        assert (
            aux_mask.dtype == torch.bool
        ), f"aux_mask dtype should be bool, got {aux_mask.dtype}"

    def _test_aux_mask_validity_count(self, debug_cfg: DictConfig, preprocessor):
        """Test that aux_mask has correct number of True values per sample."""
        B = 4
        N = debug_cfg.data.max_seq_length
        max_output_length = debug_cfg.model.gr_output_length + 1
        total_len = N + max_output_length

        past_lens = torch.tensor([5, 10, 15, 20])
        past_ids = torch.zeros(B, total_len, dtype=torch.long)

        # Fill valid positions with non-zero ids
        for i, length in enumerate(past_lens):
            past_ids[i, :length] = torch.randint(1, 100, (length.item(),))

        past_payloads = self._create_fake_payloads(B, total_len)

        past_lens_out, seq_emb, valid_mask, aux_mask = preprocessor(
            past_lens=past_lens,
            past_ids=past_ids,
            past_payloads=past_payloads,
        )

        # aux_mask True count should equal past_lens + 1 (for aux token)
        expected_counts = past_lens + 1
        actual_counts = aux_mask.sum(dim=1)

        assert torch.equal(
            actual_counts, expected_counts
        ), f"aux_mask counts mismatch: expected {expected_counts}, got {actual_counts}"

    def _test_valid_mask_padding(self, debug_cfg: DictConfig, preprocessor):
        """Test that valid_mask correctly masks padding positions."""
        B = 2
        N = debug_cfg.data.max_seq_length
        max_output_length = debug_cfg.model.gr_output_length + 1
        total_len = N + max_output_length

        past_lens = torch.tensor([5, 10])
        past_ids = torch.zeros(B, total_len, dtype=torch.long)

        # Fill valid positions
        for i, length in enumerate(past_lens):
            past_ids[i, :length] = torch.randint(1, 100, (length.item(),))

        past_payloads = self._create_fake_payloads(B, total_len)

        _, _, valid_mask, _ = preprocessor(
            past_lens=past_lens,
            past_ids=past_ids,
            past_payloads=past_payloads,
        )

        # valid_mask should be 1 for aux token (position 0) and valid items
        # After preprocessor shift, valid positions are: [aux, item1, ..., itemN]
        # valid_mask is based on past_ids != 0

        # Check that padding positions have mask = 0
        for i, length in enumerate(past_lens):
            # After aux token is added, valid length becomes length + 1
            # But the last position is removed, so valid positions are [0, length]
            valid_count = valid_mask[i].sum().item()
            # Should have length + 1 valid positions (aux + items)
            assert (
                valid_count == length + 1
            ), f"Sample {i}: expected {length + 1} valid positions, got {valid_count}"

    def _create_fake_payloads(self, B: int, N: int) -> dict:
        """Helper to create fake payloads for testing."""
        return {
            "content_embedding": torch.randn(B, N, 250),
            "category_id": torch.randint(0, 50, (B, N)),
            "created_at": torch.randint(0, 10000, (B, N)),
            "words_count": torch.randint(0, 100, (B, N)),
            "age": torch.randint(0, 100, (B, N)),
            "hour_of_day": torch.randint(1, 25, (B, N)),
            "day_of_week": torch.randint(1, 8, (B, N)),
            "environment": torch.randint(1, 5, (B,)),
            "deviceGroup": torch.randint(1, 5, (B,)),
            "os": torch.randint(1, 5, (B,)),
            "country": torch.randint(1, 5, (B,)),
            "region": torch.randint(1, 5, (B,)),
            "referrer_type": torch.randint(1, 5, (B,)),
        }


# =============================================================================
# Test: Autoregressive supervision signal alignment
# =============================================================================


class TestAutoregressiveAlignment:
    """Test that supervision signals are correctly aligned for autoregressive training."""

    @pytest.fixture
    def model(self, debug_cfg: DictConfig):
        """Create model instance."""
        model_config = deepcopy(debug_cfg.model)
        data_config = deepcopy(debug_cfg.data)

        cwd = os.getcwd()
        model_config.preprocessor.feature_counts = os.path.join(
            cwd, "user_data/processed/feature_counts.json"
        )
        data_config.train_file = os.path.join(
            cwd, "user_data/processed/sasrec_format_by_user_train.csv"
        )
        data_config.test_file = os.path.join(
            cwd, "user_data/processed/sasrec_format_by_user_test.csv"
        )
        data_config.embedding_file = os.path.join(
            cwd, "user_data/processed/article_embedding.pt"
        )
        data_config.data_preprocessor.data_dir = os.path.join(cwd, "tcdata")
        data_config.data_preprocessor.output_dir = os.path.join(
            cwd, "user_data/processed"
        )

        return hydra.utils.instantiate(
            model_config, datamodule=data_config, _recursive_=False
        )

    def _test_supervision_ids_shift(self, debug_cfg: DictConfig, model, fake_batch):
        """
        Test that supervision_ids are correctly shifted for next-item prediction.

        For autoregressive training:
        - output[i] should predict item at position i+1
        - supervision_ids[i] should equal past_ids[i+1]
        """
        device = torch.device("cpu")
        max_output_length = debug_cfg.model.gr_output_length + 1

        seq_features, target_ids = get_sequential_features(
            fake_batch, device, max_output_length
        )

        # Place target_id in sequence (simulating training_step behavior)
        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lens.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        all_ids = seq_features.past_ids

        # Compute supervision_ids as done in training_step
        supervision_ids = all_ids[:, 1:]  # shift left by 1

        # Verify shift correctness
        B, N = all_ids.shape
        for b in range(min(B, 3)):  # check first 3 samples
            for i in range(N - 1):
                assert supervision_ids[b, i] == all_ids[b, i + 1], (
                    f"Shift error at sample {b}, position {i}: "
                    f"supervision_ids={supervision_ids[b, i]}, all_ids[i+1]={all_ids[b, i+1]}"
                )

    def _test_target_id_in_supervision(self, debug_cfg: DictConfig, model, fake_batch):
        """
        Test that target_id appears in supervision_ids at the correct position.

        After scatter_:
        - past_ids[past_lens] = target_id
        After shift:
        - supervision_ids[past_lens - 1] = target_id
        """
        device = torch.device("cpu")
        max_output_length = debug_cfg.model.gr_output_length + 1

        seq_features, target_ids = get_sequential_features(
            fake_batch, device, max_output_length
        )

        original_past_lens = seq_features.past_lens.clone()

        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lens.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        supervision_ids = seq_features.past_ids[:, 1:]

        B = supervision_ids.shape[0]
        for b in range(B):
            # target should be at position (past_lens - 1) in supervision_ids
            # because supervision_ids = all_ids[:, 1:] and target is at all_ids[:, past_lens]
            target_pos = original_past_lens[b] - 1
            if target_pos >= 0:
                assert (
                    supervision_ids[b, target_pos] == target_ids[b, 0]
                ), f"Sample {b}: target_id not found at expected position {target_pos}"

    def _test_supervision_lengths_consistency(
        self, debug_cfg: DictConfig, model, fake_batch
    ):
        """
        Test that supervision_lengths are correctly computed.

        supervision_lengths should equal valid_lengths - 1
        """
        device = torch.device("cpu")
        max_output_length = debug_cfg.model.gr_output_length + 1

        seq_features, target_ids = get_sequential_features(
            fake_batch, device, max_output_length
        )

        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lens.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        # Forward pass to get valid_lengths
        encoded_embeddings, valid_lengths, _ = model.forward(seq_features)

        # Compute supervision_lengths as in training_step
        supervision_lengths = valid_lengths - 1
        supervision_lengths = torch.clamp(supervision_lengths, min=0)

        # valid_lengths should be past_lens + 1 (aux token added)
        expected_valid_lengths = seq_features.past_lens + 1

        # Check valid_lengths
        assert torch.equal(
            valid_lengths, expected_valid_lengths
        ), f"valid_lengths mismatch: expected {expected_valid_lengths}, got {valid_lengths}"

        # Check supervision_lengths = past_lens
        assert torch.equal(
            supervision_lengths, seq_features.past_lens
        ), f"supervision_lengths should equal past_lens"


# =============================================================================
# Test: Edge cases for masking
# =============================================================================


class TestMaskingEdgeCases:
    """Test edge cases in masking operations."""

    def _test_single_item_sequence(self):
        """Test masking with sequence of length 1."""
        B, N, D = 1, 10, 4

        dense_tensor = torch.randn(B, N, D)
        aux_mask = torch.zeros(B, N, dtype=torch.bool)
        aux_mask[0, 0] = True  # only first position valid
        lengths = torch.tensor([1])

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=N
        )

        assert new_lengths.item() == 1
        assert torch.allclose(masked_tensor[0, 0], dense_tensor[0, 0])

    def _test_variable_lengths_batch(self):
        """Test masking with highly variable sequence lengths in a batch."""
        B, N, D = 4, 20, 3

        dense_tensor = torch.arange(B * N * D).float().reshape(B, N, D)
        lengths = torch.tensor([1, 5, 10, 20])  # very different lengths

        aux_mask = torch.zeros(B, N, dtype=torch.bool)
        for i, length in enumerate(lengths):
            aux_mask[i, :length] = True

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=N
        )

        assert torch.equal(new_lengths, lengths)

        # Verify each sample's valid positions
        for i, length in enumerate(lengths):
            assert torch.allclose(
                masked_tensor[i, :length], dense_tensor[i, :length]
            ), f"Sample {i} values not preserved correctly"

    def _test_max_length_sequence(self, debug_cfg: DictConfig):
        """Test with sequences at maximum length."""
        N = debug_cfg.data.max_seq_length
        max_output = debug_cfg.model.gr_output_length + 1
        total_len = N + max_output
        B, D = 2, 4

        dense_tensor = torch.randn(B, total_len, D)
        lengths = torch.tensor([total_len, total_len])
        aux_mask = torch.ones(B, total_len, dtype=torch.bool)

        masked_tensor, new_lengths = ops.mask_dense_by_aux_mask(
            dense_tensor, aux_mask, lengths, max_lengths=total_len
        )

        assert torch.equal(new_lengths, lengths)
        assert torch.allclose(masked_tensor, dense_tensor)


# =============================================================================
# Test: Dense to Jagged conversion with masking
# =============================================================================


class TestDenseToJaggedWithMasking:
    """Test dense_to_jagged operations used in training."""

    def _test_jagged_lengths_consistency(self):
        """Test that jagged conversion respects lengths."""
        B, N, D = 3, 8, 4

        lengths = torch.tensor([2, 5, 3])
        dense_tensor = torch.arange(B * N * D).float().reshape(B, N, D)

        offsets = ops.asynchronous_complete_cumsum(lengths)
        jagged_tensor = ops.dense_to_jagged(dense_tensor, offsets)

        # Total elements should equal sum of lengths
        expected_total = lengths.sum().item()
        assert (
            jagged_tensor.shape[0] == expected_total
        ), f"Jagged tensor length mismatch: expected {expected_total}, got {jagged_tensor.shape[0]}"

    def _test_jagged_roundtrip(self):
        """Test that dense -> jagged -> dense preserves values."""
        B, N, D = 2, 6, 3

        lengths = torch.tensor([4, 6])
        dense_tensor = torch.randn(B, N, D)

        # Mask out positions beyond length
        for i, length in enumerate(lengths):
            dense_tensor[i, length:] = 0

        offsets = ops.asynchronous_complete_cumsum(lengths)
        jagged_tensor = ops.dense_to_jagged(dense_tensor, offsets)

        recovered_dense = ops.jagged_to_padded_dense(
            jagged_tensor, offsets, max_lengths=N, padding_value=0.0
        )

        # Check valid positions match
        for i, length in enumerate(lengths):
            assert torch.allclose(
                recovered_dense[i, :length], dense_tensor[i, :length]
            ), f"Sample {i}: roundtrip values don't match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
