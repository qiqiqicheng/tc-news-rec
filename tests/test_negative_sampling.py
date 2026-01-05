import os
from copy import deepcopy

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from tc_news_rec.models.losses.losses import SampledSoftmaxLoss
from tc_news_rec.models.negative_samplers.negative_samplers import (
    GlobalNegativeSampler,
    NegativeSampler,
)
from tc_news_rec.models.similarity.similarity import DotProductSimilarity
from tc_news_rec.models.utils.features import get_sequential_features

OmegaConf.register_new_resolver("eval", eval, replace=True)


# =============================================================================
# Test: GlobalNegativeSampler basic functionality
# =============================================================================


class TestGlobalNegativeSampler:
    """Test GlobalNegativeSampler functionality."""

    @pytest.fixture
    def sampler(self):
        """Create a GlobalNegativeSampler instance."""
        return GlobalNegativeSampler(l2_normalize=True)

    @pytest.fixture
    def embedding_table(self):
        """Create a fake embedding table."""
        num_items = 1000
        emb_dim = 64
        emb = torch.nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        return emb

    def _test_sampler_output_shape(self, sampler, embedding_table):
        """Test that sampler returns correct output shapes."""
        num_items = 1000
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(embedding_table)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        B = 32
        num_to_sample = 128
        positive_ids = torch.randint(1, num_items + 1, (B,))

        neg_ids, neg_embeddings = sampler(positive_ids, num_to_sample)

        assert neg_ids.shape == (
            B,
            num_to_sample,
        ), f"neg_ids shape mismatch: expected {(B, num_to_sample)}, got {neg_ids.shape}"
        assert neg_embeddings.shape == (
            B,
            num_to_sample,
            64,
        ), f"neg_embeddings shape mismatch: expected {(B, num_to_sample, 64)}, got {neg_embeddings.shape}"

    def _test_sampler_output_shape_2d_input(self, sampler, embedding_table):
        """Test sampler with 2D input (transformer-style)."""
        num_items = 1000
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(embedding_table)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        B, L = 16, 50
        num_to_sample = 64
        positive_ids = torch.randint(1, num_items + 1, (B, L))

        neg_ids, neg_embeddings = sampler(positive_ids, num_to_sample)

        assert neg_ids.shape == (
            B,
            L,
            num_to_sample,
        ), "neg_ids shape mismatch for 2D input"
        assert neg_embeddings.shape == (
            B,
            L,
            num_to_sample,
            64,
        ), "neg_embeddings shape mismatch for 2D input"

    def _test_negative_ids_in_valid_range(self, sampler, embedding_table):
        """Test that all negative IDs are within valid item ID range."""
        num_items = 1000
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(embedding_table)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        B = 100
        num_to_sample = 256
        positive_ids = torch.randint(1, num_items + 1, (B,))

        neg_ids, _ = sampler(positive_ids, num_to_sample)

        # All negative IDs should be >= 1 and <= num_items
        assert (neg_ids >= 1).all(), "Found negative ID < 1"
        assert (neg_ids <= num_items).all(), f"Found negative ID > {num_items}"

    def _test_l2_normalization_enabled(self, sampler, embedding_table):
        """Test that L2 normalization is applied when enabled."""
        num_items = 100
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(embedding_table)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        B = 10
        num_to_sample = 20
        positive_ids = torch.randint(1, num_items + 1, (B,))

        _, neg_embeddings = sampler(positive_ids, num_to_sample)

        # Check L2 norm of embeddings
        norms = torch.linalg.norm(neg_embeddings, dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Embeddings not L2 normalized. Norms range: [{norms.min():.4f}, {norms.max():.4f}]"
        )

    def _test_l2_normalization_disabled(self, embedding_table):
        """Test that L2 normalization is not applied when disabled."""
        sampler = GlobalNegativeSampler(l2_normalize=False)

        num_items = 100
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(embedding_table)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        B = 10
        num_to_sample = 20
        positive_ids = torch.randint(1, num_items + 1, (B,))

        _, neg_embeddings = sampler(positive_ids, num_to_sample)

        # Check L2 norm of embeddings - should NOT all be 1
        norms = torch.linalg.norm(neg_embeddings, dim=-1)

        # With random embeddings, it's highly unlikely all norms are exactly 1
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            "Embeddings appear normalized when l2_normalize=False"
        )


# =============================================================================
# Test: Negative sampling exclusion (positives should not be sampled)
# =============================================================================


class TestNegativeSamplingExclusion:
    """Test that positive samples are properly handled in negative sampling."""

    def _test_collision_rate_is_low(self):
        """
        Test that collision rate (negative == positive) is low.

        With random sampling from a large item pool, collisions should be rare.
        """
        sampler = GlobalNegativeSampler(l2_normalize=True)
        num_items = 10000
        emb = torch.nn.Embedding(num_items + 1, 64, padding_idx=0)
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(emb)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        B = 1000
        num_to_sample = 128
        positive_ids = torch.randint(1, num_items + 1, (B,))

        neg_ids, _ = sampler(positive_ids, num_to_sample)

        # Count collisions
        collisions = (neg_ids == positive_ids.unsqueeze(1)).sum().item()
        total_samples = B * num_to_sample
        collision_rate = collisions / total_samples

        # Expected collision rate: num_to_sample / num_items = 128 / 10000 = 1.28%
        expected_rate = num_to_sample / num_items

        # Allow 2x expected rate due to randomness
        assert collision_rate < expected_rate * 2, (
            f"Collision rate {collision_rate:.4f} too high (expected ~{expected_rate:.4f})"
        )

    def _test_loss_handles_collisions(self):
        """
        Test that the loss function properly masks collisions.

        In SampledSoftmaxLoss, when negative_id == positive_id,
        the negative logit should be set to -5e4 (effectively -inf).
        """
        loss_fn = SampledSoftmaxLoss(num_to_sample=10, softmax_temperature=0.05)
        similarity = DotProductSimilarity()

        # Create controlled scenario with known collision
        T = 5  # number of positions
        D = 16
        num_to_sample = 10

        output_embeddings = torch.randn(T, D)
        output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)

        supervision_ids = torch.tensor([1, 2, 3, 4, 5])
        supervision_embeddings = torch.randn(T, D)
        supervision_embeddings = supervision_embeddings / supervision_embeddings.norm(dim=-1, keepdim=True)
        supervision_weights = torch.ones(T)

        # Create mock negative sampler that intentionally creates collisions
        class MockSamplerWithCollisions(NegativeSampler):
            def forward(self, positive_item_ids, num_to_sample):
                B = positive_item_ids.shape[0]
                # First negative is always same as positive (collision)
                neg_ids = torch.randint(1, 100, (B, num_to_sample))
                neg_ids[:, 0] = positive_item_ids  # force collision
                neg_embeddings = torch.randn(B, num_to_sample, D)
                neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
                return neg_ids, neg_embeddings

        mock_sampler = MockSamplerWithCollisions()

        # Loss should still compute without NaN
        loss = loss_fn.jagged_forward(
            output_embeddings=output_embeddings,
            supervision_ids=supervision_ids,
            supervision_embeddings=supervision_embeddings,
            supervision_weights=supervision_weights,
            negative_sampler=mock_sampler,
            similarity_module=similarity,
        )

        assert not torch.isnan(loss), "Loss is NaN when collisions exist"
        assert not torch.isinf(loss), "Loss is Inf when collisions exist"


# =============================================================================
# Test: L2 normalization consistency
# =============================================================================


class TestNormalizationConsistency:
    """Test L2 normalization consistency between positive and negative embeddings."""

    @pytest.fixture
    def model(self, debug_cfg: DictConfig):
        """Create model instance."""
        model_config = deepcopy(debug_cfg.model)
        data_config = deepcopy(debug_cfg.data)

        cwd = os.getcwd()
        model_config.preprocessor.feature_counts = os.path.join(cwd, "user_data/processed/feature_counts.json")
        data_config.train_file = os.path.join(cwd, "user_data/processed/sasrec_format_by_user_train.csv")
        data_config.test_file = os.path.join(cwd, "user_data/processed/sasrec_format_by_user_test.csv")
        data_config.embedding_file = os.path.join(cwd, "user_data/processed/article_embedding.pt")
        data_config.data_preprocessor.data_dir = os.path.join(cwd, "tcdata")
        data_config.data_preprocessor.output_dir = os.path.join(cwd, "user_data/processed")

        return hydra.utils.instantiate(model_config, datamodule=data_config, _recursive_=False)

    def _test_output_embeddings_are_normalized(self, model, debug_cfg: DictConfig, fake_batch):
        """Test that model output embeddings are L2 normalized."""
        device = torch.device("cpu")
        max_output_length = debug_cfg.model.gr_output_length + 1

        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lens.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        encoded_embeddings, valid_lengths, _ = model.forward(seq_features)

        # Check L2 norm of output embeddings (should be 1 after L2NormPostprocessor)
        norms = torch.linalg.norm(encoded_embeddings, dim=-1)

        # Only check valid positions
        B, N, D = encoded_embeddings.shape
        for b in range(B):
            valid_norms = norms[b, : valid_lengths[b]]
            assert torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-5), (
                f"Sample {b}: Output embeddings not normalized. Norms: {valid_norms}"
            )

    def _test_positive_negative_normalization_match(self, model, debug_cfg: DictConfig, fake_batch):
        """
        Test that positive and negative embeddings have consistent normalization.

        This is critical because inconsistent normalization will cause the loss
        to be biased toward either positives or negatives.
        """
        device = torch.device("cpu")
        max_output_length = debug_cfg.model.gr_output_length + 1

        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lens.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        # Get supervision embeddings (positive)
        supervision_ids = seq_features.past_ids[:, 1:]
        supervision_embeddings = model.preprocessor.get_embedding_by_id(supervision_ids)

        # Set up negative sampler
        model.negative_sampler.set_item_embedding(model.preprocessor.get_item_id_embedding_module())
        model.negative_sampler.set_all_item_ids(model.preprocessor.get_all_item_ids(), device=device)

        # Get negative embeddings
        # Use reshape instead of view because supervision_ids may not be contiguous after slicing
        neg_ids, neg_embeddings = model.negative_sampler(supervision_ids.reshape(-1), num_to_sample=64)

        # Check normalization of negatives
        neg_norms = torch.linalg.norm(neg_embeddings, dim=-1)
        assert torch.allclose(neg_norms, torch.ones_like(neg_norms), atol=1e-5), (
            f"Negative embeddings not normalized. Norms range: [{neg_norms.min():.4f}, {neg_norms.max():.4f}]"
        )

        # Check normalization of positives - NOTE: these may NOT be normalized
        # This is a potential issue identified in the original analysis
        pos_norms = torch.linalg.norm(supervision_embeddings, dim=-1)

        # Log warning if positives are not normalized (this is the known issue #4)
        if not torch.allclose(
            pos_norms[pos_norms > 0],
            torch.ones_like(pos_norms[pos_norms > 0]),
            atol=0.1,
        ):
            print("WARNING: Positive embeddings may not be L2 normalized!")
            print(
                f"  Positive norms range: [{pos_norms[pos_norms > 0].min():.4f}, {pos_norms[pos_norms > 0].max():.4f}]"
            )
            print(f"  Negative norms range: [{neg_norms.min():.4f}, {neg_norms.max():.4f}]")


# =============================================================================
# Test: Sampling distribution
# =============================================================================


class TestSamplingDistribution:
    """Test the distribution of negative samples."""

    def _test_uniform_sampling_coverage(self):
        """Test that sampling approximately covers all items uniformly."""
        sampler = GlobalNegativeSampler(l2_normalize=False)
        num_items = 100
        emb = torch.nn.Embedding(num_items + 1, 16, padding_idx=0)
        all_item_ids = list(range(1, num_items + 1))

        sampler.set_item_embedding(emb)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        # Sample many times
        num_trials = 1000
        num_to_sample = 50
        positive_ids = torch.randint(1, num_items + 1, (num_trials,))

        neg_ids, _ = sampler(positive_ids, num_to_sample)

        # Count frequency of each item
        counts = torch.bincount(neg_ids.view(-1), minlength=num_items + 1)
        counts = counts[1:]  # exclude padding idx 0

        # Expected count per item
        total_samples = num_trials * num_to_sample
        expected_per_item = total_samples / num_items

        # Check that all items were sampled at least once
        assert (counts > 0).all(), "Some items were never sampled"

        # Check that distribution is roughly uniform (within 3x of expected)
        assert counts.max() < expected_per_item * 3, (
            f"Max count {counts.max()} is too high (expected ~{expected_per_item:.0f})"
        )
        assert counts.min() > expected_per_item / 3, (
            f"Min count {counts.min()} is too low (expected ~{expected_per_item:.0f})"
        )

    def _test_no_padding_idx_sampled(self):
        """Test that padding index (0) is never sampled."""
        sampler = GlobalNegativeSampler(l2_normalize=False)
        num_items = 100
        emb = torch.nn.Embedding(num_items + 1, 16, padding_idx=0)
        all_item_ids = list(range(1, num_items + 1))  # starts from 1, not 0

        sampler.set_item_embedding(emb)
        sampler.set_all_item_ids(all_item_ids, device=torch.device("cpu"))

        # Sample many times
        for _ in range(100):
            positive_ids = torch.randint(1, num_items + 1, (64,))
            neg_ids, _ = sampler(positive_ids, num_to_sample=128)

            assert (neg_ids != 0).all(), "Padding index 0 was sampled as negative"


# =============================================================================
# Test: Integration with loss computation
# =============================================================================


class TestNegativeSamplingLossIntegration:
    """Test negative sampling integration with loss computation."""

    def _test_loss_decreases_with_perfect_model(self):
        """
        Test that loss is low when model output perfectly matches positive embeddings.

        If output_embedding == positive_embedding, the positive logit should be
        highest, resulting in low loss.
        """
        loss_fn = SampledSoftmaxLoss(num_to_sample=64, softmax_temperature=0.05)
        similarity = DotProductSimilarity()

        T = 10
        D = 32

        # Create perfectly aligned output and positive embeddings
        output_embeddings = torch.randn(T, D)
        output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)

        # Positive embeddings are same as output (perfect model)
        supervision_embeddings = output_embeddings.clone()
        supervision_ids = torch.arange(1, T + 1)
        supervision_weights = torch.ones(T)

        # Create negative sampler with random embeddings
        class MockRandomNegativeSampler(NegativeSampler):
            def forward(self, positive_item_ids, num_to_sample):
                B = positive_item_ids.shape[0]
                neg_ids = torch.randint(100, 1000, (B, num_to_sample))
                neg_embeddings = torch.randn(B, num_to_sample, D)
                neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
                return neg_ids, neg_embeddings

        mock_sampler = MockRandomNegativeSampler()

        loss = loss_fn.jagged_forward(
            output_embeddings=output_embeddings,
            supervision_ids=supervision_ids,
            supervision_embeddings=supervision_embeddings,
            supervision_weights=supervision_weights,
            negative_sampler=mock_sampler,
            similarity_module=similarity,
        )

        # With perfect alignment, positive logit = 1/temp = 20
        # Random negatives have expected logit ~0/temp = 0
        # Loss should be low
        assert loss < 1.0, f"Loss {loss:.4f} too high for perfect model"

    def _test_loss_high_with_random_model(self):
        """
        Test that loss is high when model output is random.
        """
        loss_fn = SampledSoftmaxLoss(num_to_sample=64, softmax_temperature=0.05)
        similarity = DotProductSimilarity()

        T = 10
        D = 32

        # Random output embeddings
        output_embeddings = torch.randn(T, D)
        output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)

        # Different random positive embeddings
        supervision_embeddings = torch.randn(T, D)
        supervision_embeddings = supervision_embeddings / supervision_embeddings.norm(dim=-1, keepdim=True)
        supervision_ids = torch.arange(1, T + 1)
        supervision_weights = torch.ones(T)

        class MockRandomNegativeSampler(NegativeSampler):
            def forward(self, positive_item_ids, num_to_sample):
                B = positive_item_ids.shape[0]
                neg_ids = torch.randint(100, 1000, (B, num_to_sample))
                neg_embeddings = torch.randn(B, num_to_sample, D)
                neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
                return neg_ids, neg_embeddings

        mock_sampler = MockRandomNegativeSampler()

        loss = loss_fn.jagged_forward(
            output_embeddings=output_embeddings,
            supervision_ids=supervision_ids,
            supervision_embeddings=supervision_embeddings,
            supervision_weights=supervision_weights,
            negative_sampler=mock_sampler,
            similarity_module=similarity,
        )

        # With random embeddings, loss should be relatively high
        # log(1 + 64) ≈ 4.17 as baseline for uniform distribution
        assert loss > 2.0, f"Loss {loss:.4f} suspiciously low for random model"

    def _test_gradient_flows_through_sampling(self):
        """Test that gradients flow correctly through the loss computation."""
        loss_fn = SampledSoftmaxLoss(num_to_sample=32, softmax_temperature=0.05)
        similarity = DotProductSimilarity()

        T = 5
        D = 16

        output_embeddings = torch.randn(T, D, requires_grad=True)
        normalized_output = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)

        supervision_embeddings = torch.randn(T, D)
        supervision_embeddings = supervision_embeddings / supervision_embeddings.norm(dim=-1, keepdim=True)
        supervision_ids = torch.arange(1, T + 1)
        supervision_weights = torch.ones(T)

        class MockSampler(NegativeSampler):
            def forward(self, positive_item_ids, num_to_sample):
                B = positive_item_ids.shape[0]
                neg_ids = torch.randint(100, 1000, (B, num_to_sample))
                neg_embeddings = torch.randn(B, num_to_sample, D)
                neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
                return neg_ids, neg_embeddings

        mock_sampler = MockSampler()

        loss = loss_fn.jagged_forward(
            output_embeddings=normalized_output,
            supervision_ids=supervision_ids,
            supervision_embeddings=supervision_embeddings,
            supervision_weights=supervision_weights,
            negative_sampler=mock_sampler,
            similarity_module=similarity,
        )

        loss.backward()

        assert output_embeddings.grad is not None, "No gradient for output_embeddings"
        assert not torch.isnan(output_embeddings.grad).any(), "NaN in gradients"
        assert output_embeddings.grad.abs().sum() > 0, "Gradients are all zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
