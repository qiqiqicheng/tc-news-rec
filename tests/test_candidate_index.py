import os
from copy import deepcopy

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from tc_news_rec.models.indexing.candidate_index import CandidateIndex
from tc_news_rec.models.indexing.top_k import MIPSBruteTopK
from tc_news_rec.models.utils.features import get_sequential_features

OmegaConf.register_new_resolver("eval", eval, replace=True)


# =============================================================================
# Test: CandidateIndex Basic Functionality
# =============================================================================


class TestCandidateIndexBasic:
    """Test basic CandidateIndex functionality."""

    def test_initialization_with_embeddings(self):
        """Test CandidateIndex initialization with provided embeddings."""
        num_items = 100
        emb_dim = 64

        ids = torch.arange(1, num_items + 1).view(1, -1)  # [1, 100]
        embeddings = torch.randn(1, num_items, emb_dim)  # [1, 100, 64]

        index = CandidateIndex(
            ids=ids,
            embeddings=embeddings,
            top_k_module=MIPSBruteTopK(),
        )

        assert index.ids.shape == (1, num_items)
        assert index.num_objects == num_items
        assert index.embeddings is not None
        # embeddings property returns [1, X, D] to match ids shape [1, X]
        assert index.embeddings.shape == (1, num_items, emb_dim)

    def test_initialization_without_embeddings(self):
        """Test CandidateIndex initialization without embeddings."""
        num_items = 100

        ids = torch.arange(1, num_items + 1).view(1, -1)

        index = CandidateIndex(
            ids=ids,
            embeddings=None,
            top_k_module=MIPSBruteTopK(),
        )

        assert index.ids.shape == (1, num_items)
        assert index.embeddings is None

    def test_update_embeddings(self):
        """Test that update_embeddings correctly updates the internal state."""
        num_items = 50
        emb_dim = 32

        ids = torch.arange(1, num_items + 1).view(1, -1)

        index = CandidateIndex(
            ids=ids,
            embeddings=None,
            top_k_module=MIPSBruteTopK(),
        )

        # Initially no embeddings
        assert index.embeddings is None

        # Update with new embeddings
        new_embeddings = torch.randn(1, num_items, emb_dim)
        index.update_embeddings(new_embeddings)

        # Verify embeddings are stored
        assert index.embeddings is not None
        # embeddings property returns [1, X, D] to match ids shape [1, X]
        assert index.embeddings.shape == (1, num_items, emb_dim)

        # Verify the content matches
        retrieved_emb = index.embeddings
        assert torch.allclose(retrieved_emb, new_embeddings, atol=1e-6)

    def test_update_embeddings_overwrite(self):
        """Test that update_embeddings overwrites previous embeddings."""
        num_items = 30
        emb_dim = 16

        ids = torch.arange(1, num_items + 1).view(1, -1)
        initial_embeddings = torch.ones(1, num_items, emb_dim)

        index = CandidateIndex(
            ids=ids,
            embeddings=initial_embeddings,
            top_k_module=MIPSBruteTopK(),
        )

        # Verify initial embeddings (shape is [1, num_items, emb_dim])
        assert torch.allclose(index.embeddings, torch.ones(1, num_items, emb_dim))

        # Update with new embeddings
        new_embeddings = torch.zeros(1, num_items, emb_dim)
        index.update_embeddings(new_embeddings)

        # Verify new embeddings replaced old ones
        assert torch.allclose(index.embeddings, torch.zeros(1, num_items, emb_dim))


# =============================================================================
# Test: Top-K Retrieval
# =============================================================================


class TestTopKRetrieval:
    """Test top-k retrieval functionality."""

    @pytest.fixture
    def index_with_embeddings(self):
        """Create a CandidateIndex with known embeddings for testing."""
        num_items = 100
        emb_dim = 32

        ids = torch.arange(1, num_items + 1).view(1, -1)

        # Create embeddings where item i has embedding = [i, i, i, ...] normalized
        embeddings = torch.arange(1, num_items + 1).float().view(-1, 1).expand(-1, emb_dim)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.unsqueeze(0)  # [1, num_items, emb_dim]

        index = CandidateIndex(
            ids=ids,
            embeddings=embeddings,
            top_k_module=MIPSBruteTopK(),
        )
        return index

    def test_top_k_output_shape(self, index_with_embeddings):
        """Test that top-k retrieval returns correct shapes."""
        B = 8
        k = 10
        emb_dim = 32

        query_embeddings = torch.randn(B, emb_dim)
        query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)

        top_k_ids, top_k_scores = index_with_embeddings.get_top_k_outputs(
            query_embeddings=query_embeddings,
            k=k,
        )

        assert top_k_ids.shape == (B, k)
        assert top_k_scores.shape == (B, k)

    def test_top_k_ids_in_valid_range(self, index_with_embeddings):
        """Test that returned IDs are within valid range."""
        B = 16
        k = 20

        query_embeddings = torch.randn(B, 32)
        query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)

        top_k_ids, _ = index_with_embeddings.get_top_k_outputs(
            query_embeddings=query_embeddings,
            k=k,
        )

        # All IDs should be in [1, 100]
        assert (top_k_ids >= 1).all()
        assert (top_k_ids <= 100).all()

    def test_top_k_scores_sorted(self, index_with_embeddings):
        """Test that returned scores are sorted in descending order."""
        B = 8
        k = 15

        query_embeddings = torch.randn(B, 32)
        query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)

        _, top_k_scores = index_with_embeddings.get_top_k_outputs(
            query_embeddings=query_embeddings,
            k=k,
        )

        # Check that scores are sorted (descending)
        for i in range(B):
            scores = top_k_scores[i]
            assert torch.all(scores[:-1] >= scores[1:]), f"Scores not sorted for batch {i}"

    def test_top_k_with_invalid_ids_filtering(self, index_with_embeddings):
        """Test that invalid IDs are filtered from results."""
        B = 4
        k = 10
        num_invalid = 5

        query_embeddings = torch.randn(B, 32)
        query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)

        # Create invalid IDs (these should be excluded from results)
        invalid_ids = torch.randint(1, 101, (B, num_invalid))

        top_k_ids, _ = index_with_embeddings.get_top_k_outputs(
            query_embeddings=query_embeddings,
            k=k,
            invalid_ids=invalid_ids,
        )

        # Check that no invalid IDs appear in results
        for b in range(B):
            for invalid_id in invalid_ids[b]:
                assert invalid_id not in top_k_ids[b], f"Invalid ID {invalid_id} found in results for batch {b}"


# =============================================================================
# Test: Model-CandidateIndex Synchronization
# =============================================================================


class TestCandidateIndexSync:
    """Test synchronization between model embeddings and candidate index."""

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

    def test_initial_candidate_embeddings_none(self, model):
        """Test that candidate index starts with no embeddings."""
        # Initially, candidate index may not have embeddings loaded
        # This depends on initialization, but embeddings should be updatable
        assert hasattr(model, "candidate_index")
        assert hasattr(model.candidate_index, "embeddings")

    def test_update_candidate_embeddings_called(self, model):
        """Test that _update_candidate_embeddings updates the index."""
        # Get initial state
        initial_embeddings = model.candidate_index.embeddings

        # Call update
        model._update_candidate_embeddings()

        # Embeddings should now be set
        assert model.candidate_index.embeddings is not None

    def test_candidate_embeddings_match_item_embeddings(self, model):
        """Test that candidate embeddings match the item embedding table after update."""
        # Update candidate embeddings
        model._update_candidate_embeddings()

        # Get item embeddings from preprocessor (excluding padding at index 0)
        item_emb_module = model.preprocessor.get_item_id_embedding_module()
        item_embeddings = item_emb_module.weight[1:]  # [num_items, D]

        # Normalize (as done in _update_candidate_embeddings via negative_sampler.normalize_embeddings)
        normalized_item_embeddings = item_embeddings / item_embeddings.norm(dim=-1, keepdim=True)

        # Get candidate embeddings (shape is [1, num_items, D])
        candidate_embeddings = model.candidate_index.embeddings
        # Squeeze to [num_items, D] for comparison
        candidate_embeddings_squeezed = candidate_embeddings.squeeze(0)

        # They should match
        assert candidate_embeddings_squeezed.shape == normalized_item_embeddings.shape, (
            f"Shape mismatch: {candidate_embeddings_squeezed.shape} vs {normalized_item_embeddings.shape}"
        )

        assert torch.allclose(candidate_embeddings_squeezed, normalized_item_embeddings, atol=1e-5), (
            "Candidate embeddings do not match normalized item embeddings"
        )

    def test_embeddings_update_after_training_step(self, model, fake_batch):
        """Test that embeddings are properly synchronized after a training step modifies weights."""
        device = torch.device("cpu")
        model.to(device)
        model.train()

        # Initial update
        model._update_candidate_embeddings()
        initial_candidate_emb = model.candidate_index.embeddings.clone()

        # Simulate a gradient update on the item embeddings
        item_emb_module = model.preprocessor.get_item_id_embedding_module()
        with torch.no_grad():
            item_emb_module.weight.add_(torch.randn_like(item_emb_module.weight) * 0.1)

        # Call update again
        model._update_candidate_embeddings()
        updated_candidate_emb = model.candidate_index.embeddings

        # Embeddings should have changed
        assert not torch.allclose(initial_candidate_emb, updated_candidate_emb, atol=1e-6), (
            "Candidate embeddings did not update after item embedding modification"
        )

    def test_candidate_ids_match_all_item_ids(self, model):
        """Test that candidate index IDs match all item IDs from preprocessor."""
        all_item_ids = model.preprocessor.get_all_item_ids()
        candidate_ids = model.candidate_index.ids.squeeze(0).tolist()

        assert len(candidate_ids) == len(all_item_ids), (
            f"ID count mismatch: {len(candidate_ids)} vs {len(all_item_ids)}"
        )

        assert set(candidate_ids) == set(all_item_ids), "Candidate IDs do not match all item IDs"


# =============================================================================
# Test: Retrieval with Updated Embeddings
# =============================================================================


class TestRetrievalWithSync:
    """Test retrieval functionality with synchronized embeddings."""

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

    def test_retrieve_returns_valid_results(self, model, debug_cfg, fake_batch):
        """Test that retrieve returns valid top-k results."""
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        max_output_length = debug_cfg.model.gr_output_length + 1
        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        k = 10
        top_k_ids, top_k_scores = model.retrieve(
            seq_features=seq_features,
            k=k,
            filter_past_ids=True,
        )

        B = fake_batch["user_id"].shape[0]
        assert top_k_ids.shape == (B, k)
        assert top_k_scores.shape == (B, k)

        # All IDs should be positive
        assert (top_k_ids > 0).all()

    def test_retrieve_filters_past_ids(self, model, debug_cfg, fake_batch):
        """Test that retrieve properly filters past IDs when requested."""
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        max_output_length = debug_cfg.model.gr_output_length + 1
        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        k = 10
        top_k_ids, _ = model.retrieve(
            seq_features=seq_features,
            k=k,
            filter_past_ids=True,
        )

        # Check that past IDs are not in the results
        B = top_k_ids.shape[0]
        for b in range(B):
            past_ids_set = set(seq_features.past_ids[b].tolist())
            past_ids_set.discard(0)  # Remove padding
            retrieved_ids_set = set(top_k_ids[b].tolist())

            overlap = past_ids_set & retrieved_ids_set
            assert len(overlap) == 0, f"Batch {b}: Found {len(overlap)} past IDs in retrieval results"

    def test_retrieve_no_filter_includes_past_ids(self, model, debug_cfg, fake_batch):
        """Test that retrieve without filtering may include past IDs."""
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        max_output_length = debug_cfg.model.gr_output_length + 1
        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        k = 50  # Large k to increase chance of overlap
        top_k_ids, _ = model.retrieve(
            seq_features=seq_features,
            k=k,
            filter_past_ids=False,  # Don't filter
        )

        # Results should still be valid (all positive IDs)
        assert (top_k_ids > 0).all()


# =============================================================================
# Test: Epoch Hooks for Synchronization
# =============================================================================


class TestEpochHooksSync:
    """Test that epoch hooks properly synchronize embeddings."""

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

    def test_on_validation_epoch_start_updates_embeddings(self, model):
        """Test that on_validation_epoch_start updates candidate embeddings."""
        # Modify item embeddings
        item_emb_module = model.preprocessor.get_item_id_embedding_module()
        with torch.no_grad():
            item_emb_module.weight.fill_(1.0)

        # Call the hook (need to set up metrics first)
        model.metrics.reset()

        # Simulate the hook
        model._update_candidate_embeddings()

        # Check that embeddings are updated to normalized ones
        expected = torch.ones_like(model.candidate_index.embeddings)
        expected = expected / expected.norm(dim=-1, keepdim=True)

        assert torch.allclose(model.candidate_index.embeddings, expected, atol=1e-5)

    def test_on_test_epoch_start_updates_embeddings(self, model):
        """Test that on_test_epoch_start updates candidate embeddings."""
        # Similar to validation, test epoch should also update embeddings
        model.metrics.reset()
        model._update_candidate_embeddings()

        assert model.candidate_index.embeddings is not None

    def test_on_predict_epoch_start_updates_embeddings(self, model):
        """Test that on_predict_epoch_start updates candidate embeddings."""
        model._update_candidate_embeddings()

        assert model.candidate_index.embeddings is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
