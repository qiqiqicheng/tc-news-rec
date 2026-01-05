import csv
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import hydra
import lightning as L
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from tc_news_rec.models.metrics.retrieval_metrics import RetrievalMetrics
from tc_news_rec.models.utils.features import get_sequential_features

OmegaConf.register_new_resolver("eval", eval, replace=True)


# =============================================================================
# Test: RetrievalMetrics Basic Functionality
# =============================================================================


class TestRetrievalMetricsBasic:
    """Test basic RetrievalMetrics functionality."""

    def test_initialization(self):
        """Test RetrievalMetrics initialization."""
        k = 10
        at_k_list = [1, 5, 10]

        metrics = RetrievalMetrics(k=k, at_k_list=at_k_list)

        assert metrics.k == k
        assert metrics.at_k_list == at_k_list

    def test_update_accumulates_data(self):
        """Test that update accumulates predictions and targets."""
        metrics = RetrievalMetrics(k=10, at_k_list=[1, 5, 10])

        # First batch
        top_k_ids_1 = torch.randint(1, 100, (8, 10))
        target_ids_1 = torch.randint(1, 100, (8,))
        metrics.update(top_k_ids=top_k_ids_1, target_ids=target_ids_1)

        assert len(metrics.top_k_ids) == 1
        assert len(metrics.target_ids) == 1

        # Second batch
        top_k_ids_2 = torch.randint(1, 100, (8, 10))
        target_ids_2 = torch.randint(1, 100, (8,))
        metrics.update(top_k_ids=top_k_ids_2, target_ids=target_ids_2)

        assert len(metrics.top_k_ids) == 2
        assert len(metrics.target_ids) == 2

    def test_reset_clears_state(self):
        """Test that reset clears accumulated state."""
        metrics = RetrievalMetrics(k=10, at_k_list=[1, 5, 10])

        # Add some data
        top_k_ids = torch.randint(1, 100, (8, 10))
        target_ids = torch.randint(1, 100, (8,))
        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

        assert len(metrics.top_k_ids) > 0

        # Reset
        metrics.reset()

        assert len(metrics.top_k_ids) == 0
        assert len(metrics.target_ids) == 0

    def test_compute_returns_dict(self):
        """Test that compute returns a dictionary with expected keys."""
        k = 10
        at_k_list = [1, 5, 10]
        metrics = RetrievalMetrics(k=k, at_k_list=at_k_list)

        top_k_ids = torch.randint(1, 100, (16, 10))
        target_ids = torch.randint(1, 100, (16,))
        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

        results = metrics.compute()

        # Check expected keys
        assert "mrr" in results
        for at_k in at_k_list:
            assert f"ndcg@{at_k}" in results
            assert f"hr@{at_k}" in results


# =============================================================================
# Test: NDCG Computation
# =============================================================================


class TestNDCGComputation:
    """Test NDCG (Normalized Discounted Cumulative Gain) computation."""

    def test_ndcg_perfect_at_1(self):
        """Test NDCG when target is always at position 1."""
        metrics = RetrievalMetrics(k=10, at_k_list=[1, 5, 10])

        # Target is always the first result
        top_k_ids = torch.tensor([
            [5, 2, 3, 4, 6, 7, 8, 9, 10, 11],
            [15, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        ])
        target_ids = torch.tensor([5, 15])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # When target is at rank 1, NDCG = 1/log2(2) = 1.0
        assert torch.isclose(results["ndcg@1"], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(results["ndcg@5"], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(results["ndcg@10"], torch.tensor(1.0), atol=1e-5)

    def test_ndcg_at_position_2(self):
        """Test NDCG when target is at position 2."""
        metrics = RetrievalMetrics(k=10, at_k_list=[1, 5, 10])

        # Target is at position 2
        top_k_ids = torch.tensor([
            [1, 5, 3, 4, 6, 7, 8, 9, 10, 11],
        ])
        target_ids = torch.tensor([5])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # At rank 2, NDCG = 1/log2(3) ≈ 0.6309
        expected_ndcg = 1.0 / torch.log2(torch.tensor(3.0))
        assert torch.isclose(results["ndcg@5"], expected_ndcg, atol=1e-4)
        assert torch.isclose(results["ndcg@10"], expected_ndcg, atol=1e-4)

        # At @1, target is not found, so NDCG@1 = 0
        assert torch.isclose(results["ndcg@1"], torch.tensor(0.0), atol=1e-5)

    def test_ndcg_not_in_top_k(self):
        """Test NDCG when target is not in top-k."""
        metrics = RetrievalMetrics(k=5, at_k_list=[1, 5])

        # Target is not in top-k
        top_k_ids = torch.tensor([
            [1, 2, 3, 4, 6],  # target 5 not present
        ])
        target_ids = torch.tensor([5])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # NDCG should be 0 when target is not found
        assert torch.isclose(results["ndcg@1"], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(results["ndcg@5"], torch.tensor(0.0), atol=1e-5)


# =============================================================================
# Test: Hit Rate (HR) Computation
# =============================================================================


class TestHitRateComputation:
    """Test Hit Rate computation."""

    def test_hr_all_hits(self):
        """Test HR when all targets are found."""
        metrics = RetrievalMetrics(k=10, at_k_list=[5, 10])

        # All targets are in top-5
        top_k_ids = torch.tensor([
            [5, 2, 3, 4, 6, 7, 8, 9, 10, 11],
            [15, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            [25, 22, 23, 24, 26, 27, 28, 29, 30, 31],
        ])
        target_ids = torch.tensor([5, 15, 25])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        assert torch.isclose(results["hr@5"], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(results["hr@10"], torch.tensor(1.0), atol=1e-5)

    def test_hr_no_hits(self):
        """Test HR when no targets are found."""
        metrics = RetrievalMetrics(k=5, at_k_list=[1, 5])

        # No targets in top-k
        top_k_ids = torch.tensor([
            [1, 2, 3, 4, 6],
            [11, 12, 13, 14, 16],
        ])
        target_ids = torch.tensor([100, 200])  # Not in top-k

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        assert torch.isclose(results["hr@1"], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(results["hr@5"], torch.tensor(0.0), atol=1e-5)

    def test_hr_partial_hits(self):
        """Test HR with partial hits."""
        metrics = RetrievalMetrics(k=5, at_k_list=[5])

        # 2 out of 4 targets found
        top_k_ids = torch.tensor([
            [5, 2, 3, 4, 6],  # target 5 found
            [1, 2, 3, 4, 6],  # target 15 not found
            [25, 2, 3, 4, 6],  # target 25 found
            [1, 2, 3, 4, 6],  # target 35 not found
        ])
        target_ids = torch.tensor([5, 15, 25, 35])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # HR@5 = 2/4 = 0.5
        assert torch.isclose(results["hr@5"], torch.tensor(0.5), atol=1e-5)


# =============================================================================
# Test: MRR Computation
# =============================================================================


class TestMRRComputation:
    """Test Mean Reciprocal Rank computation."""

    def test_mrr_all_at_rank_1(self):
        """Test MRR when all targets are at rank 1."""
        metrics = RetrievalMetrics(k=10, at_k_list=[10])

        top_k_ids = torch.tensor([
            [5, 2, 3, 4, 6, 7, 8, 9, 10, 11],
            [15, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        ])
        target_ids = torch.tensor([5, 15])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # MRR = mean(1/1, 1/1) = 1.0
        assert torch.isclose(results["mrr"], torch.tensor(1.0), atol=1e-5)

    def test_mrr_varying_ranks(self):
        """Test MRR with targets at different ranks."""
        metrics = RetrievalMetrics(k=10, at_k_list=[10])

        # Targets at ranks 1, 2, 5
        top_k_ids = torch.tensor([
            [5, 2, 3, 4, 6, 7, 8, 9, 10, 11],  # rank 1
            [1, 15, 3, 4, 6, 7, 8, 9, 10, 11],  # rank 2
            [1, 2, 3, 4, 25, 7, 8, 9, 10, 11],  # rank 5
        ])
        target_ids = torch.tensor([5, 15, 25])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # MRR = mean(1/1, 1/2, 1/5) = mean(1.0, 0.5, 0.2) = 1.7/3 ≈ 0.5667
        expected_mrr = (1.0 + 0.5 + 0.2) / 3
        assert torch.isclose(results["mrr"], torch.tensor(expected_mrr), atol=1e-4)

    def test_mrr_not_found(self):
        """Test MRR when target is not in top-k."""
        metrics = RetrievalMetrics(k=5, at_k_list=[5])

        top_k_ids = torch.tensor([
            [1, 2, 3, 4, 6],
        ])
        target_ids = torch.tensor([100])  # Not found

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # MRR should be 0 when target not found
        assert torch.isclose(results["mrr"], torch.tensor(0.0), atol=1e-5)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestMetricsEdgeCases:
    """Test edge cases for metrics computation."""

    def test_single_sample(self):
        """Test metrics with a single sample."""
        metrics = RetrievalMetrics(k=5, at_k_list=[1, 5])

        top_k_ids = torch.tensor([[5, 2, 3, 4, 6]])
        target_ids = torch.tensor([5])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        assert torch.isclose(results["hr@1"], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(results["mrr"], torch.tensor(1.0), atol=1e-5)

    def test_large_batch(self):
        """Test metrics with a large batch."""
        metrics = RetrievalMetrics(k=100, at_k_list=[10, 50, 100])

        B = 1000
        top_k_ids = torch.randint(1, 10000, (B, 100))
        target_ids = torch.randint(1, 10000, (B,))

        # Ensure some hits by placing target at random positions
        for i in range(B // 2):
            pos = torch.randint(0, 100, (1,)).item()
            top_k_ids[i, pos] = target_ids[i]

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # Just verify computation doesn't crash and returns valid values
        assert 0 <= results["hr@100"] <= 1
        assert 0 <= results["mrr"] <= 1

    def test_duplicate_ids_in_top_k(self):
        """Test metrics when top-k contains duplicate IDs."""
        metrics = RetrievalMetrics(k=5, at_k_list=[5])

        # Top-k has duplicates (shouldn't happen in practice but test robustness)
        top_k_ids = torch.tensor([[5, 5, 5, 4, 6]])
        target_ids = torch.tensor([5])

        metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        results = metrics.compute()

        # Should still find target at rank 1
        assert torch.isclose(results["hr@5"], torch.tensor(1.0), atol=1e-5)


# =============================================================================
# Test: CSVLogger Integration
# =============================================================================


class TestCSVLoggerIntegration:
    """Test CSVLogger integration with metrics logging."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_csv_logger_creation(self, temp_log_dir):
        """Test CSVLogger can be created with the same config as csv.yaml."""
        logger = CSVLogger(
            save_dir=temp_log_dir,
            name="csv/",
            prefix="",
        )

        assert logger is not None
        assert logger.save_dir == temp_log_dir

    def test_csv_logger_logs_scalars(self, temp_log_dir):
        """Test that CSVLogger correctly logs scalar metrics."""
        logger = CSVLogger(
            save_dir=temp_log_dir,
            name="test_run",
            prefix="",
        )

        # Log some metrics
        logger.log_metrics({"train/loss": 0.5, "step": 0})
        logger.log_metrics({"train/loss": 0.4, "step": 1})
        logger.log_metrics({"val/hr@10": 0.3, "val/ndcg@10": 0.2, "step": 1})

        logger.save()

        # Check that CSV file was created
        log_dir = Path(temp_log_dir) / "test_run" / "version_0"
        metrics_file = log_dir / "metrics.csv"

        assert metrics_file.exists(), f"Metrics file not found at {metrics_file}"

        # Read and verify CSV content
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) >= 2, f"Expected at least 2 rows, got {len(rows)}"

    def test_csv_logger_with_lightning_module(self, temp_log_dir, debug_cfg):
        """Test CSVLogger integration with a simple Lightning module."""

        class SimpleModule(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 1)
                self.metrics = RetrievalMetrics(k=10, at_k_list=[1, 5, 10])

            def training_step(self, batch, batch_idx):
                # Use the layer to create a loss that requires grad
                output = self.layer(batch.squeeze(0))
                loss = output.mean()
                # on_epoch=True ensures the metric is logged to CSV at epoch end
                self.log(
                    "train/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                return loss

            def validation_step(self, batch, batch_idx):
                # Simulate retrieval metrics
                B = 4
                top_k_ids = torch.randint(1, 100, (B, 10))
                target_ids = torch.randint(1, 100, (B,))
                # Ensure some hits
                top_k_ids[:, 0] = target_ids
                self.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

            def on_validation_epoch_end(self):
                results = self.metrics.compute()
                for k, v in results.items():
                    self.log(f"val/{k}", v, prog_bar=True, logger=True)
                self.metrics.reset()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        # Create logger
        logger = CSVLogger(
            save_dir=temp_log_dir,
            name="lightning_test",
            prefix="",
        )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=2,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            limit_train_batches=3,
            limit_val_batches=2,
        )

        # Create dummy data
        train_data = [torch.randn(4, 10) for _ in range(3)]
        val_data = [torch.randn(4, 10) for _ in range(2)]

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1)

        # Train
        module = SimpleModule()
        trainer.fit(module, train_loader, val_loader)

        # Verify CSV output
        log_dir = Path(temp_log_dir) / "lightning_test" / "version_0"
        metrics_file = log_dir / "metrics.csv"

        assert metrics_file.exists(), f"Metrics file not found at {metrics_file}"

        with open(metrics_file) as f:
            content = f.read()

        # Check that expected metrics are logged
        assert "train/loss" in content or "train_loss" in content
        # Validation metrics should also be present
        assert "val" in content

    def test_csv_logger_hyperparameters(self, temp_log_dir):
        """Test that CSVLogger can log hyperparameters."""
        logger = CSVLogger(
            save_dir=temp_log_dir,
            name="hparams_test",
            prefix="",
        )

        # Log hyperparameters
        hparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 100,
        }
        logger.log_hyperparams(hparams)
        logger.save()

        # Check that hparams file was created
        log_dir = Path(temp_log_dir) / "hparams_test" / "version_0"
        hparams_file = log_dir / "hparams.yaml"

        assert hparams_file.exists(), f"Hparams file not found at {hparams_file}"


# =============================================================================
# Test: Full Model Metrics Logging
# =============================================================================


class TestFullModelMetricsLogging:
    """Test metrics logging with the full recommendation model."""

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

    def test_model_metrics_update(self, model, debug_cfg, fake_batch):
        """Test that model correctly updates metrics during validation."""
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        max_output_length = debug_cfg.model.gr_output_length + 1
        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        # Simulate validation step
        model.metrics.reset()
        top_k_ids, _ = model.retrieve(seq_features, k=model.k, filter_past_ids=True)
        model.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

        # Compute metrics
        results = model.metrics.compute()

        assert "mrr" in results
        assert len([k for k in results.keys() if k.startswith("ndcg@")]) > 0
        assert len([k for k in results.keys() if k.startswith("hr@")]) > 0

    def test_model_metrics_values_in_range(self, model, debug_cfg, fake_batch):
        """Test that computed metrics are in valid range [0, 1]."""
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        max_output_length = debug_cfg.model.gr_output_length + 1
        seq_features, target_ids = get_sequential_features(fake_batch, device, max_output_length)

        model.metrics.reset()
        top_k_ids, _ = model.retrieve(seq_features, k=model.k, filter_past_ids=True)
        model.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)

        results = model.metrics.compute()

        for key, value in results.items():
            assert 0 <= value <= 1, f"Metric {key} = {value} is out of range [0, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
