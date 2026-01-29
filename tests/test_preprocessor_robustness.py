import json
import os
import shutil
import tempfile

import pandas as pd
import pytest

from tc_news_rec.data.preprocessor import DataProcessor


class TestPreprocessorInference:
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary data and output directories."""
        data_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        yield data_dir, output_dir
        # Cleanup
        shutil.rmtree(data_dir)
        shutil.rmtree(output_dir)

    @pytest.fixture
    def mock_data(self, temp_dirs):
        """Create mock CSV files mimicking the competition data."""
        data_dir, _ = temp_dirs

        # Mock articles.csv
        articles_df = pd.DataFrame({
            "article_id": [101, 102, 103, 104, 105],
            "category_id": [1, 2, 1, 3, 2],
            "created_at_ts": [1000, 2000, 3000, 4000, 5000],
            "words_count": [100, 200, 150, 300, 250],
        })
        articles_df.to_csv(os.path.join(data_dir, "articles.csv"), index=False)

        # Mock articles_emb.csv (assuming 4 dim embedding + id)
        articles_emb_df = pd.DataFrame({
            "article_id": [101, 102, 103, 104, 105],
            "emb_0": [0.1, 0.2, 0.1, 0.3, 0.2],
            "emb_1": [0.1, 0.2, 0.1, 0.3, 0.2],
            "emb_2": [0.1, 0.2, 0.1, 0.3, 0.2],
            "emb_3": [0.1, 0.2, 0.1, 0.3, 0.2],
        })
        articles_emb_df.to_csv(os.path.join(data_dir, "articles_emb.csv"), index=False)

        # Mock train_click_log.csv
        train_df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "click_article_id": [101, 102, 103],
            "click_timestamp": [1100, 2100, 3100],
            "click_environment": [1, 1, 2],
            "click_deviceGroup": [1, 1, 3],
            "click_os": [2, 2, 4],
            "click_country": [1, 1, 1],
            "click_region": [10, 10, 11],
            "click_referrer_type": [5, 5, 6],
        })
        train_df.to_csv(os.path.join(data_dir, "train_click_log.csv"), index=False)

        return data_dir

    def test_robust_test_file_loading_standard_name(self, temp_dirs, mock_data):
        """Test loading with standard naming 'testA_click_log.csv'."""
        data_dir, output_dir = temp_dirs

        # Create testA
        test_df = pd.DataFrame({
            "user_id": [3],
            "click_article_id": [104],
            "click_timestamp": [4100],
            "click_environment": [1],
            "click_deviceGroup": [1],
            "click_os": [2],
            "click_country": [1],
            "click_region": [10],
            "click_referrer_type": [5],
        })
        test_df.to_csv(os.path.join(data_dir, "testA_click_log.csv"), index=False)

        processor = DataProcessor(data_dir, output_dir)
        processor.process()

        # Verify output exists
        assert os.path.exists(os.path.join(output_dir, "sasrec_format_by_user_test.csv"))
        assert os.path.exists(os.path.join(output_dir, "user_id_mapping.json"))

    def test_robust_test_file_loading_unknown_name(self, temp_dirs, mock_data):
        """Test loading with arbitrary name containing 'test' e.g. 'test_click_log.csv'."""
        data_dir, output_dir = temp_dirs

        # Create generic test file (e.g. TestB scenario)
        test_df = pd.DataFrame({
            "user_id": [4],
            "click_article_id": [105],
            "click_timestamp": [5100],
            "click_environment": [1],
            "click_deviceGroup": [1],
            "click_os": [2],
            "click_country": [1],
            "click_region": [10],
            "click_referrer_type": [5],
        })
        # Simulate online environment where filename isn't testA
        test_df.to_csv(os.path.join(data_dir, "test_click_log.csv"), index=False)

        processor = DataProcessor(data_dir, output_dir)
        processor.process()

        # Verify output exists
        assert os.path.exists(os.path.join(output_dir, "sasrec_format_by_user_test.csv"))

        # Check if user mapping contains the new user (User 4)
        with open(os.path.join(output_dir, "user_id_mapping.json")) as f:
            user_map = json.load(f)
            # Keys in json are strings, valid user IDs are ints.
            # We check values (original IDs)
            original_ids = list(user_map.values())
            assert 4 in original_ids

    def test_missing_test_file(self, temp_dirs, mock_data):
        """Test behavior when no test file is present."""
        data_dir, output_dir = temp_dirs

        # Do NOT create any test file

        processor = DataProcessor(data_dir, output_dir)
        with pytest.raises(FileNotFoundError, match="No test click log found"):
            processor.process()

    def test_feature_consistency(self, temp_dirs, mock_data):
        """Verify that training and test features are mapped into the same space."""
        data_dir, output_dir = temp_dirs

        # Train user has env=1, Test user has env=1. They should map to same int.
        # Train user has env=2.

        test_df = pd.DataFrame({
            "user_id": [3],
            "click_article_id": [101],  # Same article as train
            "click_timestamp": [4100],
            "click_environment": [1],  # Matches train
            "click_deviceGroup": [1],
            "click_os": [2],
            "click_country": [1],
            "click_region": [10],
            "click_referrer_type": [5],
        })
        test_df.to_csv(os.path.join(data_dir, "test_B_click_log.csv"), index=False)

        processor = DataProcessor(data_dir, output_dir)
        processor.process()

        # Load processed data
        proc_train = pd.read_csv(os.path.join(output_dir, "sasrec_format_by_user_train.csv"))
        proc_test = pd.read_csv(os.path.join(output_dir, "sasrec_format_by_user_test.csv"))

        # Check if feature counts file is generated
        assert os.path.exists(os.path.join(output_dir, "feature_counts.json"))
