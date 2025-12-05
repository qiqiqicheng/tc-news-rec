import pandas as pd
import numpy as np
import torch
import os

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)


class DataProcessor:
    def __init__(self, data_dir: str, output_dir: str) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process(self):
        # Load data
        log.info("Loading data...")
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_click_log.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "testA_click_log.csv"))
        articles_df = pd.read_csv(os.path.join(self.data_dir, "articles.csv"))
        articles_emb_df = pd.read_csv(os.path.join(self.data_dir, "articles_emb.csv"))

        # 1. Mappings
        log.info("Creating mappings...")
        # User ID
        all_users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()
        user_map = {u: i + 1 for i, u in enumerate(sorted(all_users))}

        # Article ID (Item ID)
        all_articles = articles_df["article_id"].unique()
        item_map = {i: k + 1 for k, i in enumerate(sorted(all_articles))}

        # Category ID
        all_cats = articles_df["category_id"].unique()
        category_map = {c: k + 1 for k, c in enumerate(sorted(all_cats))}

        # Other categorical features
        cat_cols = [
            "click_environment",
            "click_deviceGroup",
            "click_os",
            "click_country",
            "click_region",
            "click_referrer_type",
        ]
        cat_maps = {}
        for col in cat_cols:
            uniques = pd.concat([train_df[col], test_df[col]]).unique()
            cat_maps[col] = {v: k + 1 for k, v in enumerate(sorted(uniques))}

        # Apply mappings
        log.info("Applying mappings...")
        train_df["user_id_mapped"] = train_df["user_id"].map(user_map)
        test_df["user_id_mapped"] = test_df["user_id"].map(user_map)

        train_df["item_id_mapped"] = train_df["click_article_id"].map(item_map)
        test_df["item_id_mapped"] = test_df["click_article_id"].map(item_map)

        articles_df["category_id_mapped"] = articles_df["category_id"].map(category_map)

        for col in cat_cols:
            train_df[col] = train_df[col].map(cat_maps[col])
            test_df[col] = test_df[col].map(cat_maps[col])

        # 2. Discretization (Bucketing)
        log.info("Discretizing features...")
        # Words count
        articles_df["words_count_bucket"] = (
            pd.qcut(articles_df["words_count"], q=100, labels=False, duplicates="drop")
            + 1
        )

        num_time_buckets = 10000

        # Combine all timestamps (created_at and click_time) to ensure consistent mapping
        all_ts = pd.concat(
            [
                articles_df["created_at_ts"],
                train_df["click_timestamp"],
                test_df["click_timestamp"],
            ]
        )

        # Compute buckets on combined data
        ts_labels = (
            pd.qcut(all_ts, q=num_time_buckets, labels=False, duplicates="drop") + 1
        )

        # Split back and assign
        len_articles = len(articles_df)
        len_train = len(train_df)

        # Use .values to avoid index alignment issues
        articles_df["created_at_bucket"] = ts_labels[:len_articles].values
        train_df["timestamp_bucket"] = ts_labels[
            len_articles : len_articles + len_train
        ].values
        test_df["timestamp_bucket"] = ts_labels[len_articles + len_train :].values

        # Merge article features into logs
        log.info("Merging article features...")
        article_feats = articles_df[
            [
                "article_id",
                "category_id_mapped",
                "created_at_bucket",
                "words_count_bucket",
            ]
        ]

        train_df = train_df.merge(
            article_feats, left_on="click_article_id", right_on="article_id", how="left"
        )
        test_df = test_df.merge(
            article_feats, left_on="click_article_id", right_on="article_id", how="left"
        )

        # Sort by click_timestamp (original) in ascending order
        train_df = train_df.sort_values("click_timestamp", ascending=True)
        test_df = test_df.sort_values("click_timestamp", ascending=True)

        # 3. Aggregation
        log.info("Aggregating sequences...")

        def aggregate_user_data(df, output_filename):
            # Sort by user and timestamp
            df = df.sort_values(["user_id_mapped", "click_timestamp"], ascending=True)

            # Group by user
            grouped = df.groupby("user_id_mapped")

            # Aggregate sequences
            # Using string join for efficiency
            agg_funcs = {
                "item_id_mapped": lambda x: ",".join(map(str, x)),
                "timestamp_bucket": lambda x: ",".join(map(str, x)),
                "category_id_mapped": lambda x: ",".join(map(str, x)),
                "created_at_bucket": lambda x: ",".join(map(str, x)),
                "words_count_bucket": lambda x: ",".join(map(str, x)),
                "click_environment": lambda x: (
                    x.mode()[0] if not x.mode().empty else x.iloc[0]
                ),
                "click_deviceGroup": lambda x: (
                    x.mode()[0] if not x.mode().empty else x.iloc[0]
                ),
                "click_os": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
                "click_country": lambda x: (
                    x.mode()[0] if not x.mode().empty else x.iloc[0]
                ),
                "click_region": lambda x: (
                    x.mode()[0] if not x.mode().empty else x.iloc[0]
                ),
                "click_referrer_type": lambda x: (
                    x.mode()[0] if not x.mode().empty else x.iloc[0]
                ),
            }

            result = grouped.agg(agg_funcs)

            # Rename columns
            result.rename(
                columns={
                    "item_id_mapped": "sequence_item_ids",
                    "timestamp_bucket": "sequence_timestamps",
                    "category_id_mapped": "sequence_category_ids",
                    "created_at_bucket": "sequence_created_at_ts",
                    "words_count_bucket": "sequence_words_count",
                    "click_environment": "environment",
                    "click_deviceGroup": "deviceGroup",
                    "click_os": "os",
                    "click_country": "country",
                    "click_region": "region",
                    "click_referrer_type": "referrer_type",
                },
                inplace=True,
            )

            # Reset index to make user_id a column, and create a new numerical index
            result = result.reset_index()
            result.to_csv(
                os.path.join(self.output_dir, output_filename), index_label="index"
            )

        aggregate_user_data(train_df, "sasrec_format_by_user_train.csv")
        aggregate_user_data(test_df, "sasrec_format_by_user_test.csv")

        # 4. Embeddings
        log.info("Processing embeddings...")
        emb_dict = {}
        ids = articles_emb_df["article_id"].values
        emb_cols = [c for c in articles_emb_df.columns if c.startswith("emb_")]
        # Sort columns to ensure order emb_0, emb_1, ...
        emb_cols.sort(key=lambda x: int(x.split("_")[1]))
        embs = articles_emb_df[emb_cols].values

        for i, art_id in enumerate(ids):
            if art_id in item_map:
                mapped_id = item_map[art_id]
                emb_dict[mapped_id] = torch.tensor(embs[i], dtype=torch.float32)

        torch.save(emb_dict, os.path.join(self.output_dir, "article_embedding.pt"))
        log.info("Done.")

    def processed_train_csv(self) -> str:
        path = os.path.join(self.output_dir, "sasrec_format_by_user_train.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed train CSV not found at {path}. Please run process() first."
            )
        return path

    def processed_test_csv(self) -> str:
        path = os.path.join(self.output_dir, "sasrec_format_by_user_test.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed test CSV not found at {path}. Please run process() first."
            )
        return path
