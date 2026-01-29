import json
import os

import numpy as np
import pandas as pd
import torch

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)


class DataProcessor:
    def __init__(self, data_dir: str, output_dir: str) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process(self):
        # Check if we are in inference mode (mappings exist)
        item_map_path = os.path.join(self.output_dir, "item_id_mapping.json")
        is_inference = os.path.exists(item_map_path)

        if is_inference:
            log.info("Found existing mappings. Running in INFERENCE mode.")
        else:
            log.info("Mappings not found. Running in TRAINING mode.")

        # Load data
        log.info("Loading data...")

        train_df = None
        if not is_inference:
            # In training mode, we need train data
            train_path = os.path.join(self.data_dir, "train_click_log.csv")
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
            else:
                raise FileNotFoundError(f"Training data not found at {train_path}")

        # Robust test file loading: search for file containing "test"
        # In inference mode, this is our main input.
        # In training mode, it's optional (e.g. for validation)
        test_filename = None
        if os.path.exists(self.data_dir):
            for f in os.listdir(self.data_dir):
                if "test" in f and f.endswith(".csv"):
                    test_filename = f
                    break

        test_df = None
        if test_filename:
            log.info(f"Found test file: {test_filename}")
            test_df = pd.read_csv(os.path.join(self.data_dir, test_filename))
        else:
            if is_inference:
                raise FileNotFoundError("No test click log found in data directory for inference!")
            else:
                log.warning("No test file found. Skipping test processing.")

        articles_df = pd.read_csv(os.path.join(self.data_dir, "articles.csv"))
        articles_emb_df = pd.read_csv(os.path.join(self.data_dir, "articles_emb.csv"))

        # 1. Mappings
        log.info("Creating/Loading mappings...")

        if not is_inference:
            # --- TRAINING MODE: Build and Save Maps ---

            # User ID - Build and save mapping
            all_users = train_df["user_id"].unique()
            user_map = {int(u): k + 1 for k, u in enumerate(sorted(all_users))}
            reverse_user_map = {v: k for k, v in user_map.items()}
            with open(os.path.join(self.output_dir, "user_id_mapping.json"), "w") as f:
                json.dump(reverse_user_map, f, indent=4)
            log.info(f"User ID mapping saved. Total users: {len(user_map)}")

            # Article ID (Item ID)
            all_articles = articles_df["article_id"].unique()
            item_map = {int(i): k + 1 for k, i in enumerate(sorted(all_articles))}
            reverse_item_map = {v: k for k, v in item_map.items()}
            with open(os.path.join(self.output_dir, "item_id_mapping.json"), "w") as f:
                json.dump(reverse_item_map, f, indent=4)
            log.info("Item ID mapping saved.")

            # Category ID
            all_cats = articles_df["category_id"].unique()
            category_map = {c: k + 1 for k, c in enumerate(sorted(all_cats))}
            # Save category map if needed, but usually reconstruction is deterministic if articles.csv is static

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
                uniques = train_df[col].unique()
                cat_maps[col] = {v: k + 1 for k, v in enumerate(sorted(uniques))}
                # Reserve UNK ID for OOV values
                max_id = max(cat_maps[col].values())
                cat_maps[col]["<UNK>"] = max_id + 1
                # Save these maps for inference!
                with open(os.path.join(self.output_dir, f"{col}_mapping.json"), "w") as f:
                    # Convert keys to int/str for json
                    json_map = {str(k): v for k, v in cat_maps[col].items()}  # actually maps value->id
                    # We saved {val: id}.
                    # For JSON: {str(val): id}
                    json.dump(json_map, f, indent=4)
                log.info(
                    f"Category feature '{col}' mapping saved. Total values: {len(cat_maps[col]) - 1}, UNK ID: {cat_maps[col]['<UNK>']}"
                )

            # Popularity (Item Heat)
            # Calculate popularity from train clicks
            pop_counts = train_df["click_article_id"].value_counts()
            all_articles_ids = articles_df["article_id"].values
            # Align with all articles, fill missing with 0
            pop_series = pop_counts.reindex(all_articles_ids, fill_value=0)

            # Bucket into 20 bins (duplicates='drop' handles the long tail of 1s)
            # labels=False returns 0..N-1. We add 1.
            pop_buckets, pop_bin_edges = pd.qcut(pop_series, q=20, labels=False, duplicates="drop", retbins=True)
            pop_buckets = pop_buckets + 1
            popularity_map = pop_buckets.to_dict()

            # Save bucket edges for inference
            np.save(
                os.path.join(self.output_dir, "popularity_bucket_edges.npy"),
                pop_bin_edges,
            )
            with open(os.path.join(self.output_dir, "popularity_mapping.json"), "w") as f:
                # json keys must be strings
                json.dump({str(k): int(v) for k, v in popularity_map.items()}, f)
            log.info(
                f"Popularity mapping and bucket edges saved. Max bucket: {pop_buckets.max()}, Edges: {len(pop_bin_edges)}"
            )

        else:
            # --- INFERENCE MODE: Load Maps ---

            # User ID
            user_map_path = os.path.join(self.output_dir, "user_id_mapping.json")
            if os.path.exists(user_map_path):
                with open(user_map_path) as f:
                    user_map_reverse = json.load(f)
                user_map = {int(v): int(k) for k, v in user_map_reverse.items()}
                log.info(f"User ID mapping loaded. Total users: {len(user_map)}")
            else:
                log.warning("User ID mapping not found in inference mode!")
                user_map = {}

            # Item ID
            with open(os.path.join(self.output_dir, "item_id_mapping.json")) as f:
                item_map_reverse = json.load(f)
            # item_map_reverse is {id: original}. We need {original: id}
            item_map = {int(v): int(k) for k, v in item_map_reverse.items()}

            # Category ID (Rebuild from articles)
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
            cat_unk_ids = {}
            for col in cat_cols:
                map_path = os.path.join(self.output_dir, f"{col}_mapping.json")
                if os.path.exists(map_path):
                    with open(map_path) as f:
                        json_map = json.load(f)
                    # Restore: JSON keys are strings.
                    # If original values were ints, we need to convert back (except <UNK>).
                    cat_maps[col] = {}
                    for k, v in json_map.items():
                        if k == "<UNK>":
                            cat_unk_ids[col] = v
                        else:
                            cat_maps[col][int(k)] = v
                    log.info(f"Loaded mapping for '{col}': {len(cat_maps[col])} values, UNK ID: {cat_unk_ids[col]}")
                else:
                    log.warning(f"Mapping for {col} not found. Re-building from Test Data (Might be inconsistent!)")
                    # Fallback
                    uniques = test_df[col].unique()
                    cat_maps[col] = {v: k + 1 for k, v in enumerate(sorted(uniques))}
                    cat_unk_ids[col] = len(cat_maps[col]) + 1

            # Popularity (Load)
            pop_map_path = os.path.join(self.output_dir, "popularity_mapping.json")
            if os.path.exists(pop_map_path):
                with open(pop_map_path) as f:
                    pop_map_loaded = json.load(f)
                popularity_map = {int(k): v for k, v in pop_map_loaded.items()}
            else:
                log.warning("Popularity mapping not found in inference mode! Defaulting to bucket 1.")
                popularity_map = {}

        # Apply mappings
        log.info("Applying mappings...")

        def safe_map(series, mapping, default=0):
            # mapping keys might be int. series is int.
            # Use .map but fillna
            return series.map(mapping).fillna(default).astype(int)

        def safe_map_with_unk(series, mapping, unk_id):
            # Use UNK ID for OOV values instead of 0 (padding)
            return series.map(mapping).fillna(unk_id).astype(int)

        if train_df is not None:
            # No need to map user_id for model
            train_df["item_id_mapped"] = safe_map(train_df["click_article_id"], item_map)
            for col in cat_cols:
                if is_inference:
                    train_df[col] = safe_map_with_unk(train_df[col], cat_maps[col], cat_unk_ids[col])
                else:
                    train_df[col] = safe_map(train_df[col], cat_maps[col])

            # Apply popularity
            if popularity_map:
                train_df["popularity_bucket"] = safe_map(train_df["click_article_id"], popularity_map, default=1)
            else:
                train_df["popularity_bucket"] = 1

        if test_df is not None:
            # No need to map user_id for model
            test_df["item_id_mapped"] = safe_map(test_df["click_article_id"], item_map)
            for col in cat_cols:
                if is_inference:
                    test_df[col] = safe_map_with_unk(test_df[col], cat_maps[col], cat_unk_ids[col])
                else:
                    test_df[col] = safe_map(test_df[col], cat_maps[col])

            # Apply popularity
            if popularity_map:
                test_df["popularity_bucket"] = safe_map(test_df["click_article_id"], popularity_map, default=1)
            else:
                test_df["popularity_bucket"] = 1

        articles_df["category_id_mapped"] = safe_map(articles_df["category_id"], category_map)

        # 2. Discretization (Bucketing)
        log.info("Discretizing features...")

        # Words count (Static from articles)
        if not is_inference:
            # Training mode: Create and save bucket edges
            articles_df["words_count_bucket"], words_count_edges = pd.qcut(
                articles_df["words_count"], q=100, labels=False, duplicates="drop", retbins=True
            )
            articles_df["words_count_bucket"] = articles_df["words_count_bucket"] + 1
            np.save(
                os.path.join(self.output_dir, "words_count_bucket_edges.npy"),
                words_count_edges,
            )
            log.info(f"Words count bucket edges saved: {len(words_count_edges)} edges")
        else:
            # Inference mode: Load and apply bucket edges
            words_count_edges = np.load(os.path.join(self.output_dir, "words_count_bucket_edges.npy"))
            articles_df["words_count_bucket"] = np.digitize(articles_df["words_count"], words_count_edges[1:-1]) + 1
            log.info(f"Applied words count bucket edges: {len(words_count_edges)} edges")

        # Timestamp bucketing: Use 1000 buckets instead of 10000 for better stability
        num_time_buckets = 1000

        if not is_inference:
            # Training mode: Create and save timestamp bucket edges
            ts_series_list = [articles_df["created_at_ts"]]
            if train_df is not None:
                ts_series_list.append(train_df["click_timestamp"])
            if test_df is not None:
                ts_series_list.append(test_df["click_timestamp"])

            all_ts = pd.concat(ts_series_list)

            # Compute buckets on combined data
            ts_labels, ts_edges = pd.qcut(all_ts, q=num_time_buckets, labels=False, duplicates="drop", retbins=True)
            ts_labels = ts_labels + 1

            # Save timestamp bucket edges
            np.save(
                os.path.join(self.output_dir, "timestamp_bucket_edges.npy"),
                ts_edges,
            )
            log.info(f"Timestamp bucket edges saved: {len(ts_edges)} edges")

            # Split back
            offset = 0
            articles_df["created_at_bucket"] = ts_labels[offset : offset + len(articles_df)].values
            offset += len(articles_df)

            if train_df is not None:
                train_df["timestamp_bucket"] = ts_labels[offset : offset + len(train_df)].values
                offset += len(train_df)

            if test_df is not None:
                test_df["timestamp_bucket"] = ts_labels[offset:].values
        else:
            # Inference mode: Load and apply timestamp bucket edges
            ts_edges = np.load(os.path.join(self.output_dir, "timestamp_bucket_edges.npy"))
            log.info(f"Loaded timestamp bucket edges: {len(ts_edges)} edges")

            # Apply to articles
            articles_df["created_at_bucket"] = np.digitize(articles_df["created_at_ts"], ts_edges[1:-1]) + 1

            # Apply to test data
            if test_df is not None:
                test_df["timestamp_bucket"] = np.digitize(test_df["click_timestamp"], ts_edges[1:-1]) + 1

        # Merge article features into logs
        log.info("Merging article features...")
        article_feats = articles_df[
            [
                "article_id",
                "category_id_mapped",
                "created_at_bucket",
                "words_count_bucket",
                "created_at_ts",  # Keep original ts for age calculation
            ]
        ]

        if train_df is not None:
            train_df = train_df.merge(
                article_feats,
                left_on="click_article_id",
                right_on="article_id",
                how="left",
            )
            # Sort
            train_df = train_df.sort_values("click_timestamp", ascending=True)

        if test_df is not None:
            test_df = test_df.merge(
                article_feats,
                left_on="click_article_id",
                right_on="article_id",
                how="left",
            )
            test_df = test_df.sort_values("click_timestamp", ascending=True)

        def calculate_time_features(df):
            # Age: click_timestamp - created_at_ts
            df["age"] = df["click_timestamp"] - df["created_at_ts"]
            df["age"] = df["age"].apply(lambda x: max(0, x))
            dt_series = pd.to_datetime(df["click_timestamp"], unit="ms")
            df["hour_of_day"] = dt_series.dt.hour + 1
            df["day_of_week"] = dt_series.dt.dayofweek + 1
            return df

        if train_df is not None:
            train_df = calculate_time_features(train_df)
        if test_df is not None:
            test_df = calculate_time_features(test_df)

        # Discretize Age
        if not is_inference:
            # Training mode: Create and save age bucket edges
            age_list = []
            if train_df is not None:
                age_list.append(train_df["age"])
            if test_df is not None:
                age_list.append(test_df["age"])

            if age_list:
                all_age = pd.concat(age_list)
                age_labels, age_edges = pd.qcut(all_age, q=100, labels=False, duplicates="drop", retbins=True)
                age_labels = age_labels + 1

                # Save age bucket edges
                np.save(
                    os.path.join(self.output_dir, "age_bucket_edges.npy"),
                    age_edges,
                )
                log.info(f"Age bucket edges saved: {len(age_edges)} edges")

                offset = 0
                if train_df is not None:
                    train_df["age_bucket"] = age_labels[offset : offset + len(train_df)]
                    offset += len(train_df)
                if test_df is not None:
                    test_df["age_bucket"] = age_labels[offset:]
        else:
            # Inference mode: Load and apply age bucket edges
            age_edges = np.load(os.path.join(self.output_dir, "age_bucket_edges.npy"))
            log.info(f"Loaded age bucket edges: {len(age_edges)} edges")

            if test_df is not None:
                test_df["age_bucket"] = np.digitize(test_df["age"], age_edges[1:-1]) + 1

        # 3. Aggregation
        log.info("Aggregating sequences...")

        def aggregate_user_data(df, output_filename):
            # Sort by user and timestamp
            df = df.sort_values(["user_id", "click_timestamp"], ascending=True)

            # Group by user
            grouped = df.groupby("user_id")

            # Aggregate sequences
            # Using string join for efficiency
            agg_funcs = {
                "item_id_mapped": lambda x: ",".join(map(str, x)),
                "timestamp_bucket": lambda x: ",".join(map(str, x)),
                "category_id_mapped": lambda x: ",".join(map(str, x)),
                "created_at_bucket": lambda x: ",".join(map(str, x)),
                "popularity_bucket": lambda x: ",".join(map(str, x)),
                "words_count_bucket": lambda x: ",".join(map(str, x)),
                "age_bucket": lambda x: ",".join(map(str, x)),
                "hour_of_day": lambda x: ",".join(map(str, x)),
                "day_of_week": lambda x: ",".join(map(str, x)),
                "click_environment": lambda x: (x.mode()[0] if not x.mode().empty else x.iloc[0]),
                "click_deviceGroup": lambda x: (x.mode()[0] if not x.mode().empty else x.iloc[0]),
                "click_os": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
                "click_country": lambda x: (x.mode()[0] if not x.mode().empty else x.iloc[0]),
                "click_region": lambda x: (x.mode()[0] if not x.mode().empty else x.iloc[0]),
                "click_referrer_type": lambda x: (x.mode()[0] if not x.mode().empty else x.iloc[0]),
            }

            result = grouped.agg(agg_funcs)

            # Rename columns
            result.rename(
                columns={
                    "item_id_mapped": "sequence_item_ids",
                    "timestamp_bucket": "sequence_timestamps",
                    "category_id_mapped": "sequence_category_ids",
                    "created_at_bucket": "sequence_created_at_ts",
                    "popularity_bucket": "sequence_popularity",
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
            result.to_csv(os.path.join(self.output_dir, output_filename), index_label="index")

        if train_df is not None:
            aggregate_user_data(train_df, "sasrec_format_by_user_train.csv")

        if test_df is not None:
            aggregate_user_data(test_df, "sasrec_format_by_user_test.csv")

        # 4. Embeddings
        if not is_inference:
            # Training mode: Build embeddings
            log.info("Processing embeddings...")
            num_items = len(item_map) + 1  # +1 for padding idx 0
            emb_dim = 250
            embedding_mat = torch.zeros((num_items, emb_dim), dtype=torch.float32)
            for _, row in articles_emb_df.iterrows():
                original_id = int(row["article_id"])
                if original_id in item_map:
                    mapped_id = item_map[original_id]
                    emb_values = row[1:].values.astype(np.float32)
                    embedding_mat[mapped_id] = torch.tensor(emb_values, dtype=torch.float32)

            torch.save(embedding_mat, os.path.join(self.output_dir, "article_embedding.pt"))
            log.info(f"Article embeddings saved: shape {embedding_mat.shape}")
        else:
            # Inference mode: Verify embedding consistency
            emb_path = os.path.join(self.output_dir, "article_embedding.pt")
            if os.path.exists(emb_path):
                embedding_mat = torch.load(emb_path)
                expected_items = len(item_map) + 1
                if embedding_mat.shape[0] != expected_items:
                    log.warning(
                        f"Embedding dimension mismatch! Expected {expected_items} items, "
                        f"but loaded embedding has {embedding_mat.shape[0]} items. "
                        f"This may cause issues during inference."
                    )
                else:
                    log.info(
                        f"Embedding consistency verified: {embedding_mat.shape[0]} items, "
                        f"{embedding_mat.shape[1]} dimensions"
                    )
            else:
                log.warning("Article embedding file not found in inference mode!")

        # 5. Save feature statistics
        # Only in TRAINING mode
        if not is_inference:
            log.info("Saving feature statistics...")
            feature_max_values = {
                "item_id": len(item_map),
                "popularity": int(pop_buckets.max()),
                "category_id": len(category_map),
                "words_count": int(articles_df["words_count_bucket"].max()),
                "created_at": int(articles_df["created_at_bucket"].max()),
                "age": int(age_labels.max()) if age_list else 100,
                "hour_of_day": 24,
                "day_of_week": 7,
            }

            # Map original col name to final feature name
            col_name_map = {
                "click_environment": "environment",
                "click_deviceGroup": "deviceGroup",
                "click_os": "os",
                "click_country": "country",
                "click_region": "region",
                "click_referrer_type": "referrer_type",
            }

            # Include UNK ID in counts (cat_maps now includes <UNK>)
            for col, final_name in col_name_map.items():
                # Count includes normal IDs + UNK ID
                feature_max_values[final_name] = len(cat_maps[col])

            with open(os.path.join(self.output_dir, "feature_counts.json"), "w") as f:
                json.dump(feature_max_values, f, indent=4)
            log.info(f"Feature statistics saved: {feature_max_values}")
        else:
            # In inference, we might need to verify feature counts match, or just assume.
            log.info("Skipping feature statistics saving (Inference Mode).")

        log.info("Done.")

        log.info("Done.")

    def processed_train_csv(self) -> str:
        path = os.path.join(self.output_dir, "sasrec_format_by_user_train.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed train CSV not found at {path}. Please run process() first.")
        return path

    def processed_test_csv(self) -> str:
        path = os.path.join(self.output_dir, "sasrec_format_by_user_test.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed test CSV not found at {path}. Please run process() first.")
        return path
