import torch
import pandas as pd
import os
from typing import Optional, List, Tuple, Dict, Any
import lightning as L
from omegaconf import DictConfig
import hydra
from pprint import pformat

from tc_news_rec.utils.logger import RankedLogger
from tc_news_rec.data.preprocessor import DataProcessor

log = RankedLogger(__name__)


def load_data(file: str | pd.DataFrame):
    if isinstance(file, pd.DataFrame):
        return file
    elif isinstance(file, str) and file.endswith(".csv"):
        return pd.read_csv(file)
    elif isinstance(file, str) and file.endswith(".pt"):
        return torch.load(file)
    else:
        raise ValueError("ratings_file must be a csv file or pt file")


def save_data(df: pd.DataFrame, file: str):
    if file.endswith(".csv"):
        df.to_csv(file, index=False)
        log.info(f"Data saved to {file}")
    else:
        raise ValueError("Only csv format is supported for saving data.")


class TCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str | pd.DataFrame,
        embedding_data: str | torch.Tensor | Dict[int, torch.Tensor],
        padding_length: int,
        ignore_last_n: int,
        shift_id_by: int = 0,
        chronological: bool = True,
        sample_ratio: float = 1.0,
        additional_columns: Optional[List[str]] = [],
    ):
        """
        Args:
            file (str | pd.DataFrame): path or DataFrame containing the dataset
            embedding_data (str | torch.Tensor | Dict[int, torch.Tensor]): path to embedding file or loaded embedding data
            padding_length (int): _description_
            ignore_last_n (int): number of last interactions to ignore (used for creating train/valid/test sets)
            shift_id_by (int, optional): _description_. Defaults to 0.
            chronological (bool, optional): True for increase chronology. Defaults to True.
            sample_ratio (float, optional): _description_. Defaults to 1.0.
            additional_columns (Optional[List[str]], optional): _description_. Defaults to [].
        """
        super().__init__()

        self.df = load_data(file)
        self._padding_length = padding_length
        self._ignore_last_n = ignore_last_n
        self._shift_id_by = shift_id_by
        self._chronological = chronological
        self._sample_ratio = sample_ratio
        self._additional_columns = additional_columns
        self._cache = dict()

        if isinstance(embedding_data, str):
            self._embedding_data = load_data(embedding_data)
        else:
            self._embedding_data = embedding_data

        self._additional_columns_check()

    def _additional_columns_check(self):
        if self._additional_columns:
            columns_status = []
            for column in self._additional_columns:
                # check the column exists and status, like type, max, min, etc.
                column_exists = column in self.df.columns
                if not column_exists:
                    raise ValueError(
                        f"Column {column} does not exist in the ratings file."
                    )
                column_type = self.df[column].dtype
                max_value = self.df[column].max()
                min_value = self.df[column].min()
                columns_status.append(
                    {
                        "column": column,
                        "type": column_type,
                        "max": max_value,
                        "min": min_value,
                    }
                )
            log.info(f"Additional columns status: {columns_status}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        if index in self._cache:
            return self._cache[index]
        sample = self.load_sample(index)
        self._cache[index] = sample
        return sample

    def load_sample(self, index: int) -> Dict[str, torch.Tensor]:
        data = self.df.iloc[index]
        user_id = data["user_id_mapped"]

        def eval_as_list(x, ignore_last_n: int) -> List[int]:
            y = eval(x)
            y_list = [y] if isinstance(y, int) else list(y)
            if ignore_last_n > 0:
                y_list = y_list[:-ignore_last_n]
            return y_list

        def prepare_sequence_ids(
            x,
            ignore_last_n: int,
            shifted_by: int,
            sampling_kept_mask: Optional[List[bool]] = None,
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n)
            if sampling_kept_mask:
                y = [item for item, keep in zip(y, sampling_kept_mask) if keep]
            if shifted_by > 0:
                y = [item + shifted_by for item in y]
            seq_len = len(y)
            return y, seq_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        item_id_history, item_id_history_len = prepare_sequence_ids(
            data["sequence_item_ids"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        item_click_time_history, item_click_time_history_len = prepare_sequence_ids(
            data["sequence_timestamps"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        item_category_id_history, item_category_id_history_len = prepare_sequence_ids(
            data["sequence_category_ids"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        item_created_at_history, item_created_at_history_len = prepare_sequence_ids(
            data["sequence_created_at_ts"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        item_words_count_history, item_words_count_history_len = prepare_sequence_ids(
            data["sequence_words_count"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        # print(data.keys())
        item_age_history, item_age_history_len = prepare_sequence_ids(
            data["age_bucket"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        item_hour_history, item_hour_history_len = prepare_sequence_ids(
            data["hour_of_day"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        item_day_history, item_day_history_len = prepare_sequence_ids(
            data["day_of_week"],
            ignore_last_n=self._ignore_last_n,
            shifted_by=0,
            sampling_kept_mask=sampling_kept_mask,
        )
        assert (
            item_id_history_len
            == item_click_time_history_len
            == item_category_id_history_len
            == item_created_at_history_len
            == item_words_count_history_len
            == item_age_history_len
            == item_hour_history_len
            == item_day_history_len
        ), "Sequence lengths are not equal!"  # NOTE: current is full interactive length

        def truncate_or_pad(
            y: List[int], target_len: int, chronological: bool = False
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_item_ids = item_id_history[:-1]
        historical_item_click_times = item_click_time_history[:-1]
        historical_item_category_ids = item_category_id_history[:-1]
        historical_item_created_ats = item_created_at_history[:-1]
        historical_item_words_counts = item_words_count_history[:-1]
        historical_item_ages = item_age_history[:-1]
        historical_item_hours = item_hour_history[:-1]
        historical_item_days = item_day_history[:-1]

        target_item_id = item_id_history[-1]
        target_item_click_time = item_click_time_history[-1]

        if not self._chronological:  # raw data is chronological increasing
            historical_item_ids.reverse()
            historical_item_click_times.reverse()
            historical_item_category_ids.reverse()
            historical_item_created_ats.reverse()
            historical_item_words_counts.reverse()
            historical_item_ages.reverse()
            historical_item_hours.reverse()
            historical_item_days.reverse()

        max_seq_len = self._padding_length - 1
        history_len = min(len(historical_item_ids), max_seq_len)

        historical_item_ids = truncate_or_pad(
            historical_item_ids,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_click_times = truncate_or_pad(
            historical_item_click_times,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_category_ids = truncate_or_pad(
            historical_item_category_ids,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_created_ats = truncate_or_pad(
            historical_item_created_ats,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_words_counts = truncate_or_pad(
            historical_item_words_counts,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_ages = truncate_or_pad(
            historical_item_ages,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_hours = truncate_or_pad(
            historical_item_hours,
            target_len=max_seq_len,
            chronological=self._chronological,
        )
        historical_item_days = truncate_or_pad(
            historical_item_days,
            target_len=max_seq_len,
            chronological=self._chronological,
        )

        # Process user features
        user_features = [
            "environment",
            "deviceGroup",
            "os",
            "country",
            "region",
            "referrer_type",
        ]
        user_feature_ids = {
            feat: int(data[feat]) for feat in user_features if feat in data
        }

        # Retrieve embeddings
        emb_dim = 250
        if isinstance(self._embedding_data, dict) and len(self._embedding_data) > 0:
            emb_dim = next(iter(self._embedding_data.values())).shape[0]

        def get_embedding(item_id):
            if item_id == 0:
                return torch.zeros(emb_dim, dtype=torch.float32)
            if isinstance(self._embedding_data, torch.Tensor):
                return self._embedding_data[item_id]
            else:
                raise ValueError("Embedding data format not supported.")

        historical_item_embeddings = torch.stack(
            [get_embedding(i) for i in historical_item_ids]
        )
        target_item_embedding = get_embedding(target_item_id)

        sample_dict = {
            "user_id": torch.tensor(user_id, dtype=torch.int64),
            "historical_item_ids": torch.tensor(historical_item_ids, dtype=torch.int64),
            "historical_item_embeddings": historical_item_embeddings,
            "historical_item_click_times": torch.tensor(
                historical_item_click_times, dtype=torch.int64
            ),
            "historical_item_category_ids": torch.tensor(
                historical_item_category_ids, dtype=torch.int64
            ),
            "historical_item_created_ats": torch.tensor(
                historical_item_created_ats, dtype=torch.int64
            ),
            "historical_item_words_counts": torch.tensor(
                historical_item_words_counts, dtype=torch.int64
            ),
            "historical_item_ages": torch.tensor(
                historical_item_ages, dtype=torch.int64
            ),
            "historical_item_hours": torch.tensor(
                historical_item_hours, dtype=torch.int64
            ),
            "historical_item_days": torch.tensor(
                historical_item_days, dtype=torch.int64
            ),
            **user_feature_ids,
            "target_item_id": torch.tensor(target_item_id, dtype=torch.int64),
            "target_item_embedding": target_item_embedding,
            "target_item_click_time": torch.tensor(
                target_item_click_time, dtype=torch.int64
            ),
            "history_len": torch.tensor(history_len, dtype=torch.int64),
        }

        for column in self._additional_columns:  # type: ignore
            sample_dict[column] = torch.tensor(data[column], dtype=torch.int64)
        return sample_dict


class TCDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_preprocessor: DataProcessor | DictConfig,
        # dataset_cfg: DictConfig,
        train_file: str,
        test_file: str,
        embedding_file: str,
        max_seq_length: int,
        chronological: bool = False,
        sampling_ratio: float = 1.0,
        batch_size: int = 32,
        num_workers: int = os.cpu_count() // 4,  # type: ignore
        prefetch_factor: int = 4,
        random_split: bool = True,
        split_ratios: List[float] = [0.8, 0.1, 0.1],
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        # self.dataset_cfg = dataset_cfg
        self.train_file = train_file
        self.test_file = test_file
        self.embedding_file = embedding_file
        self.max_seq_length = max_seq_length
        self.chronological = chronological
        self.sampling_ratio = sampling_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.random_split = random_split
        self.split_ratios = split_ratios
        self.data_preprocessor = (
            hydra.utils.instantiate(data_preprocessor)
            if isinstance(data_preprocessor, DictConfig)
            else data_preprocessor
        )

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def instantiate_dataset(
        self,
        file: pd.DataFrame,
        embedding_data: Any,
        ignore_last_n: int = 0,
    ) -> TCDataset:
        kwargs = {}
        kwargs["padding_length"] = self.max_seq_length + 1
        kwargs["chronological"] = self.chronological
        kwargs["sample_ratio"] = self.sampling_ratio
        kwargs["file"] = file
        kwargs["embedding_data"] = embedding_data
        kwargs["ignore_last_n"] = ignore_last_n

        dataset = TCDataset(**kwargs)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        # Load data once
        print(f"Loading data from {self.train_file} and {self.test_file}")
        log.info(f"Loading data from {self.train_file} and {self.test_file}")
        train_df = load_data(self.train_file)
        test_df = load_data(self.test_file)

        print(f"Loading embeddings from {self.embedding_file}")
        log.info(f"Loading embeddings from {self.embedding_file}")
        embedding_data = load_data(self.embedding_file)

        if self.random_split:
            log.info("Using random split strategy for datasets.")
            # Shuffle and split
            full_df = train_df.sample(frac=1).reset_index(drop=True)
            n = len(full_df)
            train_len = int(n * self.split_ratios[0])
            val_len = int(n * self.split_ratios[1])

            train_df = full_df.iloc[:train_len]
            val_df = full_df.iloc[train_len : train_len + val_len]

            log.info(
                f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
            )

            if stage == "fit" or stage is None:
                self.train_dataset = self.instantiate_dataset(
                    file=train_df, embedding_data=embedding_data, ignore_last_n=0
                )
                self.val_dataset = self.instantiate_dataset(
                    file=val_df, embedding_data=embedding_data, ignore_last_n=0
                )
            if stage == "test" or stage is None:
                self.test_dataset = self.instantiate_dataset(
                    file=test_df, embedding_data=embedding_data, ignore_last_n=0
                )
        else:
            # If not random split, we assume the file contains everything and we might need logic to split by time or user
            # But for now, let's just use the same file for all if random_split is False (or raise error as per previous logic)
            # Or maybe the user intends to provide different files? But we only have one 'file' argument now.
            # Assuming random_split is the primary mode as requested.
            raise NotImplementedError(
                "Non-random split strategy is not implemented yet."
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def save_predictions(self, output_file: str, predictions: dict):
        df = self.test_dataset.df
        for key, value in predictions.items():
            df[key] = value
        save_data(df, output_file)
