import torch
import pandas as pd
import os
from typing import Optional, List, Tuple, Dict, Any
import lightning as L

from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)

def load_data(file: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(file, pd.DataFrame):
        return file
    elif isinstance(file, str) and file.endswith(".csv"):
        return pd.read_csv(file)
    else:
        raise ValueError("ratings_file must be a csv file.")

class TCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str | pd.DataFrame,
        padding_length: int,
        ignore_last_n: int,
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
        additional_columns: Optional[List[str]] = []
    ):
        """
        Args:
            file (str | pd.DataFrame): path or DataFrame containing the dataset
            padding_length (int): _description_
            ignore_last_n (int): number of last interactions to ignore (used for creating train/valid/test sets)
            shift_id_by (int, optional): _description_. Defaults to 0.
            chronological (bool, optional): _description_. Defaults to False.
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
        # TODO: load sample logic
        pass

class TCDataModule(L.LightningDataModule):
    # TODO: implement the data module
    pass