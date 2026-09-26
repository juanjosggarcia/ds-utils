from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from typing import ClassVar

@dataclass(frozen=True)
class DatasetContract(ABC):
    """
    Base class that defines the expected schema and validation pipeline
    for input datasets.
    """

    REQUIRED_COLUMNS: ClassVar[list[str]] = [...]
    """Columns required by the contract. Override in derived contracts."""

    def __post_init__(self):
        self.validate_contract()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the contract to the input dataset.

        The dataset is validated and normalized before being returned.
        """

        self.validate_dataset(df)
        df = self.normalize_dataset(df)
        return df
    
    def validate_contract(self) -> None:
        """
        Optional validates the contract definition itself.

        Override to verify the consistency of the contract configuration
        (e.g., column groups, mappings, thresholds).
        """
        pass

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validates that the input dataset complies with the contract.

        By default, checks that all required columns are present.
        Override to add dataset-specific validations.
        """

        self._validate_required_columns(df)

    def normalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the input dataset to comply with the contract.

        Override to apply dataset-specific transformations such as
        renaming columns, creating canonical fields, or converting types.
        """
        
        return df


    def _validate_required_columns(self, df: pd.DataFrame):
        """
        Validates that all required columns defined by the contract
        are present in the input dataset.
        """

        missing_cols = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns in the DataFrame: {missing_cols}. "
                f"Check for typos, spaces, or capitalization.")