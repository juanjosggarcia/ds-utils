from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from typing import ClassVar

@dataclass(frozen=True)
class DatasetContract(ABC):
    """
    Base class that defines the expected schema and validation pipeline
    for input datasets.

    Concrete subclasses must declare ``REQUIRED_COLUMNS`` and can override
    the validation and normalization methods to enforce dataset-specific
    requirements.

    The contract definition is validated upon instantiation, while ``apply()``
    validates and normalizes each DataFrame against the contract.

    Examples
    --------
    Define a contract and apply it to a dataset::

        @dataclass(frozen=True)
        class SalesContract(DatasetContract):
            REQUIRED_COLUMNS = ["date", "sales"]

        contract = SalesContract()
        validated_df = contract.apply(df)
    """

    # region Contract Definition -----------------------------------------------

    # --- Dataset Identification ---
    DF_NAME: ClassVar[str]
    """Logical name of the dataset, ideally matching its DataFrame variable name in the notebook."""

    # --- Required columns list ---
    REQUIRED_COLUMNS: ClassVar[list[str]]
    """Columns required by the contract. Override in derived contracts."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "DF_NAME" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define DF_NAME"
            )

        if "REQUIRED_COLUMNS" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define REQUIRED_COLUMNS"
            )

    # endregion Contract Definition --------------------------------------------

    # region Contract Initialization -------------------------------------------

    def __post_init__(self):
        self.validate_contract()
    
    def validate_contract(self) -> None:
        """
        Validate the contract definition.

        Override to verify the consistency of the contract configuration,
        such as column groups, mappings, or validation thresholds.
        """
        pass

    # endregion Contract Initialization ----------------------------------------

    # region Contract Functions ------------------------------------------------

    
    def add_attribute(self, name: str, value: str) -> None:
        """
        Add an attribute dynamically to the contract class.

        The attribute becomes available to all instances of the contract.
        This method does not update other contract definitions, such as
        DATA_COLUMNS, REQUIRED_COLUMNS, or USED_COLUMNS.
        """
        setattr(type(self), name, value)


    # endregion Contract Functions ---------------------------------------------

    # region Dataset Application -----------------------------------------------

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the contract to a DataFrame.

        Validates the DataFrame against the contract and normalizes it
        before returning the resulting dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate and normalize.

        Returns
        -------
        pd.DataFrame
            Dataset validated and normalized according to the contract.
        """

        self.validate_dataset(df)
        df = self.normalize_dataset(df)
        return df

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate the input dataset against the contract.

        By default, checks that all required columns are present.
        Override to implement dataset-specific validation rules.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        """

        self._validate_required_columns(df)

    def normalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the input dataset according to the contract.

        By default, returns the dataset unchanged. Override to apply
        dataset-specific transformations such as renaming columns,
        creating canonical fields, or converting data types.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to normalize.

        Returns
        -------
        pd.DataFrame
            Normalized dataset.
        """
        
        return df


    def _validate_required_columns(self, df: pd.DataFrame):
        """
        Validate that all required columns are present in the input dataset.

        Compares the DataFrame columns against ``REQUIRED_COLUMNS`` and raises
        an exception if any required columns are missing. Column names must
        match exactly, including capitalization and whitespace.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate against the contract.

        Raises
        ------
        KeyError
            If one or more required columns are missing. The exception
            includes the names of the missing columns.
        """

        missing_cols = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns in the DataFrame: {missing_cols}. "
                f"Check for typos, spaces, or capitalization.")

    # endregion Dataset Application --------------------------------------------