import numpy as np
import pandas as pd
from typing import Collection, Literal

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------

# region Others Functions ------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame, 
    required_cols: list[str] | str
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    This function checks whether the input DataFrame contains every column
    specified as required. It raises an exception with the missing column names
    to simplify debugging of incorrect schemas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    required_cols : list[str] | str
        Column name or list of column names that must exist in the DataFrame.

    Raises
    ------
    KeyError
        If one or more required columns are missing.

    Notes
    -----
    This function does not modify the DataFrame. It only validates the input
    schema and returns ``None`` when all required columns are present.

    Examples
    --------
    >>> required_cols = ["timestamp", "temperature", "humidity"]
    >>> validate_required_columns(df, required_cols)
    """

    if isinstance(required_cols, str):
        required_cols = [required_cols]

    missing_columns = [
        column
        for column in required_cols
        if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns in the DataFrame: {missing_columns}. "
                       f"Check for typos, spaces, or capitalization.")


def set_index_safe(
    df: pd.DataFrame,
    column: str,
    drop: bool = True,
) -> pd.DataFrame:
    """
    Safely set a DataFrame column as the index.

    If the column is already an index level, the DataFrame is returned
    unchanged. Otherwise, the column is set as the index.

    Args:
        df (pd.DataFrame):
            Input DataFrame.

        column (str):
            Column to set as the index.

        drop (bool, default=True):
            Whether to remove the column after setting it as the index.
            Passed directly to ``DataFrame.set_index``.

    Returns:
        pd.DataFrame:
            DataFrame with the requested index.

    Raises:
        KeyError:
            If the specified column does not exist.
    """

    if column in df.index.names:
        return df

    return df.set_index(column, drop=drop)


def reset_index_safe(
    df: pd.DataFrame,
    drop: bool = False,
) -> pd.DataFrame:
    """
    Safely reset the DataFrame index.

    If the DataFrame already has the default RangeIndex, the index is dropped
    to avoid creating an unnecessary "index" column. Otherwise, the current
    index is restored as one or more columns.

    Args:
        df (pd.DataFrame):
            Input DataFrame.

        drop (bool, default=False):
            Whether to discard the current index instead of restoring it as
            column(s). Passed directly to ``DataFrame.reset_index`` when the
            DataFrame does not have a default ``RangeIndex``.

    Returns:
        pd.DataFrame:
            DataFrame with its index safely reset.
    """

    if isinstance(df.index, pd.RangeIndex):
        return df.reset_index(drop=True)

    return df.reset_index(drop=drop)

def detect_timestamp(ts):
    """
    Detects the type of a timestamp and converts it to a readable datetime.
    
    Parameters:
    ts (str or int): The timestamp to detect. Can be epoch (seconds/milliseconds) or ISO 8601 string.
    
    Returns:
    str: Description of the timestamp type and the corresponding UTC datetime.
    """
    try:
        # Try converting to integer (epoch)
        n = int(ts)
        # If 13 digits or more, treat as milliseconds
        if len(str(n)) >= 13:
            dt = datetime.utcfromtimestamp(n / 1000)
            return f"Epoch in milliseconds: {dt} UTC"
        else:
            dt = datetime.utcfromtimestamp(n)
            return f"Epoch in seconds: {dt} UTC"
    except ValueError:
        # Not a number, try ISO 8601
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return f"ISO 8601 format: {dt} (timezone included if present)"
        except ValueError:
            return "Unknown timestamp format"
        
def convert_timestamp(ts):
    """
    Detect timestamp type and convert to datetime.
    Supports seconds, milliseconds, microseconds, nanoseconds, and ISO 8601.
    Returns pd.NaT if unknown format.
    """
    try:
        n = int(ts)
        # Heuristic: if > 10^12, treat as milliseconds
        # if n > 1e12:
        #     return pd.to_datetime(n, unit='ms', errors='coerce')
        # else:
        #     return pd.to_datetime(n, unit='s', errors='coerce')
        match n:
            case _ if n > 1e17:  # nanoseconds
                return pd.to_datetime(n, unit='ns', errors='coerce')
            case _ if n > 1e14:  # microseconds
                return pd.to_datetime(n, unit='us', errors='coerce')
            case _ if n > 1e11:  # milliseconds
                return pd.to_datetime(n, unit='ms', errors='coerce')
            case _:  # seconds
                return pd.to_datetime(n, unit='s', errors='coerce')
    
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(ts, errors='coerce', utc=True)
        except Exception:
            return pd.NaT



def get_columns_summary(
    df: pd.DataFrame,
    sort_by: (
        Sequence[str]
        | Literal[
            "nombre_columna",
            "tipo_dato",
            "num_valores_distintos",
            "num_valores_no_validos",
            "porcentaje_no_validos",
        ]
    ) = "num_valores_no_validos",
    ascending: Sequence[bool] | bool = True,
) -> pd.DataFrame:
    """
    Generate a summary of the DataFrame columns.

    The summary includes the data type, the number of distinct values,
    the number of invalid values and the percentage of invalid values
    for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    sort_by : Sequence[str] | {"nombre_columna", "tipo_dato", "num_valores_distintos", "num_valores_no_validos", "porcentaje_no_validos"}, default="num_valores_no_validos"
        Summary column or columns used to sort the result.

    ascending : Sequence[bool] | bool, default=True
        Whether to sort each corresponding column in ascending order.

        If a sequence is provided, it must have the same length as
        ``sort_by``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per input column with the following
        fields:

        - ``nombre_columna``
        - ``tipo_dato``
        - ``num_valores_distintos``
        - ``num_valores_no_validos``
        - ``porcentaje_no_validos``

    Raises
    ------
    ValueError
        If ``sort_by`` contains unsupported column names.
    """

    df = df.copy()

    valid_sort_columns = (
        "nombre_columna",
        "tipo_dato",
        "num_valores_distintos",
        "num_valores_no_validos",
        "porcentaje_no_validos",
    )

    sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)

    invalid_columns = [
        column
        for column in sort_columns
        if column not in valid_sort_columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Invalid sort_by value(s): {invalid_columns}. "
            f"Allowed values: {valid_sort_columns}."
        )

    invalid_values = count_missing_cells_by_column(df)

    df_columns_summary = pd.DataFrame(index=df.columns)

    df_columns_summary["tipo_dato"] = df.dtypes.astype(str)
    df_columns_summary["num_valores_distintos"] = df.nunique(dropna=True)
    df_columns_summary["num_valores_no_validos"] = invalid_values

    df_columns_summary["porcentaje_no_validos"] = (
        df_columns_summary["num_valores_no_validos"] / len(df) * 100
    ).round(2)

    df_columns_summary = (
        df_columns_summary
        .reset_index(names="nombre_columna")
        .sort_values(
            by=sort_by,
            ascending=ascending,
        )
        .reset_index(drop=True)
    )

    return df_columns_summary



# endregion Others Functions ---------------------------------------------------

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------