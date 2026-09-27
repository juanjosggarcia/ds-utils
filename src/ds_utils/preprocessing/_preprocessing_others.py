import numpy as np
import pandas as pd
from typing import Literal
from collections.abc import Sequence
from datetime import datetime
from pandas._libs.tslibs.nattype import NaTType

from ds_utils.preprocessing._preprocessing_counters import count_missing_cells_by_column

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

    required_cols : list[str] or str
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

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Column to set as the index.

    drop : bool, default=True
        Whether to remove the column after setting it as the index.
        Passed directly to ``DataFrame.set_index``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the requested index.

    Raises
    ------
    KeyError
        If ``column`` does not exist in the DataFrame and is not already
        an index level.
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
    to avoid creating an unnecessary ``"index"`` column. Otherwise, the current
    index is restored as one or more columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    drop : bool, default=False
        Whether to discard the current index instead of restoring it as
        column(s). Passed directly to ``DataFrame.reset_index`` when the
        DataFrame does not have a default ``RangeIndex``.

    Returns
    -------
    pd.DataFrame
        DataFrame with its index safely reset.
    """

    if isinstance(df.index, pd.RangeIndex):
        return df.reset_index(drop=True)

    return df.reset_index(drop=drop)


def detect_timestamp(ts: str | int) -> str:
    """
    Detect the type of a timestamp and convert it to a readable datetime.

    Parameters
    ----------
    ts : str or int
        Timestamp to detect. It can be a Unix epoch timestamp in seconds or
        milliseconds, or an ISO 8601 datetime string.

    Returns
    -------
    str
        Description of the detected timestamp format and its corresponding
        datetime representation in UTC.
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


def convert_timestamp(ts: str | int) -> pd.Timestamp | NaTType:
    """
    Detect the timestamp type and convert it to a datetime.

    Supports Unix epoch timestamps in seconds, milliseconds, microseconds,
    and nanoseconds, as well as ISO 8601 datetime strings.

    Parameters
    ----------
    ts : str or int
        Timestamp to convert. It can be a Unix epoch timestamp or an ISO 8601
        datetime string.

    Returns
    -------
    pd.Timestamp or pd.NaT
        Converted timestamp as a pandas ``Timestamp``. Returns ``pd.NaT`` if
        the timestamp format is invalid or cannot be converted.
    """
    try:
        n = int(ts)

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


def convert_timestamps_vectorized(
    values: pd.Series, utc: bool = True
) -> pd.Series:
    """
    Convert a Series containing numeric or string timestamps to datetime.

    Numeric timestamps are interpreted as Unix epoch values. The time unit
    is inferred independently for each value according to its magnitude,
    allowing seconds, milliseconds, microseconds, and nanoseconds to coexist
    in the same Series.

    Non-numeric values are parsed as datetime strings, including ISO 8601
    representations. Null or invalid values are converted to ``NaT``.

    Parameters
    ----------
    values : pd.Series
        Series containing timestamp values. Values may be numeric Unix epoch
        timestamps in seconds, milliseconds, microseconds, or nanoseconds,
        or datetime strings such as ISO 8601 values.

    utc : bool, default=True
        Whether to return timezone-aware datetimes in UTC. If ``True``, the
        returned Series has a ``datetime64[ns, UTC]`` dtype. If ``False``,
        the returned Series has a ``datetime64[ns]`` dtype.

    Returns
    -------
    pd.Series
        Series containing the converted timestamps. Invalid or missing
        values are represented as ``NaT``.

    Notes
    -----
    Numeric timestamps are classified independently using their magnitude:

    - Values greater than ``1e17`` are interpreted as nanoseconds.
    - Values greater than ``1e14`` and up to ``1e17`` are interpreted as
      microseconds.
    - Values greater than ``1e11`` and up to ``1e14`` are interpreted as
      milliseconds.
    - Values up to ``1e11`` are interpreted as seconds.

    A value of ``0`` is treated as a valid Unix epoch timestamp and therefore
    corresponds to ``1970-01-01 00:00:00``.

    Conversion is vectorized by timestamp unit and does not process values
    individually with ``Series.apply()``, which makes it suitable for large
    DataFrames.
    """
    if values.empty:
        dtype = "datetime64[ns, UTC]" if utc else "datetime64[ns]"
        return pd.Series(dtype=dtype, index=values.index)

    # 1. Intentar conversión numérica
    numeric = pd.to_numeric(values, errors="coerce")
    is_numeric = numeric.notna()

    # 2. Inicializar la serie de salida con NaT
    dtype = "datetime64[ns, UTC]" if utc else "datetime64[ns]"
    result = pd.Series(pd.NaT, index=values.index, dtype=dtype)

    # 3. Procesar valores numéricos (Epoch)
    if is_numeric.any():
        num_vals = numeric[is_numeric]

        # Umbrales estándar entre 1973 y 5138 d.C.
        masks = {
            "ns": num_vals > 1e17,
            "us": (num_vals > 1e14) & (num_vals <= 1e17),
            "ms": (num_vals > 1e11) & (num_vals <= 1e14),
            "s": num_vals <= 1e11,
        }

        for unit, mask in masks.items():
            if mask.any():
                target_idx = mask[mask].index
                parsed_epoch = pd.to_datetime(
                    num_vals.loc[target_idx],
                    unit=unit,
                    utc=utc,
                    errors="coerce",
                )
                result.loc[target_idx] = parsed_epoch

    # 4. Procesar valores no numéricos (ISO 8601 u otros formatos string)
    is_non_numeric = ~is_numeric
    if is_non_numeric.any():
        # Descartar nulos originales para no gastar tiempo parseando None/NaN
        raw_non_numeric = values[is_non_numeric].dropna()

        if not raw_non_numeric.empty:
            parsed_iso = pd.to_datetime(
                raw_non_numeric, utc=utc, errors="coerce"
            )
            result.loc[parsed_iso.index] = parsed_iso

    return result


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

    sort_by : Sequence[str] or str, default="num_valores_no_validos"
        Summary column or columns used to sort the result. Valid values are
        ``"nombre_columna"``, ``"tipo_dato"``, ``"num_valores_distintos"``,
        ``"num_valores_no_validos"``, and ``"porcentaje_no_validos"``.

    ascending : Sequence[bool] or bool, default=True
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