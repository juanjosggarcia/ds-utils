import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------

# region Finders Functions -----------------------------------------------------

def find_duplicates(
    df: pd.DataFrame, 
    col_time: str
) -> tuple[int, pd.DataFrame]:
    """
    Detect duplicate timestamps.

    This function identifies duplicated timestamps and returns both the
    number of unique duplicated timestamps and all rows involved in those
    duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column used to identify duplicates.

    Returns
    -------
    n_duplicate_timestamps : int
        Number of unique timestamps that appear more than once.

    df_duplicates : pd.DataFrame
        DataFrame containing every row whose timestamp is duplicated,
        including the first occurrence.
    """

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])
    
    # Boolean mask: Marca como True todas las filas duplicadas incluyendo la primera
    dup_mask = df.duplicated(subset=[col_time], keep=False)
    
    # DataFrame con todas las filas duplicadas
    df_duplicates = df[dup_mask]
    
    # Número de valores únicos que están duplicados
    n_unique_duplicates = df_duplicates[col_time].nunique()
    
    return n_unique_duplicates, df_duplicates


def find_large_gaps(
    df: pd.DataFrame, 
    col_time: str, 
    freq: str, 
    limit: int = 1
) -> pd.DataFrame:
    """
    Detect gaps between consecutive timestamps that exceed a given limit.

    A gap is reported when the time difference between two consecutive
    timestamps is greater than the maximum fillable interval defined by
    ``freq`` and ``limit``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column.

    freq : str
        Expected sampling frequency (for example ``"30s"``, ``"5min"``,
        ``"1h"``, ``"1D"``, ``"1W"``).

    limit : int, default=1
        Maximum number of consecutive periods considered fillable.
        Gaps larger than this value are returned.

    Returns
    -------
    pd.DataFrame
        DataFrame describing each detected gap with the following columns:

        - ``start``:
        Timestamp immediately before the gap.

        - ``end``:
        Timestamp immediately after the gap.

        - ``missing_periods``:
        Number of missing periods between ``start`` and ``end``.

        - ``duration``:
        Total gap duration as ``timedelta``.
    """

    # Validar freq
    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid freq '{freq}'") from exc

    # Timestamps reales
    ts_real = pd.to_datetime(df[col_time].drop_duplicates()).sort_values()

    # Diferencias entre timestamps consecutivos
    diffs = ts_real.diff().fillna(pd.Timedelta(0))

    # Gap máximo tolerable
    max_gap = offset * limit

    # Detectar gaps grandes
    mask_large_gap = diffs > max_gap

    # Construir DataFrame de gaps
    gaps = pd.DataFrame({
        "start": ts_real.shift(1)[mask_large_gap],
        "end": ts_real[mask_large_gap]
    })

    # Número de periodos faltantes
    gaps["missing_periods"] = ((gaps["end"] - gaps["start"]) / offset - 1).astype(int)
    gaps["duration"] = gaps["end"] - gaps["start"]

    return gaps.reset_index(drop=True)



def find_column_pairs(
    df: pd.DataFrame,
    prefix: str | None = None,
    suffix: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Find original columns and their corresponding replacement columns.

    A replacement column is identified by either a prefix or a suffix
    naming convention.

    For example, with ``suffix="_clean"``, ``temperature`` is paired with
    ``temperature_clean``. With ``prefix="clean_"``, ``temperature`` is
    paired with ``clean_temperature``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    prefix : str | None, default=None
        Prefix used to identify replacement columns.

    suffix : str | None, default=None
        Suffix used to identify replacement columns.

        Exactly one of ``prefix`` or ``suffix`` must be provided.

    Returns
    -------
    original_columns : list[str]
        Sorted list of original column names.

    replacement_columns : list[str]
        Sorted list of replacement column names.

    Raises
    ------
    ValueError
        If neither ``prefix`` nor ``suffix`` is provided, or if both are
        provided at the same time.
    """

    if prefix is None and suffix is None:
        raise ValueError(
            "At least one of 'prefix' or 'suffix' must be provided"
        )

    if prefix is not None and suffix is not None:
        raise ValueError(
            "'prefix' and 'suffix' cannot be used together"
        )

    existing_columns = set(df.columns)

    original_columns = []
    replacement_columns = []

    for column in existing_columns:

        if prefix is not None:

            if not column.startswith(prefix):
                continue

            original_column = column[len(prefix):]

        else:

            if not column.endswith(suffix):
                continue

            original_column = column[:-len(suffix)]

        if original_column in existing_columns:
            original_columns.append(original_column)
            replacement_columns.append(column)

    return (
        sorted(original_columns),
        sorted(replacement_columns),
    )

# endregion Finders Functions --------------------------------------------------

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------