from ds_utils.preprocessing.feature_config import FEATURES_CONFIG

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Collection, Literal, TypeAlias
from numpy.typing import ArrayLike
from collections.abc import Sequence
from pandas.tseries.frequencies import to_offset
from enum import Enum
from datetime import datetime

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region Typing Class ----------------------------------------------------------

SummaryColumn: TypeAlias = Literal[
    "nombre_columna",
    "tipo_dato",
    "num_valores_distintos",
    "num_valores_no_validos",
    "porcentaje_no_validos",
]

# VALID_SUMMARY_COLUMNS = [*SummaryColumn]

VALID_SUMMARY_COLUMNS = (
    "nombre_columna",
    "tipo_dato",
    "num_valores_distintos",
    "num_valores_no_validos",
    "porcentaje_no_validos",
)

# SummaryColumn: TypeAlias = Literal[*VALID_SUMMARY_COLUMNS]

# endregion Typing Class -------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------





# region Handlers Functions ----------------------------------------------------


def handle_duplicates(
    df: pd.DataFrame, 
    col_time: str,
    keep: Literal["first", "last"] = "first"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle duplicate timestamps.

    This function identifies duplicate timestamps, keeps the selected
    occurrence and returns both the cleaned DataFrame and the removed
    duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column used to identify duplicates.

    keep : {"first", "last"}, default="first"
        Duplicate occurrence to keep.

        - ``"first"``: keep the first occurrence.
        - ``"last"``: keep the last occurrence.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame with duplicate timestamps removed.

    df_duplicates : pd.DataFrame
        DataFrame containing the removed duplicate rows.
    """

    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")
    
    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])
    df = df.sort_values(col_time)
    
    # Eliminar duplicados según keep
    # df_clean = df.drop_duplicates(subset=[col_name], keep=keep)

    # Boolean mask: True para filas que se mantienen
    mask = ~df.duplicated(subset=[col_time], keep=keep)
    
    df_clean = df[mask].reset_index(drop=True)
    df_duplicates = df[~mask].reset_index(drop=True)
    
    return df_clean, df_duplicates


def handle_time_split(
    df: pd.DataFrame, 
    col_time: str, 
    cutoff_date: str | pd.Timestamp, 
    mode: Literal["head", "tail"] = "tail",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle a time-based DataFrame split.

    Split a DataFrame into two parts using a cutoff date. Depending on
    ``mode``, the function removes either the rows after the cutoff
    (``"tail"``) or the rows before it (``"head"``).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column.

    cutoff_date : str | pd.Timestamp
        Cutoff date used to split the DataFrame.

        - If a full timestamp is provided, the split is performed at that
        exact instant.
        - If only a date is provided (time equals ``00:00:00``), the entire
        day is included in the retained DataFrame.

    mode : {"tail", "head"}, default="tail"
        Portion of the DataFrame to remove.

        - ``"tail"``: keep rows up to the cutoff and remove later rows.
        - ``"head"``: keep rows after the cutoff and remove earlier rows.

    Returns
    -------
    df_main : pd.DataFrame
        Main DataFrame after the split.

    df_removed : pd.DataFrame
        Rows removed from the original DataFrame.
    """

    # assert mode in ["tail", "head"], "mode must be 'tail' or 'head'"
    valid_modes = ["tail", "head"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Allowed values: {valid_modes}")

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])

    df.sort_values(col_time, inplace=True)
    df.set_index(col_time, inplace=True)

    ts = pd.Timestamp(cutoff_date)
    has_time = not (ts.hour == 0 and ts.minute == 0 and ts.second == 0)

    if has_time:
        if mode == "tail":
            df_main = df[df.index <= ts]
            df_removed = df[df.index > ts]
        else:  # head
            df_main = df[df.index > ts]
            df_removed = df[df.index <= ts]
    else:
        start_next_day = ts + pd.Timedelta(days=1)
        end_of_day = start_next_day - pd.Timedelta(seconds=1)

        if mode == "tail":
            df_main = df[df.index <= end_of_day]
            df_removed = df[df.index >= start_next_day]
        else:  # head
            df_main = df[df.index >= start_next_day]
            df_removed = df[df.index <= end_of_day]

    return df_main.reset_index(), df_removed.reset_index()


def handle_time_bursts(
    df: pd.DataFrame, 
    col_time: str, 
    freq: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle timestamp bursts by removing records that are too close in time.

    This function detects records whose timestamps are closer than the
    minimum allowed interval defined by ``freq``. The first record of each
    burst is preserved and subsequent records inside the same burst are
    removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column used to detect temporal proximity.

    freq : str
        Minimum allowed time interval between consecutive kept records.
        Records with a smaller time difference are considered part of a
        temporal burst (for example "30s", "5min", "1h", "1D", "1W").

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame without burst records. The first occurrence of each burst
        is preserved.

    df_removed : pd.DataFrame
        DataFrame containing records removed because they were inside a
        temporal burst.

    Notes
    -----
    The function preserves the first occurrence of each burst and compares
    subsequent timestamps against the last retained timestamp.
    """

    # Validar frecuencia
    try:
        to_offset(freq)
    except ValueError:
        raise ValueError(f"Invalid freq '{freq}'")

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])
    df = df.sort_values(col_time).reset_index(drop=True)

    delta_thresh = pd.Timedelta(freq)

    keep_mask = [True]  # Siempre mantenemos el primer registro
    last_kept = df[col_time].iloc[0]

    for ts in df[col_time].iloc[1:]:
        diff = ts - last_kept
        if diff >= delta_thresh:
            keep_mask.append(True)
            last_kept = ts
        else:
            keep_mask.append(False)

    keep_mask = pd.Series(keep_mask)

    df_no_burst = df[keep_mask].reset_index(drop=True) # sacamos las filas por la mascara y lo ordenamos cronológicamente
    df_deleted = df[~keep_mask].reset_index(drop=True) # sacamos las filas por la negada mascara y lo ordenamos cronológicamente

    return df_no_burst, df_deleted



def handle_sensor_noise(
    df: pd.DataFrame, 
    tolerances: dict, 
    window: int = 3, 
    add_extra_features: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle sensor noise detection and correction.

    This function detects isolated sensor spikes based on configurable
    tolerances and validates suspicious changes using a forward window.
    Detected noise values are replaced by the previous clean value.
    The function also provides optional features useful for machine learning
    models.

    Args
    ----
    df : pd.DataFrame
        Input DataFrame containing sensor measurements.

    tolerances : dict
        Dictionary mapping sensor column names to their maximum allowed
        variation before being considered suspicious.

        Example:
            {"temperature": 3.0, "humidity": 10}

    window : int, default=3
        Number of future observations used to validate whether a suspicious
        change is a real signal variation or an isolated noise spike.

    add_extra_features : bool, default=False
        Whether to generate additional derived features for each sensor,
        such as differences, noise magnitude and rolling statistics.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame containing the original columns plus generated sensor
        cleaning columns and optional ML features.

    df_noise : pd.DataFrame
        DataFrame containing only rows where at least one sensor was
        classified as noisy.

    Notes
    -----
    The names of generated columns depend on the suffix configuration
    defined in ``FEATURES_CONFIG``.

    After applying this function, ``remove_redundant_sensor_noise_columns``
    can be optionally used to remove generated noise features that do not
    provide additional information because no correction was applied.
    """
        
    df = df.copy()

    # máscara global de ruido
    noise_global_mask = np.zeros(len(df), dtype=bool)
    
    for sensor in tolerances:
        if sensor not in df.columns:
            # continue
            raise ValueError(
                f"Sensor column '{sensor}' not found in DataFrame"
            )
        
        vals = df[sensor].values.copy() # numpy.array
        sensor_clean = vals.copy()
        is_noise = np.zeros(len(vals), dtype=bool)
        
        for i in range(1, len(vals)):
            diff = abs(vals[i] - sensor_clean[i-1])
            
            if diff > tolerances[sensor]:
                # Ventana de validación hacia alante, min para evitar el segmentation fault
                fin = min(i + window, len(vals))
                window_vals = vals[i:fin] # slice de numpy array
                
                # Si la ventana se mantiene lejos del valor anterior, es cambio real
                if np.all(abs(window_vals - sensor_clean[i-1]) > tolerances[sensor]):
                    # Cambio real, no hacemos nada
                    continue
                else:
                    # Pico aislado, corregimos
                    sensor_clean[i] = sensor_clean[i-1]
                    is_noise[i] = True
        
        df[f"{sensor}{FEATURES_CONFIG.CLEAN_SUFFIX}"] = sensor_clean
        df[f"{sensor}{FEATURES_CONFIG.IS_NOISE_SUFFIX}"] = is_noise

        # acumular ruido global, si alguna vez cualquier sensor marca True en esa fila, la fila queda marcada como True
        noise_global_mask |= is_noise
        
        # Features extra para ML
        if add_extra_features:
            # diff original
            diff = np.abs(vals - np.roll(vals, 1))
            diff[0] = 0

            # diff limpio
            diff_clean = np.abs(sensor_clean - np.roll(sensor_clean, 1))
            diff_clean[0] = 0

            df[f"{sensor}{FEATURES_CONFIG.DIFF_SUFFIX}"] = diff
            df[f"{sensor}{FEATURES_CONFIG.DIFF_CLEAN_SUFFIX}"] = diff_clean

            # magnitud del ruido (muy útil)
            df[f"{sensor}{FEATURES_CONFIG.NOISE_MAGNITUDE_SUFFIX}"] = np.abs(vals - sensor_clean)

            # detecta estabilidad del sensor (muy útil) **IMPORTANTE** Rolling sobre señal limpia
            rolling_std = (
                pd.Series(sensor_clean)
                .rolling(
                    window=FEATURES_CONFIG.ROLLING_WINDOW, 
                    min_periods=1
                )
                .std()
                .fillna(0)
            )

            df[f"{sensor}{FEATURES_CONFIG.ROLLING_STD_SUFFIX}"] = rolling_std.values

    # flag de ruido global, para filtrar facil si cualquier sensor tubo ruido
    df[FEATURES_CONFIG.GLOBAL_NOISE_COLUMN] = noise_global_mask

    # dataframe solo ruido
    df_noise = df[noise_global_mask].copy().reset_index(drop=True)
        
    return df, df_noise


def handle_missing_values(
    df: pd.DataFrame,
    fill_value: object | dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit missing values and optionally replace them.

    Missing values include pandas missing values (``NaN``, ``None``, ``NA`` and
    ``NaT``). Depending on ``fill_value``, the function either generates an audit
    report only or also replaces the missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    fill_value : object | dict[str, object] | None, default=None
        Replacement value applied to missing data.

        - ``None``:
        Only generate the audit report.

        - Scalar:
        Replace missing values in every column with the same value.

        - dict[str, object]:
        Replace missing values only in the specified columns using
        the corresponding replacement value.

        Example::

            {
                "src_as": 0,
                "dst_as": 0,
                "application": "Unknown",
            }

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of the input DataFrame after applying the requested replacements.

    df_audit : pd.DataFrame
        Audit table containing one row per column with missing values.

        The returned DataFrame contains the following columns:

        - ``column``:
        Column name.

        - ``n_missing``:
        Number of missing values.

        - ``dtype``:
        Pandas data type of the column.

        - ``fill_value``:
        Replacement value applied to the column. This value is ``None`` when
        no replacement was requested.

    Raises
    ------
    KeyError
        If ``fill_value`` is a dictionary containing columns that do not exist
        in the input DataFrame.

    Notes
    -----
    Only the columns specified in ``fill_value`` are audited when a dictionary is
    provided. Otherwise, every column in the DataFrame is included in the audit.
    """

    df_clean = df.copy()

    # ------------------------------------------------------------------
    # Determine which columns will be audited / modified
    # ------------------------------------------------------------------
    if fill_value is None:
        columns_to_check = list(df_clean.columns)

    elif isinstance(fill_value, dict):
        validate_required_columns(
            df_clean,
            list(fill_value.keys())
        )
        columns_to_check = list(fill_value.keys())

    else:
        columns_to_check = list(df_clean.columns)

    audit_rows = []

    # ------------------------------------------------------------------
    # Audit + optional cleaning
    # ------------------------------------------------------------------
    for column in columns_to_check:

        n_missing = int(df_clean[column].isna().sum())

        if n_missing == 0:
            continue

        replacement = (
            fill_value.get(column)
            if isinstance(fill_value, dict)
            else fill_value
        )

        audit_rows.append({
            "column": column,
            "n_missing": n_missing,
            "dtype": str(df_clean[column].dtype),
            "fill_value": replacement,
        })

        if fill_value is not None:
            df_clean[column] = df_clean[column].fillna(replacement)
            

    # # Siempre crear el DataFrame con el mismo esquema
    df_audit = (
        pd.DataFrame(
            audit_rows,
            columns=[
                "column",
                "n_missing",
                "dtype",
                "fill_value",
            ],
        )
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )

    return df_clean, df_audit



# endregion Handlers Functions -------------------------------------------------

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
    except ValueError:
        raise ValueError(f"Invalid freq '{freq}'")

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

# region Counters Functions ----------------------------------------------------

def count_missing_timestamps(
    df: pd.DataFrame, 
    col_time: str, 
    freq: str, 
    realign: bool = False
) -> tuple[int, pd.DatetimeIndex]:
    """
    Count missing timestamps within the observed time range.

    This function compares the existing timestamps against the complete
    expected sequence defined by ``freq`` and returns both the number of
    missing timestamps and their values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column.

    freq : str
        Expected sampling frequency (for example ``"30s"``, ``"5min"``,
        ``"1h"``, ``"1D"``, ``"1W"``).

    realign : bool, default=False
        Whether to round timestamps to the nearest frequency boundary
        before searching for missing timestamps. If ``False``, the
        original timestamp alignment is preserved.

    Returns
    -------
    n_missing_timestamps : int
        Number of missing timestamps within the observed time range.

    missing_timestamps : pd.DatetimeIndex
        DatetimeIndex containing all missing timestamps.
    """

    # Validar frecuencia
    try:
        to_offset(freq)
    except ValueError:
        raise ValueError(f"Invalid freq '{freq}'")

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])

    # Redondear timestamps al múltiplo de freq más cercano (opcional)
    if realign:
        df[col_time] = df[col_time].dt.round(freq)

    # Eliminar duplicados tras redondear
    df = df.drop_duplicates(subset=[col_time])

    # Ordenar
    df = df.sort_values(col_time)

    ts = pd.DatetimeIndex(df[col_time])

    # Rango completo desde min a max
    full_range = pd.date_range(
        start=ts.min(), # Primer timestamp REAL
        end=ts.max(), # Último timestamp REAL
        freq=freq
    )

    # Diferencia: timestamps que faltan
    missing_timestamps = full_range.difference(ts)

    return len(missing_timestamps), missing_timestamps


def count_edge_null_rows(
    df: pd.DataFrame, 
    mode: Literal["any", "all"] = "any"
) -> tuple[int, int]:
    """
    Count consecutive rows with missing values at DataFrame boundaries.

    This function counts the number of consecutive invalid rows at the
    beginning and the end of the DataFrame. A row is considered invalid when
    it contains missing values according to the selected validation mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    mode : {"any", "all"}, default="any"
        Rule used to determine whether a row is considered invalid.

        - "any": row is invalid if at least one value is missing.
        - "all": row is invalid only if all values are missing.

    Returns
    -------
    n_start : int
        Number of consecutive invalid rows at the beginning of the DataFrame.

    n_end : int
        Number of consecutive invalid rows at the end of the DataFrame.
    """

    valid_mode = ["any", "all"]
    if mode not in valid_mode:
        raise ValueError(f"Invalid mode '{mode}'. Allowed values: {valid_mode}")

    # mask = df.isna().all(axis=1) # mascara de True para todas de las columnas de la fila no validas
    # mask = df.isna().any(axis=1) # mascara de True para al menos una de las columnas de la fila no validas
    mask = df.isna().all(axis=1) if mode == "all" else df.isna().any(axis=1)

    n_start = mask.cumprod().sum()
    n_end = mask[::-1].cumprod().sum() # ::-1 recorre el vector en sentido contrario
 
    return int(n_start), int(n_end)


def count_rows_by_time(
    df: pd.DataFrame, 
    col_time: str, 
    freq: str ="ME"
) -> pd.DataFrame:
    """
    Count rows grouped by a time frequency.

    This function groups records by the specified time frequency and returns
    the number of rows contained in each period.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column used for grouping.

    freq : str, default="ME"
        Pandas offset alias defining the grouping frequency.
        Examples: "h", "D", "W", "ME", "YE".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the time periods and the number of rows in each
        period.

        Columns:
            - col_time:
                Time period represented as string.

            - n_rows:
                Number of records within that period.
    """
    # valid_freqs = ["D", "W", "M", "Y"]
    # if freq not in valid_freqs:
    #     raise ValueError(f"Invalid freq '{freq}'. Allowed values: {valid_freqs}")
    # Validar freq
    try:
        offset = to_offset(freq)
    except ValueError:
        raise ValueError(f"Invalid freq '{freq}'")

    df = df.copy()
    
    # Ensure datetime
    df[col_time] = pd.to_datetime(df[col_time])
    
    # Group by frequency
    grouped = (
        df
        .groupby(df[col_time].dt.to_period(offset))
        .size()
        .reset_index(name="n_rows")
    )
    # grouped = (
    #     df
    #     .groupby(df[col_date].dt.floor(freq))
    #     .size()
    #     .reset_index(name="n_rows")
    # )
    
    # Convert period to string for readability
    grouped[col_time] = grouped[col_time].astype(str)
    
    return grouped


def count_constant_columns(
    df: pd.DataFrame,
    dropna: bool = True
) -> int:
    """
    Count the number of columns containing constant values.

    A column is considered constant when it contains only one unique value.
    Missing values can optionally be ignored during the uniqueness calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    dropna : bool, default=True
        Whether missing values should be ignored when counting unique values.

    Returns
    -------
    int
        Number of columns with a single unique value.
    """

    return (df.nunique(dropna=dropna) == 1).sum()


def count_cells_with_missing(df: pd.DataFrame) -> int:
    """
    Count the number of missing cells in a DataFrame.

    Missing cells are detected using pandas missing-value rules and empty
    string detection in object columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    int
        Number of cells containing missing values.

    Notes
    -----
    Missing values include:
    - None
    - NaN
    - Empty strings or whitespace-only strings in object columns.
    """

    missing_values = df.isna().sum().sum()

    missing_empty_strings = (
        df.select_dtypes(include="object")
        .apply(lambda col: col.str.strip() == "")
    ).sum().sum()

    return int(missing_values + missing_empty_strings)


def count_rows_with_missing(df: pd.DataFrame) -> int:
    """
    Count the number of rows containing missing values.

    A row is considered missing when it contains at least one cell detected
    as missing according to pandas rules or an empty string in an object
    column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    int
        Number of rows containing at least one missing value.

    Notes
    -----
    Missing values include:
    - None
    - NaN
    - Empty strings or whitespace-only strings in object columns.
    """
    # Filas con NaN o None
    missing_values = df.isna().any(axis=1)

    # Filas con strings vacíos (solo columnas de tipo object y aplicando strip para que combierta cualquier cosa en blanco en un string vacio)
    missing_empty_strings = (
        df.select_dtypes(include="object")
        .apply(lambda col: col.str.strip() == "")
    ).any(axis=1)

    # Combina ambos
    total_missing_rows = int((missing_values | missing_empty_strings).sum())
    
    return total_missing_rows


def count_columns_with_missing(df: pd.DataFrame) -> int:
    """
    Count the number of columns containing missing values.

    A column is considered to contain missing values when at least one cell
    is detected as missing according to pandas rules or is an empty string
    in an object column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    int
        Number of columns containing at least one missing value.

    Notes
    -----
    Missing values include:
    - None
    - NaN
    - Empty strings or whitespace-only strings in object columns.
    """

    # Columns with pandas missing values
    missing_values = df.isna().any(axis=0)

    # Columns with empty or whitespace-only strings
    missing_empty_strings = (
        df.select_dtypes(include="object")
        .apply(lambda col: col.str.strip() == "")
    ).any(axis=0)

    total_missing_columns = int(
        (missing_values | missing_empty_strings).sum()
    )

    return total_missing_columns


def count_missing_cells_by_column(df: pd.DataFrame) -> pd.Series:
    """
    Count missing cells for each DataFrame column.

    Missing values are detected using pandas missing-value rules and empty
    string detection in object columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Number of missing cells for each column. The index contains the
        column names.

    Notes
    -----
    Missing values include:
    - None
    - NaN
    - Empty strings or whitespace-only strings in object columns.
    """

    # Missing values detected by pandas (NaN, None, NaT...)
    missing_values = df.isna().sum()

    # Empty or whitespace-only strings in object columns
    # missing_empty_strings = (
    #     df.select_dtypes(include="object")
    #     .apply(lambda column: column.str.strip().eq("").sum())
    # )
    obj_cols = df.select_dtypes(include="object")

    # Se fuerza una Series cuando no hay columnas object, ya que pandas puede devolver
    # un DataFrame vacío en ese caso y romper la suma posterior con missing_values.
    if obj_cols.empty:
        missing_empty_strings = pd.Series(0, index=df.columns)
    else:
        missing_empty_strings = obj_cols.apply(
            lambda column: column.str.strip().eq("").sum()
        )

    # Align indexes and combine both counts
    return missing_values.add(
        missing_empty_strings,
        fill_value=0,
    ).astype(int)


# endregion Counters Functions -------------------------------------------------

# region Remove Functions ------------------------------------------------------


def remove_redundant_sensor_noise_columns(
    df: pd.DataFrame,
    tolerances: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove redundant sensor noise columns.

    This function removes the generated sensor noise cleaning columns when the
    cleaned sensor signal is identical to the original signal, meaning that no
    noise correction was applied.

    Columns generated during sensor noise handling are kept only for sensors
    where at least one correction was performed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing original sensor columns and generated noise
        handling columns.

    tolerances : dict
        Dictionary containing the sensors previously processed by the sensor
        noise handler.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame without redundant sensor noise columns.

    removed_columns : list[str]
        List of removed column names.

    Notes
    -----
    The names of generated columns depend on the suffix configuration
    defined in ``FEATURES_CONFIG``.

    This function is intended to be used together with
    ``handle_sensor_noise``. It removes the auxiliary columns generated during
    the noise handling process when they do not provide additional information.
    """

    df = df.copy()

    removed_columns = []

    for sensor in tolerances:

        clean_col = f"{sensor}{FEATURES_CONFIG.CLEAN_SUFFIX}"

        if clean_col not in df.columns:
            continue

        # If the clean signal is identical to the original one,
        # no correction has been applied.
        if not df[sensor].equals(df[clean_col]):
            continue

        candidate_columns = [
            clean_col,
            f"{sensor}{FEATURES_CONFIG.IS_NOISE_SUFFIX}",
            f"{sensor}{FEATURES_CONFIG.DIFF_CLEAN_SUFFIX}",
            f"{sensor}{FEATURES_CONFIG.NOISE_MAGNITUDE_SUFFIX}",
            f"{sensor}{FEATURES_CONFIG.ROLLING_STD_SUFFIX}",
        ]

        existing_columns = [
            column
            for column in candidate_columns
            if column in df.columns
        ]

        if existing_columns:
            df = df.drop(columns=existing_columns)
            removed_columns.extend(existing_columns)

    return df, removed_columns


def remove_constant_columns(
    df: pd.DataFrame,
    dropna: bool = True,
    excluded_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns containing constant values.

    A column is considered constant when it contains only one unique value.
    Missing values can optionally be ignored during the uniqueness check.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    dropna : bool, default=True
        Whether missing values should be ignored when counting unique values.

    excluded_columns : list[str] or None, default=None
        Columns excluded from the constant-value evaluation.
        These columns are always preserved.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame without constant columns.

    removed_columns : list[str]
        List of removed column names.
    """

    df = df.copy()

    excluded_columns = set(excluded_columns or [])

    constant_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
        and df[column].nunique(dropna=dropna) == 1
    ]

    df = df.drop(columns=constant_columns)

    return df, constant_columns


def remove_replaceable_columns(
    df: pd.DataFrame,
    replacement_prefix: str | None = None,
    replacement_suffix: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove original columns that have a replacement column.

    A column is considered replaced when another column exists following
    the provided prefix or suffix naming convention.

    For example, with ``replacement_suffix="_clean"``, the presence of
    ``temperature_clean`` causes the removal of ``temperature``.
    With ``replacement_prefix="clean_"``, the presence of ``clean_temperature`` 
    causes the removal of ``temperature``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    replacement_prefix : str | None, default=None
        Prefix used to identify replacement columns.
        Example: ``"clean_"`` matches ``clean_temperature``.

    replacement_suffix : str | None, default=None
        Suffix used to identify replacement columns.
        Example: ``"_clean"`` matches ``temperature_clean``.

        Exactly one of ``replacement_prefix`` or ``replacement_suffix`` must be provided.

    Returns
    -------
    df_cleaned : pd.DataFrame
        DataFrame without the original columns that have a replacement.

    removed_columns : list[str]
        Names of the removed original columns.

    Raises
    ------
    ValueError
        If neither ``replacement_prefix`` nor ``replacement_suffix`` is provided, or if both are
        provided at the same time.
    """

    # Evaluamos si viene alguno de los 2 parametros de la funcion
    if replacement_prefix is None and replacement_suffix is None:
        raise ValueError(
            "At least one of 'replacement_prefix' or 'replacement_suffix' must be provided"
        )
    # Evaluamos que No vengan los 2 parametros de la funcion juntos
    if replacement_prefix is not None and replacement_suffix is not None:
        raise ValueError(
            "'replacement_prefix' and 'replacement_suffix' cannot be used together"
        )

    df = df.copy()

    existing_columns = set(df.columns)
    removed_columns = []

    for column in existing_columns:

        if replacement_prefix is not None:
            if not column.startswith(replacement_prefix):
                continue

            original_column = column[len(replacement_prefix):]

        else:
            if not column.endswith(replacement_suffix):
                continue

            original_column = column[:-len(replacement_suffix)]

        if original_column in existing_columns:
            removed_columns.append(original_column)

    removed_columns = sorted(removed_columns)

    df = df.drop(columns=removed_columns)

    return df, removed_columns


# endregion Remove Functions ---------------------------------------------------

# region Features Functions ----------------------------------------------------

def resample_time(
    df: pd.DataFrame, 
    col_time: str, 
    freq: str, 
    alignment: Literal["preserve", "round"] = "round",
    fill_method: Literal["ffill", "bfill"] | None = None, 
    fill_limit: int = 6
) -> pd.DataFrame:
    """
    Resample a DataFrame to a fixed time frequency.

    This function aligns timestamps to the requested frequency, inserts
    missing timestamps and preserves the original observations without
    applying any aggregation. Missing values introduced during resampling
    can optionally be filled using forward or backward filling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column.

    freq : str
        Target sampling frequency (for example ``"30s"``, ``"5min"``,
        ``"1h"``, ``"1D"``, ``"1W"``).

    alignment : {"preserve", "round"}, default="round"
        Strategy used to align timestamps before resampling.

        - ``"preserve"``
        Preserve the original sampling offset. This assumes the timestamps
        are already regularly spaced and only their origin does not match
        the requested frequency. The first timestamp is used as the
        resampling origin, so the original timestamps are preserved.

        Example::

            Original: 10:01, 10:06, 10:11
            Result:   10:01, 10:06, 10:11

        - ``"round"``
        Round each timestamp to the nearest multiple of ``freq`` before
        resampling. The original timestamp values are replaced by their
        rounded values. This option is intended for datasets with small
        timestamp deviations (jitter).

        Example::

            Original: 10:01, 10:06, 10:17
            Rounded:  10:00, 10:05, 10:15
            Result:   10:00, 10:05, 10:10, 10:15

    fill_method : {"ffill", "bfill"} or None, default=None
        Method used to fill missing values introduced during resampling.

        - ``"ffill"``: propagate the previous valid observation.
        - ``"bfill"``: propagate the next valid observation.
        - ``None``: keep missing values.

    fill_limit : int, default=6
        Maximum number of consecutive missing periods to fill.

    Returns
    -------
    pd.DataFrame
        DataFrame resampled to the requested frequency.

    Notes
    -----
    ``"preserve"`` should only be used when the timestamps are already
    sampled at a constant frequency. If the timestamps are irregular,
    observations may not match the generated time grid and can be discarded
    during resampling.
    """
    
    if alignment not in ("preserve", "round"):
        raise ValueError("alignment must be 'preserve' or 'round'")

    if fill_method not in ("ffill", "bfill", None):
        raise ValueError("fill_method must be 'ffill', 'bfill' or None")

    # Validar frecuencia
    try:
        to_offset(freq)
    except ValueError:
        raise ValueError(f"Invalid freq '{freq}'")

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])

    # Mostrar si hay duplicados para no falsear resample
    if df[col_time].duplicated().any():
        raise ValueError(f"'{col_time}' column contains duplicate timestamps")

    # Alinear según parámetro
    if alignment == "round":
        df[col_time] = df[col_time].dt.round(freq)
        # Resolver colisiones si varios timestamps caen en el mismo bucket se conserva el primero
        df = df.groupby(col_time).first().reset_index()
    elif alignment == "preserve":
        # en este caso no tocamos los timestamps
        df = df.sort_values(col_time).set_index(col_time)


    # Resample usando asfreq para mantener valores originales sin agregación
    if alignment == "round":
        df_resampled = df.set_index(col_time).resample(freq).asfreq()
    elif alignment == "preserve":
        df_resampled = df.resample(freq, origin="start").asfreq()

    # # Seleccionar agregación por tipo, No se conserva el valor original
    # agg_dict = {}
    # for c in df.columns:
    #     if c == col_date:
    #         continue
    #     if pd.api.types.is_bool_dtype(df[c]):
    #         agg_dict[c] = 'max'
    #     elif pd.api.types.is_numeric_dtype(df[c]):
    #         agg_dict[c] = 'mean'
    #     else:
    #         # columnas no numéricas se dejan como primer valor
    #         agg_dict[c] = 'first'

    # # Resample con agregación
    # if alignment == "round":
    #     df_resampled = df.set_index(col_date).resample(freq).agg(agg_dict)
    # elif alignment == "start":
    #     df_resampled = df.resample(freq, origin="start").agg(agg_dict)


    # Rellenar si se solicita
    if fill_method is not None:
        df_resampled = getattr(df_resampled, fill_method)(limit=fill_limit)

    df_resampled = df_resampled.reset_index()
    return df_resampled



def prepare_time_features(
    df: pd.DataFrame, 
    col_time: str, 
    hour_linear: bool = False, 
    hour_cyclic: bool = False, 
    weekday_linear: Literal["number", "text", "both"] | None = None,
    weekday_cyclic: bool = False
) -> pd.DataFrame:
    """
    Prepare datetime information and generate time-based features.

    This function converts the input time column into a pandas datetime column
    and optionally generates temporal features commonly used in machine learning
    models, such as hour, weekday and cyclic encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the column containing timestamp values.

    hour_linear : bool, default=False
        Whether to add the hour value as an integer feature (0-23).

    hour_cyclic : bool, default=False
        Whether to add cyclic hour encoding using sine and cosine transformations.

    weekday_linear : {"number", "text", "both"} or None, default=None
        Type of weekday feature to generate.

        - "number": Add weekday as an integer value (0-6, Monday-Sunday).
        - "text": Add weekday name as a categorical feature.
        - "both": Add both numeric and text weekday features.

    weekday_cyclic : bool, default=False
        Whether to add cyclic weekday encoding using sine and cosine
        transformations.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original columns, the converted datetime column
        and any requested time-based features.

    Notes
    -----
    The timestamp column is always converted to pandas datetime format, even if
    no additional time features are requested.

    Cyclic encoding is useful for representing periodic variables where the
    first and last values are close in meaning, such as hours of the day or
    days of the week.
    """

    valid_week_linear = [None, "number", "text", "both"]
    if weekday_linear not in valid_week_linear:
        raise ValueError(f"Invalid week_linear '{weekday_linear}'. Allowed values: {valid_week_linear}")

    df = df.copy()
    
    # 1. Base conversion (Mandatory)
    # Time ISO 8601
    # df[time_col] = pd.to_datetime(df[time_col])
    # # Time Epoch
    # df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
    # 1. Convertir la columna de tiempo en objeto 'datetime'
    df[col_time] = df[col_time].apply(convert_timestamp)

    # 1a. Comprobar que todos los valores convertidos son datetime
    if not pd.api.types.is_datetime64_any_dtype(df[col_time]):
        raise TypeError(f"La columna '{col_time}' no se pudo convertir a 'datetime'.")
    
    # 2. Linear Hour (0-23)
    if hour_linear or hour_cyclic:
        df['hour'] = df[col_time].dt.hour
        
    # 3. Cyclical Hour (Sin/Cos)
    if hour_cyclic:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        # We can drop 'hour' if only cyclic was requested, but usually linear is kept
        if not hour_linear:
            df = df.drop(columns=['hour'])


    # 4. Linear Weekday (0, 1, ...), (Monday, Tuesday, ...)
    if weekday_linear:
        weekday_num = df[col_time].dt.weekday  # 0=Monday .. 6=Sunday
        weekday_text = pd.Categorical(
            df[col_time].dt.day_name(),
            categories=[
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ],
            ordered=True
        )

        match weekday_linear:
            case "number":
                df["weekday"] = weekday_num

            case "text":
                df["weekday_text"] = weekday_text

            case "both":
                df["weekday"] = weekday_num
                df["weekday_text"] = weekday_text

            case _:
                raise ValueError(f"Invalid week_linear '{weekday_linear}'. Allowed values: {valid_week_linear}")

    # 5. Cyclical Weekday (Sin/Cos)
    if weekday_cyclic:
        # dt.weekday is 0 (Monday) to 6 (Sunday)
        weekday = df[col_time].dt.weekday
        df['week_sin'] = np.sin(2 * np.pi * weekday / 7)
        df['week_cos'] = np.cos(2 * np.pi * weekday / 7)

    # 6. Reordenacion para que 'time_col' sea la primera columna
    df = df[[col_time] + [col for col in df.columns if col != col_time]]
        
    return df


# endregion Features Functions -------------------------------------------------

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




# endregion Others Functions ---------------------------------------------------



# region ML Functions ----------------------------------------------------------


def prepare_dataframe_for_ml(
    df: pd.DataFrame,
    fillna_cols: Collection[str] | None = None,
    excluded_cols: Collection[str] | None = None,
    bool_as_int: bool = False,
    # categorical_strategy: str = "category",  # "category", "label", "onehot", "drop"
    categorical_strategy: Literal["category", "label", "onehot", "drop"] = "category"
) -> pd.DataFrame:
    """
    Prepare a DataFrame for Machine Learning models.

    The function converts unsupported data types into ML-friendly
    representations while preserving numeric columns whenever possible.

    The following transformations are applied:

    - Boolean columns are kept as ``bool`` or converted to ``0``/``1``.
    - Object and string columns are inspected to detect boolean-like and
    numeric values.
    - Remaining categorical columns are processed according to
    ``categorical_strategy``.
    - Datetime columns are converted to integer timestamps.
    - Missing values in selected numeric columns can optionally be replaced.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    fillna_cols : Collection[str] | None, default=None
        Names of numeric columns whose missing values should be replaced
        with ``0``.

    excluded_cols : Collection[str] | None, default=None
        Columns excluded from every transformation. These columns are
        returned unchanged.

    bool_as_int : bool, default=False
        Whether boolean values should be converted to integers.

        - ``False``:
        Keep boolean values as ``bool``.

        - ``True``:
        Convert boolean values to ``0`` and ``1``.

    categorical_strategy : {"category", "label", "onehot", "drop"}, default="category"
        Strategy used to process categorical columns.

        - ``"category"``:
        Convert columns to the pandas ``category`` dtype.

        - ``"label"``:
        Encode categories as integer labels.

        - ``"onehot"``:
        Replace each categorical column with one-hot encoded columns.

        - ``"drop"``:
        Remove categorical columns from the DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame prepared for Machine Learning.

    Notes
    -----
    Object and string columns are first inspected for boolean-like and
    numeric values before being treated as categorical variables.
    """


    valid_categorical_strategy = ["category", "label", "onehot", "drop"]
    if categorical_strategy not in valid_categorical_strategy:
        raise ValueError(f"Invalid categorical_strategy '{categorical_strategy}'. Allowed values: {valid_categorical_strategy}")

    df = df.copy()

    bool_type = int if bool_as_int else bool
    excluded_cols = set(excluded_cols or [])

    for col in list(df.columns):

        if col in excluded_cols:
            continue

        col_data = df[col]

        # --- Boolean ---
        if pd.api.types.is_bool_dtype(col_data):
            df[col] = col_data.astype(bool_type)

        # --- Object / String ---
        elif (
            pd.api.types.is_object_dtype(col_data)
            or pd.api.types.is_string_dtype(col_data)
        ):
            
            # ------------------------------------------------------------------
            # 1) Detect boolean-like values
            # ------------------------------------------------------------------
            non_null = col_data.dropna() # Crea una copia de la columna SIN valores nulos (NaN)

            # --- Detect boolean-like (VERSIÓN DEFINITIVA) ---
            if len(non_null) > 0:
                # 1. Limpiamos: pasamos a string, quitamos espacios y ponemos en minúscula
                # Esto convierte "" en un string vacío limpio y "  True  " en "true" y fillna para convertir los nulos al string "nan"
                # cleaned_data = col_data.fillna("nan").astype(str).str.strip().str.lower()
                cleaned_data = (
                    col_data.fillna("nan")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                
                # 2. Definimos qué valores aceptamos como "booleano" 
                # Incluimos el string vacío '' y 'nan' porque suelen venir de celdas vacías
                valid_bools_str = {"true", "false", "1", "0", "1.0", "0.0", "nan", "none", "null", ""}
                
                unique_vals = set(cleaned_data.unique())

                if unique_vals.issubset(valid_bools_str):
                    # 3. Mapeo estricto
                    # Lo que no esté aquí (como "" o "nan") se convertirá en NaN por el .map()
                    bool_map = {
                        "true": True, "1": True, "1.0": True,
                        "false": False, "0": False, "0.0": False
                    }
                    
                    mapped_col = cleaned_data.map(bool_map)
                    
                    # 4. Decisión final: los vacíos/nulos los tratamos como False (común en ML)
                    if bool_as_int:
                        df[col] = mapped_col.fillna(False).astype(int)
                    else:
                        df[col] = mapped_col.fillna(False).astype(bool)
                    continue
                

            # ------------------------------------------------------------------
            # 2) Try strict numeric conversion
            #
            # pd.to_numeric(errors="raise") converts the entire column only if
            # every non-null value is a valid numeric representation.
            #
            # Accepted examples:
            #   - Integers:        "1", "-5", "+10"
            #   - Floats:          "3.14", "-0.5"
            #   - Scientific:      "1e3", "2.5E-4"
            #   - Missing values:  None, np.nan, "NaN", "nan"
            #   - Infinity:        np.inf, "inf", "-inf"
            #
            # If any value cannot be interpreted as numeric (e.g. "hello",
            # "abc", "12a", "1,5"), a ValueError is raised and the column is
            # treated as categorical.
            # ------------------------------------------------------------------
            try:
                df[col] = pd.to_numeric(col_data, errors="raise")
                continue

            except Exception:
                pass

            # ------------------------------------------------------------------
            # 3) Handle categorical columns
            # ------------------------------------------------------------------
            if categorical_strategy == "category":
                df[col] = col_data.astype("category")

            elif categorical_strategy == "label":
                df[col] = col_data.astype("category").cat.codes

            elif categorical_strategy == "onehot":
                dummies = pd.get_dummies(col_data, prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

            elif categorical_strategy == "drop":
                df = df.drop(columns=[col])

            else:
                raise ValueError(f"Invalid categorical_strategy: {categorical_strategy}")

        # --- Datetime ---
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # df[col] = col_data.view("int64") # Esto peta en versiones modernas
            df[col] = col_data.astype("int64")

        # --- Numeric / Category already fine ---
        else:
            continue

    # --- Fill NaNs ---
    if fillna_cols:
        for col in fillna_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0) # Rellena todo los NaN numericos con 0

    return df


# endregion ML Functions ---------------------------------------------------


# region Aux Functions ---------------------------------------------------------

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

# endregion Aux Functions ------------------------------------------------------


# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------