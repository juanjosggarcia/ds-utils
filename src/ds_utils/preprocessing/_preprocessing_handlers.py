from ds_utils.preprocessing._preprocessing_others import validate_required_columns
from ds_utils.preprocessing.preprocessing_config import FEATURES_CONFIG

import numpy as np
import pandas as pd
from typing import Literal
from pandas.tseries.frequencies import to_offset

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

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
    except ValueError as exc:
        raise ValueError(f"Invalid freq '{freq}'") from exc

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

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------