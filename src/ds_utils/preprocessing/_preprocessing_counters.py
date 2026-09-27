import numpy as np
import pandas as pd
from typing import Literal
from pandas.tseries.frequencies import to_offset

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def _get_empty_string_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a boolean mask identifying empty or whitespace-only strings.

    Columns with ``object`` or ``string`` dtype are evaluated. Within those
    columns, only values that are actually strings are checked.

    Empty strings include values such as ``""``, ``" "``, or ``"   "``.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame where ``True`` indicates that the cell contains
        an empty or whitespace-only string.
    """
    string_columns = df.select_dtypes(include=["object", "string"])

    return string_columns.apply(lambda col: col.str.strip() == "")

# endregion Aux Functions ------------------------------------------------------

# region Counters Functions ----------------------------------------------------

def count_missing_timestamps(
    df: pd.DataFrame, 
    col_time: str, 
    freq: str, 
    realign: bool = False
) -> tuple[int, pd.DatetimeIndex]:
    """
    Count missing timestamps within the observed time range.

    The function builds the expected timestamp sequence between the minimum
    and maximum observed timestamps using ``freq`` and returns the timestamps
    that are not present in the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column.

    freq : str
        Expected sampling frequency, expressed as a pandas offset alias.
        Examples include ``"30s"``, ``"5min"``, ``"1h"``, ``"1D"``, and ``"1W"``.

    realign : bool, default=False
        If ``True``, round timestamps to the nearest frequency boundary before
        checking for missing timestamps. Timestamps that become duplicates
        after realignment are collapsed.

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
    except ValueError as exc:
        raise ValueError(f"Invalid freq '{freq}'") from exc

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])

    if df.empty:
        return 0, pd.DatetimeIndex([])

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

    valid_mode = ("any", "all")
    if mode not in valid_mode:
        raise ValueError(f"Invalid mode '{mode}'. Allowed values: {valid_mode}")

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

    # Validar freq
    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid freq '{freq}'") from exc

    df = df.copy()
    
    # Ensure datetime
    df[col_time] = pd.to_datetime(df[col_time])

    try:
        periods = df[col_time].dt.to_period(offset)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid period frequency '{freq}'") from exc
    
    # Group by frequency
    grouped = (
        df
        .groupby(periods)
        .size()
        .reset_index(name="n_rows")
    )
    
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

    return int((df.nunique(dropna=dropna) == 1).sum())


def count_cells_with_missing(df: pd.DataFrame) -> int:
    """
    Count the number of missing cells in a DataFrame.

    Missing cells are detected using pandas missing-value rules an empty or
    whitespace-only string in an ``object`` or ``string`` column.

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
    - ``None``
    - ``NaN`` / ``np.nan``
    - ``pd.NaT``
    - ``pd.NA``
    - Empty strings or whitespace-only strings in ``object`` or ``string`` 
      columns.
    """

    # Total de celdas en todo el Dataframe que pandas considera valores faltantes ("None", "NaN/np.nan", "pd.NaT", "pd.NA")
    missing_values = df.isna().sum().sum()

    # Total de celdas con strings y objects vacíos o en blanco
    missing_empty_strings = _get_empty_string_mask(df).sum().sum()

    # Devolvemos el total de celdas
    return int(missing_values + missing_empty_strings)


def count_rows_with_missing(df: pd.DataFrame) -> int:
    """
    Count the number of rows containing missing values.

    A row is considered to contain missing values when at least one cell
    contains a pandas-recognized missing value or an empty or
    whitespace-only string in an ``object`` or ``string`` column.

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
    - ``None``
    - ``NaN`` / ``np.nan``
    - ``pd.NaT``
    - ``pd.NA``
    - Empty strings or whitespace-only strings in ``object`` or ``string`` 
      columns.
    """

    # Total de filas que pandas considera valores faltantes ("None", "NaN/np.nan", "pd.NaT", "pd.NA")
    missing_values = df.isna().any(axis=1)

    # Total de filas con strings y objects vacíos o en blanco
    missing_empty_strings = _get_empty_string_mask(df).any(axis=1)

    # Combina ambos
    total_missing_rows = int((missing_values | missing_empty_strings).sum())
    
    return total_missing_rows


def count_columns_with_missing(df: pd.DataFrame) -> int:
    """
    Count the number of columns containing missing values.

    A column is considered to contain missing values when at least one cell
    contains a pandas-recognized missing value or an empty or
    whitespace-only string in an ``object`` or ``string`` column.

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
    - ``None``
    - ``NaN`` / ``np.nan``
    - ``pd.NaT``
    - ``pd.NA``
    - Empty strings or whitespace-only strings in ``object`` or ``string``
      columns.
    """

    # Total de columnas que pandas considera valores faltantes ("None", "NaN/np.nan", "pd.NaT", "pd.NA")
    missing_values = df.isna().any(axis=0)

    # Total de columnas con strings y objects vacíos o en blanco
    missing_empty_strings = _get_empty_string_mask(df).any(axis=0)

    total_missing_columns = int((missing_values | missing_empty_strings).sum())

    return total_missing_columns


def count_missing_cells_by_column(df: pd.DataFrame) -> pd.Series:
    """
    Count missing cells for each DataFrame column.

    Missing values are detected using pandas missing-value rules and empty
    or whitespace-only strings in ``object`` or ``string`` columns.

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
    - ``None``
    - ``NaN`` / ``np.nan``
    - ``pd.NaT``
    - ``pd.NA``
    - Empty strings or whitespace-only strings in ``object`` or ``string``
      columns.
    """

    # Total de celdas por columna que pandas considera valores faltantes ("None", "NaN/np.nan", "pd.NaT", "pd.NA")
    missing_values = df.isna().sum()

    # Total de celdas por columna con strings y objects vacíos o en blanco
    missing_empty_strings = _get_empty_string_mask(df).sum()

    # Add para alinear por nombre de columna y poner cero donde no hay máscara de strings
    return missing_values.add(
        missing_empty_strings,
        fill_value=0,
    ).astype(int)


# endregion Counters Functions -------------------------------------------------

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------