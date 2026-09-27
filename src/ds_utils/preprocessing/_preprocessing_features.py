import numpy as np
import pandas as pd
from typing import Literal
from pandas.tseries.frequencies import to_offset

from ds_utils.preprocessing._preprocessing_others import convert_timestamps_vectorized, convert_timestamp

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------

# region Features Functions ----------------------------------------------------

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
    df[col_time] = convert_timestamps_vectorized(df[col_time], utc=True)

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

    The function inserts missing timestamps according to the requested
    frequency without applying any aggregation to the original observations.
    Missing values introduced by resampling can optionally be filled using
    forward or backward filling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    col_time : str
        Name of the datetime column.

    freq : str
        Target sampling frequency, expressed as a pandas frequency alias.
        Examples include ``"30s"``, ``"5min"``, ``"1h"``, ``"1D"``, and
        ``"1W"``.

    alignment : {"preserve", "round"}, default="round"
        Strategy used to align timestamps before resampling.

        - ``"preserve"``
          Preserve the original timestamp offset. The first timestamp is
          used as the origin of the resampling grid, so regularly spaced
          timestamps keep their original offset.

          Example::

              Original: 10:01, 10:06, 10:11
              Result:   10:01, 10:06, 10:11

          This mode assumes that the input timestamps are already regularly
          spaced. If they are irregular, observations that do not match the
          generated time grid may be omitted during resampling.

        - ``"round"``
          Round each timestamp to the nearest multiple of ``freq`` before
          resampling. This replaces the original timestamp values with their
          rounded values. If multiple timestamps become equal after rounding,
          the first non-null value for each column is retained.

          This mode is intended for datasets with small timestamp deviations
          (jitter).

          Example::

              Original:  10:01, 10:06, 10:17
              Rounded:   10:00, 10:05, 10:15
              Resampled: 10:00, 10:05, 10:10, 10:15

          The ``10:10`` timestamp is introduced by the resampling step; it
          is not created by the rounding operation.

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

    Raises
    ------
    ValueError
        If ``alignment`` or ``fill_method`` is invalid, if ``freq`` is not
        a valid pandas frequency, or if the input contains duplicate
        timestamps before alignment.

    Notes
    -----
    When ``alignment="round"``, timestamps that collide after rounding are
    grouped together and resolved using ``DataFrameGroupBy.first()``.
    """
    
    if alignment not in ("preserve", "round"):
        raise ValueError("alignment must be 'preserve' or 'round'")

    if fill_method not in ("ffill", "bfill", None):
        raise ValueError("fill_method must be 'ffill', 'bfill' or None")

    # Validar frecuencia
    try:
        to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid freq '{freq}'") from exc

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


    # Rellenar si se solicita
    if fill_method is not None:
        df_resampled = getattr(df_resampled, fill_method)(limit = fill_limit)

    df_resampled = df_resampled.reset_index()
    return df_resampled

# endregion Features Functions -------------------------------------------------

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------