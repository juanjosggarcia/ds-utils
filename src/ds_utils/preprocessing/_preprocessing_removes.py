from ds_utils.preprocessing.preprocessing_config import FEATURES_CONFIG

import numpy as np
import pandas as pd

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------

# region Remove Functions ------------------------------------------------------


def remove_redundant_sensor_noise_columns(
    df: pd.DataFrame,
    tolerances: dict[str, float],
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

    tolerances : dict[str, float]
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
        Whether to exclude missing values when counting unique values.

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

    replacement_prefix : str or None, default=None
        Prefix used to identify replacement columns.
        Example: ``"clean_"`` matches ``clean_temperature``.

    replacement_suffix : str or None, default=None
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

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------