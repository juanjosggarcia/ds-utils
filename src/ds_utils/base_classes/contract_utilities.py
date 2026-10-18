import keyword
from dataclasses import dataclass

import pandas as pd

from ds_utils.base_classes.dataset_contract import DatasetContract
from ds_utils.base_classes.contract_config import CONTRACT_CONFIG


def _get_column_mapping(
    df: pd.DataFrame,
    index: str | None = None,
) -> dict[str, str]:
    """
    Build a mapping between contract constant names and DataFrame columns.

    Column names are converted into uppercase Python constant names using
    the configured column prefix. The designated index column, if provided,
    is mapped to the configured index constant.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be mapped.

    index : str or None, default=None
        Optional column to designate as the dataset index.

    Returns
    -------
    dict[str, str]
        Mapping from contract constant names to their corresponding
        DataFrame column names, preserving the original column order.

    Raises
    ------
    ValueError
        If the specified index column does not exist, a column name cannot
        be converted into a valid Python identifier, or multiple columns
        generate the same constant name.

    Notes
    -----
    The names of generated column constants depend on the naming
    conventions defined in ``CONTRACT_CONFIG``.
    """
    if index is not None and index not in df.columns:
        raise ValueError(f"Column '{index}' does not exist in the DataFrame.")

    mapping = {}

    for col in df.columns:
        if col == index:
            constant_name = CONTRACT_CONFIG.INDEX_CONSTANT
        else:
            constant_name = f"{CONTRACT_CONFIG.COLUMN_PREFIX}{col.upper()}"

        if not constant_name.isidentifier() or keyword.iskeyword(constant_name):
            raise ValueError(
                f"Column '{col}' cannot be converted into a valid "
                f"Python constant name: '{constant_name}'."
            )

        if constant_name in mapping:
            raise ValueError(
                f"Multiple columns generate the same contract constant "
                f"'{constant_name}'. Check the DataFrame column names."
            )

        mapping[constant_name] = col

    return mapping


def _generate_base_columns(
    mapping: dict[str, str],
    index: str | None = None,
) -> list[str]:
    """
    Generate the base dataset columns section for a contract.

    The generated section contains the optional index constant and the
    constants representing the original dataset columns. Region markers
    and comments are included to produce a ready-to-use contract definition.

    Parameters
    ----------
    mapping : dict[str, str]
        Mapping between contract constant names and DataFrame column names,
        preserving the original column order.

    index : str or None, default=None
        Optional column designated as the dataset index. Its constant is
        generated separately from the raw column constants.

    Returns
    -------
    list[str]
        Lines defining the base dataset columns section, including region
        markers, comments, and constant assignments.

    Notes
    -----
    The index constant name is defined by ``CONTRACT_CONFIG``.
    """
    lines = [
        "# region BASE DATASET COLUMNS ----------------------------------------------",
        "",
    ]

    if index is not None:
        lines.extend([
            "# --- Index ---",
            f"{CONTRACT_CONFIG.INDEX_CONSTANT} = {index!r}",
            "",
        ])

    lines.append("# --- RAW Columns ---")

    for constant_name, col in mapping.items():
        if col != index:
            lines.append(f"{constant_name} = {col!r}")

    lines.extend([
        "",
        "# endregion BASE DATASET COLUMNS -------------------------------------------",
    ])

    return lines


def _generate_column_groups(
    mapping: dict[str, str],
    index: str | None = None,
) -> list[str]:
    """
    Generate the column groups section for a dataset contract.

    The generated section defines the data, required, and used column
    groups. The designated index is excluded from the data columns and
    included in the required columns when provided.

    Parameters
    ----------
    mapping : dict[str, str]
        Mapping between contract constant names and DataFrame column names,
        preserving the original column order.

    index : str or None, default=None
        Optional column designated as the dataset index.

    Returns
    -------
    list[str]
        Lines defining the column groups section, including list
        declarations, documentation strings, comments, and region markers.

    Notes
    -----
    The names of the generated column groups depend on the naming
    conventions defined in ``CONTRACT_CONFIG``.
    """
    data_columns = [
        constant_name
        for constant_name, col in mapping.items()
        if col != index
    ]

    lines = [
        "# region COLUMN GROUPS -----------------------------------------------------",
        "",
        "# --- Original data columns list ---",
        f"{CONTRACT_CONFIG.DATA_COLUMNS} = [",
        *[f"    {constant_name}," for constant_name in data_columns],
        "]",
        "'''Core data columns from the source dataset.'''",
        "",
        "# --- Required columns list ---",
        f"{CONTRACT_CONFIG.REQUIRED_COLUMNS} = [",
    ]

    if index is not None:
        lines.append(f"    {CONTRACT_CONFIG.INDEX_CONSTANT},")

    lines.append(f"    *{CONTRACT_CONFIG.DATA_COLUMNS},")

    lines.extend([
        "]",
        "'''Columns that must exist in the input dataset. Their existence must be validated.'''",
        "",
        "# --- Used columns list ---",
        f"{CONTRACT_CONFIG.USED_COLUMNS} = [",
        f"    *{CONTRACT_CONFIG.DATA_COLUMNS}",
        "]",
        "'''Columns processed throughout the pipeline.'''",
        "",
        "# endregion COLUMN GROUPS --------------------------------------------------",
    ])

    return lines


def generate_contract_columns(
    df: pd.DataFrame,
    index: str | None = None,
) -> None:
    """
    Generate and print the complete column definition block for a contract.

    The output combines the base dataset columns section and the column
    groups section into a single block of Python code. The generated code
    can be copied directly into a DatasetContract subclass.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be used to generate the contract
        constants and column groups.

    index : str or None, default=None
        Optional column to designate as the dataset index. When provided,
        its constant is generated separately, excluded from DATA_COLUMNS,
        and included in REQUIRED_COLUMNS.

    Returns
    -------
    None
        Prints the generated Python code to standard output.

    Raises
    ------
    ValueError
        If the specified index column does not exist, a column name cannot
        be converted into a valid Python identifier, or multiple columns
        generate the same constant name.

    Notes
    -----
    The generated constants and column group names follow the naming
    conventions defined in ``CONTRACT_CONFIG``.
    """
    mapping = _get_column_mapping(df, index)

    lines = [
        *_generate_base_columns(mapping, index),
        "",
        *_generate_column_groups(mapping, index),
    ]

    print("\n".join(lines))




def create_contract_from_dataframe(
    df: pd.DataFrame,
    class_name: str = "DataFrameContract",
    index: str | None = None,
) -> DatasetContract:
    """
    Create a DatasetContract instance dynamically from a DataFrame.

    All DataFrame columns are treated as required. This utility is intended
    for initial data exploration and rapid prototyping, allowing data
    scientists to create a contract without manually defining a subclass.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame used to infer the contract columns.

    class_name : str, default="DataFrameContract"
        Name of the dynamically generated contract class.

    index : str or None, default=None
        Optional column to designate as the dataset index.

    Returns
    -------
    DatasetContract
        Instance of the dynamically generated contract class.

    Raises
    ------
    ValueError
        If the class name is not a valid Python identifier, the specified
        index column does not exist, a column name cannot be converted into
        a valid Python identifier, or multiple columns generate the same
        constant name.

    Notes
    -----
    The contract schema is inferred from the provided DataFrame. Therefore,
    this function should not be used to define the canonical contract for
    production pipelines, where the expected schema must be established
    independently of the incoming data.

    For production use, define an explicit DatasetContract subclass with
    the expected columns. This ensures that unexpected changes in the
    input schema are detected at the pipeline entry point instead of
    silently becoming part of the contract.

    The generated column constant names follow the naming conventions
    defined in ``CONTRACT_CONFIG``.
    """
    if not class_name.isidentifier() or keyword.iskeyword(class_name):
        raise ValueError(f"'{class_name}' is not a valid Python class name.")

    mapping = _get_column_mapping(df, index)

    attributes = {
        "__module__": __name__,
        "REQUIRED_COLUMNS": list(df.columns),
        **mapping,
    }

    contract_class = dataclass(frozen=True)(
        type(class_name, (DatasetContract,), attributes)
    )

    return contract_class()
