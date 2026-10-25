import json
from typing import Any
from pathlib import Path

import pandas as pd

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def _validate_input_file(file_path: Path) -> None:
    """
    Validate that the input file exists and is a regular file.

    Parameters
    ----------
    file_path : Path
        Path to the input file.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    IsADirectoryError
        If the specified path is a directory.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise IsADirectoryError(
            f"The specified path is not a file: {file_path}"
        )


def _prepare_output_file(
    file_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Prepare the destination path for file export.

    Parameters
    ----------
    file_path : Path
        Destination path.
        
    overwrite : bool, default=False
        Whether to allow overwriting an existing file.

    Raises
    ------
    FileExistsError
        If the destination exists and overwrite is False.
    """
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Cannot export file: '{file_path}' already exists. "
            "Set overwrite=True to overwrite it."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)


def _preview_imported_dataframe(
    df: pd.DataFrame,
    file_path: Path,
) -> None:
    """
    Display a preview of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to preview.
    file_path : Path
        Source file path, used in the preview header.

    Raises
    ------
    ImportError
        If the optional EDA module or its dependencies are unavailable.
    """
    try:
        from ds_utils.eda.preview_utilities import preview_dataframe
        from ds_utils.eda.preview_config import Alignment

    except ImportError as exc:
        raise ImportError(
            "DataFrame preview requires the optional EDA module "
            "and its dependencies."
        ) from exc

    preview_dataframe(
        df=df,
        header=(
            f"Preview of imported dataset '{file_path.name}'. "
            f"Shape: {df.shape[0]} rows and {df.shape[1]} columns"
        ),
        header_color=None,
        header_align=Alignment.LEFT,
        rows=5,
    )

# region Aux Functions ---------------------------------------------------------

# region DataFrame Functions ---------------------------------------------------

# region CSV Functions ---------------------------------------------------------

def import_csv_to_df(
    file_path: Path,
    preview_df: bool = False,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Import a CSV file and return its contents as a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file to import.

    preview_df : bool, default=False
        If True, display a preview of the imported DataFrame.
        Requires the optional EDA module and its dependencies.

    **read_csv_kwargs : Any
        Additional keyword arguments passed directly to
        pandas.read_csv().

    Returns
    -------
    pd.DataFrame
        DataFrame containing the imported data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ImportError
        If preview_df is True and the EDA module or one of its
        dependencies cannot be imported.
    Exception
        If an unexpected error occurs while importing the file
        or executing an optional operation.
    """
    _validate_input_file(file_path)

    df = pd.read_csv(
        file_path, 
        **read_csv_kwargs
    )

    if preview_df:
        _preview_imported_dataframe(df, file_path)

    return df



def export_df_to_csv(
    df: pd.DataFrame,
    file_path: Path,
    overwrite: bool = False,
    **to_csv_kwargs: Any,
) -> None:
    """
    Export a DataFrame to a CSV file.

    Existing files are not overwritten unless explicitly allowed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.

    file_path : Path
        Destination path for the CSV file.

    overwrite : bool, default=False
        Whether to overwrite the destination file if it already exists.
        If False, a FileExistsError is raised when the file exists.

    **to_csv_kwargs : Any
        Additional keyword arguments passed directly to
        pandas.DataFrame.to_csv().

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the destination file already exists and overwrite is False.

    OSError
        If the destination directory cannot be created or the file
        cannot be written.
    """
    _prepare_output_file(file_path, overwrite=overwrite)

    df.to_csv(
        file_path, 
        **to_csv_kwargs
    )


# endregion CSV Functions ------------------------------------------------------

# region Parquet Functions -----------------------------------------------------

def import_parquet_to_df(
    file_path: Path,
    preview_df: bool = False,
    **read_parquet_kwargs: Any,
) -> pd.DataFrame:
    """
    Import a Parquet file and return its contents as a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the Parquet file to import.

    preview_df : bool, default=False
        If True, display a preview of the imported DataFrame.
        Requires the optional EDA module and its dependencies.

    **read_parquet_kwargs : Any
        Additional keyword arguments passed directly to
        pandas.read_parquet().

    Returns
    -------
    pd.DataFrame
        DataFrame containing the imported data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    IsADirectoryError
        If the specified path is not a file.
    ImportError
        If PyArrow is unavailable or the optional preview dependencies
        cannot be imported.
    """
    _validate_input_file(file_path)

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires PyArrow. "
            "Install it with: pip install pyarrow"
        ) from exc

    df = pd.read_parquet(
        file_path,
        engine="pyarrow",
        **read_parquet_kwargs,
    )

    if preview_df:
        _preview_imported_dataframe(df, file_path)

    return df


def export_df_to_parquet(
    df: pd.DataFrame,
    file_path: Path,
    overwrite: bool = False,
    **to_parquet_kwargs: Any,
) -> None:
    """
    Export a DataFrame to a Parquet file.

    Existing files are not overwritten unless explicitly allowed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.

    file_path : Path
        Destination path for the Parquet file.

    overwrite : bool, default=False
        Whether to overwrite the destination file if it already exists.
        If False, a FileExistsError is raised when the file exists.

    **to_parquet_kwargs : Any
        Additional keyword arguments passed directly to
        pandas.DataFrame.to_parquet().

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the destination already exists and overwrite is False.
    ImportError
        If PyArrow is unavailable.
    OSError
        If the destination directory cannot be created or the file
        cannot be written.
    """
    _prepare_output_file(file_path, overwrite=overwrite)

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires PyArrow. "
            "Install it with: pip install pyarrow"
        ) from exc

    df.to_parquet(
        file_path,
        engine="pyarrow",
        **to_parquet_kwargs,
    )

# endregion Parquet Functions --------------------------------------------------

# endregion DataFrame Functions ------------------------------------------------


# region Python Objects Functions ----------------------------------------------

# region JSON Functions --------------------------------------------------------

def import_json_to_python(file_path: Path) -> Any:
    """
    Import a JSON file and return its contents as a Python object.

    Parameters
    ----------
    file_path : Path
        Path to the JSON file to import.

    Returns
    -------
    Any
        Python object deserialized from the JSON file. Depending on
        the JSON structure, this may be a dictionary, list, string,
        integer, float, boolean, or None.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    IsADirectoryError
        If the specified path is not a file.
    json.JSONDecodeError
        If the file does not contain valid JSON.
    OSError
        If an operating system error occurs while reading the file.
    """
    _validate_input_file(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def export_python_to_json(
    data: Any,
    file_path: Path,
    overwrite: bool = False,
    **dump_json_kwargs,
) -> None:
    """
    Export a Python object to a JSON file.

    Parameters
    ----------
    data : Any
        Python object to serialize. The object must be compatible
        with the JSON serialization rules or a custom encoder must
        be provided through dump_json_kwargs.

    file_path : Path
        Destination path for the JSON file.

    overwrite : bool, default=False
        Whether to overwrite the destination file if it already exists.
        If False, a FileExistsError is raised when the file exists.

    **dump_json_kwargs
        Additional keyword arguments passed directly to json.dump().

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the destination file already exists and overwrite is False.
    TypeError
        If the Python object is not JSON serializable.
    ValueError
        If the data contains a value that cannot be serialized under
        the selected JSON options.
    OSError
        If the destination directory cannot be created or the file
        cannot be written.
    """
    _prepare_output_file(file_path, overwrite=overwrite)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            **dump_json_kwargs,
        )

# endregion JSON Functions -----------------------------------------------------

# endregion Python Objects Functions -------------------------------------------