from pathlib import Path

import pandas as pd

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Aux Functions ---------------------------------------------------------


# region CSV Functions ---------------------------------------------------------

def import_csv(
    file_path: Path,
    preview_df: bool = False,
    **read_csv_kwargs,
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

    **read_csv_kwargs
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

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Import the DataFrame from CSV.
        df = pd.read_csv(file_path, **read_csv_kwargs)

    except FileNotFoundError as e:
        print(f"File error: {e}")
        raise

    except Exception as e:
        print(f"An unexpected error occurred while importing the file: {e}")
        raise

    if preview_df:
        try:
            from ds_utils.eda.preview_utilities import preview_dataframe
            from ds_utils.eda.preview_config import Alignment

        except ImportError as e:
            raise ImportError(
                "Unable to display the DataFrame preview. "
                "This feature requires the 'ds_utils.eda' module "
                "and its dependencies. Please check that they are "
                "installed and available."
            ) from e

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

    return df




def export_csv(
    df: pd.DataFrame,
    file_path: Path,
    **to_csv_kwargs,
) -> None:
    """
    Export a DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.

    file_path : Path
        Destination path for the CSV file.

    **to_csv_kwargs
        Additional keyword arguments passed directly to
        pandas.DataFrame.to_csv().

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the destination directory cannot be created or the file
        cannot be written.
    Exception
        If an unexpected error occurs during the export.
    """

    try:
        # Create the destination directory if it does not exist.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the DataFrame to CSV.
        df.to_csv(file_path, **to_csv_kwargs)

    except OSError as e:
        print(f"File error while exporting the DataFrame: {e}")
        raise

    except Exception as e:
        print(f"An unexpected error occurred while exporting the DataFrame: {e}")
        raise


# endregion CSV Functions ------------------------------------------------------