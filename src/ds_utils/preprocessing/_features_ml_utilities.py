import numpy as np
import pandas as pd
from typing import Collection, Literal

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------

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


# endregion ML Functions -------------------------------------------------------

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------